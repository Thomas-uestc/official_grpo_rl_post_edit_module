# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Image Edit Dataset for training image editing models with parquet data support
Optimized with chunked loading to handle large datasets efficiently
"""

import json
import os
import base64
import io
import threading
import time
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import glob
from collections import OrderedDict
from queue import Queue
import weakref


class ImageEditDataset(Dataset):
    """Dataset for image editing tasks with parquet data support and chunked loading optimization"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        processor,
        max_length: int = 512,
        image_size: int = 1024,
        data_format: str = "parquet",  # "parquet" or "jsonl"
        image_dir: Optional[str] = None,  # For backward compatibility with jsonl
        edited_image_dir: Optional[str] = None,  # For backward compatibility with jsonl
        parquet_pattern: Optional[str] = None,  # Pattern for parquet files, e.g., "*.parquet"
        # Chunked loading parameters
        cache_size: int = 5,  # Number of parquet files to keep in memory
        prefetch_size: int = 3,  # Number of files to prefetch
        enable_prefetch: bool = True,  # Enable background prefetching
    ):
        """
        Initialize the dataset with chunked loading optimization
        
        Args:
            data_path: Path to data file (parquet file or directory with parquet files, or JSONL file)
            tokenizer: Text tokenizer
            processor: Image processor
            max_length: Maximum sequence length
            image_size: Image size for processing
            data_format: Data format ("parquet" or "jsonl")
            image_dir: Directory containing original images (for jsonl format)
            edited_image_dir: Directory containing preliminary edited images (for jsonl format)
            parquet_pattern: Pattern for parquet files when data_path is a directory
            cache_size: Number of parquet files to keep in memory cache
            prefetch_size: Number of files to prefetch in background
            enable_prefetch: Enable background prefetching
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.data_format = data_format
        self.image_dir = image_dir
        self.edited_image_dir = edited_image_dir
        self.parquet_pattern = parquet_pattern or "*.parquet"
        
        # Chunked loading parameters
        self.cache_size = cache_size
        self.prefetch_size = prefetch_size
        self.enable_prefetch = enable_prefetch
        
        # Initialize chunked loading components
        self.file_index = []  # File index for efficient lookup
        self.file_cache = OrderedDict()  # LRU cache for loaded files
        self.image_cache = {}  # Cache for decoded images
        self.prefetch_queue = Queue(maxsize=prefetch_size) if enable_prefetch else None
        self.prefetch_thread = None
        self.stop_prefetch = False
        self.cache_lock = threading.Lock()
        
        # Load file index (lightweight operation)
        self._build_file_index()
        
        # Start prefetch thread if enabled
        if self.enable_prefetch:
            self._start_prefetch_thread()
        
        print(f"[ImageEditDataset] ✅ Initialized with {len(self.file_index)} files, "
              f"cache_size={cache_size}, prefetch_size={prefetch_size}")
        print(f"[ImageEditDataset] 📊 Memory optimization: LRU cache + background prefetch enabled")
        
    def _build_file_index(self):
        """Build lightweight file index for efficient data access"""
        if self.data_format == "parquet":
            self._build_parquet_file_index()
        else:
            self._build_jsonl_file_index()
    
    def _build_parquet_file_index(self):
        """Build index for parquet files"""
        if os.path.isfile(self.data_path):
            parquet_files = [self.data_path]
        elif os.path.isdir(self.data_path):
            pattern = os.path.join(self.data_path, self.parquet_pattern)
            parquet_files = glob.glob(pattern)
            parquet_files.sort()  # Ensure consistent ordering
        else:
            raise ValueError(f"Invalid data_path: {self.data_path}")
        
        print(f"[ImageEditDataset] 🔍 Building index for {len(parquet_files)} parquet files...")
        
        current_idx = 0
        for i, parquet_file in enumerate(parquet_files):
            try:
                # Only read metadata to get row count (very fast)
                df_info = pd.read_parquet(parquet_file, columns=['omni_edit_id'])
                file_info = {
                    'file_path': parquet_file,
                    'start_idx': current_idx,
                    'end_idx': current_idx + len(df_info),
                    'sample_count': len(df_info),
                    'loaded': False
                }
                self.file_index.append(file_info)
                current_idx += len(df_info)
                print(f"[ImageEditDataset] 📁 [{i+1:2d}/{len(parquet_files)}] {os.path.basename(parquet_file)}: {len(df_info):4d} samples (idx: {current_idx-len(df_info):4d}-{current_idx-1:4d})")
            except Exception as e:
                print(f"[ImageEditDataset] ❌ Error indexing {parquet_file}: {e}")
                continue
        
        print(f"[ImageEditDataset] ✅ Index complete: {current_idx} total samples across {len(self.file_index)} files")
    
    def _build_jsonl_file_index(self):
        """Build index for JSONL file (backward compatibility)"""
        if not os.path.exists(self.data_path):
            raise ValueError(f"JSONL file not found: {self.data_path}")
        
        # Count lines in JSONL file
        with open(self.data_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f if line.strip())
        
        file_info = {
            'file_path': self.data_path,
            'start_idx': 0,
            'end_idx': line_count,
            'sample_count': line_count,
            'loaded': False
        }
        self.file_index.append(file_info)
        print(f"[ImageEditDataset] 📄 Indexed JSONL file: {line_count} samples")
    
    def _find_file_by_index(self, idx: int) -> Dict[str, Any]:
        """Find the file containing the given index"""
        for file_info in self.file_index:
            if file_info['start_idx'] <= idx < file_info['end_idx']:
                return file_info
        raise IndexError(f"Index {idx} out of range")
    
    def _load_file_with_cache(self, file_path: str) -> List[Dict[str, Any]]:
        """Load file data with LRU cache management"""
        with self.cache_lock:
            # Check if file is already in cache
            if file_path in self.file_cache:
                # Move to end (most recently used)
                self.file_cache.move_to_end(file_path)
                print(f"[ImageEditDataset] 🎯 Cache HIT: {os.path.basename(file_path)} "
                      f"(cache: {len(self.file_cache)}/{self.cache_size})")
                return self.file_cache[file_path]
            
            # Load new file
            print(f"[ImageEditDataset] 📥 Loading file: {os.path.basename(file_path)}")
            data = self._load_single_parquet_file(file_path)
            
            # If cache is full, remove least recently used file
            if len(self.file_cache) >= self.cache_size:
                oldest_file, _ = self.file_cache.popitem(last=False)
                print(f"[ImageEditDataset] 🗑️  Evicted from cache: {os.path.basename(oldest_file)}")
            
            # Add to cache
            self.file_cache[file_path] = data
            print(f"[ImageEditDataset] ✅ Added to cache: {os.path.basename(file_path)} "
                  f"(cache: {len(self.file_cache)}/{self.cache_size}, samples: {len(data)})")
            
            return data
    
    def _load_single_parquet_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load a single parquet file and convert to list of dictionaries"""
        try:
            df = pd.read_parquet(file_path)
            data = []
            
            # Convert DataFrame to list of dictionaries
            for _, row in df.iterrows():
                # Parse edit_prompt if it's a string representation of a list
                edit_prompt = row['edit_prompt']
                if isinstance(edit_prompt, str):
                    try:
                        # Try to parse as a list representation
                        import ast
                        edit_prompt = ast.literal_eval(edit_prompt)
                    except:
                        # If parsing fails, treat as single string
                        edit_prompt = [edit_prompt]
                
                # Extract short and detailed instructions
                if isinstance(edit_prompt, list) and len(edit_prompt) >= 2:
                    short_instruction = edit_prompt[0]
                    detailed_instruction = edit_prompt[1]
                else:
                    short_instruction = str(edit_prompt)
                    detailed_instruction = str(edit_prompt)
                
                data.append({
                    'omni_edit_id': row['omni_edit_id'],
                    'task': row['task'],
                    'width': row['width'],
                    'height': row['height'],
                    'src_img_b64': row['src_img_b64'],
                    'edit_prompt': edit_prompt,
                    'short_instruction': short_instruction,
                    'detailed_instruction': detailed_instruction,
                    'qwen_edited_img_b64': row['qwen_edited_img_b64'],
                    'gpu_id': row['gpu_id'],
                    'sample_idx': row['sample_idx']
                })
            
            return data
            
        except Exception as e:
            print(f"[ImageEditDataset] Error loading parquet file {file_path}: {e}")
            return []
    
    def _start_prefetch_thread(self):
        """Start background prefetch thread"""
        def prefetch_worker():
            while not self.stop_prefetch:
                try:
                    # Get next file to prefetch
                    next_file = self._predict_next_file()
                    if next_file and next_file not in self.file_cache:
                        # Load file in background
                        data = self._load_single_parquet_file(next_file)
                        # Add to prefetch queue
                        if not self.prefetch_queue.full():
                            self.prefetch_queue.put((next_file, data))
                            print(f"[ImageEditDataset] 🔮 Prefetched: {os.path.basename(next_file)} "
                                  f"(queue: {self.prefetch_queue.qsize()}/{self.prefetch_size})")
                    else:
                        time.sleep(0.1)  # Short sleep if no work to do
                except Exception as e:
                    print(f"[ImageEditDataset] ⚠️  Prefetch error: {e}")
                    time.sleep(1.0)  # Longer sleep on error
        
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        print(f"[ImageEditDataset] 🚀 Started prefetch thread (prefetch_size={self.prefetch_size})")
    
    def _predict_next_file(self) -> Optional[str]:
        """Predict which file will be needed next (simple heuristic)"""
        # Simple prediction: if we have cached files, predict the next file in sequence
        if len(self.file_cache) > 0:
            cached_files = list(self.file_cache.keys())
            if cached_files:
                last_cached_file = cached_files[-1]
                # Find next file in sequence
                for i, file_info in enumerate(self.file_index):
                    if file_info['file_path'] == last_cached_file and i + 1 < len(self.file_index):
                        return self.file_index[i + 1]['file_path']
        return None
    
    def _decode_image_with_cache(self, base64_str: str) -> Image.Image:
        """Decode image with caching for frequently used images"""
        if base64_str in self.image_cache:
            return self.image_cache[base64_str]
        
        # Decode image
        image = self._decode_base64_image(base64_str)
        
        # Add to cache (simple cache without size limit for now)
        self.image_cache[base64_str] = image
        
        return image
    
    def _decode_base64_image(self, base64_str: str) -> Image.Image:
        """Decode base64 string to PIL Image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_str)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Resize if needed
            if image.size[0] > self.image_size or image.size[1] > self.image_size:
                image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (self.image_size, self.image_size), color='white')
    
    def __del__(self):
        """Cleanup method to stop prefetch thread"""
        self.stop_prefetch = True
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)
    
    def cleanup(self):
        """Explicit cleanup method"""
        self.stop_prefetch = True
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)
        print(f"[ImageEditDataset] 🧹 Cleanup completed - prefetch thread stopped")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics for monitoring"""
        with self.cache_lock:
            return {
                "cache_size": len(self.file_cache),
                "max_cache_size": self.cache_size,
                "cached_files": [os.path.basename(f) for f in self.file_cache.keys()],
                "prefetch_queue_size": self.prefetch_queue.qsize() if self.prefetch_queue else 0,
                "max_prefetch_size": self.prefetch_size,
                "total_files": len(self.file_index),
                "total_samples": self.__len__()
            }
    
    def _find_image_by_index(self, directory: str, idx: int, suffix: str) -> str:
        """Find image file by index in directory (for jsonl format)"""
        try:
            # List all files in directory
            files = os.listdir(directory)
            
            # Sort files to ensure consistent ordering
            files.sort()
            
            # Find file that starts with the index
            target_prefix = f"{idx:05d}"
            for file in files:
                if file.startswith(target_prefix) and file.endswith(suffix):
                    return os.path.join(directory, file)
            
            # If not found, try alternative naming patterns
            # For edited images, they might have double prefix like "00000_00000_..."
            if suffix == "_qie.png":
                for file in files:
                    if f"{target_prefix}_{target_prefix}" in file and file.endswith(suffix):
                        return os.path.join(directory, file)
            
            # Fallback: use the sorted file at the given index
            if idx < len(files):
                return os.path.join(directory, files[idx])
            
            raise FileNotFoundError(f"No image found for index {idx} in {directory}")
            
        except Exception as e:
            print(f"Error finding image for index {idx} in {directory}: {e}")
            # Return a dummy path that will be handled by _load_image
            return os.path.join(directory, f"{idx:05d}{suffix}")

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image from file path (for jsonl format)"""
        try:
            image = Image.open(image_path).convert('RGB')
            # Resize if needed
            if image.size[0] > self.image_size or image.size[1] > self.image_size:
                image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (self.image_size, self.image_size), color='white')
    
    def _create_prompt(self, edit_instruction: str, num_images: int = 2) -> str:
        """Create prompt for the model with CoT and re-edit instruction generation"""
        # System prompt for CoT and re-edit instruction generation
        system_prompt = (
        "You are a helpful assistant for visual thinking, design, and editing. "
        "Given a source image, an editing instruction, and the resulting edited image, do TWO tasks: "
        "1) Produce up to two short reasoning lines (≤30 words each) covering: "
        "(a) instruction compliance & subject integrity, "
        "(b) visual realism (geometry, lighting, physics), "
        "(c) contextual consistency (scene logic, attribute coherence), "
        "(d) environmental consistency (e.g., sunny sky but wet ground), "
        "(e) cultural/traditional consistency (e.g., Japanese wedding with Western dress). "
        "If no major issues, use these lines to briefly confirm compliance/realism. One sentence per line. "
        "2) Produce exactly one concise, actionable re-editing instruction that explicitly states a clear modification action (e.g., change, adjust, replace, enhance, remove). "
        "OUTPUT FORMAT (STRICT): "
        "Each block MUST be on its own separate line with no other content. Example:"
        "<CoT>The image successfully follows the instruction and maintains subject integrity.</CoT>\n"
        "<CoT>The lighting and shadows appear realistic and physically consistent.</CoT>\n"
        "<Re_edit>Adjust the brightness of the background to better match the foreground lighting.</Re_edit>\n"
        "RULES: "
        "- Output ONLY the tag blocks, each on its own line "
        "- Use exactly this format: <CoT>content</CoT> and <Re_edit>content</Re_edit> "
        "- Maximum 2 CoT blocks, minimum 1 CoT block "
        "- Exactly 1 Re_edit block "
        "- Each line ≤30 words "
        "- Use imperative voice in Re_edit (e.g., 'Change...', 'Remove...', 'Adjust...') "
        "- CRITICAL: Each tag must start on a new line with NO preceding text "
        "- CRITICAL: Each tag must end on its line with NO following text "
        "REMEMBER: Output ONLY the tags, each on its own line, nothing else! "
        "- No JSON, code fences, explanations, or extra text "
        "- Do not name categories; just write the reasoning content "
        )
        
        # User text with editing instruction
        user_text = (
            f"<desired_editing_instruction>{edit_instruction}</desired_editing_instruction>\n"
            "Return reasoning and one re-editing instruction as specified."
        )
        
        # Add image placeholders at the end of the text
        # This ensures the number of <image> tokens matches the number of images
        for _ in range(num_images):
            user_text += "<image>"
        
        # Combine system prompt and user text
        prompt = f"System: {system_prompt}\n\nUser: {user_text}"
        return prompt
    
    def __len__(self) -> int:
        """Return total number of samples across all files"""
        if self.file_index:
            return self.file_index[-1]['end_idx']
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset using chunked loading"""
        # Find the file containing this index
        file_info = self._find_file_by_index(idx)
        file_path = file_info['file_path']
        
        # Load file data with cache
        file_data = self._load_file_with_cache(file_path)
        
        # Get the specific item from the loaded file
        local_idx = idx - file_info['start_idx']
        item = file_data[local_idx]
        
        # Debug info for data access pattern
        if idx % 100 == 0:  # Print every 100th access to avoid spam
            print(f"[ImageEditDataset] 📊 Accessing sample {idx} from {os.path.basename(file_path)} "
                  f"(local_idx: {local_idx}, cache: {len(self.file_cache)}/{self.cache_size})")
        
        # Process the item based on data format
        if self.data_format == "parquet":
            return self._process_parquet_item(item)
        else:
            return self._process_jsonl_item(item, idx)
    
    def _process_parquet_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a parquet item with proper image placeholder handling"""
        # Use the detailed instruction as the edit instruction
        edit_instruction = item["detailed_instruction"]
        original_description = f"Image editing task: {item['task']}"
        
        # Create prompt without image placeholders
        prompt = self._create_prompt(edit_instruction, num_images=0)  # No image placeholders in prompt
        
        # Process images and text together for multimodal input
        try:
            # Build structured messages by adding image placeholders at the end
            # text_content itself doesn't contain <image> placeholders
            text_content = prompt.strip()
            
            # Build content list: text first, then image placeholders
            content_list = [{"type": "text", "text": text_content}]
            
            # Add image placeholders at the end
            for _ in range(2):  # Add 2 image placeholders
                content_list.append({"type": "image"})
            
            messages = [
                {
                    "role": "user",
                    "content": content_list
                }
            ]
            
            # Use processor's apply_chat_template to generate proper prompt
            formatted_prompt = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Decode images and process together (simplified - no lazy decode)
            original_image = self._decode_image_with_cache(item["src_img_b64"])
            edited_image = self._decode_image_with_cache(item["qwen_edited_img_b64"])
            
            # Process images and text together
            processed_inputs = self.processor(
                text=[formatted_prompt],
                images=[original_image, edited_image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = processed_inputs["input_ids"].squeeze(0)
            attention_mask = processed_inputs["attention_mask"].squeeze(0)
            
            # Generate raw_prompt_ids
            raw_prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
            
            # Extract position_ids if available
            if "position_ids" in processed_inputs:
                position_ids = processed_inputs["position_ids"].squeeze(0)
            else:
                # Fallback: generate position_ids manually
                if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                    from verl.models.transformers.qwen2_vl import get_rope_index
                    try:
                        position_ids = get_rope_index(
                            self.processor,
                            input_ids=input_ids,
                            image_grid_thw=processed_inputs.get("image_grid_thw"),
                            video_grid_thw=None,
                            second_per_grid_ts=None,
                            attention_mask=attention_mask,
                        )
                    except Exception as e:
                        print(f"Warning: Failed to generate Qwen2-VL position_ids: {e}")
                        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
                        position_ids = position_ids.unsqueeze(0).repeat(3, 1)
                else:
                    position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
                    position_ids = position_ids.unsqueeze(0)
                    
        except Exception as e:
            print(f"Error processing multimodal input: {e}")
            # Fallback to text-only processing
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor(prompt_tokens, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            
            # Generate position_ids based on processor type
            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                from verl.models.transformers.qwen2_vl import get_rope_index
                try:
                    position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids,
                        image_grid_thw=None,
                        video_grid_thw=None,
                        second_per_grid_ts=None,
                        attention_mask=attention_mask,
                    )
                except Exception as e:
                    print(f"Warning: Failed to generate Qwen2-VL position_ids: {e}")
                    position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
                    position_ids = position_ids.unsqueeze(0).repeat(3, 1)
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
                position_ids = position_ids.unsqueeze(0)
        
        # Pad to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            if position_ids.shape[0] == 3:  # Qwen2-VL format (3, seq_length)
                position_ids = position_ids[:, :self.max_length]
            else:  # Standard format (1, seq_length)
                position_ids = position_ids[:, :self.max_length]
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            
            # Pad position_ids
            if position_ids.shape[0] == 3:  # Qwen2-VL format (3, seq_length)
                pad_pos_ids = torch.zeros(3, padding_length, dtype=position_ids.dtype)
                position_ids = torch.cat([position_ids, pad_pos_ids], dim=-1)
            else:  # Standard format (1, seq_length)
                pad_pos_ids = torch.zeros(1, padding_length, dtype=position_ids.dtype)
                position_ids = torch.cat([position_ids, pad_pos_ids], dim=-1)
        
        # Prepare return data
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "edit_instructions": edit_instruction,
            "expected_reasoning": item.get("reasoning", ""),
            "original_descriptions": original_description,
            "omni_edit_id": item.get("omni_edit_id", ""),
            "task": item.get("task", ""),
            "short_instruction": item.get("short_instruction", edit_instruction),
        }
        
        # Return decoded images for training compatibility
        result.update({
            "original_images": original_image,
            "preliminary_edited_images": edited_image,
            "multi_modal_data": {
                "images": [original_image, edited_image]
            }
        })
        
        return result
    
    def _process_jsonl_item(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Process a JSONL item (backward compatibility)"""
        # Load images from file paths
        original_image_path = self._find_image_by_index(self.image_dir, idx, ".png")
        edited_image_path = self._find_image_by_index(self.edited_image_dir, idx, "_qie.png")
        
        original_image = self._load_image(original_image_path)
        edited_image = self._load_image(edited_image_path)
        
        edit_instruction = item["edit_instruction"]
        original_description = item["original_description"]
        
        # Create prompt without image placeholders
        prompt = self._create_prompt(edit_instruction, num_images=0)  # No image placeholders in prompt
        
        # Process images and text together for multimodal input
        try:
            # Build structured messages by adding image placeholders at the end
            # text_content itself doesn't contain <image> placeholders
            text_content = prompt.strip()
            
            # Build content list: text first, then image placeholders
            content_list = [{"type": "text", "text": text_content}]
            
            # Add image placeholders at the end
            for _ in range(2):  # Add 2 image placeholders
                content_list.append({"type": "image"})
            
            messages = [
                {
                    "role": "user",
                    "content": content_list
                }
            ]
            
            # Use processor's apply_chat_template to generate proper prompt
            formatted_prompt = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Process images and formatted prompt
            processed_inputs = self.processor(
                text=[formatted_prompt],
                images=[original_image, edited_image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = processed_inputs["input_ids"].squeeze(0)
            attention_mask = processed_inputs["attention_mask"].squeeze(0)
            
            # Generate raw_prompt_ids
            raw_prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
            
            # Extract position_ids if available
            if "position_ids" in processed_inputs:
                position_ids = processed_inputs["position_ids"].squeeze(0)
            else:
                # Fallback: generate position_ids manually
                if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                    from verl.models.transformers.qwen2_vl import get_rope_index
                    try:
                        position_ids = get_rope_index(
                            self.processor,
                            input_ids=input_ids,
                            image_grid_thw=processed_inputs.get("image_grid_thw"),
                            video_grid_thw=None,
                            second_per_grid_ts=None,
                            attention_mask=attention_mask,
                        )
                    except Exception as e:
                        print(f"Warning: Failed to generate Qwen2-VL position_ids: {e}")
                        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
                        position_ids = position_ids.unsqueeze(0).repeat(3, 1)
                else:
                    position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
                    position_ids = position_ids.unsqueeze(0)
                    
        except Exception as e:
            print(f"Error processing multimodal input for item {idx}: {e}")
            # Fallback to text-only processing
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor(prompt_tokens, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            
            # Generate position_ids based on processor type
            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                from verl.models.transformers.qwen2_vl import get_rope_index
                try:
                    position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids,
                        image_grid_thw=None,
                        video_grid_thw=None,
                        second_per_grid_ts=None,
                        attention_mask=attention_mask,
                    )
                except Exception as e:
                    print(f"Warning: Failed to generate Qwen2-VL position_ids: {e}")
                    position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
                    position_ids = position_ids.unsqueeze(0).repeat(3, 1)
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
                position_ids = position_ids.unsqueeze(0)
        
        # Pad to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            if position_ids.shape[0] == 3:  # Qwen2-VL format (3, seq_length)
                position_ids = position_ids[:, :self.max_length]
            else:  # Standard format (1, seq_length)
                position_ids = position_ids[:, :self.max_length]
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            
            # Pad position_ids
            if position_ids.shape[0] == 3:  # Qwen2-VL format (3, seq_length)
                pad_pos_ids = torch.zeros(3, padding_length, dtype=position_ids.dtype)
                position_ids = torch.cat([position_ids, pad_pos_ids], dim=-1)
            else:  # Standard format (1, seq_length)
                pad_pos_ids = torch.zeros(1, padding_length, dtype=position_ids.dtype)
                position_ids = torch.cat([position_ids, pad_pos_ids], dim=-1)
        
        # Return the processed data
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "original_images": original_image,
            "preliminary_edited_images": edited_image,
            "edit_instructions": edit_instruction,
            "expected_reasoning": item.get("reasoning", ""),
            "original_descriptions": original_description,
            "multi_modal_data": {
                "images": [original_image, edited_image]
            },
        }


def create_image_edit_dataloader(
    data_path: str,
    tokenizer,
    processor,
    batch_size: int = 4,
    max_length: int = 512,
    image_size: int = 1024,
    num_workers: int = 4,
    shuffle: bool = True,
    data_format: str = "parquet",
    image_dir: Optional[str] = None,
    edited_image_dir: Optional[str] = None,
    parquet_pattern: Optional[str] = None,
    # Chunked loading parameters
    cache_size: int = 5,
    prefetch_size: int = 3,
    enable_prefetch: bool = True,
):
    """Create dataloader for image editing dataset with chunked loading optimization"""
    dataset = ImageEditDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,
        image_size=image_size,
        data_format=data_format,
        image_dir=image_dir,
        edited_image_dir=edited_image_dir,
        parquet_pattern=parquet_pattern,
        # Chunked loading parameters
        cache_size=cache_size,
        prefetch_size=prefetch_size,
        enable_prefetch=enable_prefetch,
    )
    
    def collate_fn(batch):
        """Custom collate function for image editing data"""
        # Separate tensor and non-tensor data
        tensor_data = {}
        non_tensor_data = {}
        
        # Collect tensor data
        tensor_keys = ["input_ids", "attention_mask", "position_ids"]
        for key in tensor_keys:
            tensor_data[key] = torch.stack([item[key] for item in batch])
        
        # Collect non-tensor data and convert to numpy arrays
        non_tensor_keys = [
            "raw_prompt_ids", "original_images", "preliminary_edited_images", 
            "edit_instructions", "expected_reasoning", 
            "original_descriptions", "multi_modal_data",
            "omni_edit_id", "task", "short_instruction"
        ]
        
        for key in non_tensor_keys:
            if key in batch[0]:  # Check if key exists in first item
                # Convert list to numpy array for DataProto compatibility
                non_tensor_data[key] = np.array([item[key] for item in batch], dtype=object)
        
        return {
            **tensor_data,
            **non_tensor_data
        }
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Add cache monitoring wrapper
    class CacheMonitoringWrapper:
        def __init__(self, dataloader, dataset):
            self.dataloader = dataloader
            self.dataset = dataset
            self.batch_count = 0
        
        def __iter__(self):
            for batch in self.dataloader:
                self.batch_count += 1
                # Print cache stats every 10 batches
                if self.batch_count % 10 == 0:
                    stats = self.dataset.get_cache_stats()
                    print(f"[ImageEditDataset] 📈 Batch {self.batch_count}: "
                          f"Cache {stats['cache_size']}/{stats['max_cache_size']}, "
                          f"Prefetch queue: {stats['prefetch_queue_size']}/{stats['max_prefetch_size']}")
                yield batch
        
        def __len__(self):
            return len(self.dataloader)
    
    return CacheMonitoringWrapper(dataloader, dataset)