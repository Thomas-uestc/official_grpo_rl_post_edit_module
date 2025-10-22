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
Qwen-Image-Edit Worker implementation with FSDP support
"""

import os
import torch
import torch.distributed as dist
from PIL import Image
from diffusers import QwenImageEditPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Optional, Dict, Any
import numpy as np
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from accelerate import init_empty_weights
from codetiming import Timer

from .config import ImageEditConfig
from ...single_controller.base.decorator import Dispatch, register
from ...single_controller.base import Worker
from ...protocol import DataProto
from ...utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_model, load_fsdp_model
from ...utils.model_utils import print_model_size
from ...utils.torch_dtypes import PrecisionType


class QwenImageEditWorker(Worker):
    """Qwen-Image-Edit model worker implementation with FSDP support"""
    
    def __init__(self, config: ImageEditConfig, role: str = "image_edit"):
        print(f"[DEBUG] QwenImageEditWorker.__init__() called with role: {role}")
        
        try:
            print("[DEBUG] Calling super().__init__()...")
            super().__init__()
            print("[DEBUG] super().__init__() completed")
            
            print("[DEBUG] Setting instance variables...")
            self.config = config
            self.role = role
            self.pipeline = None
            self.fsdp_module = None
            self.device = torch.device(config.device)
            self._use_param_offload = config.enable_cpu_offload
            print(f"[DEBUG] Instance variables set - device: {self.device}")
            
            # Initialize distributed if not already done
            print("[DEBUG] Checking distributed initialization...")
            if not dist.is_initialized():
                print("[DEBUG] Initializing distributed process group...")
                # Use environment variables set by Ray
                master_addr = os.environ.get("MASTER_ADDR", "localhost")
                master_port = os.environ.get("MASTER_PORT", "12355")
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
                rank = int(os.environ.get("RANK", "0"))
                
                print(f"[DEBUG] Distributed config - MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}")
                print(f"[DEBUG] Distributed config - WORLD_SIZE: {world_size}, RANK: {rank}")
                
                dist.init_process_group(
                    backend="nccl",
                    init_method=f"tcp://{master_addr}:{master_port}",
                    world_size=world_size,
                    rank=rank
                )
                print("[DEBUG] Distributed process group initialized")
            else:
                print("[DEBUG] Distributed already initialized")
                
            # Improve numerical stability (following original worker pattern)
            print("[DEBUG] Setting CUDA backend options...")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            print("[DEBUG] CUDA backend options set")
            
            print("[DEBUG] QwenImageEditWorker.__init__() completed successfully!")
            
        except Exception as e:
            print(f"[ERROR] Exception in QwenImageEditWorker.__init__(): {type(e).__name__}: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise
            
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize Qwen-Image-Edit pipeline - 简化版本，专用GPU"""
        self.print_rank0(f"[DEBUG] Starting init_model() on {self.device}...")
        
        try:
            # Step 1: Check config
            self.print_rank0(f"[DEBUG] Config check - torch_dtype: {self.config.torch_dtype}")
            
            # Step 2: Determine torch dtype
            self.print_rank0("[DEBUG] Determining torch dtype...")
            if self.config.torch_dtype is None:
                torch_dtype = torch.bfloat16
                self.print_rank0("[DEBUG] Using default torch_dtype: bfloat16")
            else:
                self.print_rank0(f"[DEBUG] Converting torch_dtype: {self.config.torch_dtype}")
                torch_dtype = PrecisionType.to_dtype(self.config.torch_dtype)
                self.print_rank0(f"[DEBUG] Converted torch_dtype: {torch_dtype}")
            
            # Step 3: Check model path
            self.print_rank0(f"[DEBUG] Model path: {self.config.model_path}")
            if not os.path.exists(self.config.model_path):
                raise RuntimeError(f"Model path does not exist: {self.config.model_path}")
            
            # Step 4: Check CUDA availability
            self.print_rank0(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self.print_rank0(f"[DEBUG] CUDA device count: {torch.cuda.device_count()}")
                self.print_rank0(f"[DEBUG] Current CUDA device: {torch.cuda.current_device()}")
            
            # Step 5: Load pipeline
            self.print_rank0("[DEBUG] Starting QwenImageEditPipeline.from_pretrained()...")
            self.pipeline = QwenImageEditPipeline.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
                trust_remote_code=True,
                device_map="cuda"  # 使用diffusers支持的策略
            )
            self.print_rank0("[DEBUG] QwenImageEditPipeline.from_pretrained() completed successfully")
            
            # Step 6: Check pipeline components
            self.print_rank0("[DEBUG] Checking pipeline components...")
            if hasattr(self.pipeline, 'transformer'):
                self.print_rank0("[DEBUG] Pipeline has transformer component")
            if hasattr(self.pipeline, 'vae'):
                self.print_rank0("[DEBUG] Pipeline has VAE component")
            if hasattr(self.pipeline, 'text_encoder'):
                self.print_rank0("[DEBUG] Pipeline has text_encoder component")
            
            # # Step 7: Freeze parameters
            # self.print_rank0("[DEBUG] Freezing pipeline parameters...")
            # param_count = 0
            # for param in self.pipeline.parameters():
            #     param.requires_grad = False
            #     param_count += 1
            # self.print_rank0(f"[DEBUG] Frozen {param_count} parameters")
            
            # Step 8: Check GPU memory
            self.print_rank0("[DEBUG] Checking GPU memory...")
            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info()
                self.print_rank0(f"[DEBUG] GPU memory after model init: {(total_mem - free_mem) / (1024**3):.2f} GB / {total_mem / (1024**3):.2f} GB")
            
            self.print_rank0("[DEBUG] init_model() completed successfully!")
            return True
            
        except Exception as e:
            self.print_rank0(f"[ERROR] Exception in init_model(): {type(e).__name__}: {e}")
            import traceback
            self.print_rank0(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to load Qwen-Image-Edit model: {e}")
        
    def _wrap_with_fsdp(self, model):
        """Wrap model with FSDP for memory efficiency"""
        if not self.config.enable_fsdp:
            return model
            
        # Define mixed precision policy
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(self.config.torch_dtype or "bfloat16"),
            reduce_dtype=PrecisionType.to_dtype(self.config.torch_dtype or "bfloat16"),
            buffer_dtype=PrecisionType.to_dtype(self.config.torch_dtype or "bfloat16"),
        )
        
        # Get auto wrap policy
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}")
        
        # Get sharding strategy
        if self.config.fsdp_sharding_strategy == "FULL_SHARD":
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif self.config.fsdp_sharding_strategy == "SHARD_GRAD_OP":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.NO_SHARD
        
        # CPU offload configuration
        cpu_offload = None
        if self.config.fsdp_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        
        # Sync module states configuration
        sync_module_states = self.config.fsdp_sync_module_states
        
        # Wrap model with FSDP
        fsdp_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=False,
        )
        
        self.print_rank0(f"Model wrapped with FSDP using {self.config.fsdp_sharding_strategy} strategy")
        return fsdp_model
        
    def preprocess_image(self, image) -> Image.Image:
        """Preprocess image for the pipeline using thumbnail strategy to preserve aspect ratio"""
        # 🔍 DEBUG: 简洁的preprocess_image输入检查
        self.print_rank0("=" * 60)
        self.print_rank0("🔍 DEBUG: preprocess_image 输入检查")
        self.print_rank0("=" * 60)
        self.print_rank0(f"[DEBUG] preprocess_image input type: {type(image)}")
        if hasattr(image, '__len__'):
            self.print_rank0(f"[DEBUG] preprocess_image input length: {len(image)}")
        if hasattr(image, 'size'):
            self.print_rank0(f"[DEBUG] preprocess_image input size: {image.size}")
        
        # 安全显示图像类型
        img_type = type(image)
        if img_type.__name__ == 'Image':
            self.print_rank0(f"[DEBUG] preprocess_image input: PIL.Image ({image.mode}, {image.size})")
        elif img_type.__name__ == 'list':
            self.print_rank0(f"[DEBUG] preprocess_image input: LIST (nested!) length: {len(image)}")
            if len(image) > 0:
                nested_type = type(image[0])
                self.print_rank0(f"[DEBUG] nested image type: {nested_type}")
        else:
            self.print_rank0(f"[DEBUG] preprocess_image input: {img_type.__name__} (not PIL.Image!)")
        self.print_rank0("=" * 60)
        
        # Handle potential nested structure - extract actual PIL.Image from list
        if isinstance(image, list):
            if len(image) > 0:
                image = image[0]
                self.print_rank0(f"[DEBUG] Extracted image from list: {type(image)}")
            else:
                raise ValueError("Empty image list in preprocess_image")
        
        # Ensure we have a PIL.Image object
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(image)} in preprocess_image")
        
        # ═══════════════════════════════════════════════════════
        # 新策略：使用 thumbnail() 保持纵横比（仿照 ImageEditDataset）
        # Qwen-Image-Edit Pipeline 支持任意纵横比，无需强制正方形
        # ═══════════════════════════════════════════════════════
        target_size = self.config.image_size
        
        # 使用 thumbnail 缩放（保持纵横比，只缩小不放大）
        if image.size[0] > target_size or image.size[1] > target_size:
            self.print_rank0(f"[DEBUG] Resizing image from {image.size} using thumbnail strategy...")
            image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            self.print_rank0(f"[DEBUG] Image after thumbnail: {image.size} (aspect ratio preserved)")
        else:
            self.print_rank0(f"[DEBUG] Image size {image.size} within limits, no resizing needed")
        
        self.print_rank0(f"[DEBUG] Final preprocessed image size: {image.size}")
        return image
    
    def postprocess_image(self, image: Image.Image) -> Image.Image:
        """Postprocess image after generation"""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    
    def _process_single_image(self, image, instruction: str) -> Image.Image:
         """Process a single image with edit instruction"""
         try:
             self.print_rank0(f"[DEBUG] _process_single_image() called with instruction: {instruction[:50]}...")
             
             # 🔍 DEBUG: 简洁的输入参数检查
             self.print_rank0("=" * 60)
             self.print_rank0("🔍 DEBUG: _process_single_image 输入参数检查")
             self.print_rank0("=" * 60)
             self.print_rank0(f"[DEBUG] Input image type: {type(image)}")
             if hasattr(image, '__len__'):
                 self.print_rank0(f"[DEBUG] Input image length: {len(image)}")
             if hasattr(image, 'size'):
                 self.print_rank0(f"[DEBUG] Input image size: {image.size}")
             
             # 安全显示图像类型
             img_type = type(image)
             if img_type.__name__ == 'Image':
                 self.print_rank0(f"[DEBUG] Input image: PIL.Image ({image.mode}, {image.size})")
             elif img_type.__name__ == 'list':
                 self.print_rank0(f"[DEBUG] Input image: LIST (nested!) length: {len(image)}")
                 if len(image) > 0:
                     nested_type = type(image[0])
                     self.print_rank0(f"[DEBUG] nested image type: {nested_type}")
             else:
                 self.print_rank0(f"[DEBUG] Input image: {img_type.__name__} (not PIL.Image!)")
             
             self.print_rank0(f"[DEBUG] Input instruction type: {type(instruction)}")
             inst_str = str(instruction)
             self.print_rank0(f"[DEBUG] Input instruction: {inst_str[:50]}{'...' if len(inst_str) > 50 else ''}")
             self.print_rank0("=" * 60)
             
             # Handle potential nested structure - extract actual PIL.Image from list
             if isinstance(image, list):
                 if len(image) > 0:
                     self.print_rank0(f"[DEBUG] Extracting image from list: {type(image[0])}")
                     image = image[0]
                     self.print_rank0(f"[DEBUG] Extracted image type: {type(image)}")
                     self.print_rank0(f"[DEBUG] Extracted image value: {repr(image)}")
                 else:
                     raise ValueError("Empty image list in _process_single_image")
             
             # Ensure we have a PIL.Image object
             if not isinstance(image, Image.Image):
                 self.print_rank0(f"[ERROR] Expected PIL.Image, got {type(image)}")
                 raise TypeError(f"Expected PIL.Image, got {type(image)} in _process_single_image")
             
             self.print_rank0(f"[DEBUG] Final image type: {type(image)}")
             self.print_rank0(f"[DEBUG] Final image size: {image.size}")
             
             # Preprocess image
             self.print_rank0("[DEBUG] Preprocessing image...")
             processed_image = self.preprocess_image(image)
             self.print_rank0(f"[DEBUG] Image preprocessed to size: {processed_image.size}")
             
             # Prepare inputs for the pipeline - Qwen-Image-Edit uses different parameters
             self.print_rank0("[DEBUG] Preparing pipeline inputs...")
             inputs = {
                 "image": processed_image,
                 "prompt": instruction,
                 "generator": torch.manual_seed(0),  # Use fixed seed for reproducibility
                 "num_inference_steps": self.config.num_inference_steps,
                 "guidance_scale": self.config.guidance_scale
             }
             self.print_rank0(f"[DEBUG] Pipeline inputs prepared - num_inference_steps: {self.config.num_inference_steps}, guidance_scale: {self.config.guidance_scale}")
             
             # Generate edited image
             self.print_rank0("[DEBUG] Starting image generation...")
             with torch.inference_mode():
                 output = self.pipeline(**inputs)
                 edited_image = output.images[0]
             self.print_rank0("[DEBUG] Image generation completed")
             
             # Postprocess image
             self.print_rank0("[DEBUG] Postprocessing image...")
             edited_image = self.postprocess_image(edited_image)
             self.print_rank0(f"[DEBUG] Image postprocessed to size: {edited_image.size}")
             
             self.print_rank0("[DEBUG] _process_single_image() completed successfully!")
             return edited_image
             
         except Exception as e:
             self.print_rank0(f"[ERROR] Exception in _process_single_image(): {type(e).__name__}: {e}")
             import traceback
             self.print_rank0(f"[ERROR] Traceback: {traceback.format_exc()}")
             # Return original image as fallback
             return image
    
    def _move_to_gpu(self):
        """Move Qwen-Image-Edit pipeline to GPU for inference"""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call init_model() first.")
        
        self.print_rank0("Moving Qwen-Image-Edit pipeline to GPU...")
        
        # Clear GPU cache first to make room
        torch.cuda.empty_cache()
        
        # Move pipeline to GPU
        self.pipeline = self.pipeline.to(self.device)
        
        # If using FSDP, load models from CPU offload
        if self.config.enable_fsdp and self.config.fsdp_cpu_offload:
            if hasattr(self.pipeline, 'transformer') and hasattr(self.pipeline.transformer, '_fsdp_wrapped_module'):
                load_fsdp_model(self.pipeline.transformer)
            if hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, '_fsdp_wrapped_module'):
                load_fsdp_model(self.pipeline.vae)
            if hasattr(self.pipeline, 'text_encoder') and hasattr(self.pipeline.text_encoder, '_fsdp_wrapped_module'):
                load_fsdp_model(self.pipeline.text_encoder)
        
        # 直接打印GPU内存使用情况，避免函数调用问题
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            self.print_rank0(f"GPU memory after moving Qwen-Image-Edit to GPU: {(total_mem - free_mem) / (1024**3):.2f} GB / {total_mem / (1024**3):.2f} GB")
    
    def _move_to_cpu(self):
        """Move Qwen-Image-Edit pipeline back to CPU to free GPU memory"""
        if self.pipeline is None:
            return
        
        self.print_rank0("Moving Qwen-Image-Edit pipeline back to CPU...")
        
        # If using FSDP, offload models to CPU
        if self.config.enable_fsdp and self.config.fsdp_cpu_offload:
            if hasattr(self.pipeline, 'transformer') and hasattr(self.pipeline.transformer, '_fsdp_wrapped_module'):
                offload_fsdp_model(self.pipeline.transformer)
            if hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, '_fsdp_wrapped_module'):
                offload_fsdp_model(self.pipeline.vae)
            if hasattr(self.pipeline, 'text_encoder') and hasattr(self.pipeline.text_encoder, '_fsdp_wrapped_module'):
                offload_fsdp_model(self.pipeline.text_encoder)
        
        # Move pipeline to CPU
        self.pipeline = self.pipeline.to("cpu")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        # 直接打印GPU内存使用情况，避免函数调用问题
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            self.print_rank0(f"GPU memory after moving Qwen-Image-Edit to CPU: {(total_mem - free_mem) / (1024**3):.2f} GB / {total_mem / (1024**3):.2f} GB")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def edit_images_batch(self, data: DataProto) -> DataProto:
        """
        Edit images using Qwen-Image-Edit pipeline with data parallel processing
        
        Args:
            data: DataProto containing images and edit instructions
            
        Returns:
            DataProto containing edited images
        """
        self.print_rank0("[DEBUG] edit_images_batch() called with data parallel processing")
        
        if self.pipeline is None:
            self.print_rank0("[ERROR] Pipeline not initialized. Call init_model() first.")
            raise RuntimeError("Pipeline not initialized. Call init_model() first.")
        
        try:
            # Extract data
            self.print_rank0("[DEBUG] Extracting data from DataProto...")
            images = data.non_tensor_batch["preliminary_edited_images"]
            edit_instructions = data.non_tensor_batch["re_edit_instructions"]
            self.print_rank0(f"[DEBUG] Extracted {len(images)} images and {len(edit_instructions)} instructions")
            
            # 🔍 DEBUG: 简洁的数据格式检查
            self.print_rank0("=" * 60)
            self.print_rank0("🔍 DEBUG: 数据提取后的格式检查")
            self.print_rank0("=" * 60)
            self.print_rank0(f"[DEBUG] images type: {type(images)}, length: {len(images)}")
            if len(images) > 0:
                img_type = type(images[0])
                self.print_rank0(f"[DEBUG] first image type: {img_type}")
                if hasattr(images[0], '__len__'):
                    self.print_rank0(f"[DEBUG] first image length: {len(images[0])}")
                if hasattr(images[0], 'size'):
                    self.print_rank0(f"[DEBUG] first image size: {images[0].size}")
                # 安全显示：只显示类型和关键属性，不显示完整内容
                if img_type.__name__ == 'Image':
                    self.print_rank0(f"[DEBUG] first image: PIL.Image ({images[0].mode}, {images[0].size})")
                else:
                    self.print_rank0(f"[DEBUG] first image: {img_type.__name__} (not PIL.Image!)")
            
            self.print_rank0(f"[DEBUG] edit_instructions type: {type(edit_instructions)}, length: {len(edit_instructions)}")
            if len(edit_instructions) > 0:
                self.print_rank0(f"[DEBUG] first instruction type: {type(edit_instructions[0])}")
                # 只显示指令的前50个字符
                inst_str = str(edit_instructions[0])
                self.print_rank0(f"[DEBUG] first instruction: {inst_str[:50]}{'...' if len(inst_str) > 50 else ''}")
            self.print_rank0("=" * 60)
            
            # Validate data
            if len(images) != len(edit_instructions):
                self.print_rank0(f"[WARNING] Mismatch: {len(images)} images vs {len(edit_instructions)} instructions")
                min_len = min(len(images), len(edit_instructions))
                images = images[:min_len]
                edit_instructions = edit_instructions[:min_len]
                self.print_rank0(f"[WARNING] Truncated to {min_len} samples")
            
            # Convert numpy arrays to lists if needed, with image format adaptation
            self.print_rank0("[DEBUG] Converting numpy arrays to lists with image format adaptation...")
            
            # 🔍 DEBUG: 转换前的格式检查
            self.print_rank0("=" * 60)
            self.print_rank0("🔍 DEBUG: 数据类型转换前的格式检查")
            self.print_rank0("=" * 60)
            self.print_rank0(f"[DEBUG] images before conversion: {type(images)}")
            if len(images) > 0:
                img_type = type(images[0])
                self.print_rank0(f"[DEBUG] first image before conversion: {img_type}")
                if img_type.__name__ == 'Image':
                    self.print_rank0(f"[DEBUG] first image: PIL.Image ({images[0].mode}, {images[0].size})")
                else:
                    self.print_rank0(f"[DEBUG] first image: {img_type.__name__} (not PIL.Image!)")
            self.print_rank0(f"[DEBUG] edit_instructions before conversion: {type(edit_instructions)}")
            if len(edit_instructions) > 0:
                inst_str = str(edit_instructions[0])
                self.print_rank0(f"[DEBUG] first instruction before conversion: {inst_str[:50]}{'...' if len(inst_str) > 50 else ''}")
            self.print_rank0("=" * 60)
            
            # 🔧 图像格式适配：在转换为list之前，将原始像素数据转换为PIL.Image
            if isinstance(images, np.ndarray):
                self.print_rank0("[DEBUG] Converting images from numpy array to list with format adaptation...")
                adapted_images = []
                for i, img in enumerate(images):
                    if isinstance(img, Image.Image):
                        # 已经是PIL.Image，直接添加
                        adapted_images.append(img)
                    elif isinstance(img, np.ndarray):
                        # 原始像素数据，需要转换为PIL.Image
                        self.print_rank0(f"[DEBUG] Converting image {i} from numpy array to PIL.Image...")
                        try:
                            # 假设是RGB格式的像素数据
                            if len(img.shape) == 3 and img.shape[2] == 3:
                                # RGB格式
                                pil_img = Image.fromarray(img.astype(np.uint8), 'RGB')
                                adapted_images.append(pil_img)
                                self.print_rank0(f"[DEBUG] Successfully converted image {i} to PIL.Image: {pil_img.mode}, {pil_img.size}")
                            elif len(img.shape) == 2:
                                # 灰度格式
                                pil_img = Image.fromarray(img.astype(np.uint8), 'L')
                                adapted_images.append(pil_img)
                                self.print_rank0(f"[DEBUG] Successfully converted image {i} to PIL.Image: {pil_img.mode}, {pil_img.size}")
                            else:
                                # 其他格式，尝试直接转换
                                pil_img = Image.fromarray(img.astype(np.uint8))
                                adapted_images.append(pil_img)
                                self.print_rank0(f"[DEBUG] Successfully converted image {i} to PIL.Image: {pil_img.mode}, {pil_img.size}")
                        except Exception as e:
                            self.print_rank0(f"[ERROR] Failed to convert image {i} from numpy array: {e}")
                            # 如果转换失败，尝试使用原始图像
                            adapted_images.append(img)
                    else:
                        # 其他类型，直接添加
                        adapted_images.append(img)
                
                images = adapted_images
                self.print_rank0(f"[DEBUG] Converted {len(images)} images with format adaptation")
            
            if isinstance(edit_instructions, np.ndarray):
                edit_instructions = edit_instructions.tolist()
                self.print_rank0("[DEBUG] Converted edit_instructions from numpy array to list")
            
            # 🔍 DEBUG: 转换后的格式检查
            self.print_rank0("=" * 60)
            self.print_rank0("🔍 DEBUG: 数据类型转换后的格式检查")
            self.print_rank0("=" * 60)
            self.print_rank0(f"[DEBUG] images after conversion: {type(images)}")
            if len(images) > 0:
                img_type = type(images[0])
                self.print_rank0(f"[DEBUG] first image after conversion: {img_type}")
                if img_type.__name__ == 'Image':
                    self.print_rank0(f"[DEBUG] first image: PIL.Image ({images[0].mode}, {images[0].size})")
                elif img_type.__name__ == 'list':
                    self.print_rank0(f"[DEBUG] first image: LIST (nested!) length: {len(images[0])}")
                    if len(images[0]) > 0:
                        nested_type = type(images[0][0])
                        self.print_rank0(f"[DEBUG] nested image type: {nested_type}")
                        if nested_type.__name__ == 'Image':
                            self.print_rank0(f"[DEBUG] nested image: PIL.Image ({images[0][0].mode}, {images[0][0].size})")
                        else:
                            self.print_rank0(f"[DEBUG] nested image: {nested_type.__name__} (not PIL.Image!)")
                else:
                    self.print_rank0(f"[DEBUG] first image: {img_type.__name__} (not PIL.Image!)")
            
            self.print_rank0(f"[DEBUG] edit_instructions after conversion: {type(edit_instructions)}")
            if len(edit_instructions) > 0:
                inst_str = str(edit_instructions[0])
                self.print_rank0(f"[DEBUG] first instruction after conversion: {inst_str[:50]}{'...' if len(inst_str) > 50 else ''}")
            self.print_rank0("=" * 60)
            
            # Data parallel processing
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            self.print_rank0(f"[DEBUG] Data parallel processing - World size: {world_size}, Rank: {rank}")
            
            if world_size == 1:
                # Single GPU processing (fallback)
                self.print_rank0("[DEBUG] Single GPU processing mode")
                edited_images = []
                for i, (image, instruction) in enumerate(zip(images, edit_instructions)):
                    self.print_rank0(f"[DEBUG] Processing image {i+1}/{len(images)} with instruction: {instruction[:50]}...")
                    try:
                        edited_image = self._process_single_image(image, instruction)
                        edited_images.append(edited_image)
                        self.print_rank0(f"[DEBUG] Completed processing image {i+1}")
                    except Exception as e:
                        self.print_rank0(f"[ERROR] Failed to process image {i+1}: {e}")
                        edited_images.append(image)
                        self.print_rank0(f"[WARNING] Using original image as fallback for image {i+1}")
            else:
                # Multi-GPU data parallel processing
                self.print_rank0(f"[DEBUG] Multi-GPU data parallel processing - {world_size} GPUs")
                
                # Split data across GPUs
                chunk_size = len(images) // world_size
                start_idx = rank * chunk_size
                end_idx = start_idx + chunk_size if rank < world_size - 1 else len(images)
                
                local_images = images[start_idx:end_idx]
                local_instructions = edit_instructions[start_idx:end_idx]
                
                self.print_rank0(f"[DEBUG] Rank {rank}: Processing images {start_idx}-{end_idx-1} ({len(local_images)} images)")
                
                # Process local data
                local_results = []
                for i, (image, instruction) in enumerate(zip(local_images, local_instructions)):
                    global_idx = start_idx + i
                    self.print_rank0(f"[DEBUG] Rank {rank}: Processing image {global_idx+1}/{len(images)} with instruction: {instruction[:50]}...")
                    try:
                        edited_image = self._process_single_image(image, instruction)
                        local_results.append(edited_image)
                        self.print_rank0(f"[DEBUG] Rank {rank}: Completed processing image {global_idx+1}")
                    except Exception as e:
                        self.print_rank0(f"[ERROR] Rank {rank}: Failed to process image {global_idx+1}: {e}")
                        local_results.append(image)
                        self.print_rank0(f"[WARNING] Rank {rank}: Using original image as fallback for image {global_idx+1}")
                
                # Gather results from all GPUs
                self.print_rank0(f"[DEBUG] Rank {rank}: Gathering results from all GPUs...")
                all_results = [None] * world_size
                dist.all_gather_object(all_results, local_results)
                
                # Merge results in correct order
                edited_images = []
                for gpu_results in all_results:
                    if gpu_results is not None:
                        edited_images.extend(gpu_results)
                
                self.print_rank0(f"[DEBUG] Rank {rank}: Merged {len(edited_images)} results from all GPUs")
            
            # Add edited images to original data
            self.print_rank0("[DEBUG] Adding edited images to DataProto...")
            data.non_tensor_batch["final_edited_images"] = np.array(edited_images, dtype=object)
            
            # Debug: Check data format before returning
            self.print_rank0(f"[DEBUG] data type before return: {type(data)}")
            self.print_rank0(f"[DEBUG] data has non_tensor_batch: {hasattr(data, 'non_tensor_batch')}")
            if hasattr(data, 'non_tensor_batch'):
                self.print_rank0(f"[DEBUG] data.non_tensor_batch keys: {list(data.non_tensor_batch.keys())}")
                if "final_edited_images" in data.non_tensor_batch:
                    final_images = data.non_tensor_batch["final_edited_images"]
                    self.print_rank0(f"[DEBUG] final_edited_images type: {type(final_images)}")
                    self.print_rank0(f"[DEBUG] final_edited_images length: {len(final_images)}")
                    if len(final_images) > 0:
                        self.print_rank0(f"[DEBUG] first final_image type: {type(final_images[0])}")
            else:
                self.print_rank0(f"[DEBUG] data attributes: {dir(data)}")
            
            self.print_rank0("[DEBUG] edit_images_batch() completed successfully!")
            return data
            
        except Exception as e:
            self.print_rank0(f"[ERROR] Exception in edit_images_batch(): {type(e).__name__}: {e}")
            import traceback
            self.print_rank0(f"[ERROR] Traceback: {traceback.format_exc()}")
            # 返回原始数据
            data.non_tensor_batch["final_edited_images"] = data.non_tensor_batch["preliminary_edited_images"]
            return data
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def process_batch(self, data: DataProto) -> DataProto:
        """
        Process a batch of image editing requests (alias for edit_images_batch)
        
        Args:
            data: DataProto containing images and edit instructions
            
        Returns:
            DataProto with edited images added
        """
        self.print_rank0(f"[DEBUG] process_batch() called with data type: {type(data)}")
        result = self.edit_images_batch(data)
        self.print_rank0(f"[DEBUG] process_batch() returning data type: {type(result)}")
        return result