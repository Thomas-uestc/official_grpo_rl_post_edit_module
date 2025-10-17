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


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from ..py_functional import is_package_available


if is_package_available("wandb"):
    import wandb  # type: ignore


if is_package_available("swanlab"):
    import swanlab  # type: ignore


@dataclass
class GenerationLogger(ABC):
    @abstractmethod
    def log(self, samples: List[Tuple[str, str, str, float]], step: int, context: str = "val", images: List[dict] = None) -> None: ...


@dataclass
class ConsoleGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int, context: str = "val", images: List[dict] = None) -> None:
        prefix = "[TRAIN]" if context == "train" else "[VAL]"
        for inp, out, lab, score in samples:
            print(f"{prefix} [prompt] {inp}\n{prefix} [output] {out}\n{prefix} [ground_truth] {lab}\n{prefix} [score] {score}\n")


@dataclass
class WandbGenerationLogger(GenerationLogger):
    def __init__(self, config: dict = None):
        # Initialize separate persistent tables for train and validation
        self.validation_table = None
        self.training_table = None
        if config and "trainer" in config and "experiment_name" in config["trainer"]:
            self.experiment_name = config["trainer"]["experiment_name"]
        else:
            self.experiment_name = "default_experiment"
        print(f"[WandbGenerationLogger] Initialized for experiment: {self.experiment_name}")
    
    def log(self, samples: List[Tuple[str, str, str, float]], step: int, context: str = "val", images: List[dict] = None) -> None:
        # Determine table based on context
        if context == "train":
            table_attr = "training_table"
            log_key = f"train/generations_{self.experiment_name}"
            image_log_key = f"train/images_{self.experiment_name}"
        else:
            table_attr = "validation_table"
            log_key = f"val/generations_{self.experiment_name}"
            image_log_key = f"val/images_{self.experiment_name}"
        
        # Create column names for current samples (excluding input column)
        columns = ["step"] + sum(
            [[f"output_{i + 1}", f"label_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))],
            [],
        )
        
        # Get current table
        current_table = getattr(self, table_attr)
        
        # Create new table if first call
        if current_table is None:
            current_table = wandb.Table(columns=columns)
            setattr(self, table_attr, current_table)
        
        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=current_table.data)
        
        # Add new row with all data (excluding input from each sample)
        row_data = [step]
        for sample in samples:
            # sample format: (input, output, label, score)
            # We only want (output, label, score)
            _, output, label, score = sample
            row_data.extend([output, label, score])
        
        new_table.add_data(*row_data)
        
        # Log the updated table and update reference
        if wandb.run:
            wandb.log({log_key: new_table}, step=step)
            setattr(self, table_attr, new_table)
        else:
            print(f"[WandbGenerationLogger] Warning: wandb not initialized, skipping table logging")
        
        # Log images if provided
        if images is not None and len(images) > 0:
            self._log_images(images, step, image_log_key)
    
    def _log_images(self, images: List[dict], step: int, log_key: str) -> None:
        """Log images to wandb"""
        try:
            from PIL import Image
            import numpy as np
            import base64
            import io
            
            # 检查wandb是否已初始化
            if not wandb.run:
                print(f"[WandbGenerationLogger] Warning: wandb not initialized, skipping image logging")
                return
            
            wandb_images = []
            for i, image_dict in enumerate(images):
                sample_images = {}
                
                # Process each image type
                for img_type, img_data in image_dict.items():
                    if img_data is not None:
                        try:
                            # Convert to PIL Image
                            pil_image = self._ensure_pil_image(img_data)
                            if pil_image is not None:
                                sample_images[img_type] = wandb.Image(pil_image, caption=f"{img_type}")
                        except Exception as e:
                            print(f"[WandbGenerationLogger] Error processing {img_type} for sample {i}: {e}")
                            sample_images[img_type] = wandb.Image(
                                Image.new('RGB', (100, 100), color='red'), 
                                caption=f"{img_type} (Error)"
                            )
                    else:
                        # Create placeholder for missing images
                        sample_images[img_type] = wandb.Image(
                            Image.new('RGB', (100, 100), color='gray'), 
                            caption=f"{img_type} (Missing)"
                        )
                
                wandb_images.append(sample_images)
            
            # Log all images for this step
            if wandb_images:
                # 将图像字典转换为wandb可以处理的格式
                wandb_log_data = {}
                for i, sample_images in enumerate(wandb_images):
                    for img_type, wandb_img in sample_images.items():
                        key = f"{log_key}_sample_{i+1}_{img_type}"
                        wandb_log_data[key] = wandb_img
                
                wandb.log(wandb_log_data, step=step)
                print(f"[WandbGenerationLogger] Logged {len(wandb_images)} image sets to {log_key}")
                
        except Exception as e:
            print(f"[WandbGenerationLogger] Error logging images: {e}")
    
    def _ensure_pil_image(self, image_data):
        """Convert various image data types to PIL Image"""
        try:
            from PIL import Image
            import numpy as np
            import base64
            import io
            
            if image_data is None:
                return None
            
            # Handle PIL Image
            if isinstance(image_data, Image.Image):
                return image_data
            
            # Handle numpy array
            if isinstance(image_data, np.ndarray):
                if image_data.dtype != np.uint8:
                    image_data = (image_data * 255).astype(np.uint8)
                return Image.fromarray(image_data)
            
            # Handle list (assumed to be RGB values)
            if isinstance(image_data, list):
                if len(image_data) > 0 and isinstance(image_data[0], list):
                    # 2D list
                    np_array = np.array(image_data)
                    if np_array.dtype != np.uint8:
                        np_array = (np_array * 255).astype(np.uint8)
                    return Image.fromarray(np_array)
                else:
                    # 1D list, try to reshape
                    np_array = np.array(image_data)
                    if np_array.dtype != np.uint8:
                        np_array = (np_array * 255).astype(np.uint8)
                    # Try to reshape to square image
                    size = int(np.sqrt(len(np_array)))
                    if size * size == len(np_array):
                        return Image.fromarray(np_array.reshape(size, size))
                    else:
                        return Image.fromarray(np_array)
            
            # Handle base64 string
            if isinstance(image_data, str):
                try:
                    # Remove data URL prefix if present
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    return Image.open(io.BytesIO(image_bytes))
                except Exception:
                    pass
            
            # Handle bytes
            if isinstance(image_data, bytes):
                return Image.open(io.BytesIO(image_data))
            
            print(f"[WandbGenerationLogger] Unsupported image data type: {type(image_data)}")
            return None
            
        except Exception as e:
            print(f"[WandbGenerationLogger] Error converting image data: {e}")
            return None


@dataclass
class SwanlabGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int, context: str = "val", images: List[dict] = None) -> None:
        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = "\n\n---\n\n".join(
                (f"input: {sample[0]}", f"output: {sample[1]}", f"label: {sample[2]}", f"score: {sample[3]}")
            )
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        swanlab.log({"val/generations": swanlab_text_list}, step=step)


GEN_LOGGERS = {
    "console": ConsoleGenerationLogger,
    "wandb": WandbGenerationLogger,
    "swanlab": SwanlabGenerationLogger,
}


@dataclass
class AggregateGenerationsLogger:
    def __init__(self, loggers: List[str], config: dict = None):
        self.loggers: List[GenerationLogger] = []
        self.config = config

        for logger in loggers:
            if logger in GEN_LOGGERS:
                if logger == "wandb":
                    self.loggers.append(GEN_LOGGERS[logger](config))
                else:
                    self.loggers.append(GEN_LOGGERS[logger]())

    def log(self, samples: List[Tuple[str, str, str, float]], step: int, context: str = "val", images: List[dict] = None) -> None:
        for logger in self.loggers:
            logger.log(samples, step, context, images)
