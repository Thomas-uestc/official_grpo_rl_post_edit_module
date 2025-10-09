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
Image Edit Worker for Qwen-Image-Edit model integration
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import torch
from PIL import Image
import numpy as np

from ...protocol import DataProto
from ...single_controller.base import Worker
from ...single_controller.base.decorator import Dispatch, register
from .config import ImageEditConfig


class ImageEditWorker(Worker, ABC):
    """Base class for image editing workers"""
    
    def __init__(self, config: ImageEditConfig):
        super().__init__()
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = torch.device(config.device)
        
    @abstractmethod
    def init_model(self):
        """Initialize the image editing model"""
        pass
    
    @abstractmethod
    def edit_images(
        self, 
        images: List[Image.Image], 
        edit_instructions: List[str]
    ) -> List[Image.Image]:
        """
        Edit images based on instructions
        
        Args:
            images: List of PIL Images to edit
            edit_instructions: List of edit instructions
            
        Returns:
            List of edited PIL Images
        """
        pass
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def process_batch(self, data: DataProto) -> DataProto:
        """
        Process a batch of image editing requests
        
        Args:
            data: DataProto containing images and edit instructions
            
        Returns:
            DataProto with edited images added
        """
        if "preliminary_edited_images" not in data.non_tensor_batch:
            raise ValueError("preliminary_edited_images not found in data")
        if "re_edit_instructions" not in data.non_tensor_batch:
            raise ValueError("re_edit_instructions not found in data")
            
        images = data.non_tensor_batch["preliminary_edited_images"]
        instructions = data.non_tensor_batch["re_edit_instructions"]
        
        # Convert numpy arrays to lists
        if isinstance(images, np.ndarray):
            images = images.tolist()
        if isinstance(instructions, np.ndarray):
            instructions = instructions.tolist()
            
        # Process images in batches
        edited_images = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_instructions = instructions[i:i + batch_size]
            
            batch_edited = self.edit_images(batch_images, batch_instructions)
            edited_images.extend(batch_edited)
        
        # Add edited images to data
        data.non_tensor_batch["final_edited_images"] = np.array(edited_images, dtype=object)
        
        return data
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for the model"""
        # Resize image if needed
        if image.size[0] > self.config.image_size or image.size[1] > self.config.image_size:
            image.thumbnail((self.config.image_size, self.config.image_size), Image.Resampling.LANCZOS)
        
        return image
    
    def postprocess_image(self, image: Image.Image) -> Image.Image:
        """Postprocess image after editing"""
        return image
