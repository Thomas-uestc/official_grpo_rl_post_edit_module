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
Image Edit Worker config
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImageEditConfig:
    """Configuration for image editing worker"""
    
    # Model configuration
    model_path: str = "Qwen/Qwen-Image-Edit"
    trust_remote_code: bool = True
    
    # Device configuration
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    num_cpus: int = 1
    num_gpus: int = 1  # 添加GPU数量配置
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Diffusers pipeline parameters
    true_cfg_scale: float = 4.0
    negative_prompt: str = ""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    
    # FSDP configuration
    enable_fsdp: bool = True
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fsdp_cpu_offload: bool = True
    fsdp_sync_module_states: bool = True
    
    # Batch processing
    batch_size: int = 4
    max_batch_size: int = 8
    
    # Memory optimization
    enable_cpu_offload: bool = False  # 独占GPU，无需CPU offload
    enable_gradient_checkpointing: bool = False
    
    # Image processing
    image_size: int = 1024
    max_pixels: int = 1024 * 1024
    min_pixels: int = 64 * 64
    
    # API configuration for GPT-4.1 reward
    gpt41_api_key: Optional[str] = None
    gpt41_api_base: str = "https://api.openai.com/v1"
    gpt41_model: str = "gpt-4.1"
    gpt41_max_tokens: int = 1000
    gpt41_temperature: float = 0.0
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Timeout configuration
    request_timeout: float = 30.0
    
    def post_init(self):
        """Post initialization validation"""
        if self.batch_size > self.max_batch_size:
            raise ValueError(f"batch_size ({self.batch_size}) cannot be larger than max_batch_size ({self.max_batch_size})")
        
        if self.gpt41_api_key is None:
            import os
            self.gpt41_api_key = os.getenv("OPENAI_API_KEY")
            if self.gpt41_api_key is None:
                print("Warning: GPT-4.1 API key not provided. Set OPENAI_API_KEY environment variable or configure gpt41_api_key.")
