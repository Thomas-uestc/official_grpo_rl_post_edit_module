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
GPT-4.1 Reward Manager for image editing evaluation
"""

import base64
import io
import json
import time
from typing import List, Dict, Any, Tuple
import requests
from PIL import Image
import torch
import numpy as np
from collections import defaultdict

from ...protocol import DataProto
from ...single_controller.base import Worker
from ...single_controller.base.decorator import Dispatch, register
from ..reward.config import RewardConfig


class GPT41RewardManager(Worker):
    """GPT-4.1 API-based reward manager for image editing evaluation"""
    
    def __init__(self, config: RewardConfig):
        # Skip parent __init__ to avoid CUDA initialization
        # super().__init__()
        print(f"[DEBUG] GPT41RewardManager: Initializing with config...")
        self.config = config
        self.api_key = config.gpt41_api_key
        self.api_base = config.gpt41_api_base
        self.model = config.gpt41_model
        self.max_tokens = config.gpt41_max_tokens
        self.temperature = config.gpt41_temperature
        self.max_retries = config.gpt41_max_retries
        self.retry_delay = config.gpt41_retry_delay
        self.request_timeout = config.gpt41_request_timeout
        
        # 创建持久化session用于连接复用
        self.session = requests.Session()
        
        # 配置session的代理和adapter
        self._setup_session()
        
        print(f"[DEBUG] GPT41RewardManager: Initialized successfully - API base: {self.api_base}, Model: {self.model}, Max retries: {self.max_retries}")
        
    def _setup_session(self):
        """设置session的代理和连接配置"""
        import os
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        
        # 配置代理
        http_proxy = os.getenv('http_proxy') or os.getenv('HTTP_PROXY')
        https_proxy = os.getenv('https_proxy') or os.getenv('HTTPS_PROXY')
        
        if http_proxy or https_proxy:
            proxies = {}
            if http_proxy:
                proxies['http'] = http_proxy
            if https_proxy:
                proxies['https'] = https_proxy
            self.session.proxies.update(proxies)
            print(f"[DEBUG] _setup_session: Configured proxies: {proxies}")
        
        # 配置连接池和重试策略
        retry_strategy = Retry(
            total=0,  # 不在urllib3层面重试，由我们的逻辑控制
            backoff_factor=0
        )
        
        adapter = HTTPAdapter(
            pool_connections=1,  # 连接池大小
            pool_maxsize=2,      # 最大连接数
            max_retries=retry_strategy,
            pool_block=False
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置默认头部
        self.session.headers.update({
            'Connection': 'keep-alive',
            'User-Agent': 'VERL-GPT41-RewardManager/1.0'
        })
    
    def __del__(self):
        """清理session资源"""
        if hasattr(self, 'session'):
            try:
                self.session.close()
                print(f"[DEBUG] GPT41RewardManager: Session closed")
            except Exception as e:
                print(f"[DEBUG] GPT41RewardManager: Error closing session: {e}")
        
    def _ensure_pil_image(self, image_data) -> Image.Image:
        """Ensure image data is a PIL Image object"""
        print(f"[DEBUG] _ensure_pil_image: Processing image data of type: {type(image_data)}")
        
        if isinstance(image_data, Image.Image):
            print(f"[DEBUG] _ensure_pil_image: Already PIL Image, size: {image_data.size}")
            return image_data
        elif isinstance(image_data, np.ndarray):
            print(f"[DEBUG] _ensure_pil_image: Converting numpy array, shape: {image_data.shape}, dtype: {image_data.dtype}")
            # Convert numpy array to PIL Image
            if image_data.dtype == np.uint8:
                pil_img = Image.fromarray(image_data)
            else:
                # Convert to uint8 if needed
                if image_data.max() <= 1.0:
                    image_data = (image_data * 255).astype(np.uint8)
                else:
                    image_data = image_data.astype(np.uint8)
                pil_img = Image.fromarray(image_data)
            print(f"[DEBUG] _ensure_pil_image: Converted to PIL Image, size: {pil_img.size}")
            return pil_img
        elif isinstance(image_data, list):
            print(f"[DEBUG] _ensure_pil_image: Converting list to PIL Image, list length: {len(image_data)}")
            # Handle list format (from Ray serialization)
            # Convert list back to numpy array first
            try:
                np_array = np.array(image_data)
                print(f"[DEBUG] _ensure_pil_image: List converted to numpy array, shape: {np_array.shape}, dtype: {np_array.dtype}")
                if np_array.dtype == np.uint8:
                    pil_img = Image.fromarray(np_array)
                else:
                    # Convert to uint8 if needed
                    if np_array.max() <= 1.0:
                        np_array = (np_array * 255).astype(np.uint8)
                    else:
                        np_array = np_array.astype(np.uint8)
                    pil_img = Image.fromarray(np_array)
                print(f"[DEBUG] _ensure_pil_image: List converted to PIL Image, size: {pil_img.size}")
                return pil_img
            except Exception as e:
                print(f"[ERROR] _ensure_pil_image: Failed to convert list to PIL Image: {e}")
                raise ValueError(f"Failed to convert list to PIL Image: {e}")
        elif isinstance(image_data, str):
            print(f"[DEBUG] _ensure_pil_image: Converting base64 string, length: {len(image_data)}")
            # Assume it's a base64 string
            import base64
            import io
            image_bytes = base64.b64decode(image_data)
            pil_img = Image.open(io.BytesIO(image_bytes))
            print(f"[DEBUG] _ensure_pil_image: Base64 converted to PIL Image, size: {pil_img.size}")
            return pil_img
        else:
            print(f"[ERROR] _ensure_pil_image: Unsupported image data type: {type(image_data)}")
            raise ValueError(f"Unsupported image data type: {type(image_data)}")
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string with compression"""
        print(f"[DEBUG] _encode_image: Encoding PIL Image, size: {image.size}")
        
        # 根据OpenAI文档建议优化图片尺寸和质量
        max_size = 512  # 进一步减小尺寸，减少POST请求大小
        if max(image.size) > max_size:
            print(f"[DEBUG] _encode_image: Resizing image from {image.size} to reduce payload size")
            # 保持宽高比的缩放
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"[DEBUG] _encode_image: Resized image size: {image.size}")
        
        # 确保图片是RGB模式（避免RGBA等模式导致的问题）
        if image.mode != 'RGB':
            print(f"[DEBUG] _encode_image: Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        # 使用更激进的JPEG压缩质量，减少POST请求大小
        image.save(buffer, format='JPEG', quality=60, optimize=True)
        image_bytes = buffer.getvalue()
        encoded = base64.b64encode(image_bytes).decode('utf-8')
        print(f"[DEBUG] _encode_image: Encoded to base64, length: {len(encoded)}")
        return encoded
    
    def _create_evaluation_prompt(
        self, 
        original_image: Image.Image,
        final_image: Image.Image,
        edit_instruction: str,
        original_description: str = "",
        dimension: str = "overall"
    ) -> str:
        """Create evaluation prompt for GPT-4.1 for specific dimension"""
        
        # 定义5个评估维度的prompt模板
        dimension_prompts = {
            "physical_geometric": """
            You are an image editing reward model evaluator assessing **Physical & Geometric Consistency**.  
            Output only a single score between 0.000 and 10.000 (rounded to three decimals).  
            No explanations, symbols, or extra text.

            ⸻
            **Goal:**  
            Judge if the edited image obeys natural physical and geometric rules.  
            If the edit intentionally requires unrealistic results → no penalty.  
            If unrelated to physical/geometric aspects → stay neutral (5.000 baseline).

            ⸻
            **Scoring Logic:**  
            Start from baseline 5.000 (neutral).  
            Adjust ± up to 3 points per criterion, average all, clip [0.000, 10.000].

            **Criteria:**
            1. Lighting & Shadow – direction/intensity realism  
             +3 if coherent, –3 if inconsistent.  
            2. Contact & Support – object touches surfaces (no floating/penetration).  
             +3 if realistic, –3 if not.  
            3. Scale & Perspective – proportion and depth alignment.  
             +3 if accurate, –3 if off.  
            4. Reflection / Material – mirror/water physics and material coherence.  
             +3 if plausible, –3 if contradictory.  
            5. Motion & Gravity – movement/posture fits inertia and gravity.  
             +3 if plausible, –3 if not.

            ⸻
            **Procedure:**
            1. Each criterion starts at 5.000.  
            2. Apply adjustments ± up to 3.  
            3. Average → clip [0, 10] → round 3 decimals.

            ⸻
            **Restrictions:**  
            Only evaluate physical & geometric plausibility.  
            Ignore aesthetics, semantics, and instruction compliance.

            ⸻
            **Output:**  
            Single numeric score only (no extra text).
            """,
            "environment_context": """
            You are an image editing reward model evaluator assessing **Environment & Context Consistency**.  
            Output only a single score between 0.000 and 10.000 (rounded to three decimals).  
            No explanations, symbols, or extra text.

            ⸻
            **Goal:**  
            Judge whether the edited image remains consistent with environmental conditions, time, climate, and contextual cues.  
            If the instruction intentionally breaks these rules → no penalty.  
            If irrelevant to environment/context → stay neutral (5.000 baseline).

            ⸻
            **Scoring Logic:**  
            Start from baseline 5.000 (neutral).  
            Adjust ± up to 3 points per criterion, then average and clip to [0.000, 10.000].

            **Criteria:**
            1. Weather & Climate – temperature, humidity, and weather realism.  
             +3 if logical, −3 if inconsistent.  
            2. Lighting & Time – lighting direction and tone match scene time.  
             +3 if coherent, −3 if off.  
            3. Environmental Elements – plants, terrain, water, architecture coherence.  
             +3 if harmonious, −3 if contradictory.  
            4. Atmosphere & Context – emotional tone fits environment.  
             +3 if unified, −3 if mismatched.  
            5. Temporal Continuity – transitions between times/seasons are natural.  
             +3 if smooth, −3 if conflicting.

            ⸻
            **Procedure:**
            1. Start each criterion at 5.000.  
            2. Apply ± adjustments (up to 3).  
            3. Take the mean → clip [0, 10] → round to three decimals.

            ⸻
            **Restrictions:**  
            Only assess environmental and contextual realism.  
            Ignore aesthetics, subjects, or instruction correctness.

            ⸻
            **Output:**  
            Single numeric score only (no text).
""",
            "cultural_social": """
            You are an image editing reward model evaluator assessing **Cultural & Social Norm Alignment**.  
            Output only a single score between 0.000 and 10.000 (rounded to three decimals).  
            No explanations, symbols, or extra text.

            ⸻
            **Goal:**  
            Judge whether the edited image aligns with cultural knowledge, social behavior norms, and human commonsense.  
            If the instruction intentionally introduces surreal or unrealistic content → no penalty.  
            If unrelated to culture or social context → stay neutral (5.000 baseline).

            ⸻
            **Scoring Logic:**  
            Start from baseline 5.000 (neutral).  
            Adjust ± up to 3 points per criterion, average all, clip to [0.000, 10.000].

            **Criteria:**
            1. Social Behavior – gestures, postures, and interactions follow social logic.  
             +3 if natural, −3 if implausible.  
            2. Cultural Symbols – clothing, architecture, and text match cultural context.  
             +3 if appropriate, −3 if mismatched.  
            3. Gender & Roles – attire and activity fit expected roles.  
             +3 if coherent, −3 if contradictory.  
            4. Etiquette & Scene – behavior and dress suit the setting (wedding, office, etc.).  
             +3 if appropriate, −3 if off.  
            5. Safety & Ethics – conforms to societal and ethical safety norms.  
             +3 if proper, −3 if violating.

            ⸻
            **Procedure:**
            1. Each criterion starts at 5.000.  
            2. Apply ± adjustments up to 3.  
            3. Average → clip [0, 10] → round to three decimals.

            ⸻
            **Restrictions:**  
            Only evaluate cultural and social logic.  
            Ignore aesthetics, physics, and edit accuracy.

            ⸻
            **Output:**  
            Single numeric score only (no text).
            """,
            "logical_causal": """
            You are an image editing reward model evaluator assessing **Logical & Causal Consistency**.  
            Output only a single score between 0.000 and 10.000 (rounded to three decimals).  
            No explanations, symbols, or extra text.

            ⸻
            **Goal:**  
            Judge whether the edited image follows logical reasoning and cause-effect relationships.  
            If the instruction intentionally depicts surreal or illogical effects → no penalty.  
            If unrelated to logic or causality → stay neutral (5.000 baseline).

            ⸻
            **Scoring Logic:**  
            Start from baseline 5.000 (neutral).  
            Adjust ± up to 3 points per criterion, average all, clip to [0.000, 10.000].

            **Criteria:**
            1. Action–Outcome Logic – action results match effects (e.g., spilled water → wet area).  
             +3 if coherent, −3 if contradictory.  
            2. Event Transition – before–after states are continuous and plausible.  
             +3 if natural, −3 if broken.  
            3. Causal Chain – conditions produce logical effects (rain → wet ground).  
             +3 if realistic, −3 if implausible.  
            4. Actor–Object Relation – agent’s action aligns with target response.  
             +3 if spatially coherent, −3 if inconsistent.  
            5. Temporal Flow – cause precedes effect with clear time logic.  
             +3 if consistent, −3 if reversed.

            ⸻
            **Procedure:**
            1. Each criterion starts at 5.000.  
            2. Apply ± adjustments (up to 3).  
            3. Average → clip [0, 10] → round to three decimals.

            ⸻
            **Restrictions:**  
            Only assess logical and causal reasoning.  
            Ignore aesthetics, physics, and cultural aspects.

            ⸻
            **Output:**  
            Single numeric score only (no text).
            """,
            "target_attribution": """
            You are an image editing reward model evaluator assessing **Target Attribution & Referential Reasoning Consistency**.  
            Output only a single score between 0.000 and 10.000 (rounded to three decimals).  
            No explanations, symbols, or extra text.

            ⸻
            **Goal:**  
            Judge whether the edited image correctly identifies and modifies the intended target, maintaining accurate attributes, position, and relations.  
            If the instruction intentionally uses abstract or ambiguous references → no penalty.  
            If unrelated to referential reasoning → stay neutral (5.000 baseline).

            ⸻
            **Scoring Logic:**  
            Start from baseline 5.000 (neutral).  
            Adjust ± up to 3 points per criterion, average all, clip to [0.000, 10.000].

            **Criteria:**
            1. Target Identification – correct object located and edited.  
             +3 if exact, −3 if wrong or swapped.  
            2. Spatial Reasoning – spatial terms (“left,” “behind,” etc.) correctly applied.  
             +3 if accurate, −3 if misinterpreted.  
            3. Attribute Consistency – edited attributes (color, pose, etc.) match instruction.  
             +3 if faithful, −3 if mismatched.  
            4. Referential Resolution – relational references (“the cat near the window”) resolved correctly.  
             +3 if precise, −3 if confused.  
            5. Edit Scope – edit limited to referenced region without altering others.  
             +3 if isolated, −3 if overextended.

            ⸻
            **Procedure:**
            1. Each criterion starts at 5.000.  
            2. Apply ± adjustments (up to 3).  
            3. Average → clip [0, 10] → round to three decimals.

            ⸻
            **Restrictions:**  
            Only assess referential reasoning and target attribution.  
            Ignore aesthetics, physics, and realism.

            ⸻
            **Output:**  
            Single numeric score only (no text).
            """
        }
        
        # 获取对应维度的prompt，如果维度不存在则使用overall
        if dimension in dimension_prompts:
            dimension_prompt = dimension_prompts[dimension]
        else:
            # 保持原有的overall评估逻辑作为fallback
            dimension_prompt = f"""You are an image editing reward model evaluator, responsible for assessing **Human Aesthetic Preference** and **Environment-Environment Consistency** only.  
Your only task is to score a given edited image based on these two aspects and output a single number between 0.000 and 10.000 (with 3 decimal places).  
You must not output any explanation, symbols, or extra text.  

⸻

Evaluation Objectives (focus only on two aspects):  
1. **Human Aesthetic Preference**: Does the image appear visually appealing, harmonious, and natural?  
   - Color harmony  
   - Composition balance  
   - Style consistency with common aesthetic intuition  

2. **Environment-Environment Consistency**: Are physical and natural conditions internally coherent?  
   - Weather, humidity, and temperature consistency  
   - Lighting intensity and direction correctness  
   - Penalize contradictions (e.g., sunny sky with wet reflective pavement, low temperature with no frost or mist)  

⸻

Output Format (strictly enforced):  
- Output a single number only 
- No units, no percentage signs, no explanations, no spaces, no extra symbols."""
        
        # 构建完整的prompt
        full_prompt = f"""{dimension_prompt}

Original image description:  
{original_description}  

Editing instruction:  
{edit_instruction} 

Please evaluate the edited image and output only a single score between 0.000 and 10.000:"""
        
        return full_prompt
    
    def _call_gpt41_api(
        self, 
        original_image: Image.Image,
        final_image: Image.Image,
        edit_instruction: str,
        original_description: str = "",
        dimension: str = "overall"
    ) -> Dict[str, Any]:
        """Call GPT-4.1 API for evaluation"""
        
        print(f"[DEBUG] _call_gpt41_api: Starting API call for evaluation")
        print(f"[DEBUG] _call_gpt41_api: Edit instruction: {edit_instruction[:100]}...")
        print(f"[DEBUG] _call_gpt41_api: Original description: {original_description[:100]}...")
        
        # Encode images
        print(f"[DEBUG] _call_gpt41_api: Encoding images...")
        original_image_b64 = self._encode_image(original_image)
        final_image_b64 = self._encode_image(final_image)
        
        # Create prompt
        print(f"[DEBUG] _call_gpt41_api: Creating evaluation prompt for dimension: {dimension}")
        prompt = self._create_evaluation_prompt(
            original_image, final_image, edit_instruction, original_description, dimension
        )
        print(f"[DEBUG] _call_gpt41_api: Prompt created, length: {len(prompt)}")
        print(f"[DEBUG] _call_gpt41_api: Prompt created, content: {prompt}")
        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,  # 使用配置中的模型名称
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{original_image_b64}",
                                "detail": "low"  # 使用低细节模式减少处理时间
                            }
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{final_image_b64}",
                                "detail": "low"  # 使用低细节模式减少处理时间
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        print(f"[DEBUG] _call_gpt41_api: Making API request to {self.api_base}/chat/completions")
        print(f"[DEBUG] _call_gpt41_api: Model: {payload['model']}")
        print(f"[DEBUG] _call_gpt41_api: Payload size: {len(str(payload))} chars")
        print(f"[DEBUG] _call_gpt41_api: Number of images: 2")
        print(f"[DEBUG] _call_gpt41_api: Original image base64 length: {len(original_image_b64)}")
        print(f"[DEBUG] _call_gpt41_api: Final image base64 length: {len(final_image_b64)}")
        
        # 验证base64编码是否有效
        try:
            base64.b64decode(original_image_b64[:100])  # 测试解码前100个字符
            base64.b64decode(final_image_b64[:100])
            print(f"[DEBUG] _call_gpt41_api: Base64 encoding validation passed")
        except Exception as e:
            print(f"[ERROR] _call_gpt41_api: Base64 encoding validation failed: {e}")
            return {"overall_score": 5.0}
        
        # Make API call with retries
        for attempt in range(self.max_retries):
            try:
                print(f"[DEBUG] _call_gpt41_api: Attempt {attempt + 1}/{self.max_retries}")
                start_time = time.time()
                
                # 优化请求头部，避免代理问题
                request_headers = headers.copy()
                request_headers.update({
                    'Authorization': f"Bearer {self.api_key}",
                    'Content-Type': 'application/json',
                    'Connection': 'keep-alive',
                    'Accept': 'application/json',
                    'Accept-Encoding': 'gzip, deflate'
                })
                
                # 移除可能导致代理问题的头部
                request_headers.pop('Expect', None)
                request_headers.pop('Transfer-Encoding', None)
                
                # 使用session发送请求（连接复用）
                response = self.session.post(
                    f"{self.api_base}/chat/completions",
                    headers=request_headers,
                    json=payload,
                    timeout=(self.request_timeout, self.request_timeout * 3),
                    stream=False  # 禁用流式传输，避免chunked编码问题
                )
                
                request_time = time.time() - start_time
                print(f"[DEBUG] _call_gpt41_api: Request completed in {request_time:.2f}s, status: {response.status_code}")
                
                # Print response content for debugging
                if response.status_code != 200:
                    print(f"[ERROR] _call_gpt41_api: HTTP {response.status_code} - Response content: {response.text}")
                    print(f"[ERROR] _call_gpt41_api: Request headers: {request_headers}")
                    print(f"[ERROR] _call_gpt41_api: Used session proxies: {self.session.proxies}")
                
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                print(f"[DEBUG] _call_gpt41_api: Received response: {content}")
                
                # Parse numeric score from response
                try:
                    # Extract numeric score from response
                    score = float(content)
                    # Ensure score is in valid range [0, 10]
                    score = max(0.0, min(10.0, score))
                    print(f"[DEBUG] _call_gpt41_api: Parsed score: {score}")
                    return {"overall_score": score}
                except (ValueError, TypeError) as e:
                    print(f"[WARNING] _call_gpt41_api: Failed to parse score '{content}': {e}")
                    # Fallback if score parsing fails
                    return {"overall_score": 5.0}
                    
            except requests.exceptions.RequestException as e:
                error_type = type(e).__name__
                print(f"[ERROR] _call_gpt41_api: Request failed on attempt {attempt + 1}: {error_type}: {e}")
                
                # 详细的错误类型分析
                if "timeout" in str(e).lower():
                    print(f"[ERROR] _call_gpt41_api: Timeout error - check network connectivity")
                elif "connection" in str(e).lower():
                    print(f"[ERROR] _call_gpt41_api: Connection error - check proxy settings")
                elif "ssl" in str(e).lower():
                    print(f"[ERROR] _call_gpt41_api: SSL error - check certificate settings")
                
                if attempt < self.max_retries - 1:
                    retry_delay = self.retry_delay * (2 ** attempt)
                    print(f"[DEBUG] _call_gpt41_api: Retrying in {retry_delay}s...")
                    
                    # 在重试前测试基本连接性（使用session）
                    try:
                        test_response = self.session.get("https://api.openai.com/v1/models", 
                                                        timeout=(10, 30))
                        print(f"[DEBUG] _call_gpt41_api: Connection test status: {test_response.status_code}")
                    except Exception as test_e:
                        print(f"[DEBUG] _call_gpt41_api: Connection test failed: {test_e}")
                    
                    time.sleep(retry_delay)  # Exponential backoff
                    continue
                else:
                    print(f"[ERROR] _call_gpt41_api: All retries exhausted, returning default score")
                    # Return default scores on final failure
                    return {"overall_score": 5.0}
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        Compute rewards for a batch of image editing results using 5 dimensions
        
        Args:
            data: DataProto containing original images, final images, and instructions
            
        Returns:
            Tuple of (reward_tensor, reward_metrics)
        """
        print(f"[DEBUG] compute_reward: Starting multi-dimensional reward computation")
        print(f"[DEBUG] compute_reward: Available keys in data.non_tensor_batch: {list(data.non_tensor_batch.keys())}")
        
        # Extract data - handle both image editing and standard data
        if "final_edited_images" in data.non_tensor_batch:
            print(f"[DEBUG] compute_reward: Using final_edited_images data flow")
            # Image editing data flow
            original_images = data.non_tensor_batch["original_images"]
            final_images = data.non_tensor_batch["final_edited_images"]
            edit_instructions = data.non_tensor_batch["edit_instructions"]
        else:
            print(f"[DEBUG] compute_reward: Using preliminary_edited_images data flow")
            # Standard data flow - use preliminary edited images as final images
            original_images = data.non_tensor_batch["original_images"]
            final_images = data.non_tensor_batch["preliminary_edited_images"]
            edit_instructions = data.non_tensor_batch["edit_instructions"]
        
        print(f"[DEBUG] compute_reward: Data types - original_images: {type(original_images)}, final_images: {type(final_images)}, edit_instructions: {type(edit_instructions)}")
        
        # Extract original descriptions if available (from dataset "original_description" field)
        original_descriptions = data.non_tensor_batch.get("original_descriptions", [])
        print(f"[DEBUG] compute_reward: Original descriptions available: {len(original_descriptions)}")
        
        # Convert numpy arrays to lists
        if isinstance(original_images, np.ndarray):
            print(f"[DEBUG] compute_reward: Converting original_images from numpy array to list")
            original_images = original_images.tolist()
        if isinstance(final_images, np.ndarray):
            print(f"[DEBUG] compute_reward: Converting final_images from numpy array to list")
            final_images = final_images.tolist()
        if isinstance(edit_instructions, np.ndarray):
            print(f"[DEBUG] compute_reward: Converting edit_instructions from numpy array to list")
            edit_instructions = edit_instructions.tolist()
        if isinstance(original_descriptions, np.ndarray):
            print(f"[DEBUG] compute_reward: Converting original_descriptions from numpy array to list")
            original_descriptions = original_descriptions.tolist()
        
        # Ensure original_descriptions has the same length as other data
        if len(original_descriptions) == 0:
            print(f"[DEBUG] compute_reward: Padding original_descriptions with empty strings")
            original_descriptions = [""] * len(original_images)
        elif len(original_descriptions) != len(original_images):
            print(f"[DEBUG] compute_reward: Adjusting original_descriptions length from {len(original_descriptions)} to {len(original_images)}")
            # Pad or truncate to match batch size
            if len(original_descriptions) < len(original_images):
                original_descriptions.extend([""] * (len(original_images) - len(original_descriptions)))
            else:
                original_descriptions = original_descriptions[:len(original_images)]
        
        # Define 5 evaluation dimensions
        dimensions = ["physical_geometric", "environment_context", "cultural_social", "logical_causal", "target_attribution"]
        
        # Initialize reward tensor and metrics
        batch_size = len(original_images)
        print(f"[DEBUG] compute_reward: Processing batch of size {batch_size} with {len(dimensions)} dimensions")
        
        # Store scores for each dimension
        dimension_scores = {dim: torch.zeros(batch_size, dtype=torch.float32, device='cpu') for dim in dimensions}
        reward_metrics = defaultdict(list)
        
        # Evaluate each sample across all dimensions
        for i in range(batch_size):
            print(f"[DEBUG] compute_reward: Processing sample {i+1}/{batch_size}")
            try:
                # Ensure images are PIL Image objects
                print(f"[DEBUG] compute_reward: Converting images for sample {i}")
                original_img = self._ensure_pil_image(original_images[i])
                final_img = self._ensure_pil_image(final_images[i])
                
                sample_scores = {}
                
                # Evaluate each dimension separately
                for dim in dimensions:
                    print(f"[DEBUG] compute_reward: Evaluating sample {i} for dimension: {dim}")
                    try:
                        evaluation = self._call_gpt41_api(
                            original_img,
                            final_img,
                            edit_instructions[i],
                            original_descriptions[i] if i < len(original_descriptions) else "",
                            dimension=dim
                        )
                        
                        score = evaluation.get("overall_score", 5.0)
                        sample_scores[dim] = score
                        dimension_scores[dim][i] = score
                        
                        print(f"[DEBUG] compute_reward: Sample {i} {dim} score: {score}")
                        
                    except Exception as e:
                        print(f"[ERROR] compute_reward: Error evaluating sample {i} dimension {dim}: {e}")
                        # Fallback score for this dimension
                        fallback_score = 5.0
                        sample_scores[dim] = fallback_score
                        dimension_scores[dim][i] = fallback_score
                
                # Store individual dimension scores in metrics
                for dim in dimensions:
                    reward_metrics[f"gpt_{dim}"].append(sample_scores[dim])
                
                # Calculate overall average score for this sample
                overall_score = sum(sample_scores.values()) / len(sample_scores)
                reward_metrics["overall"].append(overall_score)
                
                print(f"[DEBUG] compute_reward: Sample {i} completed - Overall: {overall_score:.3f}")
                
            except Exception as e:
                print(f"[ERROR] compute_reward: Error evaluating sample {i}: {e}")
                # Fallback scores for all dimensions
                fallback_score = 5.0
                for dim in dimensions:
                    dimension_scores[dim][i] = fallback_score
                    reward_metrics[f"gpt_{dim}"].append(fallback_score)
                reward_metrics["overall"].append(fallback_score)
        
        # Calculate final reward tensor as average of all dimensions
        reward_tensor = torch.stack(list(dimension_scores.values())).mean(dim=0)
        
        print(f"[DEBUG] compute_reward: Multi-dimensional batch processing completed")
        print(f"[DEBUG] compute_reward: Final reward tensor shape: {reward_tensor.shape}")
        print(f"[DEBUG] compute_reward: Reward metrics keys: {list(reward_metrics.keys())}")
        print(f"[DEBUG] compute_reward: Average overall score: {reward_tensor.mean().item():.3f}")
        
        # Add dimension-specific statistics
        for dim in dimensions:
            avg_score = dimension_scores[dim].mean().item()
            print(f"[DEBUG] compute_reward: Average {dim} score: {avg_score:.3f}")
        
        return reward_tensor, dict(reward_metrics)
