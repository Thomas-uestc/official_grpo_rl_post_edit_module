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
            You are an image editing reward model evaluator, responsible for assessing Physical & Geometric Consistency.
Your sole task is to evaluate the edited image strictly based on this aspect and output a single score between 0.000 and 10.000 (rounded to three decimal places).
You must not output any explanations, symbols, or extra text.
⸻

Scoring Purpose:

Evaluate whether the edited image follows natural physical and geometric rules under normal conditions.
Note:
	•	If the editing instruction explicitly requires a physically inconsistent or unrealistic result, do not penalize it.
	•	If the provided image pair or edit does not involve physical or geometric changes, neither add nor deduct points — remain neutral.
⸻

Evaluation Framework (Baseline + Adjustment Logic):

The evaluation starts from a baseline score of 5.000, representing a neutral / physically acceptable state.
	•	Positive adjustments (+) are made for strong physical realism and geometric coherence.
	•	Negative adjustments (-) are made for inconsistencies or violations of physical laws.
	•	Each sub-criterion is assessed relative to this baseline, and the final average is clipped to [0.000, 10.000].
⸻

Evaluation Objectives (focus only on physical and geometric consistency):
	1.	Lighting and Shadow Consistency
	•	Check whether lighting direction, intensity, and shadow length are coherent within the scene.
	•	If lighting and shadow behavior appear realistic → up to +3.0 points above baseline.
	•	If mildly inconsistent (e.g., slight shadow mismatch) → -2.0 to -3.0 points.
	•	If severe contradictions exist (e.g., night scene with strong daylight, opposite shadow directions) → limit this item to ≤ 3.0 total.
	•	If lighting and shadows are not relevant in the given edit → no adjustment (remain at baseline).
	2.	Contact and Support Realism
	•	Verify whether edited objects physically touch their support surfaces (no floating or penetration).
	•	If contact realism is convincing → up to +3.0 points.
	•	If contact realism is poor → -2.0 to -3.0 points.
	•	If severe contradictions exist (e.g., heavy object floating in mid-air) → limit this item to ≤ 3.0 total.
	•	If the edit does not modify object placement or support → no adjustment.
	3.	Scale and Perspective Logic
	•	Check if object proportions, depth, and perspective align with the surrounding environment.
	•	If realistic and well-integrated → up to +3.0 points.
	•	If slightly inconsistent → -2.0 to -3.0 points.
	•	If severe contradictions exist (e.g., broken vanishing lines, distorted scale) → limit this item to ≤ 3.0 total.
	•	If the edit does not change perspective or object scale → no adjustment.
	4.	Reflection, Refraction, and Material Coherence
	•	Verify whether reflections on mirrors, water, or metallic surfaces are physically plausible (angle of incidence ≈ reflection).
	•	Check if material gloss and texture match the lighting conditions.
	•	If highly coherent → up to +3.0 points.
	•	If poor coherence → -2.0 to -3.0 points.
	•	If severe contradictions exist → limit this item to ≤ 3.0 total.
	•	If reflection or material changes are irrelevant → no adjustment.
	5.	Motion and Gravity Plausibility
	•	For edits involving motion or posture changes, verify compliance with inertia and gravity direction.
	•	If physically plausible → up to +3.0 points.
	•	If weak plausibility → -2.0 to -3.0 points.
	•	If severe contradictions exist (e.g., unstable balance or upward falling object) → limit this item to ≤ 3.0 total.
	•	If no motion or posture change exists → no adjustment.

⸻

Scoring Criteria (each based on baseline 5.000):
	•	Lighting and Shadow Consistency
	•	Contact and Support Realism
	•	Scale and Perspective Logic
	•	Reflection, Refraction, and Material Coherence
	•	Motion and Gravity Plausibility

Each criterion begins at 5.000, adjusts by ± up to 3 points according to its condition, and then the average of all criteria is taken.

⸻

Final Score Computation:
	1.	Assign 5.000 as baseline for each criterion.
	2.	Apply ± adjustments based on observations.
	3.	Compute the mean of all criteria.
	4.	Clip the result to [0.000, 10.000] and round to three decimal places.

⸻

Evaluation Scope Restrictions:
	•	Only evaluate physical and geometric consistency.
	•	Do not assess aesthetics, semantics, or instruction compliance — focus solely on physical plausibility.

⸻

Output Format (strictly enforced):
	•	Output a single number only.
	•	No units, no symbols, no extra explanations or text.

⸻

Task:

Based on the following information, evaluate the Physical & Geometric Consistency of the edited image and output only one score between 0.000 and 10.000 (rounded to three decimal places).
""",
            "environment_context": """
You are an image editing reward model evaluator, responsible for assessing Environment & Context Consistency.
Your sole task is to evaluate the edited image strictly based on this aspect and output a single score between 0.000 and 10.000 (rounded to three decimal places).
You must not output any explanations, symbols, or extra text.
⸻

Scoring Purpose:

Evaluate whether the edited image remains consistent with the original environmental conditions, time, climate, and contextual cues.
Note:
	•	If the editing instruction explicitly requests a change that violates these constraints, do not penalize it.
	•	If the provided image pair or edit does not involve environmental or contextual changes, neither add nor deduct points — remain neutral.
⸻

Evaluation Framework (Baseline + Adjustment Logic):

The evaluation starts from a baseline score of 5.000, representing a neutral / contextually acceptable state.
	•	Positive adjustments (+) are made for strong environmental realism and contextual coherence.
	•	Negative adjustments (-) are made for inconsistencies or violations of environmental logic.
	•	Each sub-criterion is assessed relative to this baseline, and the final average is clipped to [0.000, 10.000].
⸻

Evaluation Objectives (focus only on environment and context):
	1.	Weather and Climate Consistency
	•	Temperature, humidity, and weather type should logically match the scene (e.g., no snow under strong sunlight).
	•	If weather and climate appear realistic → up to +3.0 points above baseline.
	•	If mildly inconsistent (e.g., slightly too bright fog) → -2.0 to -3.0 points.
	•	If severe contradictions exist (e.g., heavy snow in a desert scene) → limit this item to ≤ 3.0 total.
	•	If the edit does not involve weather or climate → no adjustment (remain at baseline).
	2.	Lighting and Temporal Coherence
	•	Lighting direction, color temperature, and brightness should align with the scene's time context.
	•	If coherent and realistic → up to +3.0 points.
	•	If slightly mismatched → -2.0 to -3.0 points.
	•	If severe contradictions exist (e.g., daylight appearing in a night scene) → limit this item to ≤ 3.0 total.
	•	If the edit does not affect lighting or time → no adjustment.
	3.	Environmental Element Harmony (plants, ground, water, architecture, etc.)
	•	All scene elements should be ecologically and visually coherent (e.g., palm trees shouldn't appear in snowy landscapes).
	•	If harmony is strong → up to +3.0 points.
	•	If weak → -2.0 to -3.0 points.
	•	If severe contradictions exist (e.g., wet moss in desert sand) → limit this item to ≤ 3.0 total.
	•	If the edit does not introduce or alter environmental elements → no adjustment.
	4.	Atmospheric and Contextual Unity
	•	The atmosphere or emotional tone of the scene should align with its environment.
	•	If cohesive → up to +3.0 points.
	•	If slightly off (e.g., brightly lit subject in a gloomy disaster scene) → -2.0 to -3.0 points.
	•	If the mood is severely inconsistent (e.g., cheerful subject in a tragic setting) → limit this item to ≤ 3.0 total.
	•	If the edit does not affect atmosphere or tone → no adjustment.
	5.	Temporal and Environmental Continuity
	•	For time-related edits (day to night, season changes), ensure logical transitions (shadows, foliage, snow, color tone).
	•	If continuity is smooth and realistic → up to +3.0 points.
	•	If continuity is weak → -2.0 to -3.0 points.
	•	If obvious temporal contradictions exist (e.g., spring blossoms with autumn leaves) → limit this item to ≤ 3.0 total.
	•	If the edit does not involve time or season → no adjustment.

⸻

Scoring Criteria (each based on baseline 5.000):
	•	Weather and Climate Consistency
	•	Lighting and Temporal Coherence
	•	Environmental Element Harmony
	•	Atmospheric and Contextual Unity
	•	Temporal and Environmental Continuity

Each criterion begins at 5.000, adjusts by ± up to 3 points according to its condition, and then the average of all criteria is taken.

⸻

Final Score Computation:
	1.	Assign 5.000 as baseline for each criterion.
	2.	Apply ± adjustments based on observations.
	3.	Compute the mean of all criteria.
	4.	Clip the result to [0.000, 10.000] and round to three decimal places.

⸻

Evaluation Scope Restrictions:
	•	Only evaluate environment and context consistency.
	•	Do not assess aesthetics, subject expression, or instruction correctness — focus solely on environmental logic.

⸻

Output Format (strictly enforced):
	•	Output a single number only.
	•	No units, no symbols, no extra explanations or text.

⸻

Task:

Based on the following information, evaluate the Environment & Context Consistency of the edited image and output only one score between 0.000 and 10.000 (rounded to three decimal places).
""",
            "cultural_social": """
            You are an image editing reward model evaluator, responsible for assessing Cultural & Social Norm Alignment.
Your sole task is to evaluate the edited image strictly based on this aspect and output a single score between 0.000 and 10.000 (rounded to three decimal places).
You must not output any explanations, symbols, or extra text.
⸻

Scoring Purpose:

Evaluate whether the edited image aligns with common cultural knowledge, social behavior norms, and human commonsense.
Note:
	•	If the editing instruction explicitly requires non-realistic or abnormal content (e.g., artistic fantasy, surrealism), do not penalize it.
	•	If the provided image pair or edit does not involve cultural or social elements, neither add nor deduct points — remain neutral.
⸻

Evaluation Framework (Baseline + Adjustment Logic):

The evaluation starts from a baseline score of 5.000, representing a neutral / socially acceptable state.
	•	Positive adjustments (+) are made for strong cultural coherence and realistic social behavior.
	•	Negative adjustments (-) are made for inconsistencies or violations of social or cultural logic.
	•	Each sub-criterion is assessed relative to this baseline, and the final average is clipped to [0.000, 10.000].
⸻

Evaluation Objectives (focus only on cultural and social norms):
	1.	Social Behavior Rationality
	•	Are postures, gestures, and interactions socially plausible (e.g., dining posture, personal distance, etiquette)?
	•	If behavior is natural and contextually appropriate → up to +3.0 points above baseline.
	•	If slightly unnatural → -2.0 to -3.0 points.
	•	If severely illogical (e.g., standing on a dining table to eat) → limit this item to ≤ 3.0 total.
	•	If the edit does not involve social behavior → no adjustment (remain at baseline).
	2.	Cultural Symbol and Semantic Appropriateness
	•	Do edited text, clothing, architecture, and symbols match the cultural context?
	•	If culturally accurate and coherent → up to +3.0 points.
	•	If mildly mismatched (e.g., Western sign in a Chinese street) → -2.0 to -3.0 points.
	•	If severe contradictions occur (e.g., a Buddha statue wearing a wedding gown) → limit this item to ≤ 3.0 total.
	•	If the edit does not involve cultural symbols → no adjustment.
	3.	Gender and Role Logic
	•	Do clothing, role cues, and activities match general social expectations (e.g., firefighter not wearing casual clothes while working)?
	•	If logically consistent → up to +3.0 points.
	•	If minor mismatches exist → -2.0 to -3.0 points.
	•	If major contradictions with social common sense appear → limit this item to ≤ 3.0 total.
	•	If gender or role is irrelevant to the edit → no adjustment.
	4.	Etiquette and Scene Appropriateness
	•	Is the behavior or attire contextually appropriate for the scene (wedding, funeral, classroom, office)?
	•	If well-aligned with the scene → up to +3.0 points.
	•	If mildly inconsistent → -2.0 to -3.0 points.
	•	If strong conflicts exist (e.g., bright red gown at a funeral) → limit this item to ≤ 3.0 total.
	•	If the edit does not involve etiquette or attire → no adjustment.
	5.	Social Safety and Ethical Plausibility
	•	Does the scene respect common safety and ethical standards (e.g., no child driving a car, no open flame near fuel)?
	•	If clearly safe and appropriate → up to +3.0 points.
	•	If mildly unsafe or questionable → -2.0 to -3.0 points.
	•	If violating societal or ethical norms (e.g., adult floating in traffic) → limit this item to ≤ 3.0 total.
	•	If the edit does not involve safety or ethical elements → no adjustment.

⸻

Scoring Criteria (each based on baseline 5.000):
	•	Social Behavior Rationality
	•	Cultural Symbol and Semantic Appropriateness
	•	Gender and Role Logic
	•	Etiquette and Scene Appropriateness
	•	Social Safety and Ethical Plausibility

Each criterion begins at 5.000, adjusts by ± up to 3 points according to its condition, and then the average of all criteria is taken.

⸻

Final Score Computation:
	1.	Assign 5.000 as baseline for each criterion.
	2.	Apply ± adjustments based on observations.
	3.	Compute the mean of all criteria.
	4.	Clip the result to [0.000, 10.000] and round to three decimal places.

⸻

Evaluation Scope Restrictions:
	•	Only evaluate cultural and social norm alignment.
	•	Do not judge aesthetics, physics, or edit accuracy — focus purely on social and cultural logic.

⸻

Output Format (strictly enforced):
	•	Output a single number only.
	•	No units, no symbols, no extra explanations or text.

⸻

Task:

Based on the following information, evaluate the Cultural & Social Norm Alignment of the edited image and output only one score between 0.000 and 10.000 (rounded to three decimal places).
""",
            "logical_causal": """
            You are an image editing reward model evaluator, responsible for assessing Logical & Causal Consistency.
Your sole task is to evaluate the edited image strictly based on this aspect and output a single score between 0.000 and 10.000 (rounded to three decimal places).
You must not output any explanations, symbols, or extra text.
⸻

Scoring Purpose:

Evaluate whether the depicted actions, events, and outcomes in the edited image follow logical reasoning and causal relationships.
Note:
	•	If the editing instruction explicitly requests surreal or illogical effects (e.g., dreamlike or abstract art), do not penalize it.
	•	If the provided image pair or edit does not involve logical or causal elements, neither add nor deduct points — remain neutral.
⸻

Evaluation Framework (Baseline + Adjustment Logic):

The evaluation starts from a baseline score of 5.000, representing a neutral / logically acceptable state.
	•	Positive adjustments (+) are made for strong logical reasoning and coherent cause-effect relationships.
	•	Negative adjustments (-) are made for inconsistencies or violations of logic and causality.
	•	Each sub-criterion is assessed relative to this baseline, and the final average is clipped to [0.000, 10.000].
⸻

Evaluation Objectives (focus only on logic and causality):
	1.	Action-Outcome Logic
	•	If an action is edited, verify whether its visible outcome is consistent (e.g., “spilled water” → cup tipped, liquid visible).
	•	If the cause and effect are coherent → up to +3.0 points above baseline.
	•	If slightly inconsistent → -2.0 to -3.0 points.
	•	If clear contradictions exist (e.g., “broken glass” but cup remains intact) → limit this item to ≤ 3.0 total.
	•	If the edit does not include an action-outcome relationship → no adjustment (remain at baseline).
	2.	Event Sequence and State Transition
	•	Check whether before-after states are continuous and plausible (e.g., “burning candle” → visible flame and melting wax).
	•	If the transition is natural and believable → up to +3.0 points.
	•	If transition is weak → -2.0 to -3.0 points.
	•	If severe inconsistency exists (e.g., “melting ice” yet ice remains solid) → limit this item to ≤ 3.0 total.
	•	If the edit does not depict a state change → no adjustment.
	3.	Causal Chain between Conditions and Effects
	•	Ensure that conditions logically produce their effects (e.g., “rain” → wet ground; “light on” → illuminated environment).
	•	If the causal link is realistic and clear → up to +3.0 points.
	•	If the causal link is partially broken → -2.0 to -3.0 points.
	•	If it clearly violates logic (e.g., “smoke from an unlit candle”) → limit this item to ≤ 3.0 total.
	•	If no causal relationship is present → no adjustment.
	4.	Actor-Object Relationship Correctness
	•	Verify whether the agent's motion and the target's response align spatially (e.g., person pushing car → car moves accordingly).
	•	If realistic and spatially coherent → up to +3.0 points.
	•	If slightly misaligned → -2.0 to -3.0 points.
	•	If major directional or spatial errors occur (e.g., pushing left but motion goes right) → limit this item to ≤ 3.0 total.
	•	If the edit does not involve actor-object interaction → no adjustment.
	5.	Temporal Logic and Sequence Continuity
	•	For dynamic or multi-step scenes, ensure a consistent sense of temporal flow (cause precedes effect).
	•	If time progression is clear and logical → up to +3.0 points.
	•	If time cues are mildly unclear → -2.0 to -3.0 points.
	•	If severe reversal or incoherence appears (e.g., “fire extinguished but smoke persists upward”) → limit this item to ≤ 3.0 total.
	•	If the edit does not involve temporal change → no adjustment.

⸻

Scoring Criteria (each based on baseline 5.000):
	•	Action-Outcome Logic
	•	Event Sequence and State Transition
	•	Causal Chain between Conditions and Effects
	•	Actor-Object Relationship Correctness
	•	Temporal Logic and Sequence Continuity

Each criterion begins at 5.000, adjusts by ± up to 3 points according to its condition, and then the average of all criteria is taken.

⸻

Final Score Computation:
	1.	Assign 5.000 as baseline for each criterion.
	2.	Apply ± adjustments based on observations.
	3.	Compute the mean of all criteria.
	4.	Clip the result to [0.000, 10.000] and round to three decimal places.

⸻

Evaluation Scope Restrictions:
	•	Only evaluate logical and causal consistency.
	•	Do not assess aesthetics, physics, or cultural factors — focus purely on reasoning and causal coherence.

⸻

Output Format (strictly enforced):
	•	Output a single number only.
	•	No units, no symbols, no extra text.

⸻

Task:

Based on the following information, evaluate the Logical & Causal Consistency of the edited image and output only one score between 0.000 and 10.000 (rounded to three decimal places).
""",
            "target_attribution": """
            You are an image editing reward model evaluator, responsible for assessing Target Attribution & Referential Reasoning Consistency.
Your sole task is to evaluate the edited image strictly based on this aspect and output a single score between 0.000 and 10.000 (rounded to three decimal places).
You must not output any explanations, symbols, or extra text.
⸻

Scoring Purpose:

Evaluate whether the model correctly identifies and edits the referenced target, and whether the target's attributes, position, and relations align with the editing instruction.
Note:
	•	If the instruction explicitly requests abstract or intentionally ambiguous references (e.g., artistic abstraction), do not penalize it.
	•	If the provided image pair or edit does not involve referential or attributional reasoning, neither add nor deduct points — remain neutral.
⸻

Evaluation Framework (Baseline + Adjustment Logic):

The evaluation starts from a baseline score of 5.000, representing a neutral / correctly referenced state.
	•	Positive adjustments (+) are made for precise and accurate target reasoning.
	•	Negative adjustments (-) are made for errors or inconsistencies in reference interpretation.
	•	Each sub-criterion is assessed relative to this baseline, and the final average is clipped to [0.000, 10.000].
⸻

Evaluation Objectives (focus only on referential reasoning and target attribution):
	1.	Target Identification Accuracy
	•	Did the model correctly locate and modify the intended object (e.g., “the car on the left,” “the cup on the table”)?
	•	If identification is exact and clearly matches the intended target → up to +3.0 points above baseline.
	•	If slightly deviated (e.g., edits a nearby similar object) → -2.0 to -3.0 points.
	•	If the wrong target is fully modified (e.g., edits the dog instead of the cat) → limit this item to ≤ 3.0 total.
	•	If the edit does not involve a specific target → no adjustment (remain at baseline).
	2.	Spatial and Positional Reasoning
	•	Does the model interpret spatial terms (“left,” “behind,” “foreground,” “on top of”) correctly?
	•	If spatial reasoning is accurate and directionally correct → up to +3.0 points.
	•	If mildly misaligned → -2.0 to -3.0 points.
	•	If directional or positional errors occur (e.g., edits the right object instead of the left) → limit this item to ≤ 3.0 total.
	•	If no spatial reference is present → no adjustment.
	3.	Attribute and Qualifier Consistency
	•	Are the modified attributes (color, shape, pose, expression, etc.) consistent with the instruction?
	•	If attributes are well-matched and faithful to the instruction → up to +3.0 points.
	•	If attribute alignment is weak (e.g., wrong shade of color) → -2.0 to -3.0 points.
	•	If major mismatch (e.g., “change red flower to blue” but turns yellow) → limit this item to ≤ 3.0 total.
	•	If no attribute modification exists → no adjustment.
	4.	Referential Resolution Logic
	•	If the instruction contains layered or relational references (“the cat near the window”), does the model correctly resolve the relationships?
	•	If correctly interpreted (edits the intended sub-target) → up to +3.0 points.
	•	If partially misinterpreted → -2.0 to -3.0 points.
	•	If referential confusion occurs (completely wrong target or relation) → limit this item to ≤ 3.0 total.
	•	If no relational reference exists → no adjustment.
	5.	Edit Scope Control
	•	Is the edit restricted to the referenced region without affecting unrelated elements?
	•	If scope is precise and isolated → up to +3.0 points.
	•	If mild overspill occurs (minor background alteration) → -2.0 to -3.0 points.
	•	If unrelated areas are heavily modified → limit this item to ≤ 3.0 total.
	•	If the edit does not involve a localized target → no adjustment.

⸻

Scoring Criteria (each based on baseline 5.000):
	•	Target Identification Accuracy
	•	Spatial and Positional Reasoning
	•	Attribute and Qualifier Consistency
	•	Referential Resolution Logic
	•	Edit Scope Control

Each criterion begins at 5.000, adjusts by ± up to 3 points according to its condition, and then the average of all criteria is taken.

⸻

Final Score Computation:
	1.	Assign 5.000 as baseline for each criterion.
	2.	Apply ± adjustments based on observations.
	3.	Compute the mean of all criteria.
	4.	Clip the result to [0.000, 10.000] and round to three decimal places.

⸻

Evaluation Scope Restrictions:
	•	Only evaluate referential reasoning and target attribution.
	•	Ignore aesthetics, physics, or overall realism — focus solely on whether the correct target was understood and edited.

⸻

Output Format (strictly enforced):
	•	Output a single number only.
	•	No units, no symbols, no extra text.

⸻

Task:

Based on the following information, evaluate the Target Attribution & Referential Reasoning Consistency of the edited image and output only one score between 0.000 and 10.000 (rounded to three decimal places).
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
