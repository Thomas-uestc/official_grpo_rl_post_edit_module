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
            Environment & Context Consistency

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
	1.	Assign 5.000 as baseline for each c