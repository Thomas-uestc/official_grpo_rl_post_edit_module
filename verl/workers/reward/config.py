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
Reward config
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardConfig:
    def __post_init__(self):
        self.post_init()
    reward_type: str = "batch"
    reward_function: Optional[str] = None
    reward_function_kwargs: dict = field(default_factory=dict)
    skip_special_tokens: bool = True
    num_cpus: int = 1
    
    # GPT-4.1 API configuration
    use_gpt41_api: bool = False
    gpt41_api_key: Optional[str] = None
    gpt41_api_base: str = "https://api.openai.com/v1"
    gpt41_model: str = "gpt-4.1"
    gpt41_max_tokens: int = 1000
    gpt41_temperature: float = 0.0
    gpt41_max_retries: int = 3
    gpt41_retry_delay: float = 1.0
    gpt41_request_timeout: float = 30.0
    
    # Rule-based format reward configuration
    rule_re_edit_max_length_ideal: int = 50
    rule_re_edit_max_length_acceptable: int = 100
    rule_re_edit_max_length_tolerable: int = 150
    
    # Combined reward configuration - 6 dimensions
    # 5 GPT consistency evaluation dimensions (0-10 scale each)
    combined_gpt_physical_geometric_weight: float = 0.15
    combined_gpt_environment_context_weight: float = 0.15
    combined_gpt_cultural_social_weight: float = 0.15
    combined_gpt_logical_causal_weight: float = 0.15
    combined_gpt_target_attribution_weight: float = 0.15
    
    # 1 format evaluation dimension (0-5 scale, auto-scaled to 0-10)
    combined_rule_format_weight: float = 0.25
    
    # Legacy 2-dimension weights (for backward compatibility)
    combined_gpt_weight: float = 0.75  # Sum of 5 GPT dimensions
    combined_rule_weight: float = 0.25  # Format dimension weight
    
    # Combination strategy and thresholds
    combined_strategy: str = "weighted_sum"  # "weighted_sum", "gated", "multiplicative", "max_deviation"
    combined_rule_gate_threshold: float = 2.5  # Updated for 0-5 scale
    
    # Max deviation strategy weights (for max_deviation strategy)
    max_deviation_gpt_weight: float = 0.8  # GPT API reward weight
    max_deviation_rule_weight: float = 0.2  # Rule-based reward weight
    
    # below are auto keys
    reward_function_name: Optional[str] = field(default=None, init=False)

    def post_init(self):
        if self.reward_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.reward_function:
                self.reward_function_name = "main"
            else:
                self.reward_function, self.reward_function_name = self.reward_function.rsplit(":", maxsplit=1)

            if os.path.exists(self.reward_function):  # ray job uses absolute path
                self.reward_function = os.path.abspath(self.reward_function)
            else:
                print(f"Reward function {self.reward_function} not found.")
                self.reward_function = None
        
        # GPT-4.1 API key validation
        if self.use_gpt41_api or self.reward_type in ["gpt41", "combined"]:
            if self.gpt41_api_key is None:
                self.gpt41_api_key = os.getenv("OPENAI_API_KEY")
                if self.gpt41_api_key is None and self.reward_type in ["gpt41", "combined"]:
                    raise ValueError("GPT-4.1 API key not provided. Set OPENAI_API_KEY environment variable or configure gpt41_api_key.")
        
        # Rule-based reward validation
        if self.reward_type in ["rule_based_format", "combined"]:
            # Validate length thresholds are in ascending order
            if not (self.rule_re_edit_max_length_ideal <= self.rule_re_edit_max_length_acceptable <= self.rule_re_edit_max_length_tolerable):
                raise ValueError(f"Rule-based reward length thresholds must be in ascending order: "
                               f"ideal({self.rule_re_edit_max_length_ideal}) <= "
                               f"acceptable({self.rule_re_edit_max_length_acceptable}) <= "
                               f"tolerable({self.rule_re_edit_max_length_tolerable})")
        
        # Combined reward validation
        if self.reward_type == "combined":
            # Validate combination strategy
            valid_strategies = ["weighted_sum", "gated", "multiplicative", "max_deviation"]
            if self.combined_strategy not in valid_strategies:
                raise ValueError(f"Invalid combined_strategy '{self.combined_strategy}'. "
                               f"Must be one of: {valid_strategies}")
            
            # Validate max deviation strategy weights first (before 6-dimension validation)
            if self.combined_strategy == "max_deviation":
                if not (0.0 <= self.max_deviation_gpt_weight <= 1.0):
                    raise ValueError(f"Max deviation GPT weight must be in [0, 1], got {self.max_deviation_gpt_weight}")
                if not (0.0 <= self.max_deviation_rule_weight <= 1.0):
                    raise ValueError(f"Max deviation rule weight must be in [0, 1], got {self.max_deviation_rule_weight}")
                
                # Check weights sum to 1.0
                total_max_deviation_weight = self.max_deviation_gpt_weight + self.max_deviation_rule_weight
                if abs(total_max_deviation_weight - 1.0) > 1e-6:
                    print(f"[INFO] Normalizing max deviation weights from sum {total_max_deviation_weight:.6f} to 1.0")
                    self.max_deviation_gpt_weight /= total_max_deviation_weight
                    self.max_deviation_rule_weight /= total_max_deviation_weight
            else:
                # Validate 6-dimension weights (only for non-max_deviation strategies)
                dimension_weights = [
                    self.combined_gpt_physical_geometric_weight,
                    self.combined_gpt_environment_context_weight,
                    self.combined_gpt_cultural_social_weight,
                    self.combined_gpt_logical_causal_weight,
                    self.combined_gpt_target_attribution_weight,
                    self.combined_rule_format_weight
                ]
                
                # Check all weights are non-negative
                for i, weight in enumerate(dimension_weights):
                    if weight < 0:
                        raise ValueError(f"All dimension weights must be non-negative, got {weight} at index {i}")
                
                # Check weights sum is not zero
                total_weight = sum(dimension_weights)
                if abs(total_weight) < 1e-6:
                    raise ValueError("Sum of all dimension weights cannot be zero")
                
                # Auto-normalize weights if they don't sum to 1.0
                if abs(total_weight - 1.0) > 1e-6:
                    print(f"[INFO] Normalizing weights from sum {total_weight:.6f} to 1.0")
                    self.combined_gpt_physical_geometric_weight /= total_weight
                    self.combined_gpt_environment_context_weight /= total_weight
                    self.combined_gpt_cultural_social_weight /= total_weight
                    self.combined_gpt_logical_causal_weight /= total_weight
                    self.combined_gpt_target_attribution_weight /= total_weight
                    self.combined_rule_format_weight /= total_weight
                
                # Update legacy weights for backward compatibility
                self.combined_gpt_weight = (self.combined_gpt_physical_geometric_weight + 
                                          self.combined_gpt_environment_context_weight +
                                          self.combined_gpt_cultural_social_weight +
                                          self.combined_gpt_logical_causal_weight +
                                          self.combined_gpt_target_attribution_weight)
                self.combined_rule_weight = self.combined_rule_format_weight
            
            # Validate gate threshold (updated for 0-5 scale)
            if not (0.0 <= self.combined_rule_gate_threshold <= 5.0):
                raise ValueError(f"Combined rule gate threshold must be in [0, 5], got {self.combined_rule_gate_threshold}")
            
            print(f"[INFO] 6-Dimension Combined reward configuration:")
            print(f"  Strategy: {self.combined_strategy}")
            if self.combined_strategy == "max_deviation":
                print(f"  Max Deviation GPT Weight: {self.max_deviation_gpt_weight:.3f}")
                print(f"  Max Deviation Rule Weight: {self.max_deviation_rule_weight:.3f}")
            else:
                print(f"  GPT Physical & Geometric: {self.combined_gpt_physical_geometric_weight:.3f}")
                print(f"  GPT Environment & Context: {self.combined_gpt_environment_context_weight:.3f}")
                print(f"  GPT Cultural & Social: {self.combined_gpt_cultural_social_weight:.3f}")
                print(f"  GPT Logical & Causal: {self.combined_gpt_logical_causal_weight:.3f}")
                print(f"  GPT Target Attribution: {self.combined_gpt_target_attribution_weight:.3f}")
                print(f"  Rule Format: {self.combined_rule_format_weight:.3f}")
            if self.combined_strategy == "gated":
                print(f"  Gate threshold: {self.combined_rule_gate_threshold:.3f}")
        
        # Reward type validation
        valid_reward_types = ["batch", "sequential", "gpt41", "rule_based_format", "combined"]
        if self.reward_type not in valid_reward_types:
            print(f"[WARNING] Unknown reward_type '{self.reward_type}'. Valid types: {valid_reward_types}")
        
        print(f"[INFO] Reward system configured: {self.reward_type}")
