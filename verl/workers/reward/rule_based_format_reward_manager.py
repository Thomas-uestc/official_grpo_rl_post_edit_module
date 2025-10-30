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
Rule-Based Format Reward Manager for image editing evaluation
"""

import re
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from ...protocol import DataProto
from ...single_controller.base import Worker
from ...single_controller.base.decorator import Dispatch, register
from ..reward.config import RewardConfig


class RuleBasedFormatRewardManager(Worker):
    """Rule-based format reward manager for image editing evaluation"""
    
    def __init__(self, config: RewardConfig, tokenizer=None):
        # Skip parent __init__ to avoid CUDA initialization
        # super().__init__()
        print(f"[DEBUG] RuleBasedFormatRewardManager: Initializing with config and tokenizer...")
        self.config = config
        self.tokenizer = tokenizer
        
        # 编译正则表达式以提高性能
        self.cot_pattern = re.compile(r'<CoT>(.*?)</CoT>', re.DOTALL)
        self.re_edit_pattern = re.compile(r'<Re_edit>(.*?)</Re_edit>', re.DOTALL)
        self.all_tags_pattern = re.compile(r'(<CoT>.*?</CoT>|<Re_edit>.*?</Re_edit>)', re.DOTALL)
        
        # 配置参数
        self.re_edit_max_length_ideal = getattr(config, 'rule_re_edit_max_length_ideal', 50)
        self.re_edit_max_length_acceptable = getattr(config, 'rule_re_edit_max_length_acceptable', 100)
        self.re_edit_max_length_tolerable = getattr(config, 'rule_re_edit_max_length_tolerable', 150)
        
        print(f"[DEBUG] RuleBasedFormatRewardManager: Initialized successfully")
        print(f"[DEBUG] Re_edit length thresholds - Ideal: {self.re_edit_max_length_ideal}, "
              f"Acceptable: {self.re_edit_max_length_acceptable}, Tolerable: {self.re_edit_max_length_tolerable}")
        
    def has_required_tags(self, predict_str: str) -> bool:
        """检查是否存在必需的标签"""
        has_cot = bool(self.cot_pattern.search(predict_str))
        has_re_edit = bool(self.re_edit_pattern.search(predict_str))
        return has_cot and has_re_edit
    
    def check_tag_counts(self, predict_str: str) -> bool:
        """
        检查标签数量是否符合要求
        注意：新版本不再限制具体数量，只要存在至少1个CoT和1个Re_edit即可
        保留此方法是为了兼容性，实际检查逻辑已简化
        """
        cot_matches = self.cot_pattern.findall(predict_str)
        re_edit_matches = self.re_edit_pattern.findall(predict_str)
        
        # 新规则：只要至少有1个CoT和1个Re_edit即可，不限制上限
        cot_count_valid = len(cot_matches) >= 1
        re_edit_count_valid = len(re_edit_matches) >= 1
        
        return cot_count_valid and re_edit_count_valid
    
    def check_no_external_content(self, predict_str: str) -> bool:
        """
        检查是否存在标签外的内容
        允许标签块之间存在换行符，但不允许其他内容
        """
        # 找到所有标签的位置
        matches = list(self.all_tags_pattern.finditer(predict_str))
        
        if not matches:
            return False
        
        # 检查标签前是否有非空白内容
        first_match = matches[0]
        before_first = predict_str[:first_match.start()].strip()
        if before_first:
            return False
        
        # 检查标签后是否有非空白内容  
        last_match = matches[-1]
        after_last = predict_str[last_match.end():].strip()
        if after_last:
            return False
        
        # 检查标签之间是否只有换行符/空白字符
        for i in range(len(matches) - 1):
            current_end = matches[i].end()
            next_start = matches[i + 1].start()
            between_content = predict_str[current_end:next_start]
            
            # 只允许空白字符（包括换行符、空格、制表符等）
            if between_content.strip():
                return False
        
        return True
    
    def check_blocks_on_separate_lines(self, predict_str: str) -> Tuple[bool, int]:
        """
        检查每个block是否独立成行
        
        Returns:
            Tuple[bool, int]: (是否所有block都独立成行, 违规block数量)
        """
        # 按行分割文本
        lines = predict_str.split('\n')
        violation_count = 0
        
        # 找到所有标签
        matches = list(self.all_tags_pattern.finditer(predict_str))
        
        for match in matches:
            tag_text = match.group(0)
            tag_start = match.start()
            tag_end = match.end()
            
            # 找到这个标签所在的行
            current_pos = 0
            tag_line_found = False
            
            for line_idx, line in enumerate(lines):
                line_start = current_pos
                line_end = current_pos + len(line)
                
                # 检查标签是否在这一行中
                if line_start <= tag_start < line_end:
                    # 检查这一行是否只包含这个标签（除了空白字符）
                    line_content = line.strip()
                    tag_content = tag_text.strip()
                    
                    if line_content != tag_content:
                        # 这一行包含了标签以外的内容
                        violation_count += 1
                        print(f"[DEBUG] Block violation found - Line: '{line.strip()}', Expected: '{tag_content}'")
                    
                    tag_line_found = True
                    break
                
                # 移动到下一行的开始位置（+1是为了换行符）
                current_pos = line_end + 1
            
            if not tag_line_found:
                # 标签跨行了，这也是违规
                violation_count += 1
                print(f"[DEBUG] Block spans multiple lines: '{tag_text[:50]}...'")
        
        all_blocks_separate = violation_count == 0
        return all_blocks_separate, violation_count
    
    def check_hard_requirements(self, predict_str: str) -> bool:
        """
        检查所有硬性要求，任一不满足直接返回False
        
        硬性要求（新版本，已移除标签数量限制）:
        1. 必须存在<CoT></CoT>和<Re_edit></Re_edit>标签（至少各1个）
        2. 不能存在标签外的内容（允许换行符）
        """
        # 要求1: 必须存在<CoT></CoT>和<Re_edit></Re_edit>标签
        if not self.has_required_tags(predict_str):
            return False
        
        # 要求2: 不能存在标签外的内容（允许换行符）
        if not self.check_no_external_content(predict_str):
            return False
            
        return True
    
    def calculate_re_edit_length_penalty(self, predict_str: str) -> float:
        """
        计算Re_edit长度惩罚
        
        新版本支持多个<Re_edit>标签:
        - 计算所有<Re_edit>标签的平均长度
        - 基于平均长度应用惩罚规则
        """
        re_edit_matches = self.re_edit_pattern.findall(predict_str)
        if not re_edit_matches:
            return 0.0
        
        # 计算所有Re_edit内容的平均长度
        total_length = sum(len(content.strip()) for content in re_edit_matches)
        avg_length = total_length / len(re_edit_matches)
        
        # 长度惩罚规则 (0-5分制) - 基于平均长度
        if avg_length <= self.re_edit_max_length_ideal:  # 理想长度
            return 0.0
        elif avg_length <= self.re_edit_max_length_acceptable:  # 可接受长度
            return 0.5
        elif avg_length <= self.re_edit_max_length_tolerable:  # 过长但可容忍
            return 1.5
        else:  # 严重过长
            return 2.5
    
    def calculate_line_separation_penalty(self, predict_str: str) -> Tuple[float, int]:
        """
        计算block独立成行惩罚
        
        Returns:
            Tuple[float, int]: (惩罚分数, 违规block数量)
        """
        all_blocks_separate, violation_count = self.check_blocks_on_separate_lines(predict_str)
        
        if all_blocks_separate:
            return 0.0, 0
        
        # 每个违规block惩罚1.0分
        penalty = float(violation_count) * 1.0
        return penalty, violation_count
    
    def calculate_soft_requirements_score(self, predict_str: str) -> Tuple[float, Dict[str, Any]]:
        """
        计算软性要求得分，满分5.0
        
        Returns:
            Tuple[float, Dict]: (最终得分, 详细惩罚信息)
        """
        base_score = 5.0  # 硬性要求通过后的基础分
        penalty_details = {}
        
        # 要求4: Re_edit内容长度检查
        re_edit_penalty = self.calculate_re_edit_length_penalty(predict_str)
        penalty_details["re_edit_length_penalty"] = re_edit_penalty
        
        # 要求5: Block独立成行检查
        line_separation_penalty, violation_count = self.calculate_line_separation_penalty(predict_str)
        penalty_details["line_separation_penalty"] = line_separation_penalty
        penalty_details["line_violation_count"] = violation_count
        
        # 计算总惩罚
        total_penalty = re_edit_penalty + line_separation_penalty
        penalty_details["total_penalty"] = total_penalty
        
        final_score = base_score - total_penalty
        return max(0.0, final_score), penalty_details  # 确保不小于0
    
    def image_edit_format_reward(self, predict_str: str) -> Dict[str, Any]:
        """
        图像编辑格式奖励计算
        
        评分规则（新版本 - 支持多标签）:
        - 硬性要求不满足: 直接返回0.0
          1. 必须至少包含1个<CoT>和1个<Re_edit>标签
          2. 不能有标签外的内容
        - 硬性要求满足: 基础分5.0，然后根据软性要求扣减
          1. Re_edit平均长度惩罚（基于所有Re_edit标签的平均长度）
          2. Block独立成行惩罚
        
        注意: 不再限制标签数量上限，支持多个CoT和Re_edit标签
        
        Args:
            predict_str: 模型预测输出字符串
            
        Returns:
            Dict: 包含详细评分信息的字典
        """
        result = {
            "overall_score": 0.0,
            "hard_requirements_passed": False,
            "has_required_tags": False,
            "tag_counts_valid": False,
            "no_external_content": False,
            "blocks_on_separate_lines": False,
            "re_edit_length_penalty": 0.0,
            "line_separation_penalty": 0.0,
            "line_violation_count": 0,
            "total_penalty": 0.0,
            "re_edit_content_length": 0,
            "cot_count": 0,
            "re_edit_count": 0,
            "error": None
        }
        
        try:
            # 预处理：移除首尾空白
            predict_str = predict_str.strip()
            
            # 硬性要求检查
            result["has_required_tags"] = self.has_required_tags(predict_str)
            result["tag_counts_valid"] = self.check_tag_counts(predict_str)
            result["no_external_content"] = self.check_no_external_content(predict_str)
            
            # 检查block独立成行（软性要求，但需要记录）
            blocks_separate, violation_count = self.check_blocks_on_separate_lines(predict_str)
            result["blocks_on_separate_lines"] = blocks_separate
            result["line_violation_count"] = violation_count
            
            # 统计标签数量
            cot_matches = self.cot_pattern.findall(predict_str)
            re_edit_matches = self.re_edit_pattern.findall(predict_str)
            result["cot_count"] = len(cot_matches)
            result["re_edit_count"] = len(re_edit_matches)
            
            # 记录Re_edit内容平均长度（支持多个Re_edit标签）
            if re_edit_matches:
                total_length = sum(len(content.strip()) for content in re_edit_matches)
                avg_length = total_length / len(re_edit_matches)
                result["re_edit_content_length"] = int(avg_length)  # 记录平均长度（取整）
            
            # 检查硬性要求
            hard_requirements_passed = self.check_hard_requirements(predict_str)
            result["hard_requirements_passed"] = hard_requirements_passed
            
            if not hard_requirements_passed:
                result["overall_score"] = 0.0
                return result
            
            # 软性要求评分（包括新的独立成行惩罚）
            soft_score, penalty_details = self.calculate_soft_requirements_score(predict_str)
            
            # 更新结果中的惩罚信息
            result["re_edit_length_penalty"] = penalty_details["re_edit_length_penalty"]
            result["line_separation_penalty"] = penalty_details["line_separation_penalty"]
            result["total_penalty"] = penalty_details["total_penalty"]
            result["overall_score"] = soft_score
            
            return result
            
        except Exception as e:
            # 异常情况返回0分
            result["error"] = str(e)
            result["overall_score"] = 0.0
            return result
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        计算一批数据的格式奖励
        
        Args:
            data: DataProto containing model outputs
            
        Returns:
            Tuple of (reward_tensor, reward_metrics)
        """
        print(f"[DEBUG] RuleBasedFormatRewardManager.compute_reward: Starting reward computation")
        print(f"[DEBUG] Available keys in data.batch: {list(data.batch.keys()) if hasattr(data, 'batch') and data.batch is not None else 'No batch data'}")
        print(f"[DEBUG] Available keys in data.non_tensor_batch: {list(data.non_tensor_batch.keys())}")
        
        # 提取模型输出文本 - 模型输出通常在data.batch中而不是non_tensor_batch中
        model_outputs = None
        
        # 首先检查data.batch中的responses字段（最常见的情况）
        if hasattr(data, 'batch') and data.batch is not None and "responses" in data.batch:
            print(f"[DEBUG] Found 'responses' in data.batch")
            model_outputs = data.batch["responses"]
            print(f"[DEBUG] Using data.batch['responses'] for model outputs, type: {type(model_outputs)}")
        elif hasattr(data, 'batch') and data.batch is not None and "prompts" in data.batch:
            # 如果没有responses，检查是否有prompts（可能包含完整的对话）
            print(f"[DEBUG] Found 'prompts' in data.batch, checking if it contains responses")
            prompts = data.batch["prompts"]
            if isinstance(prompts, list) and len(prompts) > 0:
                # prompts可能包含了完整的对话，我们需要提取响应部分
                model_outputs = prompts  # 先尝试使用prompts
                print(f"[DEBUG] Using data.batch['prompts'] for model outputs, type: {type(model_outputs)}")
        elif "responses" in data.non_tensor_batch:
            # 备选：检查non_tensor_batch中的responses字段
            model_outputs = data.non_tensor_batch["responses"]
            print(f"[DEBUG] Using data.non_tensor_batch['responses'] for model outputs, type: {type(model_outputs)}")
        else:
            # 尝试其他可能的字段名
            possible_fields_batch = ["outputs", "generated_texts", "predictions", "text_outputs"]
            possible_fields_non_tensor = ["generated_texts", "outputs", "predictions", "text_outputs"]
            
            # 先检查data.batch
            if hasattr(data, 'batch') and data.batch is not None:
                for field in possible_fields_batch:
                    if field in data.batch:
                        model_outputs = data.batch[field]
                        print(f"[DEBUG] Using data.batch['{field}'] for model outputs, type: {type(model_outputs)}")
                        break
            
            # 再检查data.non_tensor_batch
            if model_outputs is None:
                for field in possible_fields_non_tensor:
                    if field in data.non_tensor_batch:
                        model_outputs = data.non_tensor_batch[field]
                        print(f"[DEBUG] Using data.non_tensor_batch['{field}'] for model outputs, type: {type(model_outputs)}")
                        break
            
            if model_outputs is None:
                batch_keys = list(data.batch.keys()) if hasattr(data, 'batch') and data.batch is not None else []
                non_tensor_keys = list(data.non_tensor_batch.keys())
                print(f"[ERROR] Could not find model outputs in data.")
                print(f"[ERROR] Available batch keys: {batch_keys}")
                print(f"[ERROR] Available non_tensor_batch keys: {non_tensor_keys}")
                # 返回全零奖励作为fallback
                batch_size = len(data.non_tensor_batch.get("original_images", [1]))
                reward_tensor = torch.zeros(batch_size, dtype=torch.float32, device='cpu')
                reward_metrics = {"overall": [0.0] * batch_size}
                return reward_tensor, reward_metrics
        
        print(f"[DEBUG] Model outputs type: {type(model_outputs)}")
        
        # 转换numpy数组为列表
        if isinstance(model_outputs, np.ndarray):
            print(f"[DEBUG] Converting model_outputs from numpy array to list")
            model_outputs = model_outputs.tolist()
        
        # 初始化奖励张量和指标
        batch_size = len(model_outputs)
        print(f"[DEBUG] Processing batch of size {batch_size}")
        reward_tensor = torch.zeros(batch_size, dtype=torch.float32, device='cpu')
        reward_metrics = defaultdict(list)
        
        # 处理每个样本
        for i in range(batch_size):
            print(f"[DEBUG] Processing sample {i+1}/{batch_size}")
            try:
                output_data = model_outputs[i]
                print(f"[DEBUG] Sample {i} raw output type: {type(output_data)}")
                
                # 解码模型输出
                print(f"[DEBUG] Sample {i} tokenizer available: {self.tokenizer is not None}")
                print(f"[DEBUG] Sample {i} output_data type: {type(output_data)}")
                print(f"[DEBUG] Sample {i} output_data is tensor: {isinstance(output_data, torch.Tensor)}")
                
                if self.tokenizer is not None and isinstance(output_data, torch.Tensor):
                    # 如果有tokenizer且输出是tensor，尝试解码
                    print(f"[DEBUG] Sample {i} attempting tokenizer decode")
                    try:
                        # 获取skip_special_tokens配置，默认为True
                        skip_special_tokens = getattr(self.config, 'skip_special_tokens', True)
                        print(f"[DEBUG] Sample {i} skip_special_tokens: {skip_special_tokens}")
                        
                        output_text = self.tokenizer.decode(output_data, skip_special_tokens=skip_special_tokens)
                        print(f"[DEBUG] Sample {i} tokenizer decode successful, decoded text length: {len(output_text)}")
                        print(f"[DEBUG] Sample {i} decoded text (first 200 chars): {output_text[:200]}...")
                    except Exception as decode_error:
                        print(f"[WARNING] Sample {i} tokenizer decode failed: {decode_error}, using string conversion")
                        output_text = str(output_data)
                elif self.tokenizer is not None and isinstance(output_data, (list, np.ndarray)):
                    # 处理列表或数组格式的token IDs
                    print(f"[DEBUG] Sample {i} attempting tokenizer decode for list/array")
                    try:
                        skip_special_tokens = getattr(self.config, 'skip_special_tokens', True)
                        if isinstance(output_data, np.ndarray):
                            output_data = output_data.tolist()
                        output_text = self.tokenizer.decode(output_data, skip_special_tokens=skip_special_tokens)
                        print(f"[DEBUG] Sample {i} tokenizer decode successful for list/array")
                    except Exception as decode_error:
                        print(f"[WARNING] Sample {i} tokenizer decode failed for list/array: {decode_error}")
                        output_text = str(output_data)
                else:
                    # 直接使用字符串转换
                    if isinstance(output_data, (list, np.ndarray)):
                        # 如果是列表或数组，取第一个元素
                        output_text = str(output_data[0]) if len(output_data) > 0 else ""
                    else:
                        output_text = str(output_data)
                
                print(f"[DEBUG] Sample {i} final output text (first 100 chars): {output_text[:100]}...")
                
                # 计算格式奖励
                evaluation = self.image_edit_format_reward(output_text)
                
                # 提取分数
                overall_score = evaluation["overall_score"]
                print(f"[DEBUG] Sample {i} evaluation completed, score: {overall_score}")
                print(f"[DEBUG] Sample {i} detailed results: {evaluation}")
                
                # 存储分数
                reward_tensor[i] = overall_score
                reward_metrics["overall"].append(overall_score)
                reward_metrics["format_score"].append(overall_score)
                reward_metrics["hard_requirements_passed"].append(float(evaluation["hard_requirements_passed"]))
                reward_metrics["blocks_on_separate_lines"].append(float(evaluation["blocks_on_separate_lines"]))
                reward_metrics["re_edit_length_penalty"].append(evaluation["re_edit_length_penalty"])
                reward_metrics["line_separation_penalty"].append(evaluation["line_separation_penalty"])
                reward_metrics["line_violation_count"].append(float(evaluation["line_violation_count"]))
                reward_metrics["total_penalty"].append(evaluation["total_penalty"])
                reward_metrics["cot_count"].append(float(evaluation["cot_count"]))
                reward_metrics["re_edit_count"].append(float(evaluation["re_edit_count"]))
                
            except Exception as e:
                print(f"[ERROR] Error evaluating sample {i}: {e}")
                # 错误时返回0分
                fallback_score = 0.0
                reward_tensor[i] = fallback_score
                reward_metrics["overall"].append(fallback_score)
                reward_metrics["format_score"].append(fallback_score)
                reward_metrics["hard_requirements_passed"].append(0.0)
                reward_metrics["blocks_on_separate_lines"].append(0.0)
                reward_metrics["re_edit_length_penalty"].append(0.0)
                reward_metrics["line_separation_penalty"].append(0.0)
                reward_metrics["line_violation_count"].append(0.0)
                reward_metrics["total_penalty"].append(0.0)
                reward_metrics["cot_count"].append(0.0)
                reward_metrics["re_edit_count"].append(0.0)
        
        print(f"[DEBUG] RuleBasedFormatRewardManager: Batch processing completed")
        print(f"[DEBUG] Final reward tensor shape: {reward_tensor.shape}")
        print(f"[DEBUG] Reward metrics keys: {list(reward_metrics.keys())}")
        print(f"[DEBUG] Average format score: {reward_tensor.mean().item():.3f}")
        
        return reward_tensor, dict(reward_metrics)
