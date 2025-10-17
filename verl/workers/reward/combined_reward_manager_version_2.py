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
Combined Reward Manager that integrates GPT-based and Rule-based rewards
"""

import torch
import numpy as np
import copy
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from ...protocol import DataProto
from ...single_controller.base import Worker
from ...single_controller.base.decorator import Dispatch, register
from ..reward.config import RewardConfig
from ..reward.gpt41_reward_manager import GPT41RewardManager
from ..reward.rule_based_format_reward_manager import RuleBasedFormatRewardManager


class CombinedRewardManager(Worker):
    """Combined reward manager that integrates GPT-based and rule-based rewards"""
    
    def __init__(self, config: RewardConfig, tokenizer=None):
        # Skip parent __init__ to avoid CUDA initialization
        # super().__init__()
        print(f"[DEBUG] CombinedRewardManager: Initializing with config and tokenizer...")
        self.config = config
        self.tokenizer = tokenizer
        
        # 确保代理环境变量在Ray worker中可用
        import os
        print(f"[DEBUG] CombinedRewardManager: Checking proxy environment variables...")
        print(f"[DEBUG] http_proxy={os.getenv('http_proxy')}, HTTP_PROXY={os.getenv('HTTP_PROXY')}")
        print(f"[DEBUG] https_proxy={os.getenv('https_proxy')}, HTTPS_PROXY={os.getenv('HTTPS_PROXY')}")
        
        # 如果环境变量缺失，设置默认代理
        if not os.getenv('http_proxy') and not os.getenv('HTTP_PROXY'):
            default_proxy = 'http://127.0.0.1:1114'
            os.environ['http_proxy'] = default_proxy
            os.environ['https_proxy'] = default_proxy
            print(f"[DEBUG] CombinedRewardManager: Set default proxy: {default_proxy}")
        
        # 6维度权重配置：5个GPT维度 + 1个格式维度
        self.gpt_physical_geometric_weight = getattr(config, 'combined_gpt_physical_geometric_weight', 0.15)
        self.gpt_environment_context_weight = getattr(config, 'combined_gpt_environment_context_weight', 0.15)
        self.gpt_cultural_social_weight = getattr(config, 'combined_gpt_cultural_social_weight', 0.15)
        self.gpt_logical_causal_weight = getattr(config, 'combined_gpt_logical_causal_weight', 0.15)
        self.gpt_target_attribution_weight = getattr(config, 'combined_gpt_target_attribution_weight', 0.15)
        self.rule_format_weight = getattr(config, 'combined_rule_format_weight', 0.25)
        
        # 存储所有权重用于归一化和访问
        self.dimension_weights = {
            'gpt_physical_geometric': self.gpt_physical_geometric_weight,
            'gpt_environment_context': self.gpt_environment_context_weight,
            'gpt_cultural_social': self.gpt_cultural_social_weight,
            'gpt_logical_causal': self.gpt_logical_causal_weight,
            'gpt_target_attribution': self.gpt_target_attribution_weight,
            'rule_format': self.rule_format_weight
        }
        
        # 确保权重之和为1.0
        total_weight = sum(self.dimension_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"[WARNING] Weight sum is {total_weight:.3f}, normalizing to 1.0")
            for key in self.dimension_weights:
                self.dimension_weights[key] = self.dimension_weights[key] / total_weight
            # 更新实例变量
            self.gpt_physical_geometric_weight = self.dimension_weights['gpt_physical_geometric']
            self.gpt_environment_context_weight = self.dimension_weights['gpt_environment_context']
            self.gpt_cultural_social_weight = self.dimension_weights['gpt_cultural_social']
            self.gpt_logical_causal_weight = self.dimension_weights['gpt_logical_causal']
            self.gpt_target_attribution_weight = self.dimension_weights['gpt_target_attribution']
            self.rule_format_weight = self.dimension_weights['rule_format']
            print(f"[DEBUG] Normalized weights: {self.dimension_weights}")
        
        # 保持向后兼容的总权重（用于旧的组合策略）
        self.gpt_weight = sum([self.gpt_physical_geometric_weight, self.gpt_environment_context_weight, 
                              self.gpt_cultural_social_weight, self.gpt_logical_causal_weight, 
                              self.gpt_target_attribution_weight])
        self.rule_weight = self.rule_format_weight
        
        # 奖励策略配置
        self.combination_strategy = getattr(config, 'combined_strategy', 'weighted_sum')  # 'weighted_sum', 'gated', 'multiplicative', 'max_deviation'
        self.rule_gate_threshold = getattr(config, 'combined_rule_gate_threshold', 2.5)  # 规则奖励门控阈值 (0-5分制的中等分数)
        
        # 最大偏差策略的权重配置
        self.max_deviation_gpt_weight = getattr(config, 'max_deviation_gpt_weight', 0.8)  # GPT API奖励权重
        self.max_deviation_rule_weight = getattr(config, 'max_deviation_rule_weight', 0.2)  # 规则奖励权重
        
        # 初始化子奖励管理器
        print(f"[DEBUG] CombinedRewardManager: Initializing GPT reward manager...")
        self.gpt_reward_manager = GPT41RewardManager(config)
        
        print(f"[DEBUG] CombinedRewardManager: Initializing rule-based reward manager...")
        self.rule_reward_manager = RuleBasedFormatRewardManager(config, tokenizer)
        
        print(f"[COMBINED_REWARD_INIT] ===== CombinedRewardManager 初始化完成 =====")
        print(f"[COMBINED_REWARD_INIT] 组合策略: {self.combination_strategy}")
        
        if self.combination_strategy == 'max_deviation':
            print(f"[COMBINED_REWARD_INIT] ===== 最大偏差策略配置 =====")
            print(f"[COMBINED_REWARD_INIT] GPT权重: {self.max_deviation_gpt_weight:.3f}")
            print(f"[COMBINED_REWARD_INIT] 规则权重: {self.max_deviation_rule_weight:.3f}")
            print(f"[COMBINED_REWARD_INIT] 权重总和: {self.max_deviation_gpt_weight + self.max_deviation_rule_weight:.3f}")
            print(f"[COMBINED_REWARD_INIT] 策略说明: 选择与5分基准偏差最大的GPT维度作为代表分数")
        else:
            print(f"[COMBINED_REWARD_INIT] 6维度权重配置:")
            for dim_key, weight in self.dimension_weights.items():
                print(f"[COMBINED_REWARD_INIT]   {dim_key}: {weight:.3f}")
            if self.combination_strategy == 'gated':
                print(f"[COMBINED_REWARD_INIT] 门控阈值: {self.rule_gate_threshold}")
        
        print(f"[COMBINED_REWARD_INIT] ===== 初始化完成 =====")
    
    def combine_rewards_weighted_sum(
        self, 
        gpt_scores: torch.Tensor, 
        rule_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        加权求和策略
        final_score = gpt_weight * gpt_score + rule_weight * rule_score
        适配新的值域：GPT分数0-10，format分数0-5
        """
        # 将format分数从0-5范围缩放到0-10范围，保持相对比例
        rule_scores_scaled = rule_scores * 2.0  # 0-5 -> 0-10
        
        combined_scores = self.gpt_weight * gpt_scores + self.rule_weight * rule_scores_scaled
        return combined_scores
    
    def combine_rewards_gated(
        self, 
        gpt_scores: torch.Tensor, 
        rule_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        门控策略：只有当规则奖励超过阈值时，才使用GPT奖励
        if rule_score >= threshold: final_score = gpt_weight * gpt_score + rule_weight * rule_score
        else: final_score = 0.0 (格式不合格直接零分)
        适配新的值域：GPT分数0-10，format分数0-5
        """
        # 创建门控掩码
        gate_mask = rule_scores >= self.rule_gate_threshold
        
        # 将format分数从0-5范围缩放到0-10范围，保持相对比例
        rule_scores_scaled = rule_scores * 2.0  # 0-5 -> 0-10
        
        # 计算加权分数
        weighted_scores = self.gpt_weight * gpt_scores + self.rule_weight * rule_scores_scaled
        
        # 应用门控：不满足格式要求的直接零分
        combined_scores = torch.where(gate_mask, weighted_scores, torch.zeros_like(weighted_scores))
        
        return combined_scores
    
    def combine_rewards_multiplicative(
        self, 
        gpt_scores: torch.Tensor, 
        rule_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        乘性策略：两个分数相乘，强调两者都要好
        final_score = (gpt_score * rule_score) ^ 0.5  # 几何平均
        适配新的值域：GPT分数0-10，format分数0-5
        """
        # 将分数标准化到[0,1]范围进行计算
        gpt_scores_normalized = gpt_scores / 10.0  # GPT分数0-10 -> 0-1
        rule_scores_normalized = rule_scores / 5.0  # format分数0-5 -> 0-1
        
        # 避免负数和零值问题
        gpt_scores_safe = torch.clamp(gpt_scores_normalized, min=1e-8, max=1.0)
        rule_scores_safe = torch.clamp(rule_scores_normalized, min=1e-8, max=1.0)
        
        # 几何平均
        combined_scores_normalized = torch.sqrt(gpt_scores_safe * rule_scores_safe)
        
        # 将结果转换回0-10范围（与GPT分数范围一致）
        combined_scores = combined_scores_normalized * 10.0
        
        return combined_scores
    
    def combine_rewards_max_deviation(
        self,
        gpt_metrics: Dict[str, List[float]],
        rule_metrics: Dict[str, List[float]]
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        最大偏差策略：选择与5分基准偏差最大的维度作为GPT代表分数
        然后与规则奖励按权重组合
        
        Args:
            gpt_metrics: GPT奖励指标，包含5个维度的分数
            rule_metrics: 规则奖励指标
            
        Returns:
            Tuple of (combined_scores, combined_metrics)
        """
        print(f"[MAX_DEVIATION] ===== 开始执行最大偏差组合策略 =====")
        print(f"[MAX_DEVIATION] 策略权重配置: GPT={self.max_deviation_gpt_weight:.3f}, Rule={self.max_deviation_rule_weight:.3f}")
        
        # 提取5个GPT维度分数
        gpt_dimensions = ["physical_geometric", "environment_context", "cultural_social", "logical_causal", "target_attribution"]
        dimension_tensors = {}
        
        batch_size = None
        print(f"[MAX_DEVIATION] 提取5个GPT维度分数:")
        for dim in gpt_dimensions:
            gpt_key = f"gpt_{dim}"
            if gpt_key in gpt_metrics:
                scores = torch.tensor(gpt_metrics[gpt_key], dtype=torch.float32, device='cpu')
                dimension_tensors[gpt_key] = scores
                if batch_size is None:
                    batch_size = len(scores)
                print(f"[MAX_DEVIATION]   {gpt_key}: 批次大小={len(scores)}, 均值={scores.mean().item():.3f}, 标准差={scores.std().item():.3f}")
                print(f"[MAX_DEVIATION]   {gpt_key}: 分数范围=[{scores.min().item():.3f}, {scores.max().item():.3f}]")
            else:
                print(f"[MAX_DEVIATION]   [WARNING] 缺少{gpt_key}，使用默认分数5.0")
                if batch_size is None:
                    # 尝试从其他指标推断batch_size
                    for key, values in gpt_metrics.items():
                        if isinstance(values, list) and len(values) > 0:
                            batch_size = len(values)
                            break
                    if batch_size is None:
                        batch_size = 1
                dimension_tensors[gpt_key] = torch.full((batch_size,), 5.0, dtype=torch.float32, device='cpu')
        
        print(f"[MAX_DEVIATION] 批次大小: {batch_size}")
        
        # 计算每个样本的最大偏差维度
        max_deviation_scores = torch.zeros(batch_size, dtype=torch.float32, device='cpu')
        max_deviation_dimensions = []
        
        print(f"[MAX_DEVIATION] ===== 开始计算每个样本的最大偏差维度 =====")
        for i in range(batch_size):
            print(f"[MAX_DEVIATION] --- 样本 {i+1}/{batch_size} ---")
            
            # 计算每个维度与5分基准的偏差绝对值
            deviations = {}
            print(f"[MAX_DEVIATION]   各维度与5分基准的偏差绝对值:")
            for dim_key, scores_tensor in dimension_tensors.items():
                deviation = abs(scores_tensor[i].item() - 5.0)
                deviations[dim_key] = deviation
                print(f"[MAX_DEVIATION]     {dim_key}: 分数={scores_tensor[i].item():.3f}, 偏差={deviation:.3f}")
            
            # 找到偏差最大的维度
            max_deviation_dim = max(deviations, key=deviations.get)
            max_deviation_score = dimension_tensors[max_deviation_dim][i].item()
            
            max_deviation_scores[i] = max_deviation_score
            max_deviation_dimensions.append(max_deviation_dim)
            
            print(f"[MAX_DEVIATION]   ✅ 选择维度: {max_deviation_dim}")
            print(f"[MAX_DEVIATION]   ✅ 最大偏差值: {deviations[max_deviation_dim]:.3f}")
            print(f"[MAX_DEVIATION]   ✅ 代表分数: {max_deviation_score:.3f}")
            print()
        
        # 提取规则奖励分数
        print(f"[MAX_DEVIATION] ===== 提取规则奖励分数 =====")
        if "overall" in rule_metrics:
            rule_scores = torch.tensor(rule_metrics["overall"], dtype=torch.float32, device='cpu')
            # 将格式分数从0-5范围缩放到0-10范围
            rule_scores_scaled = rule_scores * 2.0
            print(f"[MAX_DEVIATION] 规则分数 (0-5分制): 均值={rule_scores.mean().item():.3f}, 范围=[{rule_scores.min().item():.3f}, {rule_scores.max().item():.3f}]")
            print(f"[MAX_DEVIATION] 规则分数 (0-10分制): 均值={rule_scores_scaled.mean().item():.3f}, 范围=[{rule_scores_scaled.min().item():.3f}, {rule_scores_scaled.max().item():.3f}]")
        else:
            print(f"[MAX_DEVIATION] [WARNING] 缺少规则分数，使用默认分数5.0")
            rule_scores_scaled = torch.full((batch_size,), 5.0, dtype=torch.float32, device='cpu')
        
        # 加权组合：GPT最大偏差分数 + 规则分数
        print(f"[MAX_DEVIATION] ===== 开始加权组合 =====")
        print(f"[MAX_DEVIATION] 组合公式: 最终分数 = {self.max_deviation_gpt_weight:.3f} × GPT代表分数 + {self.max_deviation_rule_weight:.3f} × 规则分数")
        
        combined_scores = (self.max_deviation_gpt_weight * max_deviation_scores + 
                          self.max_deviation_rule_weight * rule_scores_scaled)
        
        print(f"[MAX_DEVIATION] ===== 组合结果统计 =====")
        print(f"[MAX_DEVIATION] GPT代表分数统计: 均值={max_deviation_scores.mean().item():.3f}, 范围=[{max_deviation_scores.min().item():.3f}, {max_deviation_scores.max().item():.3f}]")
        print(f"[MAX_DEVIATION] 规则分数统计: 均值={rule_scores_scaled.mean().item():.3f}, 范围=[{rule_scores_scaled.min().item():.3f}, {rule_scores_scaled.max().item():.3f}]")
        print(f"[MAX_DEVIATION] 最终组合分数统计: 均值={combined_scores.mean().item():.3f}, 范围=[{combined_scores.min().item():.3f}, {combined_scores.max().item():.3f}]")
        
        # 显示每个样本的详细组合过程
        print(f"[MAX_DEVIATION] ===== 各样本详细组合过程 =====")
        for i in range(batch_size):
            gpt_score = max_deviation_scores[i].item()
            rule_score = rule_scores_scaled[i].item()
            final_score = combined_scores[i].item()
            print(f"[MAX_DEVIATION] 样本 {i+1}: GPT代表={gpt_score:.3f} × {self.max_deviation_gpt_weight:.3f} + 规则={rule_score:.3f} × {self.max_deviation_rule_weight:.3f} = {final_score:.3f}")
        
        print(f"[MAX_DEVIATION] ===== 最大偏差组合策略执行完成 =====")
        
        # 构建详细指标
        combined_metrics = defaultdict(list)
        
        # 主要指标
        combined_metrics["overall"] = [float(x) for x in combined_scores.tolist()]
        combined_metrics["combined_score"] = [float(x) for x in combined_scores.tolist()]
        
        # 最大偏差相关指标（只包含数值类型）
        combined_metrics["max_deviation_score"] = [float(x) for x in max_deviation_scores.tolist()]
        combined_metrics["rule_format_scaled"] = [float(x) for x in rule_scores_scaled.tolist()]
        
        # 各维度原始分数
        for dim_key, scores_tensor in dimension_tensors.items():
            combined_metrics[dim_key] = [float(x) for x in scores_tensor.tolist()]
        
        # 权重信息
        for i in range(batch_size):
            combined_metrics["gpt_weight"].append(float(self.max_deviation_gpt_weight))
            combined_metrics["rule_weight"].append(float(self.max_deviation_rule_weight))
        
        # 维度选择统计（转换为数值编码）
        dimension_encoding = {
            'gpt_physical_geometric': 1,
            'gpt_environment_context': 2, 
            'gpt_cultural_social': 3,
            'gpt_logical_causal': 4,
            'gpt_target_attribution': 5
        }
        dimension_encoded = [float(dimension_encoding.get(dim, 0)) for dim in max_deviation_dimensions]
        combined_metrics["max_deviation_dimension_encoded"] = dimension_encoded
        
        print(f"[MAX_DEVIATION] ===== 维度选择统计 =====")
        print(f"[MAX_DEVIATION] 维度编码映射:")
        for dim_name, code in dimension_encoding.items():
            print(f"[MAX_DEVIATION]   {dim_name} = {code}")
        print(f"[MAX_DEVIATION] 各样本选择的维度编码: {dimension_encoded}")
        print(f"[MAX_DEVIATION] 维度选择分布:")
        for dim_name, code in dimension_encoding.items():
            count = dimension_encoded.count(code)
            print(f"[MAX_DEVIATION]   {dim_name}: {count}次")
        
        return combined_scores, dict(combined_metrics)
    
    def combine_rewards_six_dimensions(
        self,
        gpt_metrics: Dict[str, List[float]],
        rule_metrics: Dict[str, List[float]]
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        6维度加权组合策略：5个GPT维度 + 1个格式维度
        """
        print(f"[DEBUG] CombinedRewardManager: Using 6-dimension weighted combination")
        
        # 提取5个GPT维度分数
        gpt_dimensions = ["physical_geometric", "environment_context", "cultural_social", "logical_causal", "target_attribution"]
        dimension_tensors = {}
        
        batch_size = None
        for dim in gpt_dimensions:
            gpt_key = f"gpt_{dim}"
            if gpt_key in gpt_metrics:
                scores = torch.tensor(gpt_metrics[gpt_key], dtype=torch.float32, device='cpu')
                dimension_tensors[gpt_key] = scores
                if batch_size is None:
                    batch_size = len(scores)
                print(f"[DEBUG] {gpt_key} scores: mean={scores.mean().item():.3f}, std={scores.std().item():.3f}")
            else:
                print(f"[WARNING] Missing {gpt_key} in gpt_metrics, using fallback score 5.0")
                if batch_size is None:
                    # 尝试从其他指标推断batch_size
                    for key, values in gpt_metrics.items():
                        if isinstance(values, list) and len(values) > 0:
                            batch_size = len(values)
                            break
                    if batch_size is None:
                        batch_size = 1
                dimension_tensors[gpt_key] = torch.full((batch_size,), 5.0, dtype=torch.float32, device='cpu')
        
        # 提取格式维度分数
        if "overall" in rule_metrics:
            rule_scores = torch.tensor(rule_metrics["overall"], dtype=torch.float32, device='cpu')
            # 将格式分数从0-5范围缩放到0-10范围
            rule_scores_scaled = rule_scores * 2.0
            dimension_tensors["rule_format"] = rule_scores_scaled
            print(f"[DEBUG] rule_format scores: mean={rule_scores_scaled.mean().item():.3f}, std={rule_scores_scaled.std().item():.3f}")
        else:
            print(f"[WARNING] Missing rule overall scores, using fallback score 5.0")
            dimension_tensors["rule_format"] = torch.full((batch_size,), 5.0, dtype=torch.float32, device='cpu')
        
        # 计算加权组合分数
        combined_scores = torch.zeros(batch_size, dtype=torch.float32, device='cpu')
        
        for dim_key, weight in self.dimension_weights.items():
            if dim_key in dimension_tensors:
                combined_scores += weight * dimension_tensors[dim_key]
                print(f"[DEBUG] Added {dim_key} with weight {weight:.3f}")
        
        return combined_scores, dimension_tensors
    
    def combine_rewards(
        self, 
        gpt_scores: torch.Tensor, 
        rule_scores: torch.Tensor,
        gpt_metrics: Dict[str, List[float]],
        rule_metrics: Dict[str, List[float]]
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        根据配置的策略组合奖励分数
        
        Args:
            gpt_scores: GPT奖励分数张量
            rule_scores: 规则奖励分数张量
            gpt_metrics: GPT奖励指标
            rule_metrics: 规则奖励指标
            
        Returns:
            Tuple of (combined_scores, combined_metrics)
        """
        print(f"[DEBUG] CombinedRewardManager: Combining rewards using strategy '{self.combination_strategy}'")
        print(f"[DEBUG] GPT scores shape: {gpt_scores.shape}, Rule scores shape: {rule_scores.shape}")
        print(f"[DEBUG] GPT scores stats: mean={gpt_scores.mean().item():.3f}, std={gpt_scores.std().item():.3f}")
        print(f"[DEBUG] Rule scores stats: mean={rule_scores.mean().item():.3f}, std={rule_scores.std().item():.3f}")
        
        # 根据策略组合分数
        if self.combination_strategy == 'weighted_sum':
            combined_scores = self.combine_rewards_weighted_sum(gpt_scores, rule_scores)
        elif self.combination_strategy == 'gated':
            combined_scores = self.combine_rewards_gated(gpt_scores, rule_scores)
        elif self.combination_strategy == 'multiplicative':
            combined_scores = self.combine_rewards_multiplicative(gpt_scores, rule_scores)
        elif self.combination_strategy == 'max_deviation':
            # 最大偏差策略需要直接使用详细指标，跳过这里的处理
            combined_scores, combined_metrics = self.combine_rewards_max_deviation(gpt_metrics, rule_metrics)
            return combined_scores, combined_metrics
        else:
            print(f"[WARNING] Unknown combination strategy '{self.combination_strategy}', using weighted_sum")
            combined_scores = self.combine_rewards_weighted_sum(gpt_scores, rule_scores)
        
        # 组合指标
        combined_metrics = defaultdict(list)
        
        # 添加主要指标（确保是数值类型）
        combined_metrics["overall"] = [float(x) for x in combined_scores.tolist()]
        combined_metrics["combined_score"] = [float(x) for x in combined_scores.tolist()]
        
        # 添加子系统指标（带前缀，确保数值类型）
        for key, values in gpt_metrics.items():
            if isinstance(values, list) and len(values) > 0:
                # 只添加数值类型的指标，跳过字符串类型
                if isinstance(values[0], (int, float, np.integer, np.floating)):
                    combined_metrics[f"gpt_{key}"] = [float(x) for x in values]
                else:
                    print(f"[DEBUG] Skipping non-numeric GPT metric '{key}' with type {type(values[0])}")
        
        for key, values in rule_metrics.items():
            if isinstance(values, list) and len(values) > 0:
                # 只添加数值类型的指标，跳过字符串类型
                if isinstance(values[0], (int, float, np.integer, np.floating)):
                    combined_metrics[f"rule_{key}"] = [float(x) for x in values]
                else:
                    print(f"[DEBUG] Skipping non-numeric rule metric '{key}' with type {type(values[0])}")
        
        # 添加组合统计信息（只添加数值类型的指标）
        batch_size = len(combined_scores)
        combined_metrics["gpt_weight"] = [float(self.gpt_weight)] * batch_size
        combined_metrics["rule_weight"] = [float(self.rule_weight)] * batch_size
        # 不添加字符串类型的combination_strategy，避免np.mean()错误
        
        # 如果是门控策略，添加门控统计（确保都是数值类型）
        if self.combination_strategy == 'gated':
            gate_mask = rule_scores >= self.rule_gate_threshold
            gate_pass_rate = float(gate_mask.float().mean().item())
            combined_metrics["gate_pass_rate"] = [gate_pass_rate] * batch_size
            combined_metrics["gate_passed"] = [float(x) for x in gate_mask.float().tolist()]
        
        print(f"[DEBUG] Combined scores stats: mean={combined_scores.mean().item():.3f}, std={combined_scores.std().item():.3f}")
        
        return combined_scores, dict(combined_metrics)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        计算组合奖励分数
        
        Args:
            data: DataProto containing model outputs and other data
            
        Returns:
            Tuple of (combined_reward_tensor, combined_metrics)
        """
        print(f"[DEBUG] CombinedRewardManager.compute_reward: Starting combined reward computation")
        
        # 诊断数据内容
        print(f"[DEBUG] CombinedRewardManager: Diagnosing data content...")
        print(f"[DEBUG] data.batch keys: {list(data.batch.keys()) if hasattr(data, 'batch') and data.batch is not None else 'No batch data'}")
        print(f"[DEBUG] data.non_tensor_batch keys: {list(data.non_tensor_batch.keys())}")
        
        # 检查关键数据字段
        if hasattr(data, 'batch') and data.batch is not None and "responses" in data.batch:
            print(f"[DEBUG] Found responses in data.batch, type: {type(data.batch['responses'])}")
            if isinstance(data.batch['responses'], (list, tuple)) and len(data.batch['responses']) > 0:
                print(f"[DEBUG] First response type: {type(data.batch['responses'][0])}")
        
        if "original_images" in data.non_tensor_batch:
            print(f"[DEBUG] Found original_images in data.non_tensor_batch, type: {type(data.non_tensor_batch['original_images'])}")
        
        try:
            # 为每个奖励管理器创建独立的数据副本，避免相互干扰
            print(f"[DEBUG] Creating data copies for independent processing...")
            
            # 创建数据副本 - 浅拷贝应该足够，因为我们不会修改内容
            rule_data = copy.copy(data)
            gpt_data = copy.copy(data)
            
            # 先计算rule-based奖励，因为它需要访问原始的responses数据
            print(f"[DEBUG] Computing rule-based rewards with independent data copy...")
            rule_scores, rule_metrics = self.rule_reward_manager.compute_reward(rule_data)
            print(f"[DEBUG] Rule rewards computed: scores shape {rule_scores.shape}, mean {rule_scores.mean().item():.3f}")
            
            print(f"[DEBUG] Computing GPT-based rewards with independent data copy...")
            gpt_scores, gpt_metrics = self.gpt_reward_manager.compute_reward(gpt_data)
            print(f"[DEBUG] GPT rewards computed: scores shape {gpt_scores.shape}, mean {gpt_scores.mean().item():.3f}")
            
            # 验证分数张量形状一致
            if gpt_scores.shape != rule_scores.shape:
                print(f"[ERROR] Score tensor shape mismatch: GPT {gpt_scores.shape} vs Rule {rule_scores.shape}")
                # 使用较小的批次大小
                min_batch_size = min(len(gpt_scores), len(rule_scores))
                gpt_scores = gpt_scores[:min_batch_size]
                rule_scores = rule_scores[:min_batch_size]
                
                # 截断指标
                for key in gpt_metrics:
                    if isinstance(gpt_metrics[key], list):
                        gpt_metrics[key] = gpt_metrics[key][:min_batch_size]
                for key in rule_metrics:
                    if isinstance(rule_metrics[key], list):
                        rule_metrics[key] = rule_metrics[key][:min_batch_size]
            
            # 组合奖励 - 根据策略选择组合方法
            if self.combination_strategy == 'max_deviation':
                print(f"[COMBINED_REWARD] ===== 使用最大偏差策略组合奖励 =====")
                print(f"[COMBINED_REWARD] 策略: {self.combination_strategy}")
                print(f"[COMBINED_REWARD] GPT权重: {self.max_deviation_gpt_weight:.3f}")
                print(f"[COMBINED_REWARD] 规则权重: {self.max_deviation_rule_weight:.3f}")
                combined_scores, combined_metrics = self.combine_rewards_max_deviation(
                    gpt_metrics, rule_metrics
                )
                
                # 为最大偏差策略添加原始指标
                for key, values in gpt_metrics.items():
                    if isinstance(values, list) and len(values) > 0:
                        if isinstance(values[0], (int, float, np.integer, np.floating)):
                            combined_metrics[f"original_gpt_{key}"] = [float(x) for x in values]
                
                for key, values in rule_metrics.items():
                    if isinstance(values, list) and len(values) > 0:
                        if isinstance(values[0], (int, float, np.integer, np.floating)):
                            combined_metrics[f"original_rule_{key}"] = [float(x) for x in values]
                            
            else:
                print(f"[DEBUG] Combining rewards using 6-dimension approach...")
                combined_scores, dimension_tensors = self.combine_rewards_six_dimensions(
                    gpt_metrics, rule_metrics
                )
                
                # 构建组合指标
                combined_metrics = defaultdict(list)
                
                # 添加主要指标
                combined_metrics["overall"] = [float(x) for x in combined_scores.tolist()]
                combined_metrics["combined_score"] = [float(x) for x in combined_scores.tolist()]
                
                # 添加各维度分数
                for dim_key, scores_tensor in dimension_tensors.items():
                    combined_metrics[dim_key] = [float(x) for x in scores_tensor.tolist()]
                
                # 添加原始GPT和规则指标（带前缀）
                for key, values in gpt_metrics.items():
                    if isinstance(values, list) and len(values) > 0:
                        if isinstance(values[0], (int, float, np.integer, np.floating)):
                            combined_metrics[f"original_gpt_{key}"] = [float(x) for x in values]
                
                for key, values in rule_metrics.items():
                    if isinstance(values, list) and len(values) > 0:
                        if isinstance(values[0], (int, float, np.integer, np.floating)):
                            combined_metrics[f"original_rule_{key}"] = [float(x) for x in values]
                
                # 添加权重信息
                batch_size = len(combined_scores)
                for dim_key, weight in self.dimension_weights.items():
                    combined_metrics[f"weight_{dim_key}"] = [float(weight)] * batch_size
            
            print(f"[DEBUG] CombinedRewardManager: Reward computation completed successfully")
            print(f"[DEBUG] Final combined scores - mean: {combined_scores.mean().item():.3f}, "
                  f"min: {combined_scores.min().item():.3f}, max: {combined_scores.max().item():.3f}")
            
            return combined_scores, combined_metrics
            
        except Exception as e:
            print(f"[ERROR] CombinedRewardManager: Error in reward computation: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            
            # 返回默认分数 - 安全地获取batch_size
            try:
                if hasattr(data, 'non_tensor_batch') and data.non_tensor_batch:
                    first_key = list(data.non_tensor_batch.keys())[0]
                    batch_size = len(data.non_tensor_batch[first_key])
                elif hasattr(data, 'batch') and data.batch:
                    first_key = list(data.batch.keys())[0]
                    batch_size = len(data.batch[first_key])
                else:
                    batch_size = 1  # 默认批次大小
            except (IndexError, KeyError, TypeError):
                batch_size = 1  # 默认批次大小
            
            print(f"[DEBUG] Using fallback batch_size: {batch_size}")
            fallback_scores = torch.zeros(batch_size, dtype=torch.float32, device='cpu')
            fallback_metrics = {
                "overall": [0.0] * batch_size,
                "combined_score": [0.0] * batch_size,
                "error": [str(e)] * batch_size
            }
            
            return fallback_scores, fallback_metrics