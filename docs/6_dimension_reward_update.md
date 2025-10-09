# 6维度奖励系统更新说明

## 概述

已成功将原有的单一GPT奖励维度扩展为5个专门的一致性评估维度，结合1个格式奖励维度，形成完整的6维度奖励评估系统。

## 更新的5个GPT评估维度

### 1. Physical & Geometric Consistency (物理几何一致性)
- **评估内容**: 物理定律遵循、几何关系正确性、空间透视一致性
- **配置参数**: `combined_gpt_physical_geometric_weight`
- **代码标识**: `physical_geometric`

### 2. Environment & Context Consistency (环境上下文一致性)  
- **评估内容**: 环境背景一致性、上下文关联性、场景合理性
- **配置参数**: `combined_gpt_environment_context_weight`
- **代码标识**: `environment_context`

### 3. Cultural & Social Norm Alignment (文化社会规范对齐)
- **评估内容**: 文化适宜性、社会规范符合度、价值观对齐
- **配置参数**: `combined_gpt_cultural_social_weight`
- **代码标识**: `cultural_social`

### 4. Logical & Causal Consistency (逻辑因果一致性)
- **评估内容**: 逻辑关系正确性、因果链合理性、推理一致性
- **配置参数**: `combined_gpt_logical_causal_weight`
- **代码标识**: `logical_causal`

### 5. Target Attribution & Referential Reasoning Consistency (目标归因推理一致性)
- **评估内容**: 目标对象识别准确性、指代推理正确性、归因一致性
- **配置参数**: `combined_gpt_target_attribution_weight`
- **代码标识**: `target_attribution`

### 6. Format Consistency (格式一致性)
- **评估内容**: 输出格式规范性、标签正确性、结构完整性
- **配置参数**: `combined_rule_format_weight`
- **代码标识**: `rule_format`

## 主要代码更新

### 1. GPT41RewardManager (`gpt41_reward_manager.py`)
- ✅ 更新了5个维度的prompt模板占位符
- ✅ 修改了`_create_evaluation_prompt`方法支持维度参数
- ✅ 更新了`_call_gpt41_api`方法支持维度参数
- ✅ 重构了`compute_reward`方法实现多维度并行评估

### 2. CombinedRewardManager (`combined_reward_manager.py`)
- ✅ 更新了6个维度的权重配置参数
- ✅ 新增了`combine_rewards_six_dimensions`方法
- ✅ 修改了权重归一化逻辑
- ✅ 更新了指标收集和日志输出

### 3. Main配置 (`main.py`)
- ✅ 更新了配置信息显示
- ✅ 显示所有6个维度的权重配置

### 4. 配置示例 (`config_6_dimension_reward.yaml`)
- ✅ 提供了完整的6维度配置示例
- ✅ 包含详细的使用说明和权重调整建议

## 配置参数

```yaml
worker:
  reward:
    reward_type: "combined"
    
    # 5个GPT一致性维度权重
    combined_gpt_physical_geometric_weight: 0.15
    combined_gpt_environment_context_weight: 0.15
    combined_gpt_cultural_social_weight: 0.15
    combined_gpt_logical_causal_weight: 0.15
    combined_gpt_target_attribution_weight: 0.15
    
    # 格式奖励权重
    combined_rule_format_weight: 0.25
```

## 使用方法

### 1. 填写Prompt模板
在`gpt41_reward_manager.py`中的5个TODO位置填写具体的评估prompt：

```python
"physical_geometric": """
# TODO: 请在此处填写物理几何一致性评估的详细prompt
# 评估维度：Physical & Geometric Consistency
# 评分标准：物理定律遵循、几何关系正确性、空间透视一致性等
# 输出：0.000-10.000的单一数字
"""
```

### 2. 配置权重
使用配置文件或命令行参数：

```bash
# 使用配置文件
--config examples/config_6_dimension_reward.yaml

# 或命令行覆盖
--worker.reward.combined_gpt_physical_geometric_weight=0.2
```

### 3. 启动训练
```bash
python -m verl.trainer.main --config config_6_dimension_reward.yaml
```

## 系统优势

1. **精细化评估**: 5个专门的一致性维度提供全面的图像编辑质量评估
2. **灵活权重**: 可根据任务特点调整各维度重要性
3. **完整指标**: 详细的日志和指标便于分析和调优
4. **向后兼容**: 不影响现有的训练流程
5. **自动归一化**: 权重自动归一化确保总和为1.0

## 注意事项

1. **API调用成本**: 每个样本需要调用5次GPT API，成本会相应增加
2. **训练时间**: 多维度评估会增加训练时间
3. **Prompt质量**: 各维度prompt的质量直接影响评估效果
4. **权重平衡**: 需要根据具体任务调整各维度权重

## 下一步工作

1. 填写5个维度的具体评估prompt
2. 根据任务特点调整权重配置
3. 进行小规模测试验证效果
4. 根据测试结果优化prompt和权重
