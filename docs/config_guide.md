# EasyR1 Image Editing Configuration Guide

## 概述

本指南介绍如何在EasyR1框架中配置图像编辑训练，包括Qwen-Image-Edit模型和GPT-4.1奖励API的参数设置。

## 配置文件结构

### 1. 基础配置文件 (`config.yaml`)

这是默认配置文件，包含了所有可用的参数选项：

```yaml
# 数据配置
data:
  # 图像编辑数据集配置
  original_image_dir: null  # 原图目录路径
  edited_image_dir: null   # 初步编辑图像目录路径
  original_image_key: "original_images"
  preliminary_edited_image_key: "preliminary_edited_images"
  edit_instruction_key: "edit_instruction"
  expected_reasoning_key: "reasoning"
  original_description_key: "original_description"

# Worker配置
worker:
  reward:
    # GPT-4.1 API配置
    use_gpt41_api: false
    gpt41_api_key: null  # 通过OPENAI_API_KEY环境变量设置
    gpt41_api_base: "https://api.openai.com/v1"
    gpt41_model: "gpt-4.1"
    gpt41_max_tokens: 100
    gpt41_temperature: 0.0
    gpt41_max_retries: 3
    gpt41_retry_delay: 1.0
    gpt41_request_timeout: 30.0

  image_edit:
    # 模型配置
    model_path: null  # 设置为"Qwen/Qwen-Image-Edit"或本地路径
    trust_remote_code: true
    torch_dtype: "bfloat16"
    device: "cuda"
    
    # Diffusers管道参数
    true_cfg_scale: 4.0
    negative_prompt: ""
    num_inference_steps: 50
    guidance_scale: 7.5
    
    # FSDP配置
    enable_fsdp: true
    fsdp_sharding_strategy: "FULL_SHARD"
    fsdp_cpu_offload: true
    fsdp_sync_module_states: true
    
    # 批处理
    batch_size: 2
    max_batch_size: 8
    image_size: 1024

# 训练器配置
trainer:
  # 图像编辑特定设置
  enable_image_editing: false
  image_edit_validation_freq: 10
  save_edited_images: true
  max_validation_images: 5
```

### 2. 图像编辑专用配置 (`image_edit_config.yaml`)

这是针对图像编辑任务优化的完整配置示例。

## 关键参数说明

### GPT-4.1 API配置

```yaml
worker:
  reward:
    reward_type: gpt41  # 使用GPT-4.1奖励
    use_gpt41_api: true
    gpt41_model: "gpt-4.1"
    gpt41_max_tokens: 100
    gpt41_temperature: 0.0
    gpt41_max_retries: 3
    gpt41_request_timeout: 30.0
```

**重要参数**：
- `gpt41_model`: GPT模型版本
- `gpt41_max_tokens`: 最大输出token数
- `gpt41_temperature`: 生成温度（0.0为确定性输出）
- `gpt41_max_retries`: API调用重试次数
- `gpt41_request_timeout`: 请求超时时间

### Qwen-Image-Edit配置

```yaml
worker:
  image_edit:
    model_path: "Qwen/Qwen-Image-Edit"
    true_cfg_scale: 4.0
    negative_prompt: ""
    num_inference_steps: 50
    guidance_scale: 7.5
    batch_size: 2
    image_size: 1024
```

**重要参数**：
- `model_path`: 模型路径（本地或HuggingFace）
- `true_cfg_scale`: 分类器自由引导比例
- `num_inference_steps`: 推理步数（更多步数=更好质量）
- `batch_size`: 批处理大小
- `image_size`: 图像尺寸

### FSDP内存优化配置

```yaml
worker:
  image_edit:
    enable_fsdp: true
    fsdp_sharding_strategy: "FULL_SHARD"
    fsdp_cpu_offload: true
    fsdp_sync_module_states: true
```

**分片策略**：
- `FULL_SHARD`: 完全分片，内存效率最高
- `SHARD_GRAD_OP`: 只分片梯度操作
- `NO_SHARD`: 不分片

## 使用方法

### 1. 使用默认配置

```bash
python3 -m verl.trainer.main config=examples/config.yaml
```

### 2. 使用图像编辑专用配置

```bash
python3 -m verl.trainer.main config=examples/image_edit_config.yaml
```

### 3. 命令行覆盖参数

```bash
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.image_edit.model_path="Qwen/Qwen-Image-Edit" \
    worker.image_edit.enable_fsdp=true \
    worker.image_edit.batch_size=2 \
    worker.reward.reward_type=gpt41 \
    worker.reward.use_gpt41_api=true \
    data.original_image_dir="/path/to/original/images" \
    data.edited_image_dir="/path/to/edited/images"
```

### 4. 环境变量设置

```bash
export OPENAI_API_KEY="your-openai-api-key"
python3 -m verl.trainer.main config=examples/image_edit_config.yaml
```

## 参数调优建议

### 内存优化

1. **启用FSDP**：
   ```yaml
   worker.image_edit.enable_fsdp: true
   worker.image_edit.fsdp_cpu_offload: true
   ```

2. **减小批次大小**：
   ```yaml
   worker.image_edit.batch_size: 2
   worker.actor.global_batch_size: 64
   ```

3. **调整图像尺寸**：
   ```yaml
   worker.image_edit.image_size: 1024  # 或更小
   ```

### 质量优化

1. **增加推理步数**：
   ```yaml
   worker.image_edit.num_inference_steps: 50  # 或更多
   ```

2. **调整CFG比例**：
   ```yaml
   worker.image_edit.true_cfg_scale: 4.0  # 或调整
   ```

3. **优化学习率**：
   ```yaml
   worker.actor.optim.lr: 5.0e-7  # 图像编辑任务建议较低学习率
   ```

### API优化

1. **调整超时时间**：
   ```yaml
   worker.reward.gpt41_request_timeout: 30.0  # 根据网络情况调整
   ```

2. **设置重试次数**：
   ```yaml
   worker.reward.gpt41_max_retries: 3
   worker.reward.gpt41_retry_delay: 1.0
   ```

## 故障排除

### 1. 显存不足

- 启用FSDP: `worker.image_edit.enable_fsdp: true`
- 启用CPU offload: `worker.image_edit.fsdp_cpu_offload: true`
- 减小批次大小: `worker.image_edit.batch_size: 1`

### 2. API调用失败

- 检查API密钥: `export OPENAI_API_KEY="your-key"`
- 增加超时时间: `worker.reward.gpt41_request_timeout: 60.0`
- 增加重试次数: `worker.reward.gpt41_max_retries: 5`

### 3. 图像质量不佳

- 增加推理步数: `worker.image_edit.num_inference_steps: 100`
- 调整CFG比例: `worker.image_edit.true_cfg_scale: 7.5`
- 检查图像尺寸: `worker.image_edit.image_size: 1024`

## 示例配置

### 最小配置（测试用）

```yaml
worker:
  image_edit:
    model_path: "Qwen/Qwen-Image-Edit"
    enable_fsdp: true
    batch_size: 1
  reward:
    reward_type: gpt41
    use_gpt41_api: true
```

### 生产配置（高质量）

```yaml
worker:
  image_edit:
    model_path: "Qwen/Qwen-Image-Edit"
    enable_fsdp: true
    fsdp_cpu_offload: true
    batch_size: 2
    num_inference_steps: 100
    true_cfg_scale: 7.5
  reward:
    reward_type: gpt41
    use_gpt41_api: true
    gpt41_max_tokens: 200
    gpt41_request_timeout: 60.0
```

通过合理配置这些参数，你可以在EasyR1框架中实现高效的图像编辑训练。
