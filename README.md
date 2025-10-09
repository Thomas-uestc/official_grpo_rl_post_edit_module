# EasyR1: Vision-Enhanced Reinforcement Learning for Image Editing

基于VERL框架的图像编辑强化学习训练系统，使用GRPO算法和6维度混合奖励机制，专门针对多模态图像编辑任务进行优化。

## 🚀 项目概述

EasyR1是一个高效的图像编辑强化学习训练框架，具有以下核心特性：

- **多模态强化学习**: 基于Qwen2.5-VL-7B和Qwen-Image-Edit的端到端训练
- **6维度奖励系统**: 结合GPT-4o内容评估和规则格式检查的混合奖励机制
- **分布式训练**: 支持Ray框架的多GPU并行训练和推理
- **内存优化**: 集成FSDP、梯度检查点等显存优化技术
- **实验管理**: 内置Wandb日志记录和实验跟踪

## 📋 系统要求

### 软件要求
- **Python**: 3.10
- **网络**: 稳定的互联网连接（用于GPT API调用, wandb record）

## 🛠️ 环境配置

### 1. 克隆仓库
```bash
git clone https://github.com/Thomas-uestc/official_grpo_rl_post_edit_module.git
cd EasyR1_upload
```

### 2. 创建Conda环境
```bash
conda create -n yx_grpo_rl_post_edit python=3.10
conda activate yx_grpo_rl_post_edit
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

## 📦 模型和数据下载

### 1. 下载我们的SFT微调模型和训练数据
```bash
hf auth login
# enter following access token
yixuan-ding-huggingface_token

# for our sft fine-tuned qwen2.5-vl-7B
huggingface-cli download yixuan-ding-7/yx_sft_qwen2.5-vl-7B --resume-download

# for our training dataset
huggingface-cli download --repo-type dataset yixuan-ding-7/yx_grpo_train_dataset_v2 --resume-download
```

### 2. 下载qwen-image-edit图像编辑模型
```bash
huggingface-cli download Qwen/Qwen-Image-Edit --resume-download
```

## ⚙️ 配置设置

### 1. 训练脚本配置
编辑 `./examples/qwen2_5_vl_7b_image_edit_combined_v1.sh`，修改以下内容：

```bash
# OpenAI API 配置
export OPENAI_API_KEY="your-openai-api-key"

# 网络代理配置（如果需要）
export http_proxy=http://127.0.0.1:xxxx
export https_proxy=http://127.0.0.1:xxxx
# 模型路径
MODEL_PATH=/path/to/your/sft-qwen2.5-vl-7b-model
IMAGE_EDIT_MODEL_PATH=/path/to/your/qwen-image-edit-model

# 数据路径
TRAIN_DATA_PATH=/path/to/your/training-dataset
VAL_DATA_PATH=/path/to/your/validation-dataset

```

### 2. 配置文件说明
主要配置文件位于 `./examples/image_edit_config_combined.yaml`，包含：

- **数据配置**: 数据集路径、批处理大小、序列长度等
- **模型配置**: 模型路径、优化器设置、FSDP配置等
- **奖励系统**: 6维度混合奖励权重和策略
- **训练配置**: 训练轮数、保存频率、验证设置等

## 🚀 开始训练

### 1. 启动训练
```bash
bash ./examples/qwen2_5_vl_7b_image_edit_combined_v1.sh
```

## 📊 6维度奖励系统

本系统采用创新的6维度混合奖励机制：

### GPT内容评估维度（5个）
1. **物理几何一致性** (Physical & Geometric Consistency)
2. **环境上下文一致性** (Environment & Context Consistency)  
3. **文化社会规范对齐** (Cultural & Social Norm Alignment)
4. **逻辑因果一致性** (Logical & Causal Consistency)
5. **目标归因推理一致性** (Target Attribution & Referential Reasoning)

### 规则格式评估维度（1个）
6. **格式规范性** (Format Consistency)

每个维度独立评估，最终通过加权求和得到综合奖励分数。

**Record**: 训练过程数据将自动上传到我于启动代码里所配置的Wandb账户。