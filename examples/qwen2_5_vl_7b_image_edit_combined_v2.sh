#!/bin/bash

set -x
export PYTHONUNBUFFERED=1
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export OPENAI_API_KEY="your"  # replace with your openai api key


# export WANDB_API_KEY="yixuan-ding-wandb-api-key"  # wandb of Yixuan Ding, do not need to change
export WANDB_API_KEY="yixuan-ding-wandb-api-key"  # wandb of Yixuan Ding, do not need to change
export WANDB_MODE="online"                      # Wandb模式：online/offline/disabled
export WANDB_INIT_TIMEOUT=60                    # Wandb初始化超时时间（秒）
export WANDB_DISABLE_STATS=false                # 启用统计收集
export WANDB_DISABLE_META=false                 # 启用元数据收集
export WANDB_CONSOLE="off"                      # 关闭控制台输出避免混乱
export WANDB_BASE_URL="https://api.wandb.ai"    # 明确指定API地址
export WANDB_HTTP_TIMEOUT=60                    # HTTP请求超时时间

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用GPU 0-7（0,1,2,3用于训练，4,5,6,7用于图像编辑）

MODEL_PATH=/path/to/your/sft-qwen2.5-vl-7b-model  # replace it with sft fine-tuned qwen2.5-vl-7b model
IMAGE_EDIT_MODEL_PATH=/path/to/your/qwen-image-edit-model  # replace with pretrained qwen-image-edit model
TRAIN_DATA_PATH=/path/to/your/training-dataset # replace with training dataset downloaded from huggingface
VAL_DATA_PATH=/path/to/your/validation-dataset # useless

# do not need to change
ROLLOUT_BATCH_SIZE=16
MAX_RESPONSE_LENGTH=4096
DATA_SEED=43
GLOBAL_BATCH_SIZE=8
MICRO_BATCH_SIZE_PER_DEVICE_FOR_EXPERIENCE=8
MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE=2
ACTOR_LR=5.0e-9
ROLLOUT_N=4
ROLLOUT_TENSOR_PARALLEL_SIZE=4
TOTAL_EPOCHS=2
VAL_FREQ=-1
SAVE_FREQ=50
SAVE_MODEL_ONLY=true
# Max deviation strategy weights (GPT: 0.8, Rule: 0.2)
MAX_DEVIATION_GPT_WEIGHT=0.8
MAX_DEVIATION_RULE_WEIGHT=0.2

python3 -m verl.trainer.main \
    config=examples/image_edit_config_combined.yaml \
    trainer.project_name=post_reasoning_edit_easyr1_max_deviation_reward \
    trainer.experiment_name=max_deviation_gpt08_rule02 \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${VAL_DATA_PATH} \
    data.data_format=parquet \
    data.parquet_pattern="omniedit_qwen_edit_full_batch_*.parquet" \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.max_num_batched_tokens=36864 \
    data.seed=${DATA_SEED} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.micro_batch_size_per_device_for_experience=${MICRO_BATCH_SIZE_PER_DEVICE_FOR_EXPERIENCE} \
    worker.actor.micro_batch_size_per_device_for_update=${MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE} \
    worker.actor.optim.lr=${ACTOR_LR} \
    worker.actor.optim.weight_decay=1.0e-2 \
    worker.actor.optim.strategy=adamw \
    worker.actor.optim.lr_warmup_ratio=0.1 \
    worker.actor.fsdp.enable_full_shard=true \
    worker.actor.fsdp.enable_cpu_offload=false \
    worker.actor.fsdp.enable_rank0_init=true \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.tensor_parallel_size=${ROLLOUT_TENSOR_PARALLEL_SIZE} \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.prompt_length=30000 \
    worker.rollout.response_length=4096 \
    worker.rollout.max_num_batched_tokens=36864 \
    worker.image_edit.model_path=${IMAGE_EDIT_MODEL_PATH} \
    worker.image_edit.enable_fsdp=false \
    worker.image_edit.num_gpus=4 \
    worker.image_edit.batch_size=4 \
    worker.image_edit.true_cfg_scale=4.0 \
    worker.image_edit.num_inference_steps=15 \
    worker.reward.reward_type=combined \
    worker.reward.combined_strategy=max_deviation \
    worker.reward.max_deviation_gpt_weight=${MAX_DEVIATION_GPT_WEIGHT} \
    worker.reward.max_deviation_rule_weight=${MAX_DEVIATION_RULE_WEIGHT} \
    worker.reward.use_gpt41_api=true \
    worker.reward.gpt41_model=gpt-4o \
    worker.reward.gpt41_api_key=${OPENAI_API_KEY} \
    worker.reward.gpt41_api_base="https://api.openai.com/v1" \
    worker.reward.gpt41_max_tokens=100 \
    worker.reward.gpt41_temperature=0.0 \
    worker.reward.gpt41_max_retries=3 \
    worker.reward.gpt41_retry_delay=1.0 \
    worker.reward.gpt41_request_timeout=30 \
    worker.reward.rule_re_edit_max_length_ideal=50 \
    worker.reward.rule_re_edit_max_length_acceptable=100 \
    worker.reward.rule_re_edit_max_length_tolerable=150 \
    trainer.logger='["console", "wandb"]' \
    trainer.wandb_mode=${WANDB_MODE:-online} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.val_freq=${VAL_FREQ} \
    trainer.val_before_train=false \
    trainer.max_try_make_batch=10 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.save_model_only=${SAVE_MODEL_ONLY} \
    trainer.save_limit=2
