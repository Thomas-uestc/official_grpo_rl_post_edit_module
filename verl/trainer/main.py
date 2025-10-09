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

import json
import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager, GPT41RewardManager, RuleBasedFormatRewardManager, CombinedRewardManager
from ..workers.image_edit.qwen_image_edit_worker import QwenImageEditWorker
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        
        # 资源池配置 - 支持独立image_edit资源池
        global_pool_id = "global_pool"
        image_edit_pool_id = "image_edit_pool"
        
        # 根据GPU数量动态配置资源池
        n_gpus = config.trainer.n_gpus_per_node
        if n_gpus >= 6:
            # 6+ GPU环境：GPU 0,1 for training, GPU 3-5 for image_edit (GPU 2 reserved)
            training_gpus = 4
            image_edit_gpus = 4
        else:
            raise ValueError(f"Invalid number of GPUs: {n_gpus}")
        
        
        resource_pool_spec = {
            global_pool_id: [training_gpus] * config.trainer.nnodes,
            image_edit_pool_id: [image_edit_gpus] * config.trainer.nnodes,
        }
        
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
            Role.ImageEdit: image_edit_pool_id,  # image_edit使用独立资源池
        }
        
        print(f"[MAIN] Resource pool configuration:")
        print(f"  Total GPUs: {n_gpus}")
        print(f"  Training GPUs: {training_gpus}")
        print(f"  Image Edit GPUs: {image_edit_gpus}")
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # 初始化奖励管理器
        print(f"[MAIN] Initializing reward manager: {config.worker.reward.reward_type}")
        
        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        elif config.worker.reward.reward_type == "gpt41":
            RewardManager = GPT41RewardManager
        elif config.worker.reward.reward_type == "rule_based_format":
            RewardManager = RuleBasedFormatRewardManager
        elif config.worker.reward.reward_type == "combined":
            RewardManager = CombinedRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        
        # 确定奖励管理器的初始化参数
        # GPT41 只需要config参数
        # RuleBasedFormat、Combined 需要config和tokenizer参数
        # 其他也需要config和tokenizer参数
        if config.worker.reward.reward_type in ["gpt41"]:
            print(f"[MAIN] Initializing {config.worker.reward.reward_type} reward manager with config only")
            reward_fn = RemoteRewardManager.remote(config.worker.reward)
            val_reward_fn = RemoteRewardManager.remote(config.worker.reward)
        else:
            print(f"[MAIN] Initializing {config.worker.reward.reward_type} reward manager with config and tokenizer")
            reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
            val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        
        # 显示奖励系统配置信息
        if config.worker.reward.reward_type == "rule_based_format":
            print(f"[MAIN] Rule-based reward configuration:")
            print(f"  Ideal length: {getattr(config.worker.reward, 'rule_re_edit_max_length_ideal', 50)}")
            print(f"  Acceptable length: {getattr(config.worker.reward, 'rule_re_edit_max_length_acceptable', 100)}")
            print(f"  Tolerable length: {getattr(config.worker.reward, 'rule_re_edit_max_length_tolerable', 150)}")
        elif config.worker.reward.reward_type == "combined":
            print(f"[MAIN] Combined reward configuration (6-dimension):")
            print(f"  Strategy: {getattr(config.worker.reward, 'combined_strategy', 'weighted_sum')}")
            print(f"  GPT Physical & Geometric weight: {getattr(config.worker.reward, 'combined_gpt_physical_geometric_weight', 0.15)}")
            print(f"  GPT Environment & Context weight: {getattr(config.worker.reward, 'combined_gpt_environment_context_weight', 0.15)}")
            print(f"  GPT Cultural & Social weight: {getattr(config.worker.reward, 'combined_gpt_cultural_social_weight', 0.15)}")
            print(f"  GPT Logical & Causal weight: {getattr(config.worker.reward, 'combined_gpt_logical_causal_weight', 0.15)}")
            print(f"  GPT Target Attribution weight: {getattr(config.worker.reward, 'combined_gpt_target_attribution_weight', 0.15)}")
            print(f"  Rule Format weight: {getattr(config.worker.reward, 'combined_rule_format_weight', 0.25)}")
            if getattr(config.worker.reward, 'combined_strategy', 'weighted_sum') == 'gated':
                print(f"  Gate threshold: {getattr(config.worker.reward, 'combined_rule_gate_threshold', 2.5)}")
        elif config.worker.reward.reward_type == "gpt41":
            print(f"[MAIN] GPT-4.1 reward configuration:")
            print(f"  Model: {getattr(config.worker.reward, 'gpt41_model', 'gpt-4o')}")
            print(f"  Max tokens: {getattr(config.worker.reward, 'gpt41_max_tokens', 1000)}")
            print(f"  Temperature: {getattr(config.worker.reward, 'gpt41_temperature', 0.0)}")
        
        print(f"[MAIN] Reward manager initialized successfully")

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
        "env_vars": {
            # 你原来的变量
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "PYTHONUNBUFFERED": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
    }

        ray.init(runtime_env=runtime_env)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
