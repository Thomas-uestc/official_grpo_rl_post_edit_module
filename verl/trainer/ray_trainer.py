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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()
    ImageEdit = auto()  # 添加ImageEdit角色


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        print("[TRAINER] Starting init_workers()...")
        
        print("[TRAINER] Creating resource pools...")
        self.resource_pool_manager.create_resource_pool()
        print("[TRAINER] Resource pools created successfully")
        
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        print("[TRAINER] Resource pool to class mapping initialized")

        # create actor, rollout and ref
        print("[TRAINER] Creating actor/rollout/ref workers...")
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
            print("[TRAINER] Actor/rollout/ref worker class created")
        else:
            raise NotImplementedError

        # create critic
        print("[TRAINER] Creating critic workers...")
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
            print("[TRAINER] Critic worker class created")
        else:
            print("[TRAINER] Skipping critic worker creation (not using critic)")

        # create a reward model if reward_fn is None
        print("[TRAINER] Creating reward model workers...")
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls
            print("[TRAINER] Reward model worker class created")
        else:
            print("[TRAINER] Skipping reward model worker creation (using external reward function)")

        # create image edit worker if image editing is enabled
        print("[TRAINER] Creating image edit workers...")
        if hasattr(self.config.worker, 'image_edit') and self.config.worker.image_edit.model_path:
            from ..workers.image_edit.qwen_image_edit_worker import QwenImageEditWorker
            import ray
            # 使用独立的image_edit资源池
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ImageEdit)
            image_edit_cls = RayClassWithInitArgs(
                cls=ray.remote(QwenImageEditWorker), config=self.config.worker.image_edit, role="image_edit"
            )
            self.resource_pool_to_cls[resource_pool]["image_edit"] = image_edit_cls
            print("[TRAINER] Image edit worker class created")
        else:
            print("[TRAINER] Skipping image edit worker creation (not enabled)")

        # initialize WorkerGroup
        print("[TRAINER] Initializing worker groups...")
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            print(f"[TRAINER] Creating worker group for resource pool: {resource_pool}")
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)
            print(f"[TRAINER] Worker group created for resource pool: {resource_pool}")
        print("[TRAINER] All worker groups created successfully")

        print("[TRAINER] Initializing worker models...")
        if self.use_critic:
            print("[TRAINER] Initializing critic model...")
            self.critic_wg = all_wg["critic"]
            print(f"[TRAINER] About to call critic_wg.init_model()")
            try:
                self.critic_wg.init_model()
                print("[TRAINER] Critic model initialized successfully")
            except Exception as e:
                print(f"[TRAINER] ERROR initializing critic model: {e}")
                import traceback
                print(f"[TRAINER] Traceback: {traceback.format_exc()}")
                raise

        if self.use_reward_model:
            print("[TRAINER] Initializing reward model...")
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()
            print("[TRAINER] Reward model initialized successfully")

        # initialize image edit worker
        if hasattr(self.config.worker, 'image_edit') and self.config.worker.image_edit.model_path:
            print("[TRAINER] Initializing image edit model...")
            self.image_edit_wg = all_wg["image_edit"]
            print(f"[TRAINER] About to call image_edit_wg.init_model()")
            try:
                self.image_edit_wg.init_model()
                print("[TRAINER] Image edit model initialized successfully")
            except Exception as e:
                print(f"[TRAINER] ERROR initializing image edit model: {e}")
                import traceback
                print(f"[TRAINER] Traceback: {traceback.format_exc()}")
                raise

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        print("[TRAINER] Initializing actor/rollout/ref model...")
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        print(f"[TRAINER] About to call actor_rollout_ref_wg.init_model()")
        try:
            self.actor_rollout_ref_wg.init_model()
            print("[TRAINER] Actor/rollout/ref model initialized successfully")
        except Exception as e:
            print(f"[TRAINER] ERROR initializing actor/rollout/ref model: {e}")
            import traceback
            print(f"[TRAINER] Traceback: {traceback.format_exc()}")
            raise
        
        print("[TRAINER] init_workers() completed successfully!")

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        # Check if dataloader supports state_dict (StatefulDataLoader)
        if hasattr(self.train_dataloader, 'state_dict'):
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_path)
            print(f"[CHECKPOINT] Saved dataloader state to {dataloader_path}")
        else:
            # For regular DataLoader, save basic info instead
            dataloader_info = {
                "type": "DataLoader",
                "batch_size": getattr(self.train_dataloader, 'batch_size', None),
                "num_workers": getattr(self.train_dataloader, 'num_workers', None),
                "shuffle": getattr(self.train_dataloader, 'shuffle', None),
                "dataset_size": len(self.train_dataloader.dataset) if hasattr(self.train_dataloader, 'dataset') else None,
                "global_step": self.global_step
            }
            torch.save(dataloader_info, dataloader_path)
            print(f"[CHECKPOINT] Saved dataloader info (regular DataLoader) to {dataloader_path}")

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_data = torch.load(dataloader_path, weights_only=False)
            # Check if it's a state_dict (from StatefulDataLoader) or info dict (from regular DataLoader)
            if isinstance(dataloader_data, dict) and "type" in dataloader_data:
                print(f"[CHECKPOINT] Loaded dataloader info: {dataloader_data}")
                print(f"[CHECKPOINT] Regular DataLoader detected, skipping state restoration")
            else:
                # Assume it's a state_dict from StatefulDataLoader
                if hasattr(self.train_dataloader, 'load_state_dict'):
                    self.train_dataloader.load_state_dict(dataloader_data)
                    print(f"[CHECKPOINT] Restored dataloader state from {dataloader_path}")
                else:
                    print(f"[CHECKPOINT] Warning: dataloader state found but current dataloader doesn't support load_state_dict")
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _maybe_log_train_generations(
        self, batch: DataProto, step: int
    ) -> None:
        """Log a table of training samples to wandb"""
        if self.config.trainer.train_generations_to_log <= 0:
            return
        
        print(f"[TRAINER] Step {step}: Logging {self.config.trainer.train_generations_to_log} training samples to wandb...")
        
        # 检查batch中是否包含re_edit_instructions
        has_re_edit_instructions = "re_edit_instructions" in batch.non_tensor_batch
        print(f"[TRAINER] Step {step}: Has re_edit_instructions: {has_re_edit_instructions}")
        if has_re_edit_instructions:
            re_edit_count = len(batch.non_tensor_batch["re_edit_instructions"])
            print(f"[TRAINER] Step {step}: Found {re_edit_count} re_edit_instructions")
        
        try:
            # 获取batch数据
            prompts = batch.batch["prompts"]  # 原始prompt
            responses = batch.batch["responses"]  # Actor生成的responses
            response_mask = batch.batch["response_mask"]  # 响应掩码
            
            batch_size = prompts.shape[0]
            
            # 随机选择样本
            num_samples_to_log = min(self.config.trainer.train_generations_to_log, batch_size)
            rng = np.random.RandomState(step)  # 使用step作为随机种子，确保可重现
            selected_indices = rng.choice(batch_size, size=num_samples_to_log, replace=False)
            
            # 准备样本数据 (excluding input content)
            samples = []
            for idx in selected_indices:
                # 解码response（只解码有效部分）
                response_ids = responses[idx]
                response_length = torch.sum(response_mask[idx]).item()
                valid_response_ids = response_ids[:int(response_length)]
                response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                
                # 获取re_edit_instructions（如果存在）
                re_edit_instruction = ""
                if "re_edit_instructions" in batch.non_tensor_batch:
                    re_edit_instructions = batch.non_tensor_batch["re_edit_instructions"]
                    if idx < len(re_edit_instructions):
                        re_edit_instruction = str(re_edit_instructions[idx])
                        # 记录前3个样本的re_edit_instruction内容用于调试
                        if idx < 3:
                            print(f"[TRAINER] Step {step}: Sample {idx+1} re_edit_instruction: {re_edit_instruction[:100]}...")
                
                # 创建样本元组 (input, output, label, score)
                # 对于训练样本，将re_edit_instruction作为label，score可以是0.0
                # 使用空字符串作为input占位符，因为WandbGenerationLogger会忽略input
                sample = ("", response_text, re_edit_instruction, 0.0)
                samples.append(sample)
            
            # 记录到wandb - 标记为训练日志
            self.logger.log_generation(samples, step, context="train")
            print(f"[TRAINER] Step {step}: Successfully logged {len(samples)} training samples to wandb")
            
        except Exception as e:
            print(f"[TRAINER] Step {step}: Error logging training samples: {e}")
            import traceback
            print(f"[TRAINER] Traceback: {traceback.format_exc()}")

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        
        # 限制验证样本数量为20条，采用随机选择
        max_val_samples = 20
        val_samples_processed = 0
        
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            # 检查是否已达到最大验证样本数
            if val_samples_processed >= max_val_samples:
                print(f"[VALIDATION] Reached maximum validation samples ({max_val_samples}), stopping validation")
                break
            test_batch = DataProto.from_single_dict(batch_dict)
            
            # 获取当前批次的样本数量
            current_batch_size = test_batch.batch["input_ids"].shape[0]
            remaining_samples = max_val_samples - val_samples_processed
            
            # 如果当前批次超过剩余需要的样本数，随机选择部分样本
            if current_batch_size > remaining_samples:
                print(f"[VALIDATION] Batch size ({current_batch_size}) exceeds remaining samples ({remaining_samples}), randomly selecting samples")
                # 随机选择索引
                import numpy as np
                selected_indices = np.random.choice(current_batch_size, size=remaining_samples, replace=False)
                # 根据索引筛选批次（使用PyTorch索引）
                import torch
                selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long)
                tensor_data = test_batch.batch[selected_indices_tensor] if test_batch.batch is not None else None
                non_tensor_data = {key: value[selected_indices] for key, value in test_batch.non_tensor_batch.items()}
                test_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=test_batch.meta_info)
                current_batch_size = remaining_samples
            
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # Process image editing for validation (same as training)
            if hasattr(self.config.worker, 'image_edit') and self.config.worker.image_edit.model_path:
                print(f"[VALIDATION] Processing image editing for validation...")
                test_batch = self._process_image_editing(test_batch)
                print(f"[VALIDATION] Image editing completed for validation")

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # Print validation rewards
            print(f"[VALIDATION] Reward tensor shape: {reward_tensor.shape}")
            print(f"[VALIDATION] Reward tensor values: {reward_tensor.cpu().tolist()}")
            print(f"[VALIDATION] Reward metrics: {reward_metrics}")
            print(f"[VALIDATION] Average reward: {reward_tensor.mean().item():.4f}")
            print(f"[VALIDATION] Min reward: {reward_tensor.min().item():.4f}")
            print(f"[VALIDATION] Max reward: {reward_tensor.max().item():.4f}")

            # store generations (excluding input content)
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.cpu().tolist()
            # Skip sample_inputs since we don't want to log input content
            sample_outputs.extend(output_texts)
            
            # For RL training, we don't need ground_truth labels
            # Use empty strings as placeholders for logging
            if "ground_truth" in test_batch.non_tensor_batch:
                sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            else:
                # Use empty strings as labels for logging purposes (RL doesn't need true ground truth)
                sample_labels.extend([""] * len(output_texts))
            
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)
            
            # 更新已处理的样本数量
            val_samples_processed += current_batch_size
            print(f"[VALIDATION] Processed {val_samples_processed}/{max_val_samples} samples")

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations([], sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print(f"Finish validation. Total samples processed: {val_samples_processed}/{max_val_samples}")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("[TRAINER] Starting batch data generation...")
        while True:
            num_try_make_batch += 1
            print(f"[TRAINER] Attempt {num_try_make_batch}: Loading data from dataloader...")
            try:
                batch_dict = next(self.data_iterator)
                print(f"[TRAINER] Attempt {num_try_make_batch}: Data loaded successfully")
            except StopIteration:
                print("[TRAINER] Dataloader exhausted, restarting...")
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)
                print("[TRAINER] Dataloader restarted, data loaded")

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            print(f"[TRAINER] Attempt {num_try_make_batch}: DataProto created with {len(new_batch.batch)} samples")

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )
            print(f"[TRAINER] Attempt {num_try_make_batch}: Generation batch prepared")

            # generate a batch (re-edit instructions)
            print(f"[TRAINER] Attempt {num_try_make_batch}: Generating sequences...")
            print(f"[TRAINER] Attempt {num_try_make_batch}: About to call actor_rollout_ref_wg.generate_sequences")
            print(f"[TRAINER] Attempt {num_try_make_batch}: gen_batch keys: {list(gen_batch.batch.keys())}")
            print(f"[TRAINER] Attempt {num_try_make_batch}: gen_batch non_tensor keys: {list(gen_batch.non_tensor_batch.keys())}")
            print(f"[TRAINER] Attempt {num_try_make_batch}: gen_batch batch size: {len(gen_batch.batch)}")
            
            try:
                gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
                print(f"[TRAINER] Attempt {num_try_make_batch}: Sequences generated successfully")
                print(f"[TRAINER] Attempt {num_try_make_batch}: gen_batch_output keys: {list(gen_batch_output.batch.keys())}")
                print(f"[TRAINER] Attempt {num_try_make_batch}: gen_batch_output batch size: {len(gen_batch_output.batch)}")
            except Exception as e:
                print(f"[TRAINER] Attempt {num_try_make_batch}: ERROR in generate_sequences: {e}")
                print(f"[TRAINER] Attempt {num_try_make_batch}: Exception type: {type(e)}")
                import traceback
                print(f"[TRAINER] Attempt {num_try_make_batch}: Traceback: {traceback.format_exc()}")
                raise

            if self.config.algorithm.adv_estimator == "remax":
                print(f"[TRAINER] Attempt {num_try_make_batch}: Computing baseline for remax...")
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output
                print(f"[TRAINER] Attempt {num_try_make_batch}: Baseline computed for remax")

            # repeat to align with repeated responses in rollout
            print(f"[TRAINER] Attempt {num_try_make_batch}: Repeating batch for rollout alignment...")
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)
            print(f"[TRAINER] Attempt {num_try_make_batch}: Batch repeated and merged")

            # Process image editing after repeat operation
            if hasattr(self.config.worker, 'image_edit') and self.config.worker.image_edit.model_path:
                print(f"[TRAINER] Attempt {num_try_make_batch}: Processing image editing...")
                new_batch = self._process_image_editing(new_batch)
                print(f"[TRAINER] Attempt {num_try_make_batch}: Image editing completed")

            # filter group
            if self.config.algorithm.online_filtering:
                print(f"[TRAINER] Attempt {num_try_make_batch}: Applying online filtering...")
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                
                # Print online filtering rewards
                print(f"[TRAINER] Attempt {num_try_make_batch}: Online filtering rewards")
                print(f"[TRAINER] Attempt {num_try_make_batch}: Reward tensor shape: {reward_tensor.shape}")
                print(f"[TRAINER] Attempt {num_try_make_batch}: Sample-level rewards: {reward_tensor.cpu().tolist()}")
                print(f"[TRAINER] Attempt {num_try_make_batch}: Average reward: {reward_tensor.mean().item():.4f}")
                print(f"[TRAINER] Attempt {num_try_make_batch}: Min reward: {reward_tensor.min().item():.4f}")
                print(f"[TRAINER] Attempt {num_try_make_batch}: Max reward: {reward_tensor.max().item():.4f}")
                
                # Convert sample-level rewards to token-level rewards for filtering
                if reward_tensor.dim() == 1:
                    # Sample-level rewards: broadcast to token level
                    response_mask = new_batch.batch["response_mask"]
                    token_level_scores = reward_tensor.unsqueeze(1).expand(-1, response_mask.shape[1])
                    new_batch.batch["token_level_scores"] = token_level_scores
                else:
                    # Already token-level rewards
                    new_batch.batch["token_level_scores"] = reward_tensor
                
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]
                print(f"[TRAINER] Attempt {num_try_make_batch}: Online filtering applied, kept {len(kept_sample_idxs)} samples")

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            print(f"[TRAINER] Attempt {num_try_make_batch}: Current batch size: {current_batch_size}, Target: {rollout_batch_size}")
            
            if current_batch_size < rollout_batch_size:
                print(f"[TRAINER] Attempt {num_try_make_batch}: Need more data, continuing...")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"[TRAINER] Attempt {num_try_make_batch}: Continuing generation...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"[TRAINER] Attempt {num_try_make_batch}: Batch size sufficient, finishing generation...")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                final_batch = batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
                print(f"[TRAINER] Batch data generation completed with {len(final_batch)} samples")
                return final_batch

    def _process_image_editing(self, batch: DataProto, gen_batch_output: DataProto = None) -> DataProto:
        """
        Process image editing pipeline with synchronization:
        1. Extract re-edit instructions from actor output using InstructionExtractor
        2. Use image edit worker to generate final edited images (with sync barrier)
        3. Prepare data for reward computation
        4. Ensure all training GPUs wait for image editing completion
        """
        print(f"[Trainer] _process_image_editing: Starting image editing pipeline")
        print(f"[Trainer] _process_image_editing: batch keys: {list(batch.batch.keys())}")
        print(f"[Trainer] _process_image_editing: batch non_tensor keys: {list(batch.non_tensor_batch.keys())}")
        
        # Extract re-edit instructions from generated output using InstructionExtractor
        from ..utils.instruction_extractor import InstructionExtractor
        
        print(f"[Trainer] _process_image_editing: Extracting responses from batch")
        response_ids = batch.batch["responses"]
        response_length = torch.sum(batch.batch["response_mask"], dim=-1)
        print(f"[Trainer] _process_image_editing: Got {len(response_ids)} responses")
        
        # Decode model outputs
        print(f"[Trainer] _process_image_editing: Decoding model outputs...")
        model_outputs = []
        for i in range(len(response_ids)):
            cur_response_length = int(response_length[i].item())
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)
            if i < 3:  # Log first 3 for debugging
                print(f"[Trainer] _process_image_editing: Response {i+1}: {response_str[:100]}...")
        
        print(f"[Trainer] _process_image_editing: Extracting re-edit instructions...")
        # Extract re-edit instructions using our InstructionExtractor
        extractor = InstructionExtractor()
        re_edit_instructions = extractor.extract_batch_instructions(model_outputs)
        
        print(f"[Trainer] Extracted {len(re_edit_instructions)} re-edit instructions")
        for i, instruction in enumerate(re_edit_instructions[:3]):  # Log first 3 for debugging
            print(f"[Trainer] Sample {i+1}: {instruction}")
        
        # Add re-edit instructions to batch
        batch.non_tensor_batch["re_edit_instructions"] = np.array(re_edit_instructions, dtype=object)
        print(f"[Trainer] Added {len(re_edit_instructions)} re-edit instructions to batch")
        
        # Process image editing using image edit worker with synchronization
        if hasattr(self, 'image_edit_wg') and self.image_edit_wg is not None:
            print(f"[Trainer] Starting image editing for {len(re_edit_instructions)} images...")
            print(f"[Trainer] About to call image_edit_wg.process_batch")
            
            # 保存原始batch引用，以防Ray序列化问题
            original_batch = batch
            
            try:
                # 同步调用image_edit worker - 所有训练GPU会等待这里完成
                batch = self.image_edit_wg.process_batch(batch)
                print(f"[Trainer] Image editing completed successfully")
                
                # 检查Ray返回的数据格式并修复
                print(f"[Trainer] batch type after image editing: {type(batch)}")
                if isinstance(batch, list):
                    print(f"[Trainer] WARNING: Ray returned list instead of DataProto. Attempting to reconstruct...")
                    print(f"[Trainer] List length: {len(batch)}")
                    print(f"[Trainer] List content types: {[type(item) for item in batch]}")
                    
                    try:
                        from verl.protocol import DataProto
                        # 处理数据并行返回的多个DataProto
                        if len(batch) == 1 and isinstance(batch[0], DataProto):
                            # Ray返回的是 [DataProto] 结构
                            batch = batch[0]
                            print(f"[Trainer] Successfully extracted DataProto from list")
                        elif len(batch) >= 2 and all(isinstance(item, DataProto) for item in batch):
                            # 数据并行：多个DataProto是相同的（all_gather_object的结果）
                            # 只需要取第一个，避免重复数据
                            print(f"[Trainer] Detected multiple identical DataProto from data parallel processing. Taking first one...")
                            batch = batch[0]
                            print(f"[Trainer] Successfully extracted DataProto from identical list")
                        elif len(batch) >= 2:
                            # 假设list包含 [batch, non_tensor_batch, meta_info] 的结构
                            batch_data = batch[0] if batch[0] is not None else None
                            non_tensor_data = batch[1] if batch[1] is not None else {}
                            meta_data = batch[2] if len(batch) > 2 and batch[2] is not None else {}
                            
                            batch = DataProto(batch=batch_data, non_tensor_batch=non_tensor_data, meta_info=meta_data)
                            print(f"[Trainer] Successfully reconstructed DataProto from list")
                        else:
                            print(f"[Trainer] ERROR: List too short to reconstruct DataProto")
                            import sys
                            sys.exit(1)
                    except Exception as e:
                        print(f"[Trainer] ERROR: Failed to reconstruct DataProto from list: {e}")
                        print(f"[Trainer] Cannot proceed with training as data format is corrupted.")
                        import sys
                        sys.exit(1)
                elif hasattr(batch, 'non_tensor_batch'):
                    print(f"[Trainer] batch has non_tensor_batch: True")
                    print(f"[Trainer] batch.non_tensor_batch keys: {list(batch.non_tensor_batch.keys())}")
                else:
                    print(f"[Trainer] batch attributes: {dir(batch)}")
                    
            except Exception as e:
                print(f"[Trainer] ERROR in image editing: {e}")
                print(f"[Trainer] Exception type: {type(e)}")
                import traceback
                print(f"[Trainer] Traceback: {traceback.format_exc()}")
                raise
            
            print(f"[Trainer] Image editing completed. Proceeding with reward computation...")
            
            # 添加同步屏障 - 确保所有训练GPU等待image editing完成
            print(f"[Trainer] Synchronizing training GPUs...")
            self._sync_all_training_gpus()
            print(f"[Trainer] GPU synchronization completed")
            
        else:
            print(f"[Trainer] Image edit worker not available, using fallback")
            # Fallback: use preliminary edited images as final images
            if "preliminary_edited_images" in batch.non_tensor_batch:
                batch.non_tensor_batch["final_edited_images"] = batch.non_tensor_batch["preliminary_edited_images"]
                print(f"[Trainer] Using preliminary edited images as final images")
            else:
                raise ValueError("No preliminary edited images found and image edit worker not available")
        
        print(f"[Trainer] _process_image_editing: Image editing pipeline completed")
        return batch


    
    def _sync_all_training_gpus(self):
        """
        同步所有训练GPU，确保它们等待image editing完成后再进行后续处理
        """
        import torch.distributed as dist
        
        if dist.is_initialized():
            # 使用distributed barrier确保所有GPU同步
            print(f"[Trainer] Synchronizing all training GPUs...")
            dist.barrier()
            print(f"[Trainer] All training GPUs synchronized. Proceeding with parallel processing...")
        else:
            # 如果没有distributed，使用简单的同步机制
            print(f"[Trainer] No distributed training detected. Skipping GPU synchronization.")

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[TRAINER] Starting fit() - training loop initialization...")
        
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None
        print(f"[TRAINER] Training will run for {self.training_steps} steps")

        # load checkpoint before doing anything
        print("[TRAINER] Loading checkpoint...")
        self._load_checkpoint()
        main_tqdm.update(self.global_step)
        print(f"[TRAINER] Checkpoint loaded, starting from step {self.global_step}")

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            print("[TRAINER] Performing validation before training...")
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            print("[TRAINER] Pre-training validation completed")
            if self.config.trainer.val_only:
                print("[TRAINER] Validation only mode, exiting...")
                return
        else:
            print("[TRAINER] Skipping pre-training validation")

        print("[TRAINER] Starting main training loop...")
        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1
            print(f"[TRAINER] === Starting training step {self.global_step}/{self.training_steps} ===")

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                print(f"[TRAINER] Step {self.global_step}: Generating batch data...")
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()
                print(f"[TRAINER] Step {self.global_step}: Batch data generated")

                # 记录训练样本到wandb
                self._maybe_log_train_generations(batch, self.global_step)

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                print(f"[TRAINER] Step {self.global_step}: Balancing batch...")
                self._balance_batch(batch, metrics=metrics)
                print(f"[TRAINER] Step {self.global_step}: Batch balanced")

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            # compute reward
            if "token_level_scores" not in batch.batch:
                print(f"[TRAINER] Step {self.global_step}: Computing reward...")
                print(f"[TRAINER] Step {self.global_step}: batch keys: {list(batch.batch.keys())}")
                print(f"[TRAINER] Step {self.global_step}: batch non_tensor keys: {list(batch.non_tensor_batch.keys())}")
                print(f"[TRAINER] Step {self.global_step}: batch size: {len(batch.batch)}")
                print(f"[TRAINER] Step {self.global_step}: About to call reward_fn.compute_reward.remote")
                
                with timer("reward", timing_raw):
                    try:
                        reward_ref = self.reward_fn.compute_reward.remote(batch)
                        print(f"[TRAINER] Step {self.global_step}: Reward computation started successfully")
                    except Exception as e:
                        print(f"[TRAINER] Step {self.global_step}: ERROR in reward computation: {e}")
                        print(f"[TRAINER] Step {self.global_step}: Exception type: {type(e)}")
                        import traceback
                        print(f"[TRAINER] Step {self.global_step}: Traceback: {traceback.format_exc()}")
                        raise

                # recompute old_log_probs
                print(f"[TRAINER] Step {self.global_step}: Computing old log probabilities...")
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)
                print(f"[TRAINER] Step {self.global_step}: Old log probabilities computed")

                # compute ref_log_probs
                if self.use_reference_policy:
                    print(f"[TRAINER] Step {self.global_step}: Computing reference log probabilities...")
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)
                    print(f"[TRAINER] Step {self.global_step}: Reference log probabilities computed")

                # compute values
                if self.use_critic:
                    print(f"[TRAINER] Step {self.global_step}: Computing values...")
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)
                    print(f"[TRAINER] Step {self.global_step}: Values computed")

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        print(f"[TRAINER] Step {self.global_step}: Getting reward scores...")
                        print(f"[TRAINER] Step {self.global_step}: About to call ray.get(reward_ref)")
                        
                        try:
                            reward_tensor, reward_metrics = ray.get(reward_ref)
                            print(f"[TRAINER] Step {self.global_step}: Reward scores obtained successfully")
                            print(f"[TRAINER] Step {self.global_step}: reward_tensor shape: {reward_tensor.shape}")
                            print(f"[TRAINER] Step {self.global_step}: reward_metrics keys: {list(reward_metrics.keys())}")
                            
                            # Print detailed reward information
                            print(f"[TRAINER] Step {self.global_step}: Sample-level rewards: {reward_tensor.cpu().tolist()}")
                            print(f"[TRAINER] Step {self.global_step}: Average reward: {reward_tensor.mean().item():.4f}")
                            print(f"[TRAINER] Step {self.global_step}: Min reward: {reward_tensor.min().item():.4f}")
                            print(f"[TRAINER] Step {self.global_step}: Max reward: {reward_tensor.max().item():.4f}")
                            print(f"[TRAINER] Step {self.global_step}: Reward std: {reward_tensor.std().item():.4f}")
                            
                            # Convert sample-level rewards to token-level rewards
                            if reward_tensor.dim() == 1:
                                # Sample-level rewards: broadcast to token level
                                print(f"[TRAINER] Step {self.global_step}: Converting sample-level rewards to token-level")
                                response_mask = batch.batch["response_mask"]
                                token_level_scores = reward_tensor.unsqueeze(1).expand(-1, response_mask.shape[1])
                                batch.batch["token_level_scores"] = token_level_scores
                                print(f"[TRAINER] Step {self.global_step}: Token-level scores shape: {token_level_scores.shape}")
                            else:
                                # Already token-level rewards
                                batch.batch["token_level_scores"] = reward_tensor
                            
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                            metrics.update(reward_metrics)
                            print(f"[TRAINER] Step {self.global_step}: Reward metrics processed")
                        except Exception as e:
                            print(f"[TRAINER] Step {self.global_step}: ERROR getting reward scores: {e}")
                            print(f"[TRAINER] Step {self.global_step}: Exception type: {type(e)}")
                            import traceback
                            print(f"[TRAINER] Step {self.global_step}: Traceback: {traceback.format_exc()}")
                            raise

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        print(f"[TRAINER] Step {self.global_step}: Applying KL penalty...")
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                        print(f"[TRAINER] Step {self.global_step}: KL penalty applied")
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    print(f"[TRAINER] Step {self.global_step}: Computing advantages...")
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )
                    print(f"[TRAINER] Step {self.global_step}: Advantages computed")

                # update critic
                if self.use_critic:
                    print(f"[TRAINER] Step {self.global_step}: Updating critic...")
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)
                    print(f"[TRAINER] Step {self.global_step}: Critic updated")

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    print(f"[TRAINER] Step {self.global_step}: Updating actor...")
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)
                    print(f"[TRAINER] Step {self.global_step}: Actor updated")
                else:
                    print(f"[TRAINER] Step {self.global_step}: Skipping actor update (critic warmup)")

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    print(f"[TRAINER] Step {self.global_step}: Performing validation...")
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)
                    print(f"[TRAINER] Step {self.global_step}: Validation completed")

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    print(f"[TRAINER] Step {self.global_step}: Saving checkpoint...")
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()
                    print(f"[TRAINER] Step {self.global_step}: Checkpoint saved")

            # collect metrics
            print(f"[TRAINER] Step {self.global_step}: Collecting metrics...")
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()
            print(f"[TRAINER] Step {self.global_step}: Completed successfully")

        print("[TRAINER] Main training loop completed!")

        # perform validation after training
        if self.val_reward_fn is not None:
            print("[TRAINER] Performing final validation...")
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            print("[TRAINER] Saving final checkpoint...")
            self._save_checkpoint()
        
        print("[TRAINER] Training completed successfully!")
