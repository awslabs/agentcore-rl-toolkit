"""AgentCore trainer for verl.

Subclass of RayPPOTrainer that overrides init_workers() and fit() to provide
a clean AgentCore-only training loop. Rewards come from S3 via AgentCore
(no trainer-side reward function needed). Multi-segment trajectories are
supported with advantage broadcast from last-step segments.
"""

import math
import uuid
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.utils import Role
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.workers.utils.padding import left_right_2_no_padding

from agentcore_rl_toolkit.backends.verl.llm_server import AgentCoreLLMServerManager
from agentcore_rl_toolkit.backends.verl.loop_manager import AgentCoreLoopManager


def _broadcast_advantage_to_steps(
    batch: DataProto,
    last_step_batch: DataProto,
    last_step_idx: np.ndarray,
    non_last_step_idx: np.ndarray,
) -> None:
    """Broadcast advantage from last steps to all earlier steps of the same trajectory.

    Operates in-place on `batch`.
    """
    N = len(batch)
    R = batch.batch["response_mask"].shape[1]

    batch.batch["advantages"] = torch.zeros(N, R, dtype=torch.float32)
    batch.batch["returns"] = torch.zeros(N, R, dtype=torch.float32)

    # Write last-step advantages back to their original positions
    for i, idx in enumerate(last_step_idx):
        batch.batch["advantages"][idx] = last_step_batch.batch["advantages"][i]
        batch.batch["returns"][idx] = last_step_batch.batch["returns"][i]

    # Extract scalar advantage per trajectory from last steps
    traj_to_advantage = {}
    src_traj_ids = last_step_batch.non_tensor_batch["trajectory_id"]
    src_advantages = last_step_batch.batch["advantages"]
    src_masks = last_step_batch.batch["response_mask"]

    for i in range(len(last_step_batch)):
        traj_id = src_traj_ids[i]
        mask = src_masks[i].bool()
        if mask.any():
            scalar_adv = src_advantages[i][mask].mean().item()
        else:
            scalar_adv = 0.0
        traj_to_advantage[traj_id] = scalar_adv

    # Assign advantage to non-last steps
    for idx in non_last_step_idx:
        traj_id = batch.non_tensor_batch["trajectory_id"][idx]
        scalar_adv = traj_to_advantage.get(traj_id, 0.0)
        batch.batch["advantages"][idx] = scalar_adv * batch.batch["response_mask"][idx].float()
        batch.batch["returns"][idx] = batch.batch["advantages"][idx].clone()


class AgentCoreTrainer(RayPPOTrainer):
    """PPO/GRPO trainer specialized for AgentCore rollout mode.

    Key differences from the base RayPPOTrainer:
    - Rewards come from S3 (no trainer-side reward function)
    - Uses AgentCoreLLMServerManager (with model gateway) and AgentCoreLoopManager
    - Multi-segment trajectories with advantage broadcast
    - Overrides fit() with a clean agentcore-only loop
    """

    def init_workers(self):
        """Initialize workers with AgentCore-specific server and loop managers."""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError("AgentCore trainer requires hybrid engine")

        # Create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # Initialize worker groups
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # Init actor/rollout (last so vllm can better estimate KV cache memory)
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # Create AgentCore LLM server manager (handles vLLM replicas + gateway)
        self.llm_server_manager = AgentCoreLLMServerManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )
        self.llm_server_manager.start_gateway()

        # Create AgentCore loop manager
        self.async_rollout_manager = AgentCoreLoopManager(
            config=self.config,
            gateway_url=self.llm_server_manager.gateway_url,
        )

        # Create checkpoint manager
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        from verl.checkpoint_engine import CheckpointEngineManager

        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # Sleep replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _pad_batch_for_distributed(self, batch: DataProto) -> tuple[DataProto, int]:
        """Pad batch to be evenly distributable AND mini-batchable."""
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_steps = self.config.actor_rollout_ref.actor.get("ppo_mini_steps", None)
        rollout_n = self.config.actor_rollout_ref.rollout.n

        if ppo_mini_steps is not None:
            num_mini_batch = max(1, int(ppo_mini_steps))
        else:
            mini_batch_global = ppo_mini_batch_size * rollout_n
            num_mini_batch = max(1, len(batch) // mini_batch_global)

        dp_sizes = [self._get_dp_size(self.actor_rollout_wg, "actor")]
        if (
            getattr(self, "use_reference_policy", False)
            and not getattr(self, "ref_in_actor", False)
            and self.ref_policy_wg is not None
            and self.ref_policy_wg is not self.actor_rollout_wg
        ):
            dp_sizes.append(self._get_dp_size(self.ref_policy_wg, "ref"))
        agg_dp_size = math.lcm(*dp_sizes) if len(dp_sizes) > 1 else dp_sizes[0]

        divisor = agg_dp_size * num_mini_batch
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, divisor)

        if pad_size > 0:
            original_size = len(batch)
            for i in range(pad_size):
                idx = original_size + i
                batch_padded.non_tensor_batch["is_last_step"][idx] = False
                batch_padded.non_tensor_batch["trajectory_id"][idx] = "__pad__"
                batch_padded.batch["reward_tensor"][idx] = 0.0
                if "response_mask" in batch_padded.batch:
                    batch_padded.batch["response_mask"][idx] = 0
                if "rollout_log_probs" in batch_padded.batch:
                    batch_padded.batch["rollout_log_probs"][idx] = -100.0

        batch_padded.meta_info["num_mini_batch"] = num_mini_batch
        return batch_padded, pad_size

    def _update_actor(self, batch: DataProto) -> DataProto:
        """Override to pass num_mini_batch from agentcore padding."""
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        calculate_entropy = self.config.actor_rollout_ref.actor.get("calculate_entropy", False) or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle

        update_metadata = {
            "calculate_entropy": calculate_entropy,
            "epochs": ppo_epochs,
            "seed": seed,
            "dataloader_kwargs": {"shuffle": shuffle},
            "compute_loss": True,
        }

        num_mini_batch = batch.meta_info.get("num_mini_batch")
        if num_mini_batch is not None:
            update_metadata["num_mini_batch"] = num_mini_batch
            assert (
                len(batch) % num_mini_batch == 0
            ), f"Batch size {len(batch)} not divisible by num_mini_batch {num_mini_batch}"
            update_metadata["global_batch_size"] = len(batch) // num_mini_batch
        else:
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            update_metadata["mini_batch_size"] = ppo_mini_batch_size
            update_metadata["global_batch_size"] = ppo_mini_batch_size

        tu.assign_non_tensor(batch_td, **update_metadata)

        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _validate(self, merged: bool = False):
        """Validation loop for AgentCore mode.

        Dispatches validation rollouts via AgentCoreLoopManager, extracts
        rewards from the returned DataProto, handles failed sessions.
        """
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch))], dtype=object
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            test_gen_batch = test_batch
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            try:
                test_output_gen_batch = self.async_rollout_manager.generate_sequences(test_gen_batch)
            except RuntimeError as e:
                if "No rollout data received" in str(e):
                    n_sessions = len(test_batch)
                    print(f"[Validation] All rollouts failed for {n_sessions} sessions. Recording reward=0.")
                    reward_extra_infos_dict["reward"].extend([0.0] * n_sessions)
                    sample_scores.extend([0.0] * n_sessions)
                    sample_inputs.extend(["[rollout failed]"] * n_sessions)
                    sample_outputs.extend(["[rollout failed]"] * n_sessions)
                    sample_gts.extend([None] * n_sessions)
                    sample_uids.extend(test_batch.non_tensor_batch["uid"].tolist())
                    data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * n_sessions))
                    continue
                raise

            print("validation generation end")

            # Select last steps only for metric computation
            is_last_step_val = test_output_gen_batch.non_tensor_batch["is_last_step"]
            last_step_idx_val = np.where(is_last_step_val)[0]
            test_batch_result = test_output_gen_batch.select_idxs(last_step_idx_val)
            ground_truths = [None] * len(test_batch_result)
            sample_gts.extend(ground_truths)

            # Account for failed sessions
            original_uid_counts = Counter(test_gen_batch.non_tensor_batch["uid"].tolist())
            returned_uid_counts = Counter(test_batch_result.non_tensor_batch["uid"].tolist())
            n_total_missing = 0
            for uid_val, n_dispatched in original_uid_counts.items():
                n_failed = n_dispatched - returned_uid_counts.get(uid_val, 0)
                if n_failed > 0:
                    n_total_missing += n_failed
                    reward_extra_infos_dict["reward"].extend([0.0] * n_failed)
                    sample_scores.extend([0.0] * n_failed)
                    sample_inputs.extend(["[rollout failed]"] * n_failed)
                    sample_outputs.extend(["[rollout failed]"] * n_failed)
                    sample_gts.extend([None] * n_failed)
                    sample_uids.extend([uid_val] * n_failed)
            if n_total_missing > 0:
                n_affected = sum(
                    1
                    for uid_val, n_dispatched in original_uid_counts.items()
                    if returned_uid_counts.get(uid_val, 0) < n_dispatched
                )
                print(
                    f"[Validation] {n_total_missing} failed sessions across "
                    f"{n_affected} examples. Recording reward=0 for each."
                )
                data_source_lst.append(["unknown"] * n_total_missing)

            # Store generated outputs
            output_ids = test_batch_result.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            input_ids = test_batch_result.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch_result.non_tensor_batch["uid"].tolist())

            reward_tensor = test_batch_result.batch["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)

            data_source_lst.append(
                test_batch_result.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": [],
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, [])

    def fit(self):
        """AgentCore-specific training loop.

        Flow per step:
        1. Dispatch rollouts to AgentCore via loop manager
        2. Recompute old_log_probs on training model
        3. Compute ref_log_probs (if KL enabled)
        4. Compute advantage on last-step segments, broadcast to earlier segments
        5. Update actor
        6. Sync weights to vLLM replicas
        """
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger_tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # Validate before training
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger_tracker.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)

                gen_batch = batch
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # Generate rollouts via AgentCore
                    with marked_timer("gen", timing_raw, color="red"):
                        try:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        except RuntimeError as e:
                            if "No rollout data received" in str(e):
                                print(f"[Step {self.global_steps}] All rollouts failed, skipping: {e}")
                                progress_bar.update(1)
                                self.global_steps += 1
                                if self.global_steps > self.total_training_steps:
                                    break
                                continue
                            raise
                        self.checkpoint_manager.sleep_replicas()

                    timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                    gen_batch_output.meta_info.pop("timing", None)

                    batch = gen_batch_output
                    # AgentCore's generate_sequences returns a fresh DataProto, so meta_info
                    # set before generation (e.g. `temperature`) doesn't survive. The megatron
                    # forward step reads `batch["temperature"]`, so re-inject it here. Mirrors
                    # the pattern used in verl's `_update_actor` and AWSAgenticAIVerl's
                    # `_compute_old_log_prob`.
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                    if "response_mask" not in batch.batch.keys():
                        response_length = batch.batch["responses"].size(1)
                        attention_mask = batch.batch["attention_mask"]
                        batch.batch["response_mask"] = attention_mask[:, -response_length:]

                    # Pad for distributed training
                    assert self.config.algorithm.adv_estimator in (
                        AdvantageEstimator.GRPO,
                        AdvantageEstimator.REMAX,
                    ), f"AgentCore mode only supports GRPO and REMAX, got {self.config.algorithm.adv_estimator}"
                    batch, _ = self._pad_batch_for_distributed(batch)

                    # Balance batch across DP ranks
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Reward is already in batch from loop manager
                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_tensor = batch.batch["reward_tensor"]

                    # Recompute old log probs on training model
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item(),
                            "perf/mfu/actor_infer": old_log_prob_mfu,
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_probs" not in {batch.batch.keys()=}'

                    # Compute reference log probs if needed
                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # Compute advantage
                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["token_level_scores"] = reward_tensor

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        # Advantage broadcast: compute on last-step segments, broadcast to earlier ones
                        is_last_step_arr = batch.non_tensor_batch["is_last_step"]
                        last_step_idx = np.where(is_last_step_arr)[0]
                        non_last_step_idx = np.where(~is_last_step_arr)[0]

                        last_step_batch = batch.select_idxs(last_step_idx)
                        last_step_batch = compute_advantage(
                            last_step_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        _broadcast_advantage_to_steps(batch, last_step_batch, last_step_idx, non_last_step_idx)

                    # Update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    else:
                        self.checkpoint_manager.update_weights(self.global_steps)

                    # Recompute token counts excluding pad rows
                    real_mask = batch.non_tensor_batch["trajectory_id"] != "__pad__"
                    per_row_tokens = torch.sum(batch.batch["attention_mask"], dim=-1)
                    batch.meta_info["global_token_num"] = per_row_tokens[torch.from_numpy(real_mask)].tolist()

                # Validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # Training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))

                # Adjust score/reward mean for failed rollout sessions
                n_dispatched = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                is_last = batch.non_tensor_batch.get("is_last_step", None)
                n_succeeded = int(np.sum(is_last)) if is_last is not None else n_dispatched
                if n_succeeded < n_dispatched:
                    scale = n_succeeded / n_dispatched
                    for key in ("critic/score/mean", "critic/rewards/mean"):
                        if key in metrics:
                            metrics[key] *= scale

                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

                logger_tracker.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    del logger_tracker
                    progress_bar.close()
                    return
