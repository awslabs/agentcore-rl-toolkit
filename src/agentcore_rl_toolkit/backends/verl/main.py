"""Main entrypoint for AgentCore RL training with verl.

Usage:
    python -m agentcore_rl_toolkit.backends.verl.main \
        --config-path=config --config-name=agentcore_grpo \
        actor_rollout_ref.model.path=/path/to/model \
        ...
"""

import logging
import os
import socket

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local


class VerlRunner:
    """Ray remote class for executing AgentCore RL training."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def run(self, config):
        """Execute the AgentCore training workflow."""
        from pprint import pprint

        # Re-establish logging config inside the Ray actor. Other libraries
        # imported before verl may have already attached handlers to the root
        # logger, which makes verl's basicConfig() in verl/__init__.py a no-op
        # and silently drops logger.warning(...) records from this package.
        logging.basicConfig(
            format="%(levelname)s:%(asctime)s:%(message)s",
            level=logging.WARNING,
            force=True,
        )

        from verl.single_controller.ray import RayWorkerGroup, ResourcePoolManager
        from verl.trainer.ppo.ray_trainer import Role
        from verl.utils import hf_tokenizer
        from verl.workers.engine_workers import ActorRolloutRefWorker

        print(f"AgentCoreTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Setup actor rollout worker
        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        if need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"

        # Resource pool
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

        # Tokenizer
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Dataset
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            None,  # no processor needed for AgentCore mode
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            None,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Create and run trainer
        from agentcore_rl_toolkit.backends.verl.trainer import AgentCoreTrainer

        trainer = AgentCoreTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="agentcore_grpo", version_base=None)
def main(config: DictConfig):
    """Main entry point function for AgentCore RL training."""
    auto_set_device(config)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner_class = ray.remote(num_cpus=1)(VerlRunner)
    runner = runner_class.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
