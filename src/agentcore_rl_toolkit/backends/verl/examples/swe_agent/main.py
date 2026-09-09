import datetime as dt

import hydra
from omegaconf import open_dict
from verl.experimental.reward_loop.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

from agentcore_rl_toolkit.backends.experimental.verl.task_runner import (
    TaskRunnerWithRolloutSessionResources,
)

def main(config):
    auto_set_device(config)
    config.transfer_queue.enable = True

    config = migrate_legacy_reward_impl(config)

    from agentcore_rl_toolkit.aws_tools.ec2_tools import get_current_instance_type

    with open_dict(config):
        config.ec2_instance_type = get_current_instance_type()
        config.trainer.experiment_start_at = dt.datetime.now().isoformat()
        config.data.train_batch_size = (
            config.trainer.v1.separate_async.parameter_sync_step
            * config.actor_rollout_ref.actor.ppo_mini_batch_size
        )

    run_ppo(config, task_runner_class=TaskRunnerWithRolloutSessionResources)


if __name__ == "__main__":
    import sys

    config_name = sys.argv.pop()

    @hydra.main(config_path="config", config_name=config_name, version_base=None)
    def task(config):
        main(config)

    task()
