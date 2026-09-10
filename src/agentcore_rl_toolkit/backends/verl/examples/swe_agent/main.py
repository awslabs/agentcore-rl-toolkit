import datetime as dt

import hydra
from omegaconf import open_dict
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from agentcore_rl_toolkit.aws_tools.ec2_tools import get_current_instance_type
from agentcore_rl_toolkit.backends.experimental.verl.task_runner import (
    TaskRunnerWithRolloutSessionResources,
)


def main(config):
    auto_set_device(config)

    with open_dict(config):
        config.ec2_instance_type = get_current_instance_type()
        config.trainer.experiment_start_at = dt.datetime.now().isoformat()
        config.data.train_batch_size = (
            config.trainer.v1.separate_async.parameter_sync_step * config.actor_rollout_ref.actor.ppo_mini_batch_size
        )

    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    run_ppo(config, task_runner_class=TaskRunnerWithRolloutSessionResources)


@hydra.main(config_path="config", config_name="main", version_base=None)
def task(config):
    main(config)


if __name__ == "__main__":
    task()
