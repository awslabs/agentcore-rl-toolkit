import datetime as dt

import hydra
from omegaconf import open_dict
from verl.experimental.reward_loop.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

# Imported by module path rather than from the package, which star-imports only
# what every verl worker needs; this runs on the driver alone.
from agentcore_rl_toolkit.backends.experimental.verl.task_runner import (
    TaskRunnerWithRolloutSessionResources,
)

# The trainer this recipe runs -- its mixin list, and the trainer.v1.trainer_mode
# names it registers -- is declared in swe_agent_verl/trainer.py. It is not
# imported here: registration has to happen inside verl's TaskRunnerV1 actor, so
# the module is named in VERL_USE_EXTERNAL_MODULES.


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
