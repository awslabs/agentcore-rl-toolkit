"""Run verl's default v1 trainer flow with AgentCore's trainer registered in TaskRunnerV1."""

import hydra
from omegaconf import DictConfig
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


def _register_agentcore_trainer(_task_runner) -> None:
    import agentcore_rl_toolkit.backends.verl.trainer as _trainer  # noqa: F401


class _ConfiguredTaskRunnerV1:
    def __init__(self, task_runner):
        self.task_runner = task_runner

    def remote(self):
        runner = self.task_runner.remote()
        # The registry is process-local. Queue the import on TaskRunnerV1 itself;
        # stock run_ppo submits run() through the same actor handle immediately after.
        runner.__ray_call__.remote(_register_agentcore_trainer)
        return runner


class AgentCoreTaskRunnerV1:
    """Create stock TaskRunnerV1 actors and register AgentCore before run()."""

    @staticmethod
    def remote():
        return _ConfiguredTaskRunnerV1(TaskRunnerV1).remote()

    @staticmethod
    def options(**options):
        return _ConfiguredTaskRunnerV1(TaskRunnerV1.options(**options))


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    if not config.trainer.use_v1:
        raise ValueError("AgentCore's verl launcher requires trainer.use_v1=true")
    run_ppo(config, task_runner_class=AgentCoreTaskRunnerV1)


if __name__ == "__main__":
    main()
