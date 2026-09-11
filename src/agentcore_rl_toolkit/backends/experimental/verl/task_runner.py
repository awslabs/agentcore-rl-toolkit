"""verl's task runner plus the shared resources a rollout-session agent loop needs.

Pass to ``run_ppo(config, task_runner_class=...)`` in place of verl's ``TaskRunnerV1``.
It wraps rather than subclasses ``TaskRunnerV1`` (already ``@ray.remote``, so there is no
class to inherit from), and driver-only, so it is not star-imported by the package init.
"""

import ray
from omegaconf import DictConfig
from verl.trainer.main_ppo import TaskRunnerV1

from .rollout_session_resources import start_rollout_session_agent_loop_resources

__all__ = ["TaskRunnerWithRolloutSessionResources"]


@ray.remote(num_cpus=1)
class TaskRunnerWithRolloutSessionResources:
    """Sets up the run's shared rollout-session resources, then defers to verl."""

    async def run(self, config: DictConfig):
        self.config = config
        self.runner = TaskRunnerV1.remote()

        if "rollout_session_agent_loop" in self.config:
            # kept on self for the run's length: dropping the handles would collect the
            # non-detached actors mid-run (see RolloutSessionAgentLoopResources)
            self.rollout_session_resources = await start_rollout_session_agent_loop_resources(config)

        await self.runner.run.remote(config)
