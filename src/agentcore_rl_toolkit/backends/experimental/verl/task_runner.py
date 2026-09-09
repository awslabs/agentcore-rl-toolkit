"""verl's task runner plus the shared resources a rollout-session agent loop needs.

``run_ppo(config, task_runner_class=...)`` takes the runner class by parameter, so
an agent's ``main.py`` swaps this in for verl's ``TaskRunnerV1`` and gets the
cluster-wide semaphores, priority assigners, session rate limiter and EC2 monitor
set up before training starts.

It **wraps** ``TaskRunnerV1`` rather than subclassing it: ``TaskRunnerV1`` is
already ``@ray.remote``-decorated, so the name is a Ray ``ActorClass`` and there
is no class to inherit from. Composition also keeps this actor's event loop free
for the EC2 monitor to poll on, which a synchronous trainer hook could not offer.

This module is deliberately **not** star-imported by the package ``__init__``:
that ``__init__`` is loaded in every verl worker via ``VERL_USE_EXTERNAL_MODULES``
, and this class only ever runs on the driver, ``main.py`` imports it by module path.
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
            # Kept on self for the length of the run: the named actors are not
            # detached, so dropping the handles would collect them mid-run, and the
            # monitor holds the only reference to its polling task. See
            # RolloutSessionAgentLoopResources.
            self.rollout_session_resources = await start_rollout_session_agent_loop_resources(config)

        await self.runner.run.remote(config)
