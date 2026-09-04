import ray

from agentcore_rl_toolkit.backends.verl import main_ppo


class _RemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)


class _Runner:
    def __init__(self):
        self.__ray_call__ = _RemoteMethod()


class _TaskRunner:
    def __init__(self, runner):
        self.runner = runner
        self.options_kwargs = None

    def remote(self):
        return self.runner

    def options(self, **kwargs):
        self.options_kwargs = kwargs
        return self


def test_task_runner_factory_preserves_options_and_queues_registration(monkeypatch):
    runner = _Runner()
    task_runner = _TaskRunner(runner)
    monkeypatch.setattr(main_ppo, "TaskRunnerV1", task_runner)

    configured = main_ppo.AgentCoreTaskRunnerV1.options(runtime_env={"nsight": {}})
    result = configured.remote()

    assert result is runner
    assert task_runner.options_kwargs == {"runtime_env": {"nsight": {}}}
    assert runner.__ray_call__.calls == [(main_ppo._register_agentcore_trainer,)]


def test_registration_runs_inside_stock_task_runner_actor():
    def registered_trainer_name(_task_runner):
        from verl.trainer.ppo.v1 import get_trainer_cls

        return get_trainer_cls("agentcore_sync").__name__

    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)

    runner = main_ppo.AgentCoreTaskRunnerV1.remote()
    try:
        assert ray.get(runner.__ray_call__.remote(registered_trainer_name)) == "AgentCorePPOTrainerSync"
    finally:
        ray.kill(runner)
        if started_ray:
            ray.shutdown()
