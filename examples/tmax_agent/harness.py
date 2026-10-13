"""Run mini-swe-agent with commands executed in an AgentCore sandbox."""

import math
import time

from litellm import ContextWindowExceededError
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

from agentcore_rl_toolkit.sandbox import ExecTimeoutError


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Agent time budget exhausted")
    return seconds


class SandboxEnvironment(LocalEnvironment):
    def __init__(self, sandbox, deadline, **kwargs):
        super().__init__(**kwargs)
        self.sandbox = sandbox
        self.deadline = deadline
        platform = self.execute({"command": "uname -s; uname -r; uname -v; uname -m"})
        self.platform = dict(
            zip(("system", "release", "version", "machine"), platform["output"].splitlines(), strict=True)
        )

    def execute(self, action, cwd="", *, timeout=None):
        try:
            result = self.sandbox.exec(
                ["/bin/bash", "-c", action["command"]],
                cwd=cwd or self.config.cwd or None,
                env=self.config.env,
                timeout=math.ceil(min(timeout or self.config.timeout, remaining(self.deadline))),
            )
        except ExecTimeoutError as exc:
            result = exc.result
        output = {
            "output": result.stdout + result.stderr,
            "returncode": result.exit_code,
            "exception_info": "Command timed out" if result.timed_out else "",
            "extra": {
                "timed_out": result.timed_out,
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
            },
        }
        self._check_finished(output)
        return output

    def get_template_vars(self, **kwargs):
        return self.config.model_dump() | self.platform | kwargs


class DeadlineModel(LitellmModel):
    abort_exceptions = [*LitellmModel.abort_exceptions, TimeoutError]

    def __init__(self, deadline, **kwargs):
        super().__init__(**kwargs)
        self.deadline = deadline

    def _query(self, messages, **kwargs):
        # Recompute the request timeout for each of mini-swe's retry attempts.
        try:
            return super()._query(messages, **(kwargs | {"timeout": remaining(self.deadline)}))
        finally:
            remaining(self.deadline)


def run(instruction, sandbox, config, *, timeout=600, max_steps=64, record):
    settings = get_config_from_spec(builtin_config_dir / "mini.yaml")
    settings["model"]["model_kwargs"].update(
        api_base=config["base_url"],
        api_key=config.get("api_key", "EMPTY"),
        max_tokens=16384,
        temperature=0.7,
        top_p=0.95,
        num_retries=0,
    )
    settings["agent"].update(step_limit=max_steps, cost_limit=0, wall_time_limit_seconds=math.ceil(timeout))
    deadline = time.monotonic() + timeout
    environment = SandboxEnvironment(sandbox, deadline, timeout=120, **settings["environment"])
    model = DeadlineModel(
        deadline,
        model_name=f"openai/{config['model_id']}",
        cost_tracking="ignore_errors",
        **settings["model"],
    )
    agent = DefaultAgent(model, environment, **settings["agent"])
    try:
        result = agent.run(instruction.strip())
        record["stop_reason"] = result["exit_status"]
        if result["exit_status"] == "TimeExceeded":
            raise TimeoutError("Agent time budget exhausted")
    except ContextWindowExceededError:
        record["stop_reason"] = "context_limit"
    finally:
        record["trajectory"] = agent.messages
