#!/usr/bin/env python3
"""Harbor harness — agent_type "strands": Strands Agent, bash tool -> sb.exec.

SEPARATED agent, framework-run: a Strands ``Agent`` — backed by a Bedrock
``BedrockModel`` for evaluation, or by an ``OpenAIModel`` wired from the
trainer-injected ``_rollout`` config (base_url/model_id/api_key) for RL
training — reasons IN THIS PROCESS (arm64 microVM); its single tool,
``bash``, is overridden to ship each command into the task session via
``SandboxClient.exec`` and return the output:

    launcher ──invoke──▶ THIS app (Strands Agent) ──bash tool = sb.exec──▶ task runtime

The system prompt introduces exactly ONE thing — that a remote sandbox holds
the task and the ``bash`` tool runs there. The user message is EXACTLY the task
instruction. Grading is identical to the other harness, so results are directly
comparable.

Async contract (same for both harness images): the entrypoint is the
toolkit's @rollout_entrypoint — the invoke ACKS immediately and the rollout runs
as a background task whose record (or a status_code=500 error doc) is saved to
s3://<s3_bucket>/<exp_id>/<input_id>/<session_id>.json.

Payload: task_runtime_arn, instruction (required); tests_tar_b64, model,
    max_steps, agent_timeout_s, verifier_timeout_s, _rollout.
"""
from __future__ import annotations

import time

from harbor_sandbox import REGION, HarborSandboxClient
from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent, HookProvider
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel

from agentcore_rl_toolkit import AgentCoreRLApp
from agentcore_rl_toolkit.sandbox import SandboxClient

# REGION is imported from harbor_sandbox.config (.env-driven) so the model calls
# follow the deploy region, not the Dockerfile-baked AWS_REGION.
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"

# Default agent wall-clock ceiling when the payload omits a per-task budget. The
# step cap alone is not enough: a task whose every command times out could burn
# max_steps*120s. The launcher normally passes the task's own agent_timeout_s
# (task.toml [agent].timeout_sec); either way the agent stops at the budget and
# we still GRADE its partial work (faithful AgentTimeout).
DEFAULT_AGENT_TIMEOUT_S = 3300
# Verifier (test.sh) budget when the payload omits [verifier].timeout_sec, and
# the exec service ceiling it is clamped to — same as claude_code, so grading
# stays directly comparable (a hardcoded 180s silently truncated heavy verifiers).
DEFAULT_VERIFIER_TIMEOUT_S = 900
EXEC_TIMEOUT_MAX_S = 3600
# Per-read idle timeout: must outlast the longest single command (the 3600s
# verifier) so a long silent verifier isn't killed mid-run.
SANDBOX_READ_TIMEOUT_S = EXEC_TIMEOUT_MAX_S + 300  # 3900

SYSTEM_PROMPT = (
    "You are an autonomous software-engineering agent. Use the `bash` tool to run "
    "commands in a remote Linux sandbox — that is your only way to act. Complete "
    "the task, then stop."
)

app = AgentCoreRLApp()


def _resolve_sandbox(payload):
    """Start a task's sandbox. Two addressing modes:
      * legacy: payload["task_runtime_arn"] — a pre-created runtime; open a session.
      * harbor: payload["benchmark"] + payload["task_id"] — ensure the runtime via
        HarborSandboxClient.create (idempotent). With payload["lease"]=true the
        harness deletes it after the rollout (the lease pattern).

    Returns (sandbox_client, release_fn) — release_fn is a no-op unless leased.
    """
    arn = payload.get("task_runtime_arn")
    if arn:
        return SandboxClient(runtime_arn=arn, read_timeout=SANDBOX_READ_TIMEOUT_S), (lambda: None)
    bench, task = payload.get("benchmark"), payload.get("task_id")
    if not (bench and task):
        raise ValueError("payload must include either 'task_runtime_arn' or " "'benchmark' + 'task_id'")
    client = HarborSandboxClient.create(bench, task, read_timeout=SANDBOX_READ_TIMEOUT_S)
    if payload.get("lease"):
        return client, client.release  # instance path: deletes its OWN runtime
    return client, (lambda: None)


class _Budget(HookProvider):
    """Hard cap on BOTH tool count and wall clock, so a stuck agent cannot loop
    forever or burn time on a task whose every command times out. Raises once
    either budget is exceeded; the caller catches it and grades what exists."""

    class Exceeded(Exception):
        pass

    def __init__(self, limit: int, deadline: float):
        self.limit, self.deadline, self.n = limit, deadline, 0

    def register_hooks(self, registry, **_):
        registry.add_callback(BeforeToolCallEvent, self._before)

    def _before(self, event):
        self.n += 1
        if self.n > self.limit or time.time() > self.deadline:
            raise _Budget.Exceeded()


def _stage_b64(sb, b64: str, dest: str, chunk: int = 50000) -> None:
    """Decode a base64 blob to <dest> inside the box, written in <=50000-char
    slices so a large test tarball can't blow exec's 64KB command cap (the
    base64 alphabet has no single-quotes, so '...'-wrapping is safe)."""
    tmp = dest + ".b64"
    sb.exec(f"rm -f {tmp}", timeout=30)
    for i in range(0, len(b64), chunk):
        sb.exec(f"printf '%s' '{b64[i:i + chunk]}' >> {tmp}", timeout=60)
    sb.exec(f"base64 -d {tmp} > {dest} && rm -f {tmp}", timeout=60)


def grade_with_verifier(sb, tests_tar_b64: str, verifier_timeout_s: float = DEFAULT_VERIFIER_TIMEOUT_S) -> dict:
    """Run the task's own shipped verifier in the sandbox (uniform corpus
    contract: tests/test.sh -> /logs/verifier/reward.txt). The test.sh budget is
    the task's own [verifier].timeout_sec, clamped to exec's 3600s ceiling — a
    hardcoded cap would silently truncate heavy verifiers into reward=None."""
    if not tests_tar_b64:
        return {"reward": None, "rewards": None, "verifier_tail": "(no tests supplied)"}
    verify_s = min(int(verifier_timeout_s), EXEC_TIMEOUT_MAX_S)
    sb.exec("rm -rf /tests && mkdir -p /tests /logs/verifier", timeout=30)
    _stage_b64(sb, tests_tar_b64, "/tmp/_tests.tgz")
    unpack = sb.exec("tar xzf /tmp/_tests.tgz -C /tests 2>&1", timeout=120)
    run = sb.exec("bash /tests/test.sh 2>&1", timeout=verify_s)
    reward_txt = sb.exec("cat /logs/verifier/reward.txt 2>/dev/null", timeout=30)
    reward = None
    try:
        reward = int(reward_txt.stdout.strip())
    except Exception:
        pass
    return {
        "reward": reward,
        # Plural alias: training backends read result["rewards"] (e.g.
        # backends/experimental/verl/agent_loop.py); the singular key stays for
        # the eval launcher. None (verifier wrote no reward.txt) scores 0.
        "rewards": reward,
        "verifier_tail": (run.stdout or "")[-1500:],
        "unpack_err": (unpack.stdout or "")[:300] if unpack.exit_code != 0 else "",
        "verifier_budget_s": verify_s,
        "verifier_timed_out": bool(getattr(run, "timed_out", False)),
    }


def run_rollout(
    sb_client,
    instruction,
    model,
    max_steps,
    tests_tar_b64,
    agent_timeout_s,
    verifier_timeout_s=DEFAULT_VERIFIER_TIMEOUT_S,
    rollout_cfg=None,
) -> dict:
    client = sb_client
    commands: list[str] = []
    timing = {}

    t0 = time.time()
    with client.start() as sb:
        timing["start_s"] = round(time.time() - t0, 1)

        @tool
        def bash(command: str) -> str:
            """Run a shell command in the remote task sandbox and return its
            combined result (exit code, stdout, stderr)."""
            commands.append(command)
            r = sb.exec(command, timeout=120)
            return f"exit={r.exit_code}\nstdout:\n{r.stdout[:4000]}\nstderr:\n{r.stderr[:1500]}"

        rc = rollout_cfg or {}
        if rc.get("base_url"):
            # Training: the trainer injects the inference endpoint (the rollout
            # gateway) via _rollout, and the api-key slot carries the trajectory-
            # capture session key — it MUST reach the LLM client or every rollout
            # degenerates into one shared gateway session. "EMPTY" keeps plain
            # OpenAI-compatible eval endpoints (vLLM etc.) working unchanged.
            model = rc["model_id"]
            model_obj = OpenAIModel(
                client_args={"api_key": rc.get("api_key") or "EMPTY", "base_url": rc["base_url"]},
                model_id=model,
                params=rc.get("sampling_params", {}),
            )
        else:
            model_obj = BedrockModel(model_id=model, region_name=REGION, max_tokens=4096, temperature=1.0)
        t0 = time.time()
        limiter = _Budget(max_steps, deadline=t0 + agent_timeout_s)
        agent = Agent(
            model=model_obj, tools=[bash], system_prompt=SYSTEM_PROMPT, hooks=[limiter], callback_handler=None
        )
        stop = "end_turn"
        try:
            agent(instruction)
        except _Budget.Exceeded:
            stop = "max_steps" if limiter.n > limiter.limit else "max_wall"
        except Exception as e:  # still grade whatever the agent left behind
            stop = f"agent_error:{type(e).__name__}"
        timing["agent_s"] = round(time.time() - t0, 1)

        t0 = time.time()
        result = grade_with_verifier(sb, tests_tar_b64, verifier_timeout_s)
        timing["grade_s"] = round(time.time() - t0, 1)

    result.update(
        {
            "agent_type": "strands",
            "model": model,
            "stop_reason": stop,
            "steps": len(commands),
            "timing": timing,
            "commands": [c[:500] for c in commands],
        }
    )
    return result


@app.rollout_entrypoint
def invoke(payload: dict) -> dict:
    """One rollout per invocation, run as a background task by the SDK.

    Exceptions are NOT caught here on purpose: the SDK's error path saves
    {"status_code": 500, "stop_reason": str(e), "traceback": ...} to the same
    S3 key, so the launcher learns about failures the same way as results.
    """
    instruction = payload.get("instruction")
    if not instruction:
        raise ValueError("payload must include 'instruction'")
    sb_client, release = _resolve_sandbox(payload)
    try:
        return run_rollout(
            sb_client=sb_client,
            instruction=instruction,
            model=payload.get("model", DEFAULT_MODEL),
            max_steps=int(payload.get("max_steps", 25)),
            tests_tar_b64=payload.get("tests_tar_b64", ""),
            agent_timeout_s=float(payload.get("agent_timeout_s", DEFAULT_AGENT_TIMEOUT_S)),
            verifier_timeout_s=float(payload.get("verifier_timeout_s", DEFAULT_VERIFIER_TIMEOUT_S)),
            rollout_cfg=payload.get("_rollout"),
        )
    finally:
        # guarded so a delete failure (AccessDenied, or ResourceNotFound when the
        # runtime never became ready) can't REPLACE the rollout's real result or
        # exception with the cleanup error. Swallow it; the slot may leak.
        try:
            release()
        except Exception as e:  # noqa: BLE001 - cleanup best-effort
            print(f"lease release failed (ignored, slot may leak): {e!r}")


if __name__ == "__main__":
    app.run()
