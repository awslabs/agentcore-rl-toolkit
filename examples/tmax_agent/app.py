"""Trusted harness runtime: model loop, then late verifier delivery."""

import json
import os
import shlex
import time
from dataclasses import asdict

import boto3
import litellm
from harness import run, tracer
from opentelemetry import trace

from agentcore_rl_toolkit import AgentCoreRuntimeApp
from agentcore_rl_toolkit.sandbox import ExecTimeoutError, SandboxClient

# One gen_ai span per model call, without message content (each call repeats the history).
os.environ.setdefault("USE_OTEL_LITELLM_REQUEST_SPAN", "true")
litellm.callbacks = ["otel"]
litellm.turn_off_message_logging = True
# Drop the help banner litellm prints on each error; the rollout log line keeps the error.
litellm.suppress_debug_info = True

app = AgentCoreRuntimeApp()


def tail(text, limit=4000):
    return text[-limit:] if text else text


def log_rollout(record, span, started):
    verifier = record.get("verifier") or {}
    summary = {
        "event": "tmax_rollout",
        "task_id": record["task_id"],
        "session_id": record["sandbox_session_id"],
        "source": record["source"],
        "reward": record["reward"],
        "stop_reason": record.get("stop_reason"),
        "error": tail(record.get("error")),
        "steps": sum(message.get("role") == "assistant" for message in record.get("trajectory", [])),
        "duration_s": round(time.monotonic() - started, 1),
        "verifier_exit_code": verifier.get("exit_code"),
        "verifier_timed_out": verifier.get("timed_out"),
        "verifier_stdout": tail(verifier.get("stdout")),
        "verifier_stderr": tail(verifier.get("stderr")),
        "trace_id": format(span.get_span_context().trace_id, "032x"),
    }
    for key in ("reward", "stop_reason", "error", "steps"):
        if summary[key] is not None:
            span.set_attribute(f"tmax.{key}", summary[key])
    print(json.dumps(summary), flush=True)


@tracer.start_as_current_span("tmax.verify")
def verify(sandbox, s3, bucket, task):
    span = trace.get_current_span()
    # Created only after the final model turn; never added to the conversation.
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": task["verifier_key"]},
        ExpiresIn=300,
    )
    command = f"curl -fsS {shlex.quote(url)} | tar -xz -C /\nmkdir -p /logs/verifier"
    result = sandbox.exec(["/bin/bash", "-eo", "pipefail", "-c", command], timeout=120)
    if result.exit_code:
        raise RuntimeError(f"Verifier delivery failed: {result.stderr}")
    # Background services must not keep sandboxd's output pipes open.
    command = (
        "TEST_DIR=/tests bash /tests/test.sh > /logs/verifier/stdout 2> /logs/verifier/stderr\n"
        "status=$?\n"
        "cat /logs/verifier/stdout\n"
        "cat /logs/verifier/stderr >&2\n"
        'exit "$status"'
    )
    try:
        verification = sandbox.exec(["/bin/bash", "-c", command], timeout=task["verifier_timeout"])
    except ExecTimeoutError as exc:
        span.set_attribute("tmax.verifier.timed_out", True)
        return 0.0, asdict(exc.result)
    span.set_attribute("tmax.verifier.exit_code", verification.exit_code)
    reward = sandbox.exec(["cat", "/logs/verifier/reward.txt"], timeout=120)
    if reward.exit_code:
        logs = sandbox.exec(
            ["/bin/bash", "-c", "cat /logs/verifier/stdout; cat /logs/verifier/stderr >&2"], timeout=120
        )
        raise RuntimeError(f"Verifier did not finish: {logs.stdout}\n{logs.stderr}")
    value = float(reward.stdout.strip())
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Unexpected verifier reward: {value}")
    return value, asdict(verification)


@app.entrypoint
def invoke(payload, context):
    task = payload["task"]
    sandbox_client = SandboxClient(task["runtime_arn"])
    session_id = context.session_id
    # Keep a handle even if the start response is lost, for finally cleanup.
    sandbox = sandbox_client.attach(session_id)
    record = {"task_id": task["task_id"], "sandbox_session_id": session_id, "source": task["source"], "reward": 0.0}
    with tracer.start_as_current_span(
        "tmax.rollout", attributes={"tmax.task_id": task["task_id"], "session.id": session_id}
    ) as span:
        started = time.monotonic()
        try:
            rollout(task, payload, sandbox_client, sandbox, record)
        finally:
            log_rollout(record, span, started)
    return record


def rollout(task, payload, sandbox_client, sandbox, record):
    session_id = record["sandbox_session_id"]
    s3 = boto3.client("s3", region_name=os.environ["VERIFIER_REGION"])
    bucket = os.environ["VERIFIER_BUCKET"]
    try:
        sandbox_client.start(session_id)
        try:
            run(
                task["instruction"],
                sandbox,
                payload["_config"],
                timeout=task["agent_timeout"],
                command_timeout=task.get("command_timeout", 300),
                max_steps=task.get("max_steps", 64),
                record=record,
            )
        except TimeoutError as exc:
            record.update(stop_reason="timeout", error=f"{type(exc).__name__}: {exc}")
        reward, verifier = verify(sandbox, s3, bucket, task)
        record.update(reward=reward, verifier=verifier)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["stop_reason"] = "timeout" if isinstance(exc, TimeoutError) else "error"
    finally:
        sandbox.terminate()


if __name__ == "__main__":
    app.run()
