"""Trusted harness runtime: model loop, then late verifier delivery."""

import base64
import os
from dataclasses import asdict

import boto3
from harness import Bash, run

from agentcore_rl_toolkit import AgentCoreRuntimeApp
from agentcore_rl_toolkit.sandbox import ExecTimeoutError, SandboxClient

app = AgentCoreRuntimeApp()


def verify(sandbox, s3, bucket, task):
    # Created only after the final model turn; never added to the conversation.
    response = s3.get_object(Bucket=bucket, Key=task["verifier_key"])
    with response["Body"] as body:
        archive = base64.b64encode(body.read()).decode()
    command = (
        f'printf "%s" "{archive}" | base64 -d | tar -xz -C /\n'
        "mkdir -p /logs/verifier"
    )
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
        return 0.0, asdict(exc.result)
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
    s3 = boto3.client("s3", region_name=os.environ["VERIFIER_REGION"])
    bucket = os.environ["VERIFIER_BUCKET"]
    record = {"task_id": task["task_id"], "sandbox_session_id": session_id, "source": task["source"]}
    try:
        sandbox_client.start(session_id)
        bash = Bash(sandbox)
        try:
            run(task["instruction"], bash, payload["_config"], timeout=task["agent_timeout"], record=record)
        except TimeoutError as exc:
            record.update(stop_reason="timeout", error=f"{type(exc).__name__}: {exc}")
        reward, verifier = verify(sandbox, s3, bucket, task)
        record.update(reward=reward, verifier=verifier)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["stop_reason"] = "timeout" if isinstance(exc, TimeoutError) else "error"
    finally:
        sandbox.terminate()
    return record


if __name__ == "__main__":
    app.run()
