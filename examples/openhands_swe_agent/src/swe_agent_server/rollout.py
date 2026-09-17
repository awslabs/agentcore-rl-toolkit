"""Task setup and dispatch of a rollout to the requested agent backend."""

import json
import os
import subprocess
import tempfile

from swe_agent_server.observability import configure_openhands_tracing

from agentcore_rl_toolkit.rollout_session.wire import (
    RolloutDumpResponse,
    RolloutSetupRequest,
    RolloutStartRequest,
)


def run_rollout(payload: RolloutStartRequest) -> RolloutDumpResponse:
    """Dispatch a rollout to the requested agent backend.

    Backends are imported lazily so a deployment only needs the SDK for the ones it
    actually runs.
    """
    match payload.task_input["agent"]:
        case "openhands":
            # Must precede the import, which initialises OpenHands' tracing layer.
            configure_openhands_tracing()

            from swe_agent_server.open_hands_agent import rollout
        case "strands":
            from swe_agent_server.strands_agent import rollout
        case "oracle":
            from swe_agent_server.oracle_agent import rollout
        case "noop":
            from swe_agent_server.noop_agent import rollout
        case unknown:
            raise ValueError(f"Unknown agent backend: {unknown!r}")
    return rollout(payload)


def run_setup(request: RolloutSetupRequest):
    # The unpack script reads the fields it needs out of this JSON file (e.g. via jq).
    fd, task_path = tempfile.mkstemp(suffix=".json", prefix="swe_task_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(request.task_input, f)
        result = subprocess.run(
            [
                "/agent/swe_unpack.sh",
                task_path,
            ],
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(task_path)
    _check("swe_unpack.sh", result)

    # After the unpack: the patch applies to the checkout it just copied in.
    if request.task_input.get("test_patch_applied"):
        apply_test_patch(request.task_input)


def apply_test_patch(task_input: dict) -> None:
    """Put the task's test files in their post-patch state, before the agent starts.

    Only when the harness asked for it (``test_patch_applied``), because seeing the tests
    changes the task the agent faces. The script is ``test_patch_script`` from the dataset
    parquet (see ``preprocess.py``); the eval script resets and re-applies the same patch,
    so the agent still cannot pass a task by editing its tests.
    """
    script = task_input.get("test_patch_script")
    if not script:
        raise RuntimeError(
            "test_patch_applied is set but the task carries no test_patch_script: "
            "rebuild the dataset parquet with preprocess.py"
        )

    # A temp file rather than one inside the repo, so the agent never sees it.
    fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="swe_test_patch_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)
        result = subprocess.run(["/bin/bash", script_path], capture_output=True, text=True)
    finally:
        os.unlink(script_path)
    _check("test_patch_script", result)


def _check(what: str, result: subprocess.CompletedProcess) -> None:
    """Raise with both streams if ``result`` failed -- setup output is not captured anywhere else."""
    if result.returncode != 0:
        raise RuntimeError(
            f"{what} failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
