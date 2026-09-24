"""Task setup and dispatch of a rollout to the requested agent backend."""

import json
import os
import subprocess
import tempfile

from swe_agent_server.observability import configure_openhands_tracing

from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse


def run_rollout(task_input: dict) -> RolloutDumpResponse:
    """Dispatch a rollout to the requested agent backend.

    Backends are imported lazily so a deployment only needs the SDK for the ones it
    actually runs.
    """
    match task_input["agent"]:
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
    return rollout(task_input)


def run_setup(task_input: dict):
    # The unpack script reads the fields it needs out of this JSON file (e.g. via jq).
    fd, task_path = tempfile.mkstemp(suffix=".json", prefix="swe_task_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(task_input, f)
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

    # Must run after unpack: it patches the checkout unpack just created.
    if task_input.get("test_patch_applied"):
        apply_test_patch(task_input)


def apply_test_patch(task_input: dict) -> None:
    """Apply the task's test patch before the agent starts, if the harness asked for it.

    The eval script re-applies the same patch regardless, so this can't be gamed by editing tests.
    """
    script = task_input.get("test_patch_script")
    if not script:
        raise RuntimeError(
            "test_patch_applied is set but the task carries no test_patch_script: "
            "rebuild the dataset parquet with preprocess.py"
        )

    # Outside the repo, so the agent never sees the patch script.
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
