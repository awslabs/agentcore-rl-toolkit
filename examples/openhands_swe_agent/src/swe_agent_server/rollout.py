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

    The backend modules are imported lazily so a deployment only needs the SDK for
    the backend(s) it actually runs (OpenHands and Strands are separate optional
    dependencies).
    """
    match payload.task_input["agent"]:
        case "openhands":
            # Before the import, not after: importing the OpenHands agent initialises
            # its tracing layer, and both of the things that can be done about that
            # have to be settled first (swe_agent_server.observability).
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
    # Write the whole task to a temporary JSON file and hand its path to the
    # unpack script, which extracts whatever fields it needs (e.g. via jq).
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
    if result.returncode != 0:
        raise RuntimeError(
            f"swe_unpack.sh failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
