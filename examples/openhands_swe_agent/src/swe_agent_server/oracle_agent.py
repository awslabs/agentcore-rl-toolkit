"""Gold-patch baseline backend: applies the reference diff, expecting reward 1.0."""

import logging
import subprocess

from swe_agent_server.evaluation import run_evaluation
from swe_agent_server.utils import clean_metrics, exc_to_full_string

from agentcore_rl_toolkit.rollout_session.wire import (
    RolloutDumpResponse,
    RolloutStartRequest,
)


def rollout(request: RolloutStartRequest) -> RolloutDumpResponse:
    """Apply ``task_input["patch"]``, then grade the result.

    A reward other than 1.0 means the eval harness is broken. Never raises --
    failures are returned in ``exception``.
    """
    exception = None
    git_diff = None
    eval_report = None

    try:
        repo_path = request.task_input["repo_path"]
        patch = request.task_input["patch"]

        # ``-p1`` matches the repo-relative paths git produces.
        subprocess.run(
            ["git", "apply", "--verbose", "-p1", "-"],
            cwd=repo_path,
            input=patch.encode(),
            check=True,
            capture_output=True,
        )

        git_diff = subprocess.check_output(
            [
                "git",
                "--no-pager",
                "diff",
                "--no-color",
                request.task_input["base_commit"],
            ],
            cwd=repo_path,
        ).decode()

        eval_report = run_evaluation(request.task_input)

    except Exception as error:
        logging.error("Exception during oracle rollout", exc_info=error)
        exception = exc_to_full_string(error)

    return RolloutDumpResponse(
        metrics=clean_metrics(
            {
                "eval_latency_s": (eval_report.get("eval_latency_s") if eval_report is not None else None),
            }
        ),
        task_output=dict(
            git_diff=git_diff,
            eval_report=eval_report,
        ),
        reward=float(eval_report["resolved"]) if eval_report is not None else None,
        exception=exception,
    )
