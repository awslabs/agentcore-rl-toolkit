"""Empty-patch baseline backend: grades the untouched repo, expecting reward 0.0."""

import logging
import subprocess

from swe_agent_server.evaluation import run_evaluation
from swe_agent_server.utils import clean_metrics, exc_to_full_string

from agentcore_rl_toolkit.rollout_session.wire import (
    RolloutDumpResponse,
    RolloutStartRequest,
)


def rollout(request: RolloutStartRequest) -> RolloutDumpResponse:
    """Change nothing, then grade the untouched repo. Never raises -- see ``exception``.

    A reward other than 0.0 means the task grades as resolved without any work
    (already-fixed repo, broken eval script, grader misparse).
    """
    exception = None
    git_diff = None
    eval_report = None

    try:
        # Should come back empty; a non-empty diff means a dirty testbed.
        git_diff = subprocess.check_output(
            [
                "git",
                "--no-pager",
                "diff",
                "--no-color",
                request.task_input["base_commit"],
            ],
            cwd=request.task_input["repo_path"],
        ).decode()

        eval_report = run_evaluation(request.task_input)

    except Exception as error:
        logging.error("Exception during noop rollout", exc_info=error)
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
