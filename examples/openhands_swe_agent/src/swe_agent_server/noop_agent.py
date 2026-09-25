"""Empty-patch baseline backend: grades the untouched repo, expecting reward 0.0."""

import logging

from swe_agent_server.evaluation import capture_git_diff, run_evaluation
from swe_agent_server.utils import clean_metrics, exc_to_full_string

from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse


def rollout(task_input: dict) -> RolloutDumpResponse:
    """Change nothing, then grade the untouched repo. Never raises -- see ``exception``.

    A reward other than 0.0 means the task grades as resolved without any work
    (already-fixed repo, broken eval script, grader misparse).
    """
    exception = None
    git_diff = None
    eval_report = None

    try:
        # Should come back empty; a non-empty diff means a dirty testbed.
        git_diff = capture_git_diff(task_input)

        eval_report = run_evaluation(task_input)

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
