"""Unit tests for the noop agent backend.

The backend is a no-LLM baseline -- it only diffs the untouched repo and grades it
-- so what's worth covering is that it never modifies the repo (no ``git apply``),
returns the eval verdict as the reward, swallows failures into ``exception``, and
is reachable through the app's backend dispatch.
"""

import subprocess
import unittest
from unittest import mock

from swe_agent_server import noop_agent

from agentcore_rl_toolkit.rollout_session.wire import RolloutStartRequest


def _request() -> RolloutStartRequest:
    return RolloutStartRequest(
        rollout_id="r",
        task_input=dict(agent="noop", repo_path="/testbed", base_commit="HEAD"),
    )


class RolloutTest(unittest.TestCase):
    def test_grades_untouched_repo(self):
        with (
            mock.patch.object(noop_agent.subprocess, "check_output", return_value=b"") as check_output,
            mock.patch.object(
                noop_agent,
                "run_evaluation",
                return_value={"resolved": False, "eval_latency_s": 3.5},
            ),
        ):
            dump = noop_agent.rollout(_request())

        self.assertIsNone(dump.exception)
        self.assertEqual(dump.reward, 0.0)
        self.assertEqual(dump.metrics["eval_latency_s"], 3.5)
        assert dump.task_output is not None
        self.assertEqual(dump.task_output["git_diff"], "")
        # The only subprocess the backend runs is a read-only diff: no patch is
        # applied, which is the whole point of the baseline.
        (argv,), _ = check_output.call_args
        self.assertEqual(argv[:2], ["git", "--no-pager"])
        self.assertIn("diff", argv)

    def test_reports_resolved_base_commit_as_reward_one(self):
        # A task that grades as resolved with no changes is a broken task, not an
        # error: the backend reports it as reward 1.0 so the run surfaces it.
        with (
            mock.patch.object(noop_agent.subprocess, "check_output", return_value=b""),
            mock.patch.object(noop_agent, "run_evaluation", return_value={"resolved": True}),
        ):
            dump = noop_agent.rollout(_request())

        self.assertEqual(dump.reward, 1.0)

    def test_failure_is_returned_not_raised(self):
        with mock.patch.object(
            noop_agent.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            dump = noop_agent.rollout(_request())

        self.assertIsNone(dump.reward)
        assert dump.exception is not None
        self.assertIn("CalledProcessError", dump.exception)


class BackendDispatchTest(unittest.TestCase):
    def test_dispatches_to_noop_backend(self):
        from swe_agent_server import app

        sentinel = object()
        with mock.patch.object(noop_agent, "rollout", return_value=sentinel) as noop_rollout:
            result = app.run_rollout(_request())

        noop_rollout.assert_called_once()
        self.assertIs(result, sentinel)

    def test_unknown_agent_raises(self):
        from swe_agent_server import app

        with self.assertRaises(ValueError):
            app.run_rollout(RolloutStartRequest(rollout_id="r", task_input=dict(agent="nope")))


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2)
