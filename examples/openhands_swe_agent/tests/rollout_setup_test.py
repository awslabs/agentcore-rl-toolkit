"""Tests for run_setup: unpacking the task image and the optional test-patch step."""

import subprocess
import unittest
from unittest import mock

from swe_agent_server import rollout

from agentcore_rl_toolkit.rollout_session.wire import RolloutSetupRequest


def _request(**task_input) -> RolloutSetupRequest:
    return RolloutSetupRequest(
        task_input=dict(
            agent="openhands",
            docker_image_namespace="acct.dkr.ecr.us-west-2.amazonaws.com/cache",
            docker_image_uri="swebench/sweb.eval.x86_64.some_1776_task:latest",
            **task_input,
        )
    )


def _ok(*_args, **_kwargs) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


class RunSetupTest(unittest.TestCase):
    def test_unpacks_the_task_image(self):
        with mock.patch.object(rollout.subprocess, "run", side_effect=_ok) as run:
            rollout.run_setup(_request())

        (argv,), _ = run.call_args
        self.assertEqual(argv[0], "/agent/swe_unpack.sh")

    def test_unpack_failure_reports_both_streams(self):
        failed = subprocess.CompletedProcess(args=[], returncode=2, stdout="out", stderr="boom")
        with mock.patch.object(rollout.subprocess, "run", return_value=failed):
            with self.assertRaises(RuntimeError) as caught:
                rollout.run_setup(_request())

        self.assertIn("swe_unpack.sh failed with exit code 2", str(caught.exception))
        self.assertIn("boom", str(caught.exception))

    def test_test_patch_is_not_applied_by_default(self):
        # test_patch_script rides in every row; only the flag decides whether it runs.
        with (
            mock.patch.object(rollout.subprocess, "run", side_effect=_ok),
            mock.patch.object(rollout, "apply_test_patch") as apply_test_patch,
        ):
            rollout.run_setup(_request(test_patch_script="#!/bin/bash\ntrue\n"))

        apply_test_patch.assert_not_called()

    def test_test_patch_applied_runs_the_script_after_the_unpack(self):
        with mock.patch.object(rollout.subprocess, "run", side_effect=_ok) as run:
            rollout.run_setup(_request(test_patch_applied=True, test_patch_script="#!/bin/bash\ntrue\n"))

        unpack, patch = [call.args[0] for call in run.call_args_list]
        self.assertEqual(unpack[0], "/agent/swe_unpack.sh")
        self.assertEqual(patch[0], "/bin/bash")


class ApplyTestPatchTest(unittest.TestCase):
    SCRIPT = "#!/bin/bash\necho applied\n"

    def test_runs_the_task_script_and_removes_it(self):
        written = {}

        def record(argv, **_kwargs):
            with open(argv[1]) as f:
                written["path"], written["script"] = argv[1], f.read()
            return _ok()

        with mock.patch.object(rollout.subprocess, "run", side_effect=record):
            rollout.apply_test_patch(dict(test_patch_applied=True, test_patch_script=self.SCRIPT))

        self.assertEqual(written["script"], self.SCRIPT)
        # Nothing of the patch is left behind for the agent to find.
        with self.assertRaises(FileNotFoundError):
            open(written["path"])

    def test_a_script_that_fails_to_apply_fails_the_setup(self):
        # A patch that fails to apply must fail setup, not leave the agent on an unpatched checkout.
        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="patch does not apply")
        with mock.patch.object(rollout.subprocess, "run", return_value=failed):
            with self.assertRaises(RuntimeError) as caught:
                rollout.apply_test_patch(dict(test_patch_script=self.SCRIPT))

        self.assertIn("test_patch_script failed with exit code 1", str(caught.exception))
        self.assertIn("patch does not apply", str(caught.exception))

    def test_missing_script_is_a_stale_parquet_not_a_silent_skip(self):
        with mock.patch.object(rollout.subprocess, "run", side_effect=_ok) as run:
            with self.assertRaises(RuntimeError) as caught:
                rollout.apply_test_patch(dict(test_patch_applied=True))

        run.assert_not_called()
        self.assertIn("preprocess.py", str(caught.exception))


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2)
