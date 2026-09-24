#!/usr/bin/env python
"""Tests for reverting the setup stage's test-patch commit before grading, and for
excluding it from the recorded rollout patch. Asserted against a real ``git`` repo."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from swe_agent_server import evaluation

SOURCE = "calc.py"
EXISTING_TEST = "tests/test_math.py"
ADDED_TEST = "tests/test_new.py"
REMOVED_TEST = "tests/test_stale.py"

BASE_TEST = "def test_old(): pass\n"
PATCHED_TEST = "def test_old(): pass\ndef test_added(): pass\n"


class RevertTestPatchCommitTest(unittest.TestCase):
    """A checkout in the state the setup stage leaves it in, and what grading has to undo."""

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.repo)])

        self.git("init", "-q", ".")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "t")
        self.write(SOURCE, "def add(a, b): return 0\n")
        self.write(EXISTING_TEST, BASE_TEST)
        self.write(REMOVED_TEST, "def test_stale(): pass\n")
        self.git("add", "-A")
        self.git("commit", "-qm", "base")
        self.base_commit = self.git("rev-parse", "HEAD")
        # Harness's own install script commits on top of base_commit, so HEAD is never base_commit.
        self.write("installed.py", "built = True\n")
        self.git("add", "-A")
        self.git("commit", "-qm", "SWE-bench install")
        self.install_commit = self.git("rev-parse", "HEAD")

    def git(self, *args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True).strip()

    def write(self, path: str, text: str):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def apply_test_patch(self, remove_a_test=False):
        """Applies and commits the test patch under TEST_PATCH_REF, and records the diff as
        the eval script's heredoc patch."""
        self.write(EXISTING_TEST, PATCHED_TEST)
        self.write(ADDED_TEST, "def test_brand_new(): pass\n")
        paths = [EXISTING_TEST, ADDED_TEST]
        if remove_a_test:
            (self.repo / REMOVED_TEST).unlink()
            paths.append(REMOVED_TEST)
        self.git("add", "--", *paths)
        self.git("commit", "-qm", "Apply test patch", "--", *paths)
        self.git("update-ref", evaluation.TEST_PATCH_REF, "HEAD")
        self.test_patch = self.git("diff", self.install_commit, "HEAD")

    def agent_works(self):
        """The agent's source fix, plus an edit to a graded test file (agents do this)."""
        self.write(SOURCE, "def add(a, b): return a + b\n")
        self.write(EXISTING_TEST, "def test_old(): pass\ndef test_added(): assert False\n")

    def revert(self) -> str | None:
        return evaluation.revert_test_patch_commit(str(self.repo))

    def test_returns_a_modified_test_file_to_its_base_commit_form(self):
        self.apply_test_patch()
        self.agent_works()

        self.assertEqual(self.revert(), self.git("rev-parse", evaluation.TEST_PATCH_REF))

        # Setup commit's and agent's edits are both undone.
        self.assertEqual((self.repo / EXISTING_TEST).read_text(), BASE_TEST)

    def test_deletes_a_test_file_the_setup_commit_created(self):
        self.apply_test_patch()
        self.agent_works()

        self.revert()

        self.assertFalse((self.repo / ADDED_TEST).exists())

    def test_restores_a_test_file_the_test_patch_deleted(self):
        # Paths come from git, not a diff regex, so a deleted test file round-trips too.
        self.apply_test_patch(remove_a_test=True)
        self.write(REMOVED_TEST, "def test_stale(): assert False\n")

        self.revert()

        self.assertEqual((self.repo / REMOVED_TEST).read_text(), "def test_stale(): pass\n")

    def test_leaves_the_agents_source_change_as_the_only_recorded_diff(self):
        self.apply_test_patch()
        self.agent_works()

        self.revert()

        # model_patch is read from this diff; the staged revert cancels out here too.
        recorded = self.git("diff")
        self.assertIn(SOURCE, recorded)
        self.assertNotIn(EXISTING_TEST, recorded)
        self.assertNotIn(ADDED_TEST, recorded)

    def test_survives_an_agent_that_committed_over_the_graded_tests(self):
        # Restoring from the setup commit (not the index) avoids a conflicted revert
        # when the agent has committed its own version of the test file.
        self.apply_test_patch()
        self.agent_works()
        self.git("add", "-A")
        self.git("commit", "-qm", "agent work")

        self.revert()

        self.assertEqual((self.repo / EXISTING_TEST).read_text(), BASE_TEST)
        self.assertFalse((self.repo / ADDED_TEST).exists())

    def test_leaves_the_images_own_history_alone(self):
        self.apply_test_patch()
        self.agent_works()

        self.revert()

        # Revert doesn't rewind the checkout: install commit, branch, and installed files stay put.
        self.assertEqual((self.repo / "installed.py").read_text(), "built = True\n")
        self.assertNotEqual(self.git("branch", "--show-current"), "")

    def test_does_nothing_when_the_setup_stage_made_no_commit(self):
        # Baseline arm: no ref, so revert is a no-op.
        self.agent_works()
        before = self.git("status", "--porcelain")

        self.assertIsNone(self.revert())

        self.assertEqual(self.git("status", "--porcelain"), before)
        self.assertEqual(self.git("rev-parse", "HEAD"), self.install_commit)

    def test_raises_when_the_commit_cannot_be_reverted(self):
        # A failed restore must not proceed to grading. A merge commit is the cheapest
        # thing git refuses to revert unaided.
        self.git("checkout", "-q", "-b", "side", self.base_commit)
        self.write("side.py", "x = 1\n")
        self.git("add", "-A")
        self.git("commit", "-qm", "side")
        self.git("checkout", "-q", "-")
        self.git("merge", "--no-ff", "-q", "-m", "merge", "side")
        self.git("update-ref", evaluation.TEST_PATCH_REF, "HEAD")

        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.revert()

        # git's own explanation travels with the exception, so the failure is diagnosable.
        self.assertIn("is a merge but no -m option", caught.exception.output)

    # --- the recorded rollout patch -------------------------------------------------

    def capture(self) -> str:
        return evaluation.capture_git_diff({"repo_path": str(self.repo), "base_commit": self.base_commit})

    def test_the_recorded_patch_is_measured_against_the_tree_the_agent_was_handed(self):
        self.apply_test_patch()
        self.agent_works()

        recorded = self.capture()

        # Pre-existing graded tests aren't the agent's work, so they're excluded.
        self.assertNotIn(ADDED_TEST, recorded)
        self.assertNotIn("+def test_added(): pass", recorded)
        self.assertIn(SOURCE, recorded)
        # But the agent's own edit to a graded test file is still recorded.
        self.assertIn("assert False", recorded)

    def test_the_recorded_patch_falls_back_to_the_base_commit(self):
        # Baseline arm: no setup commit, so diff falls back to base_commit.
        self.agent_works()

        self.assertIn(SOURCE, self.capture())

    # --- the harness's script, unmodified ------------------------------------------

    def harness_eval_script(self) -> str:
        """The harness's own eval-script commands (unmodified), for testing revert against
        real usage."""
        return "\n".join(
            [
                "#!/bin/bash",
                "set -uxo pipefail",
                f"cd {self.repo}",
                f"git checkout {self.base_commit} {EXISTING_TEST}",
                f"git apply -v - <<'EOF_114329324912'\n{self.test_patch}\nEOF_114329324912",
                f"cat {EXISTING_TEST} {ADDED_TEST}",
            ]
        )

    def run_eval_script(self) -> subprocess.CompletedProcess:
        path = self.repo.parent / "eval.sh"
        path.write_text(self.harness_eval_script())
        self.addCleanup(path.unlink, missing_ok=True)
        return subprocess.run(["/bin/bash", str(path)], capture_output=True, text=True)

    def test_the_harness_script_grades_the_patched_tests_after_the_revert(self):
        self.apply_test_patch()
        self.agent_works()
        self.revert()

        result = self.run_eval_script()

        # Both the modified and newly-added test files got graded.
        self.assertIn("test_added", result.stdout)
        self.assertIn("test_brand_new", result.stdout)
        # Agent's source change survives the whole sequence.
        self.assertEqual((self.repo / SOURCE).read_text(), "def add(a, b): return a + b\n")

    def test_the_harness_script_cannot_grade_without_the_revert(self):
        # Without revert, git apply refuses (file already exists), so base_commit's tests run instead.
        self.apply_test_patch()
        self.agent_works()

        result = self.run_eval_script()

        self.assertIn("already exists in working directory", result.stderr + result.stdout)
        self.assertNotIn("test_added", result.stdout)


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2)
