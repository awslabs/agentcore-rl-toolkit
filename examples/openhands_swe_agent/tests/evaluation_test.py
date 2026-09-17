#!/usr/bin/env python
"""Unit tests for the two places the setup stage's test-patch commit has to be accounted
for: undoing it before the harness's own eval script runs, and keeping it out of the patch
the rollout records as the agent's work.

Asserted against a real ``git`` repo, because every claim here is a claim about what git
does -- what the revert restores, what it leaves behind for ``model_patch``, and that the
harness's script, which this repo deliberately does not modify, can then apply the test
patch the way it expects to. The setup state is built with plain git rather than by running
``preprocess.make_test_patch_script``: the contract between the stages is the ref and the
commit under it, and nothing here should depend on how that commit came to be.
"""

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
        # Every task image carries one of these on top of the base commit -- the harness
        # appends `git commit --allow-empty -am SWE-bench` to its own install script -- so
        # HEAD is never the base commit, in either arm.
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
        """What the setup stage leaves behind: the graded tests patched and committed, under
        the ref grading reads. Also records the diff of that commit, which is the test patch
        the harness's eval script carries in a heredoc."""
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
        """The agent's turn: the source change it is graded on, and edits to the graded tests
        -- which agents do make, believing they are cleaning up a diff of their own."""
        self.write(SOURCE, "def add(a, b): return a + b\n")
        self.write(EXISTING_TEST, "def test_old(): pass\ndef test_added(): assert False\n")

    def revert(self) -> str | None:
        return evaluation.revert_test_patch_commit(str(self.repo))

    def test_returns_a_modified_test_file_to_its_base_commit_form(self):
        self.apply_test_patch()
        self.agent_works()

        self.assertEqual(self.revert(), self.git("rev-parse", evaluation.TEST_PATCH_REF))

        # Both what the setup commit added to it and what the agent then did to it are gone,
        # which is what makes the harness's own apply of the same patch possible again.
        self.assertEqual((self.repo / EXISTING_TEST).read_text(), BASE_TEST)

    def test_deletes_a_test_file_the_setup_commit_created(self):
        self.apply_test_patch()
        self.agent_works()

        self.revert()

        self.assertFalse((self.repo / ADDED_TEST).exists())

    def test_restores_a_test_file_the_test_patch_deleted(self):
        # The other direction, and the reason the paths come from git rather than a regex on
        # the diff: one SWE-Gym task's test patch deletes a test file. The agent is free to
        # have put something back in its place.
        self.apply_test_patch(remove_a_test=True)
        self.write(REMOVED_TEST, "def test_stale(): assert False\n")

        self.revert()

        self.assertEqual((self.repo / REMOVED_TEST).read_text(), "def test_stale(): pass\n")

    def test_leaves_the_agents_source_change_as_the_only_recorded_diff(self):
        self.apply_test_patch()
        self.agent_works()

        self.revert()

        # ``run_evaluation`` reads ``model_patch`` from exactly this command, and the staged
        # revert cancels out in it, so the graded patch is the agent's work and nothing else.
        recorded = self.git("diff")
        self.assertIn(SOURCE, recorded)
        self.assertNotIn(EXISTING_TEST, recorded)
        self.assertNotIn(ADDED_TEST, recorded)

    def test_survives_an_agent_that_committed_over_the_graded_tests(self):
        # Some agents commit their work. Restoring the graded paths out of the setup commit
        # rather than out of the index is what keeps that from becoming a conflicted revert
        # in the middle of grading: by then the index holds the agent's version of the test
        # file, and git will not quietly overwrite it.
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

        # Nothing here rewinds the checkout: the image's install commit is still HEAD's
        # ancestor, HEAD is still on its branch, and the files it installed are untouched.
        self.assertEqual((self.repo / "installed.py").read_text(), "built = True\n")
        self.assertNotEqual(self.git("branch", "--show-current"), "")

    def test_does_nothing_when_the_setup_stage_made_no_commit(self):
        # The baseline arm: no test patch was applied, so there is no ref and grading has to
        # take the worktree exactly as the agent left it.
        self.agent_works()
        before = self.git("status", "--porcelain")

        self.assertIsNone(self.revert())

        self.assertEqual(self.git("status", "--porcelain"), before)
        self.assertEqual(self.git("rev-parse", "HEAD"), self.install_commit)

    def test_raises_when_the_commit_cannot_be_reverted(self):
        # Grading must not go on to score a worktree it failed to restore: the tests the
        # eval script would then run are not the graded ones. A merge commit is the cheapest
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

        # The graded tests were already there when the agent started, so they are not the
        # agent's work and do not belong in the dump that trains on it.
        self.assertNotIn(ADDED_TEST, recorded)
        self.assertNotIn("+def test_added(): pass", recorded)
        self.assertIn(SOURCE, recorded)
        # What the agent did to a graded test file is still its own doing, and is recorded.
        self.assertIn("assert False", recorded)

    def test_the_recorded_patch_falls_back_to_the_base_commit(self):
        # The baseline arm: no setup commit, so the tree the agent was handed is the base
        # commit's, and that is what the diff is against.
        self.agent_works()

        self.assertIn(SOURCE, self.capture())

    # --- the harness's script, unmodified ------------------------------------------

    def harness_eval_script(self) -> str:
        """The two commands of the harness's eval script that touch the test files, in its
        own order and with its own header (no ``set -e``), plus a stand-in for the test
        command: what the grader parses is whatever the test files hold at that point."""
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

        # The patch applied, so the graded tests are the ones the task is scored on -- both
        # the nodes it adds to an existing file and the file it creates.
        self.assertIn("test_added", result.stdout)
        self.assertIn("test_brand_new", result.stdout)
        # And the agent's source change survived the whole sequence, so it is what decides.
        self.assertEqual((self.repo / SOURCE).read_text(), "def add(a, b): return a + b\n")

    def test_the_harness_script_cannot_grade_without_the_revert(self):
        # Why the revert exists. ``git apply`` refuses a file that already exists and is
        # atomic about it, so with the setup commit still in place no hunk lands at all and
        # the tests that run are the base commit's -- a reward that ignores the agent.
        self.apply_test_patch()
        self.agent_works()

        result = self.run_eval_script()

        self.assertIn("already exists in working directory", result.stderr + result.stdout)
        self.assertNotIn("test_added", result.stdout)


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2)
