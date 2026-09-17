#!/usr/bin/env python
"""Unit tests for the dataset conversion: the script the setup stage runs on a task image.

Asserted against a real ``git`` repo rather than against the script's text, because what
matters is that it leaves the test files patched and the worktree clean -- whatever shape the
patch has, and whatever the image left lying around -- and that it leaves the commit where
the grading stage can find it. Undoing that commit is ``evaluation_test.py``.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest
from swe_agent_server import evaluation

# The conversion pulls in the dataset stack; only this module's import needs it.
pytest.importorskip("datasets")

# The recipe directory is on ``sys.path`` via its ``conftest.py``.
import preprocess  # noqa: E402

EXISTING_TEST = "tests/test_math.py"
ADDED_TEST = "tests/test_new.py"

TEST_PATCH = f"""diff --git a/{EXISTING_TEST} b/{EXISTING_TEST}
--- a/{EXISTING_TEST}
+++ b/{EXISTING_TEST}
@@ -1 +1,2 @@
 def test_old(): pass
+def test_added_by_patch(): pass
diff --git a/{ADDED_TEST} b/{ADDED_TEST}
new file mode 100644
--- /dev/null
+++ b/{ADDED_TEST}
@@ -0,0 +1 @@
+def test_brand_new(): pass
"""

_SPLIT_AT = f"diff --git a/{ADDED_TEST}"
# The half that touches nothing that exists at the base commit -- around 2% of SWE-Gym.
ADD_ONLY_PATCH = _SPLIT_AT + TEST_PATCH.split(_SPLIT_AT)[1]

# A file the patch creates *empty*, which is what a new test package's ``__init__.py`` is.
# It has no hunk and so no ``+++ b/`` line: the diff header is the only place its path
# appears at all.
EMPTY_FILE = "tests/pkg/__init__.py"
EMPTY_FILE_PATCH = TEST_PATCH + f"diff --git a/{EMPTY_FILE} b/{EMPTY_FILE}\nnew file mode 100644\n"

# The same shape from the other direction: a rename git recorded as a rename, so neither
# side of it has a body either.
RENAMED_TEST = "tests/test_moved.py"
RENAME_PATCH = (
    f"diff --git a/{EXISTING_TEST} b/{RENAMED_TEST}\n"
    "similarity index 100%\n"
    f"rename from {EXISTING_TEST}\n"
    f"rename to {RENAMED_TEST}\n"
)


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def git_out(*args, cwd) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd).decode().strip()


class RepoTest(unittest.TestCase):
    """A checkout standing in for the task image's ``/testbed``, and how to run a script in it."""

    def setUp(self):
        # The checkout sits in a directory of its own, so the script and the scratch files it
        # works through can go beside it and be cleaned up with it.
        root = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(root)])
        self.repo = root / "testbed"
        self.repo.mkdir()

        git("init", "-q", cwd=self.repo)
        git("config", "user.email", "t@example.com", cwd=self.repo)
        git("config", "user.name", "t", cwd=self.repo)
        (self.repo / "tests").mkdir()
        (self.repo / EXISTING_TEST).write_text("def test_old(): pass\n")
        git("add", "-A", cwd=self.repo)
        git("commit", "-qm", "base", cwd=self.repo)
        self.base_commit = git_out("rev-parse", "HEAD", cwd=self.repo)

    # Lines that only make sense inside the task image: the conda env it was built with,
    # and a global git setting a test must not write to the developer's own config.
    IMAGE_ONLY = ("miniconda3", "conda activate", "git config --global")

    def runnable(self, script: str) -> str:
        return "\n".join(line for line in script.splitlines() if not any(s in line for s in self.IMAGE_ONLY))

    def run_script(self, script: str, name="script.sh") -> subprocess.CompletedProcess:
        path = self.repo.parent / name
        path.write_text(script)
        self.addCleanup(path.unlink, missing_ok=True)
        return subprocess.run(["/bin/bash", str(path)], capture_output=True, text=True)

    def setup_script_for(self, test_patch=TEST_PATCH) -> str:
        """The setup script, retargeted from the image's ``/testbed`` at this temp checkout.

        Its scratch files move next to the checkout too, so a test that means to fail leaves
        nothing behind in the developer's own ``/tmp``.
        """
        scratch = self.repo.parent
        with (
            mock.patch.object(preprocess, "REPO_PATH", str(self.repo)),
            mock.patch.object(preprocess, "PATCH_FILE", str(scratch / "test-patch.diff")),
            mock.patch.object(preprocess, "SCRATCH_INDEX", str(scratch / "index")),
        ):
            script = preprocess.make_test_patch_script({"base_commit": self.base_commit, "test_patch": test_patch})
        return self.runnable(script)


class MakeTestPatchScriptTest(RepoTest):
    """The script the setup stage runs when the harness asks for ``test_patch_applied``."""

    def script_for(self, test_patch=TEST_PATCH) -> str:
        return self.setup_script_for(test_patch)

    def test_applies_the_patch_to_modified_and_added_test_files(self):
        result = self.run_script(self.script_for())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.repo / EXISTING_TEST).read_text(),
            "def test_old(): pass\ndef test_added_by_patch(): pass\n",
        )
        self.assertIn("test_brand_new", (self.repo / ADDED_TEST).read_text())

    def test_refuses_an_image_that_already_modified_one_of_the_test_files(self):
        # There is no reset step, so this is the one thing the image can do that the script
        # cannot absorb -- and ``git apply --index`` is what catches it, since it requires
        # the patch to apply to the index and the worktree alike. A failed rollout is the
        # right outcome: the alternative is an agent working against tests the grader will
        # replace underneath it. No SWE-Gym image observed does this.
        (self.repo / EXISTING_TEST).write_text("def test_old(): assert False\n")

        result = self.run_script(self.script_for())

        self.assertNotEqual(result.returncode, 0)
        # Atomically refused: not the modified file half-patched, and not the other path of
        # the same patch created either.
        self.assertEqual((self.repo / EXISTING_TEST).read_text(), "def test_old(): assert False\n")
        self.assertFalse((self.repo / ADDED_TEST).exists())
        self.assertEqual(git_out("log", "--format=%s", cwd=self.repo), "base")
        self.assertEqual(git_out("for-each-ref", "--format=%(refname)", preprocess.TEST_PATCH_REF, cwd=self.repo), "")

    def test_commits_the_patch_so_the_agent_starts_on_a_clean_worktree(self):
        # The whole point of the commit: nothing for the agent to mistake for its own edit,
        # and nothing of ours in the ``git diff`` the grade report is built from.
        result = self.run_script(self.script_for())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git_out("status", "--porcelain", cwd=self.repo), "")
        self.assertEqual(git_out("diff", cwd=self.repo), "")
        self.assertEqual(
            sorted(git_out("show", "--name-only", "--format=", cwd=self.repo).splitlines()),
            [EXISTING_TEST, ADDED_TEST],
        )

    def test_leaves_the_images_own_dirt_out_of_the_commit(self):
        # Why the commit's tree is built in an index of its own: an image that ships a dirty
        # or partly staged worktree must not have any of it folded into the test-patch
        # commit, and must get it back untouched afterwards.
        (self.repo / "unrelated.py").write_text("dirt\n")
        (self.repo / "staged.py").write_text("also dirt\n")
        git("add", "--", "staged.py", cwd=self.repo)

        result = self.run_script(self.script_for())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            sorted(git_out("show", "--name-only", "--format=", cwd=self.repo).splitlines()),
            [EXISTING_TEST, ADDED_TEST],
        )
        self.assertEqual(
            sorted(git_out("status", "--porcelain", cwd=self.repo).splitlines()),
            ["?? unrelated.py", "A  staged.py"],
        )

    def test_builds_on_whatever_the_image_has_committed_above_the_base_commit(self):
        # The script never names ``base_commit``: it commits on top of HEAD and hands the
        # commit itself to grading. So an image that has committed something of its own above
        # the base commit -- an install step, as upstream SWE-bench's images do and SWE-Gym's
        # do not -- keeps it, on its own branch, with or without a path to patch in it. An
        # add-only patch is around 2% of SWE-Gym, and was the case a reset step had to be
        # guarded against.
        (self.repo / "installed.py").write_text("built = True\n")
        git("add", "-A", cwd=self.repo)
        git("commit", "-qm", "image install", cwd=self.repo)
        branch = git_out("branch", "--show-current", cwd=self.repo)

        result = self.run_script(self.script_for(ADD_ONLY_PATCH))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.repo / "installed.py").read_text(), "built = True\n")
        self.assertEqual(git_out("branch", "--show-current", cwd=self.repo), branch)
        self.assertEqual(
            git_out("log", "--format=%s", cwd=self.repo).splitlines(),
            ["Apply test patch", "image install", "base"],
        )
        self.assertEqual(git_out("status", "--porcelain", cwd=self.repo), "")

    def test_commits_a_file_the_patch_creates_empty(self):
        # The path of an empty created file is in the diff header and nowhere else, and a
        # created file left out of the commit is left untracked -- so grading's revert does
        # not remove it, and the harness's own apply then dies on a file that already
        # exists, atomically, taking the whole test patch with it.
        result = self.run_script(self.script_for(EMPTY_FILE_PATCH))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git_out("status", "--porcelain", cwd=self.repo), "")
        self.assertIn(EMPTY_FILE, git_out("show", "--name-only", "--format=", cwd=self.repo).splitlines())

    def test_commits_both_sides_of_a_rename(self):
        # A pure rename has no body on either side, and the old path has to be in the commit
        # too or the revert cannot put it back.
        result = self.run_script(self.script_for(RENAME_PATCH))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git_out("status", "--porcelain", cwd=self.repo), "")
        # ``--no-renames`` because git's own rename detection would show the pair as the one
        # destination path, and what is being asserted is that both are in the commit.
        self.assertEqual(
            sorted(git_out("show", "--name-only", "--no-renames", "--format=", cwd=self.repo).splitlines()),
            sorted([EXISTING_TEST, RENAMED_TEST]),
        )

    def test_a_patch_that_does_not_apply_exits_nonzero(self):
        # The setup stage turns this into a failed rollout instead of handing the agent a
        # checkout whose tests are not the graded ones.
        conflicting = TEST_PATCH.replace(" def test_old(): pass", " def test_renamed(): pass")

        result = self.run_script(self.script_for(conflicting))

        # ``git apply``'s own status, under ``set -e`` -- and no commit and no ref, so nothing
        # half-applied is passed off as the graded tests.
        self.assertEqual(result.returncode, 1)
        self.assertEqual(git_out("log", "--format=%s", cwd=self.repo), "base")
        self.assertEqual(git_out("for-each-ref", "--format=%(refname)", preprocess.TEST_PATCH_REF, cwd=self.repo), "")

    def test_commits_a_created_test_file_the_repo_gitignores(self):
        # ``git add`` would have refused this path, and a created file left out of the commit
        # is left untracked, which defeats grading's revert. Neither ``git apply`` nor
        # ``read-tree``/``write-tree`` consults ``.gitignore``, which is the same answer the
        # harness's own eval script gives the file.
        (self.repo / ".gitignore").write_text(f"{ADDED_TEST}\n")
        git("add", "-A", cwd=self.repo)
        git("commit", "-qm", "ignore", cwd=self.repo)

        result = self.run_script(self.script_for())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git_out("status", "--porcelain", cwd=self.repo), "")
        self.assertIn(ADDED_TEST, git_out("show", "--name-only", "--format=", cwd=self.repo).splitlines())

    def test_leaves_the_commit_where_grading_can_find_it(self):
        # The handoff to ``evaluation.revert_test_patch_commit``: a ref, so the grading stage
        # neither has to guess which commit is ours nor read a marker out of the worktree.
        result = self.run_script(self.script_for())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            git_out("rev-parse", preprocess.TEST_PATCH_REF, cwd=self.repo),
            git_out("rev-parse", "HEAD", cwd=self.repo),
        )
        # Out of the way of anything that lists branches or tags for the agent to see.
        self.assertEqual(git_out("branch", "--list", "--all", cwd=self.repo).count("tpa"), 0)
        self.assertEqual(git_out("tag", "--list", cwd=self.repo), "")

    def test_grading_reads_the_ref_this_script_writes(self):
        # Two constants, one contract: the scripts are baked into the dataset, so the only
        # thing keeping the stages in step is that these two strings stay equal.
        self.assertEqual(preprocess.TEST_PATCH_REF, evaluation.TEST_PATCH_REF)


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2)
