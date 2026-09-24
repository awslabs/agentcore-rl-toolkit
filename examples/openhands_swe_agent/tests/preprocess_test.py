#!/usr/bin/env python
"""Tests for ``make_test_patch_script`` against a real git repo, checking the resulting
worktree and commit rather than the script text.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest
from swe_agent_server import evaluation

# preprocess.py imports the datasets package; skip this module without it.
pytest.importorskip("datasets")

import preprocess  # noqa: E402  (recipe dir added to sys.path by conftest.py)

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
# Just the added-file half of TEST_PATCH (~2% of SWE-Gym patches only add files).
ADD_ONLY_PATCH = _SPLIT_AT + TEST_PATCH.split(_SPLIT_AT)[1]

# An empty created file (e.g. a new package's __init__.py) has no hunk, so its path
# appears only in the diff header, not in a "+++ b/" line.
EMPTY_FILE = "tests/pkg/__init__.py"
EMPTY_FILE_PATCH = TEST_PATCH + f"diff --git a/{EMPTY_FILE} b/{EMPTY_FILE}\nnew file mode 100644\n"

# A pure rename has no body on either side either.
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
        # Own directory so the checkout and the script's scratch files are cleaned up together.
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

    # Lines only meaningful inside the task image; stripped so tests don't touch dev config.
    IMAGE_ONLY = ("miniconda3", "conda activate", "git config --global")

    def runnable(self, script: str) -> str:
        return "\n".join(line for line in script.splitlines() if not any(s in line for s in self.IMAGE_ONLY))

    def run_script(self, script: str, name="script.sh") -> subprocess.CompletedProcess:
        path = self.repo.parent / name
        path.write_text(script)
        self.addCleanup(path.unlink, missing_ok=True)
        return subprocess.run(["/bin/bash", str(path)], capture_output=True, text=True)

    def setup_script_for(self, test_patch=TEST_PATCH) -> str:
        """Build the setup script with REPO_PATH/PATCH_FILE/SCRATCH_INDEX retargeted at this checkout."""
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
        # git apply --index requires the patch to apply cleanly to worktree and index alike,
        # so it catches this without a reset step.
        (self.repo / EXISTING_TEST).write_text("def test_old(): assert False\n")

        result = self.run_script(self.script_for())

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.repo / EXISTING_TEST).read_text(), "def test_old(): assert False\n")
        self.assertFalse((self.repo / ADDED_TEST).exists())
        self.assertEqual(git_out("log", "--format=%s", cwd=self.repo), "base")
        self.assertEqual(git_out("for-each-ref", "--format=%(refname)", preprocess.TEST_PATCH_REF, cwd=self.repo), "")

    def test_commits_the_patch_so_the_agent_starts_on_a_clean_worktree(self):
        result = self.run_script(self.script_for())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git_out("status", "--porcelain", cwd=self.repo), "")
        self.assertEqual(git_out("diff", cwd=self.repo), "")
        self.assertEqual(
            sorted(git_out("show", "--name-only", "--format=", cwd=self.repo).splitlines()),
            [EXISTING_TEST, ADDED_TEST],
        )

    def test_leaves_the_images_own_dirt_out_of_the_commit(self):
        # The commit's tree is built in a separate index so pre-existing dirty/staged files
        # aren't folded in, and are left untouched afterward.
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
        # The script commits on top of HEAD (never resets to base_commit), so an image with
        # its own extra install commit keeps it.
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
        # If left untracked, grading's revert can't remove it, and the harness's later apply
        # fails on a file that already exists.
        result = self.run_script(self.script_for(EMPTY_FILE_PATCH))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git_out("status", "--porcelain", cwd=self.repo), "")
        self.assertIn(EMPTY_FILE, git_out("show", "--name-only", "--format=", cwd=self.repo).splitlines())

    def test_commits_both_sides_of_a_rename(self):
        # The old path must be in the commit too, or the revert can't restore it.
        result = self.run_script(self.script_for(RENAME_PATCH))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git_out("status", "--porcelain", cwd=self.repo), "")
        # --no-renames so both paths are listed individually, not collapsed into one.
        self.assertEqual(
            sorted(git_out("show", "--name-only", "--no-renames", "--format=", cwd=self.repo).splitlines()),
            sorted([EXISTING_TEST, RENAMED_TEST]),
        )

    def test_a_patch_that_does_not_apply_exits_nonzero(self):
        conflicting = TEST_PATCH.replace(" def test_old(): pass", " def test_renamed(): pass")

        result = self.run_script(self.script_for(conflicting))

        # set -e propagates git apply's failure before any commit or ref is written.
        self.assertEqual(result.returncode, 1)
        self.assertEqual(git_out("log", "--format=%s", cwd=self.repo), "base")
        self.assertEqual(git_out("for-each-ref", "--format=%(refname)", preprocess.TEST_PATCH_REF, cwd=self.repo), "")

    def test_commits_a_created_test_file_the_repo_gitignores(self):
        # git apply/read-tree/write-tree don't consult .gitignore, unlike a plain git add.
        (self.repo / ".gitignore").write_text(f"{ADDED_TEST}\n")
        git("add", "-A", cwd=self.repo)
        git("commit", "-qm", "ignore", cwd=self.repo)

        result = self.run_script(self.script_for())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git_out("status", "--porcelain", cwd=self.repo), "")
        self.assertIn(ADDED_TEST, git_out("show", "--name-only", "--format=", cwd=self.repo).splitlines())

    def test_leaves_the_commit_where_grading_can_find_it(self):
        result = self.run_script(self.script_for())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            git_out("rev-parse", preprocess.TEST_PATCH_REF, cwd=self.repo),
            git_out("rev-parse", "HEAD", cwd=self.repo),
        )
        # Not visible via branch/tag listing, e.g. to the agent.
        self.assertEqual(git_out("branch", "--list", "--all", cwd=self.repo).count("tpa"), 0)
        self.assertEqual(git_out("tag", "--list", cwd=self.repo), "")

    def test_grading_reads_the_ref_this_script_writes(self):
        # These two strings are the only thing keeping preprocess.py and evaluation.py in sync.
        self.assertEqual(preprocess.TEST_PATCH_REF, evaluation.TEST_PATCH_REF)


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2)
