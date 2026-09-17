"""Run a task's test suite in the container and grade it with the SWE-bench grader."""

import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TypedDict

APPLY_PATCH_PASS = ">>>>> Applied Patch"

# The graded checkout, the same one the eval script below cds into.
REPO_PATH = "/testbed"
# Where the setup stage leaves its test-patch commit, when it made one. Kept in step by
# hand with ``preprocess.TEST_PATCH_REF``, which writes it: the scripts are baked into the
# dataset, so the two stages agree by convention, the same way they agree on /testbed.
TEST_PATCH_REF = "refs/tpa/test-patch"


class EvalReport(TypedDict):
    resolved: bool
    eval_log: str | None
    eval_timeout: bool
    grade_report: dict | None
    eval_latency_s: float | None


def find_test_patch_commit(repo_path: str = REPO_PATH) -> str | None:
    """The commit the setup stage made for the test patch, or None if it made none.

    Absent in the baseline arm, where the patch is never applied before the rollout.
    """
    found = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", TEST_PATCH_REF], cwd=repo_path, capture_output=True, text=True
    )
    return found.stdout.strip() if found.returncode == 0 else None


def capture_git_diff(task_input: dict) -> str:
    """The agent's own work, as a patch against the tree the agent was handed.

    In the ``test_patch_applied`` variant that tree is the test-patch commit rather than
    ``base_commit``, and the difference is the whole point: diffing against base puts the
    graded tests into every recorded rollout patch, which contaminates the dumps as
    training and analysis data, and costs the noop backend its "an empty diff means a
    clean testbed" check. No reward depends on it either way -- both graders use
    ``model_patch`` only to check that it is not None.

    Call this *before* ``run_evaluation``, which reverts that commit; afterwards a diff
    against it would report the revert as the agent's work. Falls back to ``base_commit``
    when there is no test-patch commit, which is what the agent was handed there.
    """
    repo_path = task_input["repo_path"]
    against = find_test_patch_commit(repo_path) or task_input["base_commit"]
    return subprocess.check_output(["git", "--no-pager", "diff", "--no-color", against], cwd=repo_path).decode()


def revert_test_patch_commit(repo_path: str = REPO_PATH) -> str | None:
    """Undo the setup stage's test-patch commit, so the eval script's precondition holds.

    In the ``test_patch_applied`` variant, ``preprocess.make_test_patch_script`` applies the
    graded test files and commits them before the agent starts. The eval script the harness
    generated knows nothing about that commit: it checks the files the patch modifies out of
    ``base_commit`` and applies the patch over them, which assumes a worktree the patch was
    *not* already applied to. Run against a patched one it misgrades in silence -- ``git
    apply`` refuses a file the patch creates, and is atomic about it, so no hunk lands at
    all and the tests that get graded are the base commit's, unrelated to the agent's work.

    Undoing our own mutation restores that assumption, rather than rewriting a script this
    repo does not own: the eval script then runs byte-identical in both arms, and git
    derives the paths from the commit, so a test patch that renames, deletes or chmods a
    file needs no special case here. Returns the reverted commit, or None in the baseline
    arm, where the ref is absent because no commit was ever made.

    Order matters. The graded paths are restored from the commit first, so the revert cannot
    be refused over whatever the agent left in them -- git declines to overwrite local
    changes, and only these paths are touched, so the agent's source changes stay to be
    graded. Restoring after the revert would undo the revert instead.

    The revert is deliberately left staged: ``run_evaluation`` reads ``model_patch`` from a
    plain ``git diff``, which compares the worktree against the index, so a staged revert
    cancels out there and the recorded patch is the agent's own changes alone. Every step
    raises on failure -- this is grading, and a silent failure is a wrong reward.
    """

    def git(*args: str) -> str:
        # stderr folded into stdout so a CalledProcessError carries git's own explanation.
        return subprocess.check_output(["git", *args], cwd=repo_path, stderr=subprocess.STDOUT, text=True)

    sha = find_test_patch_commit(repo_path)
    if sha is None:
        logging.info("No test-patch commit to revert: grading the worktree as it stands")
        return None

    # --no-renames so each line is one status and one path, and a renamed test file is the
    # delete and the add it is made of -- which is what has to be undone anyway.
    changed = [
        line.split("\t")
        for line in git("diff-tree", "--no-commit-id", "--no-renames", "--name-status", "-r", sha).splitlines()
    ]
    restore = [path for status, path in changed if status != "D"]
    if restore:
        git("checkout", sha, "--", *restore)
    for status, path in changed:
        # A file the test patch deleted is not in the commit to restore from, and the revert
        # has to recreate it -- which git refuses if the agent left anything in its place.
        if status == "D":
            Path(repo_path, path).unlink(missing_ok=True)
    git("revert", "--no-commit", sha)
    # ``--no-commit`` leaves REVERT_HEAD behind, so the repo reads as mid-revert to anything
    # that looks at it later -- a `git status` while debugging a dump, most likely. ``--quit``
    # drops that state and keeps the index and worktree, which ``--abort`` would not.
    git("revert", "--quit")

    logging.info(f"Reverted the setup stage's test-patch commit {sha} before grading")
    return sha


def run_evaluation(task: dict) -> EvalReport:
    revert_test_patch_commit()

    git_diff_output_before = subprocess.check_output(["git", "diff"], cwd="/testbed").decode("utf-8").strip()
    logging.info(f"Git diff before:\n{git_diff_output_before}")

    eval_file = Path("/testbed/eval.sh")
    eval_file.write_text(task["eval_script"])

    test_start = time.perf_counter()
    # stderr carries the `set -x` trace, the only source of the `>>>>> Start/End Test
    # Output` markers swebench's grader parses.
    test_output = subprocess.check_output(["/bin/bash", "-x", "/testbed/eval.sh"], stderr=subprocess.STDOUT).decode()
    eval_latency_s = time.perf_counter() - test_start

    test_output = APPLY_PATCH_PASS + test_output

    logging.info(f"Test runtime: {eval_latency_s} seconds")

    git_diff_output_after = subprocess.check_output(["git", "diff"], cwd="/testbed").decode("utf-8").strip()

    logging.info(f"Git diff after:\n{git_diff_output_after}")
    if git_diff_output_after != git_diff_output_before:
        logging.info("Git diff changed after running eval script")

    grade_report = make_grade_report(task, test_output, git_diff_output_before)
    logging.info(f"Grade report: {grade_report}")

    return EvalReport(
        resolved=grade_report["resolved"],
        eval_timeout=False,
        eval_log=test_output,
        grade_report=grade_report,
        eval_latency_s=eval_latency_s,
    )


def make_grade_report(task, test_output, git_diff):
    if "SWE-bench" in task["data_source"]:
        sys.path.append("/agent/swebench")
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import make_test_spec
    elif "SWE-Gym" in task["data_source"]:
        sys.path.append("/agent/swegym")
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec import make_test_spec
    else:
        raise Exception(f"Unknown {task['data_source']=}")

    with tempfile.TemporaryDirectory() as d:
        log_dir = f"{d}/{task['instance_id']}"
        os.makedirs(log_dir)

        log_path = f"{log_dir}/eval_log.txt"
        with open(log_path, "w") as f:
            f.write(test_output)

        test_spec = make_test_spec(task)  # type: ignore
        pred = {
            "instance_id": task["instance_id"],
            "model_patch": git_diff,
        }
        # SWE-bench renamed the log-path kwarg (`log_path` -> `test_log_path`); both it
        # and SWE-Gym keep it 3rd positional, so pass positionally for either version.
        return get_eval_report(
            test_spec,
            pred,
            log_path,
            True,  # include_tests_status
        )[task["instance_id"]]
