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

# The graded checkout.
REPO_PATH = "/testbed"
# Must match preprocess.TEST_PATCH_REF, which writes this ref.
TEST_PATCH_REF = "refs/tpa/test-patch"


class EvalReport(TypedDict):
    resolved: bool
    eval_log: str | None
    eval_timeout: bool
    grade_report: dict | None
    eval_latency_s: float | None


def find_test_patch_commit(repo_path: str = REPO_PATH) -> str | None:
    """The setup stage's test-patch commit, or None in the baseline arm (patch never applied)."""
    found = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", TEST_PATCH_REF], cwd=repo_path, capture_output=True, text=True
    )
    return found.stdout.strip() if found.returncode == 0 else None


def capture_git_diff(task_input: dict) -> str:
    """The agent's diff against the tree it was handed.

    Diffs against the test-patch commit rather than ``base_commit`` so the recorded patch
    excludes the graded tests. Call before ``run_evaluation``, which reverts that commit.
    """
    repo_path = task_input["repo_path"]
    against = find_test_patch_commit(repo_path) or task_input["base_commit"]
    return subprocess.check_output(["git", "--no-pager", "diff", "--no-color", against], cwd=repo_path).decode()


def revert_test_patch_commit(repo_path: str = REPO_PATH) -> str | None:
    """Undo the setup stage's test-patch commit so the harness's eval script can apply the
    patch cleanly (it assumes an unpatched worktree; ``git apply`` would silently no-op
    otherwise, grading the base commit's tests instead of the agent's work).

    Restores the graded paths from the commit before reverting, so the revert can't be
    blocked by the agent's own edits to those paths. The revert is left staged
    (``--no-commit``) so it cancels out in the ``git diff`` that ``run_evaluation`` reads
    as ``model_patch``. Returns the reverted commit, or None in the baseline arm.
    """

    def git(*args: str) -> str:
        # stderr folded into stdout so a CalledProcessError carries git's own explanation.
        return subprocess.check_output(["git", *args], cwd=repo_path, stderr=subprocess.STDOUT, text=True)

    sha = find_test_patch_commit(repo_path)
    if sha is None:
        logging.info("No test-patch commit to revert: grading the worktree as it stands")
        return None

    # --no-renames: a rename becomes a delete+add, which is what has to be undone anyway.
    changed = [
        line.split("\t")
        for line in git("diff-tree", "--no-commit-id", "--no-renames", "--name-status", "-r", sha).splitlines()
    ]
    restore = [path for status, path in changed if status != "D"]
    if restore:
        git("checkout", sha, "--", *restore)
    for status, path in changed:
        # Deleted files aren't in the restore commit; recreate the empty path so revert can apply.
        if status == "D":
            Path(repo_path, path).unlink(missing_ok=True)
    git("revert", "--no-commit", sha)
    # --quit clears the mid-revert state without undoing the revert (--abort would).
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
