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


class EvalReport(TypedDict):
    resolved: bool
    eval_log: str | None
    eval_timeout: bool
    grade_report: dict | None
    eval_latency_s: float | None


def run_evaluation(task: dict) -> EvalReport:
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
