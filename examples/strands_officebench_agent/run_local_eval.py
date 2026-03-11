"""Batch local evaluation — runs all OfficeBench tasks and saves results."""

import argparse
import json
import logging
import os
import shutil
import time
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
TESTBED_DIR = "/tmp/testbed"

SYSTEM_PROMPT = (
    "You are an AI office assistant for user {username}. "
    "Today is {date} ({weekday}). The current time is {time}.\n\n"
    "Help solve the office task using the available tools.\n"
    "Task files are in {testbed}/data/. Calendar files are in {testbed}/calendar/. "
    "Email files are in {testbed}/emails/.\n\n"
    "Important:\n"
    "- Use absolute paths starting with {testbed}/ for all file operations.\n"
    "- When creating new files, place them under {testbed}/data/ unless specified otherwise.\n"
    "- When the task is complete, stop calling tools and summarize what you did."
)


def collect_all_subtasks(officebench_dir, category=None):
    """Collect all (task_id, subtask_id) pairs, sorted."""
    tasks_dir = os.path.join(officebench_dir, "tasks")
    entries = []
    for task_dir_name in os.listdir(tasks_dir):
        if category and not task_dir_name.startswith(f"{category}-"):
            continue
        subtasks_dir = os.path.join(tasks_dir, task_dir_name, "subtasks")
        if not os.path.exists(subtasks_dir):
            continue
        for f in os.listdir(subtasks_dir):
            if f.endswith(".json"):
                entries.append((task_dir_name, f.removesuffix(".json")))

    def sort_key(e):
        parts = e[0].split("-")
        try:
            return (int(parts[0]), int(parts[1]), int(e[1]))
        except (ValueError, IndexError):
            return (999, 999, 999)

    entries.sort(key=sort_key)
    return entries


def setup_local_testbed(task_dir):
    """Set up testbed from local OfficeBench task directory."""
    if os.path.exists(TESTBED_DIR):
        shutil.rmtree(TESTBED_DIR)

    testbed_src = os.path.join(task_dir, "testbed")
    if os.path.exists(testbed_src):
        shutil.copytree(testbed_src, TESTBED_DIR)
    else:
        os.makedirs(TESTBED_DIR, exist_ok=True)

    os.makedirs(os.path.join(TESTBED_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(TESTBED_DIR, "calendar"), exist_ok=True)
    os.makedirs(os.path.join(TESTBED_DIR, "emails"), exist_ok=True)

    # Copy reference/ for evaluate_exact_match
    reference_src = os.path.join(task_dir, "reference")
    if os.path.exists(reference_src):
        shutil.copytree(reference_src, os.path.join(TESTBED_DIR, "reference"))

    # Create cache/ snapshot for evaluate_diff_contain_text
    data_src = os.path.join(TESTBED_DIR, "data")
    if os.path.exists(data_src) and os.listdir(data_src):
        shutil.copytree(data_src, os.path.join(TESTBED_DIR, "cache", "data"))


def load_completed(result_path):
    """Load already-completed task IDs from an existing results file."""
    completed = set()
    if os.path.exists(result_path):
        with open(result_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    completed.add(f"{record['task_id']}/{record['subtask_id']}")
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def main():
    parser = argparse.ArgumentParser(description="Run local OfficeBench evaluation")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Bedrock model ID")
    parser.add_argument("--exp_id", type=str, default="local_eval", help="Experiment ID for results folder")
    parser.add_argument(
        "--category", type=str, default=None, choices=["1", "2", "3"], help="Only evaluate a specific category"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results (skip completed)")
    args = parser.parse_args()

    # Configure local paths BEFORE importing tools (they read env at import time)
    officebench_dir = os.environ.get("OFFICEBENCH_DIR", os.path.expanduser("~/OfficeBench"))
    os.environ["OFFICEBENCH_APPS_DIR"] = os.path.join(officebench_dir, "apps")
    os.environ["OFFICEBENCH_TESTBED_DIR"] = TESTBED_DIR
    os.environ["BYPASS_TOOL_CONSENT"] = "true"

    from reward import OfficeBenchReward
    from strands import Agent
    from strands.agent.conversation_manager import NullConversationManager
    from strands.models import BedrockModel
    from strands_tools import shell
    from tools import ALL_TOOLS

    reward_fn = OfficeBenchReward()

    def run_single_task(task_id, subtask_id, model_id):
        """Run a single task and return the result dict."""
        task_dir = os.path.join(officebench_dir, "tasks", task_id)
        config_path = os.path.join(task_dir, "subtasks", f"{subtask_id}.json")

        with open(config_path) as f:
            task_config = json.load(f)

        setup_local_testbed(task_dir)

        model = BedrockModel(model_id=model_id)

        system_prompt = SYSTEM_PROMPT.format(
            username=task_config.get("username", "User"),
            date=task_config.get("date", "unknown"),
            weekday=task_config.get("weekday", "unknown"),
            time=task_config.get("time", "unknown"),
            testbed=TESTBED_DIR,
        )

        agent = Agent(
            model=model,
            tools=[shell, *ALL_TOOLS],
            system_prompt=system_prompt,
            conversation_manager=NullConversationManager(),
        )

        response = agent(task_config["task"])
        response_text = response.message["content"][0]["text"]

        reward = reward_fn(testbed_dir=TESTBED_DIR, evaluation_config=task_config["evaluation"])

        # Collect full conversation history (all messages with tool calls)
        messages = [{"role": msg.get("role", "unknown"), "content": msg.get("content", [])} for msg in agent.messages]

        # Collect testbed file listing
        testbed_files = []
        for root, _dirs, files in os.walk(TESTBED_DIR):
            for f in files:
                path = os.path.join(root, f)
                testbed_files.append({"path": path, "size": os.path.getsize(path)})

        return {
            "task": task_config["task"],
            "response": response_text,
            "messages": messages,
            "reward": reward,
            "testbed_files": testbed_files,
        }

    # Setup results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results", args.exp_id)
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, "rollouts.jsonl")
    summary_path = os.path.join(results_dir, "summary.json")

    # Collect tasks
    entries = collect_all_subtasks(officebench_dir, category=args.category)
    if args.limit:
        entries = entries[: args.limit]

    logger.info(f"Total subtasks: {len(entries)}")
    logger.info(f"Results dir: {results_dir}")

    # Check for resume
    completed = set()
    if args.resume:
        completed = load_completed(result_path)
        logger.info(f"Resuming: {len(completed)} already completed, {len(entries) - len(completed)} remaining")
    elif os.path.exists(result_path):
        logger.error(f"Results file exists: {result_path}")
        logger.error("Use --resume to continue, or delete the file / use a different --exp_id")
        return

    # Run evaluation
    benchmark_start = time.time()
    category_stats = {}
    total_done = len(completed)

    for task_id, subtask_id in entries:
        display_id = f"{task_id}/{subtask_id}"
        category = task_id.split("-")[0]

        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0, "failed": 0, "errored": 0}

        if display_id in completed:
            continue

        category_stats[category]["total"] += 1
        total_done += 1

        record = {
            "task_id": task_id,
            "subtask_id": subtask_id,
            "category": category,
        }

        start_time = time.time()
        try:
            result = run_single_task(task_id, subtask_id, args.model_id)
            elapsed = time.time() - start_time

            record["success"] = True
            record["reward"] = result["reward"]
            record["response"] = result["response"]
            record["messages"] = result["messages"]
            record["task_description"] = result["task"]
            record["testbed_files"] = result["testbed_files"]
            record["elapsed"] = elapsed

            is_pass = result["reward"] == 1.0
            if is_pass:
                category_stats[category]["passed"] += 1
            else:
                category_stats[category]["failed"] += 1

            status = "PASS" if is_pass else "FAIL"
            logger.info(
                f"[{total_done}/{len(entries)}] {display_id} {status} (reward={result['reward']}, {elapsed:.1f}s)"
            )

        except Exception as e:
            elapsed = time.time() - start_time
            record["success"] = False
            record["error"] = traceback.format_exc()
            record["elapsed"] = elapsed
            category_stats[category]["errored"] += 1

            logger.error(f"[{total_done}/{len(entries)}] {display_id} ERROR ({elapsed:.1f}s): {e}")

        # Append result immediately
        with open(result_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Compute final stats (re-read all results for accurate stats when resuming)
    final_stats = {
        "1": {"total": 0, "passed": 0, "failed": 0, "errored": 0},
        "2": {"total": 0, "passed": 0, "failed": 0, "errored": 0},
        "3": {"total": 0, "passed": 0, "failed": 0, "errored": 0},
    }

    with open(result_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                cat = rec["category"]
                if cat not in final_stats:
                    final_stats[cat] = {"total": 0, "passed": 0, "failed": 0, "errored": 0}
                final_stats[cat]["total"] += 1
                if rec.get("success"):
                    if rec.get("reward") == 1.0:
                        final_stats[cat]["passed"] += 1
                    else:
                        final_stats[cat]["failed"] += 1
                else:
                    final_stats[cat]["errored"] += 1
            except (json.JSONDecodeError, KeyError):
                pass

    total_time = time.time() - benchmark_start

    # Build summary
    summary = {
        "model_id": args.model_id,
        "exp_id": args.exp_id,
        "total_time_seconds": total_time,
        "categories": {},
        "overall": {},
    }

    overall_total = 0
    overall_passed = 0

    for cat in sorted(final_stats):
        s = final_stats[cat]
        rate = s["passed"] / s["total"] if s["total"] > 0 else 0
        summary["categories"][f"{cat}-app"] = {
            "total": s["total"],
            "passed": s["passed"],
            "failed": s["failed"],
            "errored": s["errored"],
            "pass_rate": round(rate, 4),
        }
        overall_total += s["total"]
        overall_passed += s["passed"]

    overall_rate = overall_passed / overall_total if overall_total > 0 else 0
    summary["overall"] = {
        "total": overall_total,
        "passed": overall_passed,
        "pass_rate": round(overall_rate, 4),
    }

    # Save summary
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    for cat in sorted(final_stats):
        s = final_stats[cat]
        rate = s["passed"] / s["total"] if s["total"] > 0 else 0
        logger.info(
            f"  {cat}-app: {s['passed']}/{s['total']} ({rate:.1%}) [failed={s['failed']}, errored={s['errored']}]"
        )
    logger.info(f"  Overall: {overall_passed}/{overall_total} ({overall_rate:.1%})")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    logger.info(f"  Results: {result_path}")
    logger.info(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
