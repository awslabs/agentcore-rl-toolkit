"""Run the full OfficeBench benchmark (300 tasks) on AgentCore and produce results.

Usage:
    uv run python benchmark.py --exp_id my_run
    uv run python benchmark.py --exp_id my_run --limit 10
    uv run python benchmark.py --exp_id my_run --category 1
    uv run python benchmark.py --exp_id my_run --resume

Results are saved to:
    - results/{exp_id}/rollouts.jsonl  (per-task results, streamed)
    - results/{exp_id}/summary.json    (final scores)
    - S3: s3://{s3_output_bucket}/{exp_id}/...
"""

import argparse
import json
import logging
import time
from pathlib import Path

import boto3
import tomllib

from agentcore_rl_toolkit import RolloutClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.toml") -> dict:
    path = Path(__file__).parent / config_path
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


def list_all_subtasks(s3_uri: str) -> list[dict]:
    """List all subtask entries from S3.

    Returns sorted list of dicts: task_id, subtask_id, task_uri, testbed_uri.
    """
    path = s3_uri.replace("s3://", "").rstrip("/")
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client("s3")
    entries = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/config.json"):
                rel = key[len(prefix) :]
                parts = rel.split("/")
                if len(parts) == 3:
                    task_id, subtask_id, _ = parts
                    entries.append(
                        {
                            "task_id": task_id,
                            "subtask_id": subtask_id,
                            "task_uri": f"s3://{bucket_name}/{key}",
                            "testbed_uri": f"s3://{bucket_name}/{prefix}{task_id}/testbed.tar.gz",
                        }
                    )

    def sort_key(e):
        parts = e["task_id"].split("-")
        try:
            return (int(parts[0]), int(parts[1]), int(e["subtask_id"]))
        except (ValueError, IndexError):
            return (999, 999, 999)

    entries.sort(key=sort_key)
    return entries


def load_completed(result_path: Path) -> set:
    """Load already-completed task IDs from existing results file."""
    completed = set()
    if result_path.exists():
        with open(result_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed.add(f"{rec['task_id']}/{rec['subtask_id']}")
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def compute_summary(result_path: Path, model_id: str, exp_id: str, total_time: float) -> dict:
    """Compute final summary from the results JSONL file."""
    stats = {}
    with open(result_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                cat = rec["category"]
                if cat not in stats:
                    stats[cat] = {"total": 0, "passed": 0, "failed": 0, "errored": 0}
                stats[cat]["total"] += 1
                if rec.get("success") and rec.get("is_pass"):
                    stats[cat]["passed"] += 1
                elif rec.get("success"):
                    stats[cat]["failed"] += 1
                else:
                    stats[cat]["errored"] += 1
            except (json.JSONDecodeError, KeyError):
                pass

    summary = {
        "model_id": model_id,
        "exp_id": exp_id,
        "total_time_seconds": total_time,
        "categories": {},
        "overall": {},
    }

    overall_total = 0
    overall_passed = 0

    for cat in sorted(stats):
        s = stats[cat]
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
    return summary


def print_leaderboard(summary: dict):
    """Print results in OfficeBench leaderboard table format."""
    model = summary["model_id"]
    cats = summary["categories"]
    overall = summary["overall"]

    s1 = cats.get("1-app", {"total": 0, "passed": 0, "pass_rate": 0})
    s2 = cats.get("2-app", {"total": 0, "passed": 0, "pass_rate": 0})
    s3 = cats.get("3-app", {"total": 0, "passed": 0, "pass_rate": 0})

    header = (
        f"| {'Model':<40} | {'Single App (93)':>15} "
        f"| {'Two Apps (95)':>14} | {'Three Apps (112)':>16} | {'Overall (300)':>14} |"
    )
    sep = f"|{'-' * 42}|{'-' * 17}|{'-' * 16}|{'-' * 18}|{'-' * 16}|"
    row = (
        f"| {model:<40} "
        f"| {s1['pass_rate'] * 100:>14.2f}% "
        f"| {s2['pass_rate'] * 100:>13.2f}% "
        f"| {s3['pass_rate'] * 100:>15.2f}% "
        f"| {overall['pass_rate'] * 100:>13.2f}% |"
    )
    detail = (
        f"| {'(passed/total)':<40} "
        f"| {s1['passed']:>7}/{s1['total']:<7} "
        f"| {s2['passed']:>6}/{s2['total']:<7} "
        f"| {s3['passed']:>8}/{s3['total']:<7} "
        f"| {overall['passed']:>6}/{overall['total']:<7}|"
    )

    print()
    print(header)
    print(sep)
    print(row)
    print(detail)
    print()


def main():
    config = load_config()
    agentcore_config = config.get("agentcore", {})
    eval_config = config.get("eval", {})

    parser = argparse.ArgumentParser(description="Run OfficeBench benchmark on AgentCore")
    parser.add_argument("--agent_arn", type=str, default=agentcore_config.get("agent_arn"))
    parser.add_argument("--s3_input_bucket", type=str, default=eval_config.get("s3_input_bucket"))
    parser.add_argument("--s3_output_bucket", type=str, default=eval_config.get("s3_output_bucket"))
    parser.add_argument("--base_url", type=str, default=eval_config.get("base_url"))
    parser.add_argument("--model_id", type=str, default=eval_config.get("model_id"))
    parser.add_argument("--exp_id", type=str, default="benchmark", help="Experiment ID (also S3 folder)")
    parser.add_argument("--max_concurrent", type=int, default=100, help="Max concurrent ACR sessions")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Timeout per task (seconds)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max output tokens per turn")
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=None,
        help="Enable thinking mode with token budget (e.g. 10000)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument(
        "--category", type=str, default=None, choices=["1", "2", "3"], help="Only run a specific category"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")

    args = parser.parse_args()

    if not args.agent_arn:
        parser.error("--agent_arn required (or set agentcore.agent_arn in config.toml)")
    if not args.s3_input_bucket:
        parser.error("--s3_input_bucket required")
    if not args.s3_output_bucket:
        parser.error("--s3_output_bucket required")

    # Discover subtasks
    logger.info(f"Listing subtasks from {args.s3_input_bucket}...")
    entries = list_all_subtasks(args.s3_input_bucket)
    if not entries:
        logger.error("No subtasks found")
        return

    if args.category:
        entries = [e for e in entries if e["task_id"].split("-")[0] == args.category]
    if args.limit:
        entries = entries[: args.limit]

    logger.info(f"Total subtasks to evaluate: {len(entries)}")

    # Setup results
    results_dir = Path(__file__).parent / "results" / args.exp_id
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / "rollouts.jsonl"
    summary_path = results_dir / "summary.json"

    # Resume support
    completed = set()
    if args.resume:
        completed = load_completed(result_path)
        remaining = [e for e in entries if f"{e['task_id']}/{e['subtask_id']}" not in completed]
        logger.info(f"Resuming: {len(completed)} done, {len(remaining)} remaining")
    else:
        if result_path.exists():
            logger.error(f"Results exist: {result_path}. Use --resume or different --exp_id")
            return
        remaining = entries

    if not remaining:
        logger.info("All tasks already completed!")
    else:
        # Prepare payloads
        payloads = [{"task_uri": e["task_uri"], "testbed_uri": e["testbed_uri"]} for e in remaining]

        # Build sampling params from CLI args
        sampling_params = {}
        if args.temperature is not None:
            sampling_params["temperature"] = args.temperature
        if args.top_p is not None:
            sampling_params["top_p"] = args.top_p
        if args.max_tokens is not None:
            sampling_params["max_tokens"] = args.max_tokens
        if args.thinking_budget is not None:
            sampling_params["thinking_budget"] = args.thinking_budget

        # Create client (sampling_params passed via extra_config into _rollout)
        client_kwargs = {
            "agent_runtime_arn": args.agent_arn,
            "s3_bucket": args.s3_output_bucket,
            "exp_id": args.exp_id,
            "base_url": args.base_url,
            "model_id": args.model_id,
        }
        if sampling_params:
            client_kwargs["sampling_params"] = sampling_params
        client = RolloutClient(**client_kwargs)

        logger.info(f"Starting benchmark: max_concurrent={args.max_concurrent}, timeout={args.timeout}s")
        benchmark_start = time.time()
        done_count = len(completed)

        for item in client.run_batch(payloads, max_concurrent_sessions=args.max_concurrent, timeout=args.timeout):
            done_count += 1
            entry = remaining[item.index]
            task_id = entry["task_id"]
            subtask_id = entry["subtask_id"]
            category = task_id.split("-")[0]
            display_id = f"{task_id}/{subtask_id}"

            record = {
                "task_id": task_id,
                "subtask_id": subtask_id,
                "category": category,
                "success": item.success,
                "task_uri": entry["task_uri"],
            }

            if item.success:
                record["result"] = item.result
                record["elapsed"] = item.elapsed
                rewards = item.result.get("rewards", 0.0)
                is_pass = rewards == 1.0
                record["is_pass"] = is_pass
                logger.info(
                    f"[{done_count}/{len(entries)}] {display_id} "
                    f"{'PASS' if is_pass else 'FAIL'} (reward={rewards}, {item.elapsed:.1f}s)"
                )
            else:
                record["error"] = item.error
                record["elapsed"] = item.elapsed
                record["is_pass"] = False
                logger.warning(f"[{done_count}/{len(entries)}] {display_id} ERROR ({item.elapsed:.1f}s): {item.error}")

            with open(result_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    # Compute and save summary
    wall_clock_time = time.time() - benchmark_start if "benchmark_start" in dir() else 0
    # Re-read to handle resume case
    if result_path.exists():
        # Calculate cumulative task time (sum of all individual elapsed times)
        cumulative_elapsed = 0
        with open(result_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cumulative_elapsed += rec.get("elapsed", 0)
                except (json.JSONDecodeError, KeyError):
                    pass

        summary = compute_summary(result_path, args.model_id or "unknown", args.exp_id, wall_clock_time)
        summary["cumulative_task_time_seconds"] = cumulative_elapsed
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Print leaderboard
        logger.info("=" * 70)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 70)
        print_leaderboard(summary)

        cats = summary["categories"]
        for cat_name in sorted(cats):
            s = cats[cat_name]
            logger.info(
                f"  {cat_name}: {s['passed']}/{s['total']} ({s['pass_rate']:.1%}) "
                f"[failed={s['failed']}, errored={s['errored']}]"
            )
        o = summary["overall"]
        logger.info(f"  Overall: {o['passed']}/{o['total']} ({o['pass_rate']:.1%})")
        logger.info(f"  Wall clock time: {wall_clock_time:.1f}s ({wall_clock_time / 60:.1f}m)")
        logger.info(f"  Cumulative task time: {cumulative_elapsed:.1f}s ({cumulative_elapsed / 60:.1f}m)")
        logger.info(f"  Results: {result_path}")
        logger.info(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
