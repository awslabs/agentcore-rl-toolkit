"""Evaluate MigrationBench through background HTTP invocations."""

import argparse
import asyncio
import json
import logging
import math
import tempfile
import time
import uuid
from pathlib import Path

from eval_utils import append_result_to_file, get_s3_folder_uris, load_config, prepare_payload

from agentcore_rl_toolkit.aws_tools.agentcore_tools import close_shared_agentcore_clients
from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister, PersistentDict
from agentcore_rl_toolkit.concurrency.rate_limiter import LocalRateLimiter
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession

logger = logging.getLogger(__name__)


async def evaluate(args, folders, result_path):
    semaphore = asyncio.Semaphore(args.max_concurrent)
    starts = LocalRateLimiter(args.tps_limit)
    sampling = json.loads(args.sampling_params) if isinstance(args.sampling_params, str) else args.sampling_params

    async def run_one(index, uri):
        async with semaphore:
            sid = str(uuid.uuid4())
            started = time.monotonic()
            task = prepare_payload(
                uri, args.require_maximal_migration, args.apply_static_update, args.use_dependency_search_tool
            )
            task["_config"] = {
                "base_url": args.base_url,
                "model_id": args.model_id,
                "api_key": sid,
                "sampling_params": sampling or {},
            }
            record = {"index": index, "input_uri": uri, "session_id": sid, "success": False}
            try:
                async with AgentCoreHttpSession(
                    sid, PersistentDict(persister=NullPersister()), agentcore_runtime_arn=args.agent_arn
                ) as session:
                    await starts.wait_async()
                    async with asyncio.timeout(args.timeout):
                        dump = await session.run(task)
                if dump.exception:
                    raise RuntimeError(dump.exception)
                result = dump.task_output or {}
                reward = result.get("reward")
                if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                    raise ValueError(f"Expected a finite scalar reward, got {reward!r}")
                record.update(success=True, result=result)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed"] = time.monotonic() - started
            return record

    completed = succeeded = task_successes = 0
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run_one(i, uri)) for i, uri in enumerate(folders)]
            for future in asyncio.as_completed(tasks):
                record = await future
                append_result_to_file(result_path, record)
                completed += 1
                succeeded += int(record["success"])
                task_successes += int(record.get("result", {}).get("reward") == 1.0)
                logger.info(
                    "[%d/%d] %s elapsed=%.1fs reward=%s error=%s",
                    completed,
                    len(folders),
                    record["input_uri"],
                    record["elapsed"],
                    record.get("result", {}).get("reward"),
                    record.get("error"),
                )
    finally:
        await close_shared_agentcore_clients()
    logger.info("Completed=%d successful_invocations=%d task_successes=%d", completed, succeeded, task_successes)


def main():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    agentcore = config.get("agentcore", {})
    evaluation = config.get("eval", {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_arn", default=agentcore.get("agent_arn"))
    for field in ("s3_input_bucket", "base_url", "model_id"):
        parser.add_argument(f"--{field}", default=evaluation.get(field))
    parser.add_argument("--sampling_params", default=evaluation.get("sampling_params", {}), help="JSON object")
    parser.add_argument("--exp_id", default="eval")
    parser.add_argument("--output_dir", type=Path, default=Path(tempfile.gettempdir()) / "migration-evaluations")
    parser.add_argument("--max_concurrent", type=int, default=16)
    parser.add_argument("--tps_limit", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--limit", type=int)
    for field in ("require_maximal_migration", "apply_static_update", "use_dependency_search_tool"):
        parser.add_argument(f"--{field}", action="store_true")
    args = parser.parse_args()
    for field in ("agent_arn", "s3_input_bucket", "base_url", "model_id"):
        if not getattr(args, field):
            parser.error(f"--{field} is required (or set it in config.toml)")
    for field in ("max_concurrent", "tps_limit", "timeout"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.exp_id or Path(args.exp_id).name != args.exp_id or args.exp_id in {".", ".."}:
        parser.error("--exp_id must be a filename component")
    folders = get_s3_folder_uris(args.s3_input_bucket)[: args.limit]
    if not folders:
        parser.error("No repository folders found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{args.exp_id}.jsonl"
    with result_path.open("x"):
        pass
    asyncio.run(evaluate(args, folders, result_path))
    logger.info("Results: %s", result_path)


if __name__ == "__main__":
    main()
