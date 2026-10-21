"""Evaluation script for GSM8K math agent through background HTTP invocations.

Loads GSM8K data from HuggingFace, extracts prompt/ground_truth,
and performs bulk invocation through a deployed AgentCore runtime.

Usage:
    uv run python evaluate.py \
        --agent_arn arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/abc123 \
        --base_url http://localhost:8000/v1 \
        --model_id my-model \
        --limit 100
"""

import argparse
import asyncio
import json
import logging
import math
import tempfile
import time
import uuid
from pathlib import Path

from datasets import load_dataset

from agentcore_rl_toolkit.aws_tools.agentcore_tools import close_shared_agentcore_clients
from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister, PersistentDict
from agentcore_rl_toolkit.concurrency.rate_limiter import LocalRateLimiter
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gsm8k(split: str = "test", limit: int | None = None) -> list[dict]:
    """Load GSM8K from HuggingFace and extract {prompt, answer} pairs.

    HF answers are formatted "reasoning steps\\n#### <number>"; we split on
    "####" and strip commas from the number so ground truth matches the
    format the agent is prompted to emit.
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    examples = []
    for row in dataset:
        item: dict = row  # type: ignore[assignment]  # HF Dataset yields dicts at runtime
        answer = item["answer"].split("####")[-1].replace(",", "").strip()
        examples.append({"prompt": item["question"], "answer": answer})
        if limit and len(examples) >= limit:
            break

    return examples


async def evaluate(args, entries, result_path):
    benchmark_start = time.monotonic()
    semaphore = asyncio.Semaphore(args.max_concurrent)
    starts = LocalRateLimiter(args.tps_limit)
    sampling = {"temperature": args.temperature, "max_completion_tokens": args.max_tokens}

    async def run_one(index, entry):
        async with semaphore:
            sid = str(uuid.uuid4())
            started = time.monotonic()
            task = dict(entry)
            task["_config"] = {
                "base_url": args.base_url,
                "model_id": args.model_id,
                "api_key": sid,
                "sampling_params": sampling or {},
            }
            record = {
                "index": index,
                "session_id": sid,
                "success": False,
                "prompt": entry["prompt"],
                "ground_truth": entry["answer"],
            }
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
                record.update(success=True, result=result, reward=reward)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed"] = time.monotonic() - started
            return record

    completed = succeeded = task_successes = 0
    rewards = []
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run_one(i, entry)) for i, entry in enumerate(entries)]
            for future in asyncio.as_completed(tasks):
                record = await future
                completed += 1
                succeeded += int(record["success"])
                passed = record.get("reward") == 1.0
                task_successes += int(passed)
                if record["success"]:
                    rewards.append(record["reward"])
                with open(result_path, "a") as stream:
                    stream.write(json.dumps(record) + "\n")
                logger.info(
                    "[%d/%d] index=%d elapsed=%.1fs reward=%s error=%s",
                    completed,
                    len(entries),
                    record["index"],
                    record["elapsed"],
                    record.get("reward"),
                    record.get("error"),
                )
    finally:
        await close_shared_agentcore_clients()
    logger.info("Completed=%d successful_invocations=%d task_successes=%d", completed, succeeded, task_successes)
    logger.info(
        "Scored=%d/%d mean_reward=%s", len(rewards), len(entries), sum(rewards) / len(rewards) if rewards else None
    )
    logger.info("Total time=%.1fs results=%s", time.monotonic() - benchmark_start, result_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K math agent via AgentCore runtime")
    parser.add_argument(
        "--agent_arn",
        type=str,
        required=True,
        help="AgentCore runtime ARN",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="vLLM/SGLang server URL for model inference",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model ID for inference",
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default="gsm8k_eval",
        help="Experiment ID for organizing results (default: gsm8k_eval)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="GSM8K split to evaluate (default: test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate (for testing)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=100,
        help="Max concurrent ACR sessions (default: 100)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Timeout in seconds per request (default: 900s / 15 min)",
    )
    parser.add_argument(
        "--tps_limit",
        type=int,
        default=25,
        help="ACR invocation rate limit in TPS (default: 25, matches ACR service quota)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, matches train.sh ROLLOUT_TEMPERATURE)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Max output tokens per LLM call (default: 1024, matches train.sh MAX_RESPONSE_LEN)",
    )

    parser.add_argument(
        "--output_dir", type=Path, default=Path(tempfile.gettempdir()) / "strands_math_agent-evaluations"
    )
    args = parser.parse_args()

    for field in ("max_concurrent", "tps_limit", "timeout"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    for field in ("base_url", "model_id"):
        if not getattr(args, field):
            parser.error(f"--{field} is required")

    # Load and process GSM8K data
    logger.info(f"Loading GSM8K {args.split} split from HuggingFace...")
    examples = load_gsm8k(split=args.split, limit=args.limit)
    logger.info(f"Loaded {len(examples)} examples")

    # Setup results directory and file
    results_dir = args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{args.exp_id}.jsonl"

    if result_path.exists():
        logger.error(f"Results file already exists: {result_path}")
        logger.error("Delete the file or use a different --exp_id")
        return

    logger.info(f"Results will be written to: {result_path}")

    asyncio.run(evaluate(args, examples, result_path))


if __name__ == "__main__":
    main()
