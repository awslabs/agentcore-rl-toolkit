"""Evaluation script for GSM8K math agent using RolloutClient.run_batch().

Loads GSM8K data from HuggingFace, extracts prompt/ground_truth,
and performs bulk invocation through a deployed AgentCore runtime.

Usage:
    uv run python evaluate.py \
        --agent_arn arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/abc123 \
        --s3_bucket my-rollout-bucket \
        --base_url http://localhost:8000/v1 \
        --model_id my-model \
        --limit 100
"""

import argparse
import json
import logging
import time
from pathlib import Path

from datasets import load_dataset

from agentcore_rl_toolkit import RolloutClient

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K math agent via AgentCore runtime")
    parser.add_argument(
        "--agent_arn",
        type=str,
        required=True,
        help="AgentCore runtime ARN",
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        required=True,
        help="S3 bucket for storing rollout results",
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

    args = parser.parse_args()

    # Load and process GSM8K data
    logger.info(f"Loading GSM8K {args.split} split from HuggingFace...")
    examples = load_gsm8k(split=args.split, limit=args.limit)
    logger.info(f"Loaded {len(examples)} examples")

    # Setup results directory and file
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    result_path = results_dir / f"{args.exp_id}.jsonl"

    if result_path.exists():
        logger.error(f"Results file already exists: {result_path}")
        logger.error("Delete the file or use a different --exp_id")
        return

    logger.info(f"Results will be written to: {result_path}")

    # Create client. sampling_params flow into every request's _rollout field;
    # the agent (rl_app.py) reads them and applies to the OpenAIModel. Default
    # temperature=1.0 and max_tokens=1024 match train.sh so eval mirrors the
    # sampling distribution the policy sees during RL training.
    client = RolloutClient(
        agent_runtime_arn=args.agent_arn,
        s3_bucket=args.s3_bucket,
        exp_id=args.exp_id,
        base_url=args.base_url,
        model_id=args.model_id,
        tps_limit=args.tps_limit,
        sampling_params={
            "temperature": args.temperature,
            "max_completion_tokens": args.max_tokens,
        },
    )

    # Run batch evaluation
    logger.info(f"Starting evaluation with max_concurrent={args.max_concurrent}, timeout={args.timeout}s...")

    benchmark_start = time.time()
    completed = 0
    succeeded = 0
    failed = 0
    rewards: list[float] = []

    for item in client.run_batch(examples, max_concurrent_sessions=args.max_concurrent, timeout=args.timeout):
        completed += 1

        record = {
            "index": item.index,
            "success": item.success,
            "prompt": examples[item.index]["prompt"],
            "ground_truth": examples[item.index]["answer"],
        }

        if item.success:
            succeeded += 1
            record["result"] = item.result
            record["elapsed"] = item.elapsed
            # Use the reward produced by rl_app.py (via GSM8KReward). This is the
            # same scalar the training loop scores against, so eval and train
            # accuracy are directly comparable.
            raw = item.result.get("rewards")
            reward = float(raw) if isinstance(raw, (int, float)) else None
            if reward is None:
                logger.warning(
                    f"Index {item.index} succeeded but 'rewards' was {raw!r} — "
                    "check rl_app.py return shape. Skipping in accuracy aggregate."
                )
            else:
                rewards.append(reward)
                record["reward"] = reward
            logger.info(
                f"[{completed}/{len(examples)}] Index {item.index} completed in {item.elapsed:.1f}s - "
                f"reward: {reward}"
            )
        else:
            failed += 1
            record["error"] = item.error
            record["elapsed"] = item.elapsed
            logger.warning(
                f"[{completed}/{len(examples)}] Index {item.index} failed in {item.elapsed:.1f}s: {item.error}"
            )

        with open(result_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Summary. Accuracy is computed over examples that produced a numeric reward
    # (excludes both failed runs and successful-but-malformed responses).
    total = len(examples)
    accuracy = sum(rewards) / len(rewards) if rewards else 0.0
    total_time = time.time() - benchmark_start
    logger.info("=" * 50)
    logger.info(f"Evaluation complete: {succeeded} succeeded, {failed} failed out of {total}")
    logger.info(f"Scored: {len(rewards)} / {total}")
    logger.info(f"Accuracy: {accuracy:.4f} ({sum(rewards):.0f}/{len(rewards)})")
    logger.info(f"Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    logger.info(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
