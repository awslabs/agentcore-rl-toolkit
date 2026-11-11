"""Run a benchmark with Claude on Amazon Bedrock as the agent's model, on this host.

  python eval/claude_bench.py --data data/benchmarks/swe-explore.jsonl --out OUT --model MODEL_ID

It runs the agent's own episode (agent/rl_app.py run_episode); only the model differs. Claude uses adaptive
thinking at high effort, prompt caching and the same 16k output tokens per turn as Qwen. A refusal counts
as no answer. The output is the same as eval/full_benchmark.py's, and so is resuming. Needs
`uv sync --extra claude` and Bedrock access to the model.
"""

import argparse
import asyncio
import json
import os
import threading
from pathlib import Path

from anthropic import AsyncAnthropicBedrock
from dotenv import load_dotenv
from full_benchmark import Benchmark, load_tasks
from strands.models.anthropic import AnthropicModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PARAMS = {
    "thinking": {"type": "adaptive"},
    "output_config": {"effort": "high"},
    "cache_control": {"type": "ephemeral"},
}
# List prices per million tokens (input, output, cache read, cache write), to print the running cost.
PRICES = {
    "global.anthropic.claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.50),
    "global.anthropic.claude-sonnet-5": (2.0, 10.0, 0.20, 2.50),
}
USAGE_KEYS = ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheWriteInputTokens")


class ClaudeBenchmark(Benchmark):
    def __init__(self, rows, out, concurrency, timeout, *, model: str, region: str):
        super().__init__(rows, out, concurrency, timeout)
        self.model = model
        self.region = region
        self.usage_path = out / "usage.json"
        self.usage = dict.fromkeys(USAGE_KEYS, 0)
        if self.usage_path.exists():
            self.usage = json.loads(self.usage_path.read_text())
        self.lock = threading.Lock()

    async def run_agent(self, row: dict) -> dict:
        return await asyncio.to_thread(self.episode, row)

    def episode(self, row: dict) -> dict:
        import rl_app  # only once .env is loaded: it reads the corpus bucket and checkout cache at import

        model = AnthropicModel(model_id=self.model, max_tokens=16384, params=MODEL_PARAMS)
        model.client = AsyncAnthropicBedrock(aws_region=self.region)
        result, agent = rl_app.run_episode(row, model)
        if agent is not None:
            with self.lock:
                for key in USAGE_KEYS:
                    self.usage[key] += agent.event_loop_metrics.accumulated_usage.get(key, 0)
                self.usage_path.write_text(json.dumps(self.usage, indent=2) + "\n")
                if price := PRICES.get(self.model):
                    cost = sum(self.usage[k] * p for k, p in zip(USAGE_KEYS, price, strict=True)) / 1e6
                    print(f"{self.out.name}: ${cost:.2f} so far", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True, help="a Bedrock model or inference profile ID")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N tasks")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")  # the corpus bucket, checkout cache and region
    rows = load_tasks(args.data, args.out, args.model, args.limit)
    benchmark = ClaudeBenchmark(
        rows, args.out, args.concurrency, args.timeout, model=args.model, region=os.environ["AWS_REGION"]
    )
    asyncio.run(benchmark.run())


if __name__ == "__main__":
    main()
