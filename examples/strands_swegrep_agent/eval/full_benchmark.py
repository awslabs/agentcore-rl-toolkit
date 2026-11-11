"""Run a benchmark through the deployed agent, with the model served by vLLM on this host.

  python eval/full_benchmark.py --data data/benchmarks/swe-explore.jsonl --out OUT --model NAME

The agent calls vLLM directly (port 8399; endpoint/serve_eval.sh fixes the sampling settings). Each task
gets one attempt, and its result is appended to OUT/results.jsonl: the regions, reward, metrics and
assistant turns. A task that runs past --timeout counts as unanswered. Tasks that fail for another
reason are retried, and OUT/COMPLETED is written once every task has finished without error. Rerunning
resumes. eval/score.py scores the results.
"""

import argparse
import asyncio
import json
import os
import subprocess
import urllib.request
import uuid
from pathlib import Path

from dotenv import load_dotenv

from agentcore_rl_toolkit.concurrency.rate_limiter import LocalRateLimiter
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession
from agentcore_rl_toolkit.runtime import AgentCoreHttpClient

ROOT = Path(__file__).resolve().parents[1]
VLLM_PORT = 8399
MAX_CONTEXT_TOKENS = 65536
ATTEMPTS = 3


def latest_results(path: Path) -> dict[str, dict]:
    """instance_id -> its result; a retried task's last result wins."""
    if not path.exists():
        return {}
    return {row["instance_id"]: row for row in map(json.loads, path.read_text().splitlines())}


class Benchmark:
    """Runs every task that has no successful result yet. claude_bench.py overrides run_agent."""

    def __init__(self, rows: list[dict], out: Path, concurrency: int, timeout: float):
        self.rows = rows
        self.out = out
        self.timeout = timeout
        self.limit = asyncio.Semaphore(concurrency)

    async def run_agent(self, row: dict) -> dict:
        raise NotImplementedError

    async def run_task(self, row: dict) -> None:
        async with self.limit:
            error, result = None, {}
            deadline = asyncio.timeout(self.timeout)
            try:
                async with deadline:
                    result = await self.run_agent(row)
                error = result.get("error")
                if not error and "regions" not in result:
                    error = "the agent returned no regions: redeploy it (bash cdk/deploy.sh)"
            except Exception as e:  # noqa: BLE001 - recorded, and retried unless it is our own timeout
                error = None if deadline.expired() else repr(e)
            timed_out = deadline.expired()
        record = {
            "instance_id": row["instance_id"],
            "data_source": row["data_source"],
            "error": error,
            "timed_out": timed_out,
            "regions": result.get("regions", []),
            "reward": result.get("reward"),
            "metrics": result.get("metrics", {}),
            "messages": result.get("messages", []),
        }
        with (self.out / "results.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if error:
            print(f"{row['instance_id']}: {error}", flush=True)

    async def run(self) -> None:
        for _ in range(ATTEMPTS):
            results = latest_results(self.out / "results.jsonl")
            pending = [row for row in self.rows if results.get(row["instance_id"], {"error": "not run"})["error"]]
            if not pending:
                break
            print(f"{self.out}: {len(pending)} tasks to run", flush=True)
            await asyncio.gather(*(self.run_task(row) for row in pending))
        results = latest_results(self.out / "results.jsonl")
        failed = [row["instance_id"] for row in self.rows if results[row["instance_id"]]["error"]]
        if failed:
            raise SystemExit(f"{len(failed)} tasks failed {ATTEMPTS} times, e.g. {failed[0]}; rerun to retry them")
        (self.out / "COMPLETED").touch()


class RuntimeBenchmark(Benchmark):
    def __init__(self, rows, out, concurrency, timeout, *, arn: str, base_url: str, model: str):
        super().__init__(rows, out, concurrency, timeout)
        self.arn = arn
        self.base_url = base_url
        self.model = model

    async def run(self) -> None:
        self.client = AgentCoreHttpClient(self.arn, request_rate_limiter=LocalRateLimiter(100))
        self.session_starts = LocalRateLimiter(25)
        try:
            await super().run()
        finally:
            await self.client.aclose()

    async def run_agent(self, row: dict) -> dict:
        sid = str(uuid.uuid4())  # the runtime session, and the API key the agent calls vLLM with
        task = {**row, "_config": {"base_url": self.base_url, "model_id": self.model, "api_key": sid}}
        async with AgentCoreHttpSession.with_client(sid, self.client) as session:
            await self.session_starts.wait_async()
            dump = await session.run(task)
        if dump.exception:
            raise RuntimeError(f"agent invocation failed: {dump.exception}")
        return dump.task_output or {}


def load_tasks(data: Path, out: Path, model: str, limit: int = 0) -> list[dict]:
    """The task rows, after checking that a resumed run uses the same data and model."""
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    rows = rows[:limit] if limit else rows
    out.mkdir(parents=True, exist_ok=True)
    run = {"data": data.name, "tasks": len(rows), "model": model}
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != run:
        raise SystemExit(f"{out} holds a run of {manifest.read_text().strip()}, not {run}")
    manifest.write_text(json.dumps(run) + "\n")
    return rows


def check_vllm(base_url: str, model: str) -> None:
    with urllib.request.urlopen(f"{base_url}/models", timeout=10) as response:
        served = {m["id"]: m for m in json.load(response)["data"]}
    if model not in served or served[model].get("max_model_len") != MAX_CONTEXT_TOKENS:
        raise SystemExit(f"vLLM at {base_url} must serve {model} with max_model_len {MAX_CONTEXT_TOKENS}: {served}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True, help="the served model name")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N tasks")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    # The runtime reaches vLLM on this host's private IP.
    host = subprocess.check_output(["hostname", "-I"], text=True).split()[0]
    base_url = f"http://{host}:{VLLM_PORT}/v1"
    check_vllm(base_url, args.model)
    rows = load_tasks(args.data, args.out, args.model, args.limit)
    benchmark = RuntimeBenchmark(
        rows, args.out, args.concurrency, args.timeout,
        arn=os.environ["AGENTCORE_AGENT_ARN"], base_url=base_url, model=args.model,
    )  # fmt: skip
    asyncio.run(benchmark.run())


if __name__ == "__main__":
    main()
