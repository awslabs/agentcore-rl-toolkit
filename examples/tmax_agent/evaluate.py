"""Run one rollout per prepared task; only invoke the harness runtime."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from agentcore_rl_toolkit.runtime.client import AgentCoreHttpClient


async def evaluate(manifest, base_url, model_id, output, concurrency):
    run_id = str(uuid.uuid4())
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    async with AgentCoreHttpClient(manifest["harness_arn"]) as client:

        async def one(task):
            async with semaphore:
                session_id = str(uuid.uuid4())
                result = {"task_id": task["task_id"], "session_id": session_id}
                try:
                    handle = await client.invoke(
                        {
                            "task": task,
                            "_config": {"base_url": base_url, "model_id": model_id},
                        },
                        session_id=session_id,
                        background=True,
                    )
                    result.update(await handle.result(timeout=task["agent_timeout"] + task["verifier_timeout"] + 300))
                except Exception as exc:
                    result["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    await client.stop_session(session_id)
                results.append(result)
                output.write_text(json.dumps({"run_id": run_id, "results": results}, indent=2) + "\n")
                print(
                    json.dumps({key: result[key] for key in ("task_id", "reward", "error") if key in result}),
                    flush=True,
                )

        await asyncio.gather(*(one(task) for task in manifest["tasks"]))
    completed = [result for result in results if "reward" in result]
    print(
        json.dumps(
            {
                "completed": len(completed),
                "errors": len(results) - len(completed),
                "passed": sum(result["reward"] == 1 for result in completed),
                "mean_reward": sum(result.get("reward", 0) for result in results) / len(results) if results else 0,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-id", required=True, help="Model ID served by the endpoint (name or absolute path)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(
        evaluate(json.loads(args.manifest.read_text()), args.base_url, args.model_id, args.output, args.concurrency)
    )


if __name__ == "__main__":
    main()
