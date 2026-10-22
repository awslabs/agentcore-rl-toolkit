import argparse
import asyncio
import json
import subprocess
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import requests

from agentcore_rl_toolkit import AgentCoreHttpClient


def build_payload(task_file: str, base_url: str, model_id: str) -> dict:
    """Build the ACR invocation payload from a raw tau2-bench task file."""
    with open(task_file) as f:
        task = json.load(f)

    domain = task["user_scenario"]["instructions"]["domain"]
    return {
        "_task": {
            "domain": domain,
            "user_scenario": task["user_scenario"],
            "initial_state": task.get("initial_state"),
            "evaluation_criteria": task.get("evaluation_criteria"),
        },
        "_config": {
            "base_url": base_url,
            "model_id": model_id,
            "sampling_params": {
                "max_tokens": 4096,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            },
        },
    }


def run_local(payload: dict) -> None:
    """POST to a local rl_app.py server (localhost:8080)."""
    resp = requests.post("http://localhost:8080/invocations", json=payload)
    print(resp.status_code)
    print(json.dumps(resp.json(), indent=2))


async def run_acr(payload: dict, agent_arn: str) -> None:
    """Invoke a deployed ACR agent with its task payload.

    The host part of base_url is rewritten to the cluster's primary IP so the
    deployed container can reach the vLLM server. Port and path are preserved.
    """
    cluster_ip = subprocess.check_output("hostname -I | awk '{print $1}'", shell=True, text=True).strip()
    parsed = urlparse(payload["_config"]["base_url"])
    new_netloc = f"{cluster_ip}:{parsed.port}" if parsed.port else cluster_ip
    payload["_config"]["base_url"] = urlunparse(parsed._replace(netloc=new_netloc))

    session_id = str(uuid4())
    async with AgentCoreHttpClient(agent_arn) as client:
        try:
            result = await client.invoke(payload, session_id=session_id)
            print(json.dumps(result, indent=2))
        finally:
            await client.stop_session(session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Path to a tau2-bench task JSON (e.g. tasks/retail_example.json)")
    parser.add_argument(
        "--base-url", required=True, help="vLLM server URL for the assistant model (e.g. http://localhost:4000/v1)"
    )
    parser.add_argument(
        "--model-id", required=True, help="Assistant model id (must match the vLLM --served-model-name)"
    )
    parser.add_argument("--acr", action="store_true", help="Invoke the deployed ACR agent instead of a local server")
    parser.add_argument("--agent-arn", help="Runtime ARN (required with --acr)")
    args = parser.parse_args()

    payload = build_payload(args.task, args.base_url, args.model_id)
    if args.acr:
        if not args.agent_arn:
            parser.error("--agent-arn is required with --acr")
        asyncio.run(run_acr(payload, args.agent_arn))
    else:
        run_local(payload)


if __name__ == "__main__":
    main()
