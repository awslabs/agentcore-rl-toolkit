import argparse
import json
import subprocess
from urllib.parse import urlparse, urlunparse

import requests


def build_payload(task_file: str, base_url: str, model_id: str, s3_bucket: str) -> dict:
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
        "_rollout": {
            "exp_id": "test",
            "s3_bucket": s3_bucket,
            "session_id": "session_123",
            "input_id": "task_123",
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


def run_acr(payload: dict, agent_name: str) -> None:
    """Invoke a deployed ACR agent via the agentcore CLI.

    The host part of base_url is rewritten to the cluster's primary IP so the
    deployed container can reach the vLLM server. Port and path are preserved.
    """
    cluster_ip = subprocess.check_output("hostname -I | awk '{print $1}'", shell=True, text=True).strip()
    parsed = urlparse(payload["_rollout"]["base_url"])
    new_netloc = f"{cluster_ip}:{parsed.port}" if parsed.port else cluster_ip
    payload["_rollout"]["base_url"] = urlunparse(parsed._replace(netloc=new_netloc))

    result = subprocess.run(
        ["agentcore", "invoke", "--agent", agent_name, json.dumps(payload)],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Path to a tau2-bench task JSON (e.g. tasks/retail_example.json)")
    parser.add_argument(
        "--base-url", required=True, help="vLLM server URL for the assistant model (e.g. http://localhost:4000/v1)"
    )
    parser.add_argument(
        "--model-id", required=True, help="Assistant model id (must match the vLLM --served-model-name)"
    )
    parser.add_argument(
        "--s3-bucket", default="agentcore-rl", help="S3 bucket for rollout results (default: agentcore-rl)"
    )
    parser.add_argument("--acr", action="store_true", help="Invoke the deployed ACR agent instead of a local server")
    parser.add_argument("--agent-name", default="strands_taubench_agent_rl", help="ACR agent name (used with --acr)")
    args = parser.parse_args()

    payload = build_payload(args.task, args.base_url, args.model_id, args.s3_bucket)
    if args.acr:
        run_acr(payload, args.agent_name)
    else:
        run_local(payload)


if __name__ == "__main__":
    main()
