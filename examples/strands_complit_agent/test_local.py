"""Run the CompLit agent locally, without AgentCore Runtime: the entrypoint is called in this process.

Needs the environment service (env/serve.py) at COMPLIT_ENV_URL and AWS credentials for the Bedrock model.

    python test_local.py --split smoke2 --model_id us.anthropic.claude-opus-5 \
        --bedrock_fields '{"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}'
"""

import argparse
import json

from rl_app import invoke_agent
from tasks import load_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="smoke2")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--base_url", help="OpenAI-compatible endpoint; without it, model_id is a Bedrock model")
    parser.add_argument("--bedrock_fields", type=json.loads, help="additionalModelRequestFields of a Bedrock model")
    parser.add_argument("--simulator_model_id", help="Cumulative: the simulated researcher's Bedrock model")
    args = parser.parse_args()
    config = {"model_id": args.model_id, "base_url": args.base_url, "bedrock_fields": args.bedrock_fields}
    if args.simulator_model_id:
        config["simulator_model_id"] = args.simulator_model_id
    for task in load_split(args.split):
        result = invoke_agent({**task, "_config": config})
        print(
            json.dumps(
                {
                    "task_id": task["task_id"],
                    "gold": task["gold"],
                    **{k: result[k] for k in ("reward", "stop", "forced", "turns")},
                }
            )
        )


if __name__ == "__main__":
    main()
