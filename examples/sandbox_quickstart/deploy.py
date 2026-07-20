"""Deploy the sandbox image as an AgentCore runtime.

Reads config.toml (copy config.example.toml and fill in your values), creates
or updates the runtime, waits for the endpoint to be ready, and prints the
runtime ARN to export as SANDBOX_RUNTIME_ARN.

NOTE: temporary scaffolding, like build_and_push.sh — a future phase moves
runtime provisioning into the SDK (SandboxClient.create()), at which point
this script goes away.
"""

from pathlib import Path

import tomllib
from bedrock_agentcore_starter_toolkit.services.runtime import BedrockAgentCoreClient


def main():
    config_path = Path(__file__).parent / "config.toml"
    if not config_path.exists():
        raise SystemExit("config.toml not found — copy config.example.toml and fill in your values")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    agentcore_config = config["agentcore"]

    agentcore_client = BedrockAgentCoreClient(region=agentcore_config["region"])

    # The sandbox container needs no env vars — unlike the agent examples,
    # the image only runs the sandboxd contract shim.
    response = agentcore_client.create_agent(
        agent_name=agentcore_config["agent_name"],
        deployment_type="container",
        image_uri=agentcore_config["image_uri"],
        execution_role_arn=agentcore_config["execution_role_arn"],
        network_config={"networkMode": "PUBLIC"},
        env_vars={},
        auto_update_on_conflict=True,
    )

    agent_id = response["id"]
    agent_arn = response["arn"]

    endpoint_response = agentcore_client.wait_for_agent_endpoint_ready(agent_id=agent_id, max_wait=120)
    if agent_arn not in endpoint_response:
        raise TimeoutError(endpoint_response)

    print(f"Sandbox runtime ready: {agent_arn}")
    print(f'\nRun the demo with:\n  SANDBOX_RUNTIME_ARN="{agent_arn}" python run_sandbox.py')


if __name__ == "__main__":
    main()
