from pathlib import Path

import tomllib
from bedrock_agentcore_starter_toolkit.services.runtime import BedrockAgentCoreClient
from dotenv import dotenv_values


def main():
    # Load config
    config_path = Path(__file__).parent / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    agentcore_config = config["agentcore"]

    # Load environment variables from .env
    env_path = Path(__file__).parent / ".env"
    env_vars = dotenv_values(env_path)

    # Create the client
    agentcore_client = BedrockAgentCoreClient(region=agentcore_config["region"])

    # Build network config if provided
    network_config = {"networkMode": "PUBLIC"}
    if agentcore_config.get("subnets") and agentcore_config.get("security_groups"):
        network_config = {
            "networkMode": agentcore_config.get("network_mode", "VPC"),
            "networkModeConfig": {
                "subnets": agentcore_config["subnets"],
                "securityGroups": agentcore_config["security_groups"],
            },
        }

    response = agentcore_client.create_agent(
        agent_name=agentcore_config["agent_name"],
        deployment_type="container",
        image_uri=agentcore_config["image_uri"],
        execution_role_arn=agentcore_config["execution_role_arn"],
        network_config=network_config,
        env_vars=env_vars,
        auto_update_on_conflict=True,
    )

    agent_id = response["id"]
    agent_arn = response["arn"]

    # Wait for endpoint to be ready (handles both new deployments and existing ones)
    endpoint_response = agentcore_client.wait_for_agent_endpoint_ready(agent_id=agent_id, max_wait=120)
    if agent_arn not in endpoint_response:
        raise TimeoutError(endpoint_response)

    print(f"Agent endpoint ready: {agent_arn}")


if __name__ == "__main__":
    main()
