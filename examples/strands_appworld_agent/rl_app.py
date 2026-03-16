import logging

import requests
from appworld.environment import AppWorld, AppWorldServers
from mcp.client.streamable_http import streamable_http_client
from reward import AppWorldReward
from strands import Agent, tool
from strands.tools.mcp import MCPClient

from agentcore_rl_toolkit import AgentCoreRLApp
from agentcore_rl_toolkit.frameworks.strands.vllm_model import vLLMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = AgentCoreRLApp()
reward_fn = AppWorldReward()

SYSTEM_PROMPT = """You are an AI assistant helping {supervisor_name} ({supervisor_email}).
The current date and time is {datetime}.

You have access to the following apps: {allowed_apps}.

Instructions:
- Use the available tool functions to interact with the apps.
- Tool names follow the pattern: appname__action_name (e.g., amazon__search_products).
- Many actions require authentication. Login first using the app's login function.
- You can also use execute_python to run Python code for complex data manipulation.
- When you have completed the task, stop calling tools and summarize what you did.
- Be thorough: verify your actions succeeded by checking responses."""


@app.rollout_entrypoint
def invoke_agent(payload: dict):
    rollout_config = payload.get("_rollout", {})

    model = vLLMModel(
        client_args={"api_key": "EMPTY", "base_url": rollout_config["base_url"]},
        model_id=rollout_config["model_id"],
        params=rollout_config.get("sampling_params", {}),
    )

    # Start AppWorld servers (once per container, each ACR container handles one session)
    servers = AppWorldServers(
        remote_apis_port="{port}",
        remote_environment_port="{port}",
        remote_mcp_port="{port}",
        mcp_server_kwargs={"output_type": "content_only"},
        show_server_logs=False,
        timeout=120,
    )
    servers.start()
    server_defaults = servers.defaults

    task_id = payload["task_id"]
    env_url = server_defaults["remote_environment_url"]

    @tool
    def execute_python(code: str) -> str:
        """Execute Python code in AppWorld's sandbox environment.

        The sandbox has a pre-loaded `apis` object for calling app APIs directly:
          apis.amazon.search_products(query="laptop", ...)
          apis.spotify.login(email="...", password="...")
          apis.gmail.send_email(...)

        Standard libraries available: json, re, math, datetime, pendulum, etc.
        Use print() to see output. Variables persist across calls.

        Args:
            code: Python code to execute.
        """
        resp = requests.post(f"{env_url}/execute", json={"task_id": task_id, "code": code})
        if resp.status_code != 200:
            return f"Error: {resp.text}"
        return resp.json()["output"]

    # Initialize AppWorld for this task
    # AppWorld cleans up the output directory on init, so a fixed name is fine
    # (each ACR container handles exactly one session/task)
    with AppWorld(
        task_id=task_id,
        experiment_name="rollout",
        remote_apis_url=server_defaults["remote_apis_url"],
        remote_environment_url=server_defaults["remote_environment_url"],
        remote_mcp_url=server_defaults["remote_mcp_url"],
    ) as world:
        task = world.task
        supervisor = task.supervisor
        system_prompt = SYSTEM_PROMPT.format(
            supervisor_name=f"{supervisor['first_name']} {supervisor['last_name']}",
            supervisor_email=supervisor["email"],
            datetime=task.datetime.strftime("%Y-%m-%d %H:%M:%S"),
            allowed_apps=", ".join(task.allowed_apps),
        )

        mcp_url = server_defaults["remote_mcp_url"]
        mcp_client = MCPClient(lambda: streamable_http_client(f"{mcp_url}/mcp"))

        with mcp_client:
            mcp_tools = mcp_client.list_tools_sync()
            agent = Agent(
                model=model,
                tools=[*mcp_tools, execute_python],
                system_prompt=system_prompt,
            )
            _response = agent(task.instruction)

        # Save state (required after MCP interaction to persist in-memory DB)
        world.save()
        # Evaluate
        test_tracker = world.evaluate()

    rollout_data = model.get_token_data()
    reward = reward_fn(test_tracker=test_tracker)

    return {"rollout_data": rollout_data, "rewards": reward}


if __name__ == "__main__":
    app.run()
