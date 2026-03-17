import json
import logging

from appworld.environment import AppWorld, AppWorldServers
from reward import AppWorldReward
from strands import Agent, tool
from strands.agent.conversation_manager import NullConversationManager

from agentcore_rl_toolkit import AgentCoreRLApp
from agentcore_rl_toolkit.frameworks.strands.vllm_model import vLLMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = AgentCoreRLApp()
reward_fn = AppWorldReward()

MAX_TOOL_OUTPUT_LENGTH = 7200

SYSTEM_PROMPT = """\
You are an AI assistant that completes tasks by writing Python code.
You interact with apps (e.g., spotify, venmo, gmail) through their APIs using a Python environment.
In each step, write Python code and the environment will execute it and return the output.
Use print() to see results. Variables persist across calls.

## Discovering APIs

You have three key functions to discover available APIs:

1. List all available apps:
   print(apis.api_docs.show_app_descriptions())

2. List APIs for a specific app:
   print(apis.api_docs.show_api_descriptions(app_name="spotify"))

3. Get full specification of an API (parameters, response schema):
   print(apis.api_docs.show_api_doc(app_name="spotify", api_name="login"))

## Calling APIs

Call APIs directly: apis.<app_name>.<api_name>(**arguments)
Example: apis.spotify.login(username="user@email.com", password="pass123")

## Key instructions

- Use the "supervisor" app to get account credentials (passwords, etc.).
- Use the "phone" app to look up contact information for friends and family.
- Always check API specifications (show_api_doc) before calling an API.
- Write small chunks of code, one chunk per step. Verify results before making irreversible changes.
- Many APIs return paginated results. Loop over page_index to get all results.
- When the task is complete, call apis.supervisor.complete_task().
  If the task asks for information, pass it as: apis.supervisor.complete_task(answer=<answer>).
  If no answer is needed, just call: apis.supervisor.complete_task()"""


@app.rollout_entrypoint
def invoke_agent(payload: dict):
    rollout_config = payload.get("_rollout", {})

    model = vLLMModel(
        client_args={"api_key": "EMPTY", "base_url": rollout_config["base_url"]},
        model_id=rollout_config["model_id"],
        params=rollout_config.get("sampling_params", {}),
    )

    task_id = payload["task_id"]

    # Start AppWorld environment server (embeds APIs internally, same as `appworld serve environment`)
    with AppWorldServers(
        remote_environment_port="{port}",
        show_server_logs=False,
        timeout=120,
    ) as servers:
        server_defaults = servers.defaults

        with AppWorld(
            task_id=task_id,
            experiment_name="rollout",
            remote_environment_url=server_defaults["remote_environment_url"],
        ) as world:
            # Define the execute tool with closure over `world`
            @tool
            def execute(code: str) -> str:
                """Execute Python code in the AppWorld environment.

                The environment has a pre-loaded `apis` object for discovering and calling app APIs.
                Use print() to see output. Variables persist across calls.

                Args:
                    code: Python code to execute.
                """
                try:
                    output = world.execute(code)
                    result = json.dumps(
                        {
                            "output": output,
                            "task_completed": world.task_completed(),
                        }
                    )
                except Exception as e:
                    result = json.dumps(
                        {
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                if len(result) > MAX_TOOL_OUTPUT_LENGTH:
                    result = result[: MAX_TOOL_OUTPUT_LENGTH - len("... (truncated)")] + "... (truncated)"
                return result

            # Build user message with task context
            task = world.task
            supervisor = task.supervisor
            user_message = (
                f"Today's date is: {task.datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"My name is: {supervisor.first_name} {supervisor.last_name}. "
                f"My personal email is {supervisor.email} "
                f"and phone number is {supervisor.phone_number}.\n"
                f"Task: {task.instruction}"
            )

            agent = Agent(
                model=model,
                tools=[execute],
                system_prompt=SYSTEM_PROMPT,
                conversation_manager=NullConversationManager(),
            )

            response = agent(user_message)
            logger.info(f"Agent response: {response.message['content'][0]['text']}")

            # Save state and evaluate
            world.save()
            test_tracker = world.evaluate()

    rollout_data = model.get_token_data()
    reward = reward_fn(test_tracker=test_tracker)

    return {"rollout_data": rollout_data, "rewards": reward}


if __name__ == "__main__":
    app.run()
