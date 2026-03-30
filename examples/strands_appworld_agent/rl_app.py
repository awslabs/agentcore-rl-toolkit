import logging

from appworld.environment import AppWorld, AppWorldServers
from few_shot_example import build_example_messages
from reward import AppWorldReward
from strands import Agent, tool
from strands.models.openai import OpenAIModel

from agentcore_rl_toolkit import AgentCoreRLApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = AgentCoreRLApp()
reward_fn = AppWorldReward()

MAX_TOOL_OUTPUT_LENGTH = 12000

# Create system prompt and key instructions following
# https://github.com/StonyBrookNLP/appworld/blob/main/experiments/prompts/react_code_agent/instructions.txt

# ────── System prompt──────
SYSTEM_PROMPT = """\
I am your supervisor, and you are an AI Assistant whose job is to complete my \
day-to-day tasks fully autonomously.

To do this, you will need to interact with app(s) (e.g., spotify, venmo etc) \
using their associated APIs on my behalf. For this you will undertake a \
*multi-step conversation* using a python REPL environment. That is, you will \
write python code using the `execute` tool, the environment will execute it \
and show you the result, based on which, you will write python code for the \
next step and so on, until you've achieved the goal.

Here are three key APIs that you need to know to get more information:

1. List all available apps:
   print(apis.api_docs.show_app_descriptions())

2. List APIs for a specific app, e.g. spotify:
   print(apis.api_docs.show_api_descriptions(app_name='spotify'))

3. Get the specification of a particular API, e.g. spotify login:
   print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))

Each code execution will produce an output that you can use in subsequent calls."""

# ────── Key Instructions (included in user message) ──────
KEY_INSTRUCTIONS = """\
The above conversation you see is a demonstration example. Now start doing the real task.

**Key instructions**:

A. General instructions:
- Act fully on your own. You must make all decisions yourself and never ask \
me or anyone else to confirm or clarify.
- Never invent or guess values. For example, if I ask you to play a song, \
do not assume the ID is 123. Instead, look it up properly through the right API.
- Never leave placeholders; don't output things like "your_username". Always \
fill in the real value by retrieving it via APIs (e.g., Supervisor app for \
credentials).
- Avoid collateral damage. Only perform what I explicitly ask for.

B. App-specific instructions:
- All my personal information (credentials, addresses, cards) is stored in \
the Supervisor app, accessible via its APIs.
- Any reference to my friends, family or any other person refers to the \
people in my phone's contacts list.
- Paginated APIs: Always process all results, looping through the page_index. \
Don't stop at the first page.

C. Code-operation instructions:
- Remember you can use the variables in your code in subsequent calls.
- Remember that the email addresses, access tokens and variables in the \
demonstration example above are not valid anymore.
- Always look at API specifications (using apis.api_docs.show_api_doc) \
before calling an API.
- Write small chunks of code and only one chunk of code in every step. \
Make sure everything is working correctly before making any irreversible changes.
- The provided API documentation has both the input arguments and the output \
JSON format. Use this information when making API calls and parsing their outputs.

D. Task-completion instructions:
- You must call `apis.supervisor.complete_task()` after completing the task.
- If an answer is needed, call it with the appropriate answer argument value.
- If no answer is required, omit the answer argument (or set it to None).
- Keep answers minimal. Return only the entity, number, or direct value \
requested - not full sentences.
- Numbers must be numeric and not in words (e.g., return "10", not "ten")."""


# ── Entrypoint ───────────────────────────────────────────────────────────────


@app.rollout_entrypoint
def invoke_agent(payload: dict):
    rollout_config = payload.get("_rollout", {})

    model = OpenAIModel(
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
                except Exception as e:
                    output = f"Error: {type(e).__name__}: {e}"
                if len(output) > MAX_TOOL_OUTPUT_LENGTH:
                    output = output[: MAX_TOOL_OUTPUT_LENGTH - len("... (truncated)")] + "... (truncated)"
                if world.task_completed():
                    output += "\n\n[TASK COMPLETED] Stop and do not call any more tools."
                return output

            # Build user message with task context
            task = world.task
            supervisor = task.supervisor
            user_message = (
                f"{KEY_INSTRUCTIONS}\n\n"
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
                messages=build_example_messages(),
            )

            response = agent(user_message)
            logger.info(f"Agent response: {response.message['content'][0]['text']}")

            # Save state and evaluate
            world.save()
            test_tracker = world.evaluate()

    reward = reward_fn(test_tracker=test_tracker)

    return {"rewards": reward}


if __name__ == "__main__":
    app.run()
