import logging
import os
import sys

import yaml
from appworld_utils import AppWorldExecutor, execute_appworld
from dotenv import load_dotenv
from reward import AppWorldReward
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager

from agentcore_rl_toolkit import AgentCoreRLApp
from agentcore_rl_toolkit.frameworks.strands.vllm_model import vLLMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv()

app = AgentCoreRLApp()

# Load system prompt from YAML at module level
_prompt_path = os.path.join(os.path.dirname(__file__), "appworld_system_prompt.yaml")
with open(_prompt_path) as f:
    system_prompt = yaml.safe_load(f)["system_prompt"]

reward_fn = AppWorldReward()


@app.rollout_entrypoint
def invoke_agent(payload: dict):
    """Invoke the AppWorld agent with a payload using the rollout_entrypoint decorator.

    For RL training, the following fields are expected:
    - prompt: AppWorld task_id (e.g. "test_0")
    - _rollout: rollout config with base_url and model_id
    """
    base_url = payload["_rollout"]["base_url"]
    model_id = payload["_rollout"]["model_id"]
    params = payload["_rollout"].get("sampling_params", {})

    # Create model, executor, and agent per-invocation for concurrency safety
    model = vLLMModel(
        client_args={"api_key": "EMPTY", "base_url": base_url},
        model_id=model_id,
        params=params,
    )

    appworld_executor = AppWorldExecutor()
    execute_tool = appworld_executor.get_execute_tool()

    agent = Agent(
        model=model,
        tools=[execute_tool],
        system_prompt=system_prompt,
        conversation_manager=NullConversationManager(),
    )

    task_id = payload.get("task_id")
    logger.info("Starting AppWorld task: %s", task_id)

    result = execute_appworld(agent, task_id, appworld_executor)

    # Collect token data (prompt IDs, response IDs, logprobs) from the model
    rollout_data = model.get_token_data()

    # Compute rewards
    rewards = reward_fn(evaluation=result)

    logger.info("Task %s completed with reward: %s", task_id, rewards)

    return {"rollout_data": rollout_data, "rewards": rewards}


if __name__ == "__main__":
    app.run()
