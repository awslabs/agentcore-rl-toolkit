from reward import GSM8KReward
from strands import Agent
from strands_tools import calculator

from agentcore_rl_toolkit import AgentCoreRLApp
from agentcore_rl_toolkit.frameworks.strands.vllm_model import vLLMModel

app = AgentCoreRLApp()

system_prompt = (
    "Your task is to solve the math problem. "
    + "Use the calculator tool to compute all mathematical expressions. "
    + 'Let\'s think step by step and output the final answer after "####".'
)

reward_fn = GSM8KReward()


@app.rollout_entrypoint
def invoke_agent(payload: dict):
    """
    Invoke the math agent with a payload using the rollout_entrypoint decorator.

    For RL training, the following fields are expected:
    - prompt: question from gsm8k
    - answer: ground truth (str)
    - _rollout: rollout config with base_url and model_id

    The @rollout_entrypoint decorator automatically:
    - Executes the function in the background for non-blocking processing
    - Saves rollout data to S3 with a predictable key
    - Handles errors and saves error rollouts for client awareness
    - Works with both sync and async functions
    """
    base_url = payload["_rollout"]["base_url"]
    model_id = payload["_rollout"]["model_id"]

    model = vLLMModel(
        client_args={"api_key": "EMPTY", "base_url": base_url},
        model_id=model_id,
    )

    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt=system_prompt,
    )

    user_input = payload.get("prompt")
    answer = payload.get("answer")  # used for computing reward

    print("User input:", user_input)

    response = agent(user_input)

    # Collect token data (prompt IDs, response IDs, logprobs) from the model
    rollout_data = model.get_token_data()

    # Compute rewards
    rewards = reward_fn(response_text=response.message["content"][0]["text"], ground_truth=answer)

    # Return expected structure (dict with `rollout_data` and `rewards` keys)
    # Framework validates and normalizes values automatically
    return {"rollout_data": rollout_data, "rewards": rewards}


if __name__ == "__main__":
    app.run()
