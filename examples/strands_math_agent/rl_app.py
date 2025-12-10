from dotenv import load_dotenv
from reward import GSM8KReward
from strands import Agent
from strands_tools import calculator

from agentcore_rl_toolkit import StrandsAgentCoreRLApp, StrandsRolloutCollector

app = StrandsAgentCoreRLApp()

load_dotenv()

model = app.create_openai_compatible_model()

rollout_collector = StrandsRolloutCollector()
agent = Agent(
    model=model,
    tools=[calculator],
    system_prompt=(
        "Your task is to solve the math problem. "
        + "Use the calculator tool to compute all mathematical expressions. "
        + 'Let\'s think step by step and output the final answer after "####".'
    ),
    hooks=[rollout_collector],
)
reward_fn = GSM8KReward()


@app.rollout_entrypoint
async def invoke_agent(payload, context):
    """
    Invoke the math agent with a payload using the rollout_entrypoint decorator.

    For RL training, the following fields are expected:
    - prompt: question from gsm8k
    - answer: ground truth (str)

    The @rollout_entrypoint decorator automatically:
    - Handles asyncio.create_task() for non-blocking execution
    - Saves rollout data to S3 and notifies SQS when returned
    - Handles errors and saves error rollouts for client notification
    - Works with both sync and async functions
    """
    user_input = payload.get("prompt")
    answer = payload.get("answer")  # used for computing reward

    print("User input:", user_input)

    # Hooks auto collecting rollout data while agent is running
    response = await agent.invoke_async(user_input)

    # Gather rollouts from the collector
    rollout_data = rollout_collector.get_rollout_data()

    # Compute rewards
    rewards = reward_fn(response_text=response.message["content"][0]["text"], ground_truth=answer)

    # Return expected structure (dict with `rollout_data` and `rewards` keys)
    # Framework validates and normalizes values automatically
    return {"rollout_data": rollout_data, "rewards": rewards}


if __name__ == "__main__":
    app.run()
