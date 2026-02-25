"""
RL agent app with token-level data capture for rLLM integration.

This is an enhanced version of ``rl_app.py`` that uses ``capture_tokens=True``
to call rLLM's ``/v1/model_response`` endpoint instead of the standard
``/v1/chat/completions``.  This captures ``prompt_ids``, ``completion_ids``,
and ``logprobs`` during inference, avoiding lossy retokenization on the
trainer side.

Usage:

    # Set BASE_URL to the rLLM trainer's inference API
    export BASE_URL="http://<trainer-host>:8089/v1"
    export MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
    python rl_app_with_tokens.py

The ``_training`` dict in the payload is injected automatically by rLLM's
``AgentCoreEpisodeCollector`` when dispatching tasks.
"""

from dotenv import load_dotenv
from reward import GSM8KReward
from strands import Agent
from strands_tools import calculator

from agentcore_rl_toolkit import StrandsAgentCoreRLApp, StrandsRolloutCollector

app = StrandsAgentCoreRLApp()

load_dotenv()

# capture_tokens=True uses RLLMRemoteModel which calls /v1/model_response
# and stores prompt_ids, completion_ids, logprobs for each model call.
model = app.create_openai_compatible_model(capture_tokens=True)

rollout_collector = StrandsRolloutCollector()
agent = Agent(
    model=model,
    tools=[calculator],
    system_prompt=(
        "Your task is to solve the math problem. "
        "Use the calculator tool to compute all mathematical expressions. "
        'Let\'s think step by step and output the final answer after "####".'
    ),
    hooks=[rollout_collector],
)
reward_fn = GSM8KReward()


@app.rollout_entrypoint
async def invoke_agent(payload, context):
    """
    Invoke the math agent with token-level data capture.

    The rLLM trainer dispatches tasks to this endpoint via POST /invocations.
    The ``_training`` dict in the payload tells the decorator where to save
    the rollout data (S3 bucket, SQS queue, etc.).

    By using ``capture_tokens=True`` on the model, each model call stores
    the full ModelOutput dict including token IDs and logprobs.  We attach
    these to the rollout data so the rLLM trainer can build Steps with
    ``prompt_ids``, ``response_ids``, and ``logprobs`` directly.
    """
    user_input = payload.get("prompt")
    answer = payload.get("answer")

    print("User input:", user_input)

    # Clear any outputs from a previous invocation
    model.clear_model_outputs()

    # Hooks auto-collect rollout data (message history) while agent is running.
    # The model stores ModelOutput dicts (with token IDs) for each call.
    response = await agent.invoke_async(user_input)

    # Gather rollouts from the collector
    rollout_data = rollout_collector.get_rollout_data()

    # Attach token-level data to each turn
    model_outputs = model.get_model_outputs()
    for turn, output in zip(rollout_data, model_outputs):
        turn["model_output"] = output

    # Compute rewards
    rewards = reward_fn(response_text=response.message["content"][0]["text"], ground_truth=answer)

    return {"rollout_data": rollout_data, "rewards": rewards}


if __name__ == "__main__":
    app.run()
