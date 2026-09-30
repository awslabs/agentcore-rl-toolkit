import copy
import logging
from dataclasses import dataclass
from typing import Any, Optional

from constants import (
    ASSISTANT_INSTRUCTION,
    ASSISTANT_MODEL_DEFAULTS,
    ASSISTANT_SYSTEM_PROMPT,
    MALFORMED_TOOL_CALL_TAGS,
    ORCHESTRATOR_CONFIG,
)
from reward import TauBenchReward
from strands.models.openai import OpenAIModel
from tau2.data_model.tasks import EnvFunctionCall
from tau2.domains.airline.environment import get_environment as airline_env
from tau2.domains.retail.environment import get_environment as retail_env
from tau2.domains.telecom.environment import get_environment_workflow_policy as telecom_env
from user_simulator import TauBenchUserSimulator, UserSimulator
from utils import extract_text, make_strands_tool, run_agent_turn, to_real_world_roles

from agentcore_rl_toolkit import AgentCoreRLApp

GET_ENVIRONMENT_FUNC = {
    "airline": airline_env,
    "retail": retail_env,
    "telecom": telecom_env,
}

logger = logging.getLogger(__name__)

app = AgentCoreRLApp()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


@dataclass
class RolloutContext:
    """Everything assembled from the payload that the rollout needs."""

    base_url: str
    model_id: str
    params: dict
    task: dict
    domain: str
    assistant_model: Any
    user_simulator: UserSimulator
    env: Any
    assistant_tools: list
    assistant_system_prompt: str


def _setup_rollout(payload: dict) -> RolloutContext:
    """Assemble everything a rollout needs from the request payload.

    Parses model/task config, instantiates the assistant (vLLM) model, builds a
    fresh environment and replays its initialization actions, wraps tau-bench
    assistant tools as Strands tools, renders the assistant system prompt, and
    builds the user simulator (Bedrock), reset onto this task and environment.

    Args:
        payload: The rollout request, with a "_rollout" block (base_url, model_id,
            optional sampling_params) and a "_task" block (domain, user_scenario,
            initial_state, evaluation_criteria, ...).

    Returns:
        A populated ``RolloutContext``.
    """
    base_url = payload["_rollout"]["base_url"]
    model_id = payload["_rollout"]["model_id"]
    # Copy so we don't mutate the caller's payload while applying defaults.
    params = copy.deepcopy(payload["_rollout"].get("sampling_params", {}))
    for k, v in ASSISTANT_MODEL_DEFAULTS.items():
        if k == "extra_body":
            params.setdefault("extra_body", {})
            for ek, ev in v.items():
                params["extra_body"].setdefault(ek, ev)
        else:
            params.setdefault(k, v)

    task = payload["_task"]
    domain = task["domain"]
    api_key = payload["_rollout"].get("api_key") or "EMPTY"

    # Assistant model (RL-trained, served via vLLM)
    assistant_model = OpenAIModel(
        client_args={"api_key": api_key, "base_url": base_url},
        model_id=model_id,
        params=params,
    )

    # Fresh environment + initialize state
    env = GET_ENVIRONMENT_FUNC[domain]()
    initial_state = task.get("initial_state") or {}
    for action in initial_state.get("initialization_actions") or []:
        env.run_env_function_call(EnvFunctionCall(**action))

    # Wrap tau-bench assistant tools as Strands tools
    assistant_tools = [make_strands_tool(env, t.name, t, "assistant") for t in env.get_tools()]

    # Assistant system prompt
    assistant_system_prompt = ASSISTANT_SYSTEM_PROMPT.format(
        assistant_instruction=ASSISTANT_INSTRUCTION,
        assistant_policy=env.get_policy(),
    )

    # User simulator (Bedrock); reset binds its prompt and user tools to this task and env
    user_simulator = TauBenchUserSimulator()
    user_simulator.reset(task, env)

    return RolloutContext(
        base_url=base_url,
        model_id=model_id,
        params=params,
        task=task,
        domain=domain,
        assistant_model=assistant_model,
        user_simulator=user_simulator,
        env=env,
        assistant_tools=assistant_tools,
        assistant_system_prompt=assistant_system_prompt,
    )


def _run_conversation(ctx: RolloutContext) -> tuple[list[dict], Optional[float], Optional[str]]:
    """Drive the multi-turn user/assistant conversation to completion.

    Alternates user and assistant turns until the user emits a termination
    keyword, the assistant leaks a malformed tool-call tag, or ``max_turns`` is
    reached.

    Args:
        ctx: The rollout context holding the assistant model, tools, system
            prompt, and the user simulator.

    Returns:
        A tuple ``(global_messages, force_reward, terminated_reason)``:
          - global_messages: the full shared history (tagged with "from").
          - force_reward: a forced reward (e.g. 0.0 on malformed tool call), or
            None when the reward should be computed normally.
          - terminated_reason: why the loop ended (termination keyword,
            "malformed_tool_call"), or None if it ran to max_turns.

    Raises:
        Exception: propagates any error from the assistant agent so the caller
            can save a partial rollout with reward 0.
    """
    global_messages = [{"role": "assistant", "content": [{"text": "Hello! How can I help you?"}], "from": "assistant"}]
    force_reward = None
    terminated_reason = None

    for turn in range(ORCHESTRATOR_CONFIG.get("max_turns", 50)):
        # --- User turn ---
        user_turn = ctx.user_simulator.step(global_messages)
        global_messages.extend(user_turn.messages)
        if user_turn.terminated:
            terminated_reason = user_turn.termination_reason
            logger.info("Terminated at turn %d: %s", turn, terminated_reason)
            break

        # --- Assistant turn ---
        try:
            _, assistant_messages = run_agent_turn(
                turn,
                "assistant",
                global_messages,
                ctx.assistant_model,
                ctx.assistant_tools,
                ctx.assistant_system_prompt,
            )
        except Exception as e:
            logger.warning("Agent failed (%s), partial rollout, reward=0", e)
            raise
        global_messages.extend(assistant_messages)

        # Check for malformed tool-call tags leaking into the assistant's text
        combined_text = extract_text(global_messages[-1]).strip()
        leaked_tag = next((tag for tag in MALFORMED_TOOL_CALL_TAGS if tag in combined_text), None)
        if leaked_tag:
            terminated_reason = "malformed_tool_call"
            force_reward = 0.0
            logger.warning("Terminated at turn %d: malformed %s leaked into text, forcing reward=0", turn, leaked_tag)
            break

    return global_messages, force_reward, terminated_reason


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@app.rollout_entrypoint
def invoke_agent(payload: dict) -> dict:
    """Run one tau2-bench rollout and return its results.

    Sets up the rollout, drives the user/assistant conversation, computes the
    reward, and returns the reward, conversation, and metadata. Used as the
    ``@rollout_entrypoint``, which automatically runs this in the background,
    saves the returned dict to S3, and records errors for the client.

    Token IDs are captured server-side by the rllm-model-gateway HTTP proxy
    (no client-side token collection in agent code).

    Args:
        payload: The rollout request (see ``_setup_rollout`` for the schema).

    Returns:
        A JSON-serializable dict with keys: "rewards", "reward_info",
        "messages", and "meta". On assistant failure, a minimal
        ``{"rewards": 0.0}`` partial result.
    """
    ctx = _setup_rollout(payload)
    logger.info("Starting rollout: domain=%s, model=%s", ctx.domain, ctx.model_id)

    # Run conversation — if the rollout throws, return reward=0
    # (token IDs are captured server-side by the rllm-model-gateway, not collected here)
    try:
        global_messages, force_reward, terminated_reason = _run_conversation(ctx)
    except Exception:
        logger.exception("Rollout failed before completion")
        return {"rewards": 0.0}

    # Compute reward
    if force_reward is not None:
        rewards = force_reward
        reward_info = {"forced": True, "terminated_reason": terminated_reason}
    else:
        reward_fn = TauBenchReward(GET_ENVIRONMENT_FUNC)
        rewards, reward_info = reward_fn(env=ctx.env, task=ctx.task, domain=ctx.domain, messages=global_messages)
    logger.info(
        "Rollout complete: %d messages, reward=%.4f, terminated_reason=%s",
        len(global_messages),
        rewards,
        terminated_reason,
    )

    return {
        "rewards": rewards,
        "reward_info": reward_info,
        "messages": to_real_world_roles(global_messages),
        "meta": {
            "assistant_model": {
                "base_url": ctx.base_url,
                "model_id": ctx.model_id,
                "params": ctx.params,
            },
            "user_model": ctx.user_simulator.describe(),
            "terminated_reason": terminated_reason,
        },
    }


if __name__ == "__main__":
    app.run()
