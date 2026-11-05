"""CompLit: find the one arXiv paper a researcher describes, or say that none qualifies.

Upstream's setup with web search off: no system prompt; upstream's prompts; tools arxiv_search and read_paper
(served by env/serve.py); 8K output tokens per turn, 128K context. The reward is upstream's deterministic rule. An
unavailable environment fails the rollout instead of reaching the model.

- Single turn (Standard, Implicit, Unmet): 15 turns (upstream's limit of 30 messages). When the turns or the
  context run out, upstream's forced-answer prompt gets one more turn.
- Cumulative: the requirements arrive over 3-5 researcher turns from a simulated researcher (simulator.py), as in
  upstream's complit/cumulative.py. Per researcher turn the agent gets 10 tool calls and 15 turns; a turn without
  a text reply gets upstream's nudge for one more turn. Tool calls and results are dropped between researcher
  turns. The answer is the last reply that holds the answer JSON.
"""

import logging

import simulator as sim
from models import InvocationRequest
from prompts import CONTRACT, FORCED_ANSWER, TURN_NUDGE, build_prompt, researcher_turn
from reward import CompLitReward, parse_answer
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import BeforeModelCallEvent, HookProvider
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from tools import make_tools

from agentcore_rl_toolkit import AgentCoreRuntimeApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = AgentCoreRuntimeApp()
reward_fn = CompLitReward()

MAX_TURNS = 15
MAX_TOKENS = 8192
MAX_CONTEXT = 131072  # prompt + maximum output, the vLLM rule
FORCE_MARGIN = 512  # output tokens a forced reply gives up, so its longer prompt still fits
TOOLS_PER_RESEARCHER_TURN = 10  # Cumulative
SIMULATOR_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # Cumulative, at temperature 0; upstream: gpt-5.4-mini


class _Guard(HookProvider):
    """Stops the episode before a model call that would not fit the context, or after an infrastructure error.
    Records the history length of the last call that fit, for the forced reply."""

    def __init__(self, state: dict):
        self.state = state

    def register_hooks(self, registry, **_):
        registry.add_callback(BeforeModelCallEvent, self._before)

    def _before(self, event):
        if "infra_error" in self.state:
            event.cancel = True
        elif (event.projected_input_tokens or 0) + self.state.get("max_tokens", MAX_TOKENS) > MAX_CONTEXT:
            self.state["context_exceeded"] = True
            event.cancel = True
        else:
            self.state["fit_len"] = len(event.agent.messages)


def make_model(config: dict):
    """An OpenAI-compatible endpoint (training: the rollout gateway; vLLM) or, without base_url, a Bedrock model."""
    sampling = config.get("sampling_params", {})
    if config.get("base_url"):
        return OpenAIModel(
            client_args={"api_key": config.get("api_key") or "EMPTY", "base_url": config["base_url"]},
            model_id=config["model_id"],
            params={"max_tokens": MAX_TOKENS, **sampling},
        )
    return BedrockModel(
        model_id=config["model_id"],
        region_name=config.get("region", "us-west-2"),
        max_tokens=MAX_TOKENS,
        additional_request_fields=config.get("bedrock_fields"),
        **sampling,
    )


def set_max_tokens(agent: Agent, state: dict, n: int):
    state["max_tokens"] = n
    if isinstance(agent.model, OpenAIModel):
        agent.model.update_config(params={**agent.model.get_config().get("params", {}), "max_tokens": n})
    else:
        agent.model.update_config(max_tokens=n)


def run(agent: Agent, state: dict, prompt: str, turns: int) -> str:
    """The agent on `prompt` for up to `turns` turns -> completed | max_turns | context_exceeded | refusal. After
    context_exceeded, the history is back at the last call that fit."""
    result = agent(prompt, limits={"turns": turns})
    if state.pop("context_exceeded", False):
        del agent.messages[state["fit_len"] :]
        return "context_exceeded"
    # content_filtered: the model refused (e.g. Claude on some epidemic-modeling requests): no answer
    return {"limit_turns": "max_turns", "content_filtered": "refusal"}.get(result.stop_reason, "completed")


def force_reply(agent: Agent, state: dict, prompt: str):
    """`prompt`, then one more turn with the tools attached, on the full history or, if that does not fit, on the
    last history that fit. The prompt joins trailing tool results, so roles keep alternating."""
    set_max_tokens(agent, state, MAX_TOKENS - FORCE_MARGIN)
    for _ in range(2):
        n = len(agent.messages)
        last = agent.messages[-1]
        if last["role"] == "user":
            last["content"].append({"text": prompt})
            agent(None, limits={"turns": 1})
        else:
            agent(prompt, limits={"turns": 1})
        if not state.pop("context_exceeded", False):
            break
        del agent.messages[state["fit_len"] :]
        if state["fit_len"] >= n:  # no older history to fall back to
            break
    set_max_tokens(agent, state, MAX_TOKENS)


def text_of(message: dict) -> str:
    return "".join(b["text"] for b in message.get("content") or [] if "text" in b).strip()


def last_text(messages: list) -> str:
    """The text of the last assistant message that has any."""
    return next((t for m in reversed(messages) if m["role"] == "assistant" and (t := text_of(m))), "")


def single_turn(agent: Agent, state: dict, request: InvocationRequest) -> tuple[str, str, bool]:
    stop = run(agent, state, build_prompt(request.question, use_read_paper=True, use_web_search=False), MAX_TURNS)
    forced = stop in ("max_turns", "context_exceeded") and "infra_error" not in state
    if forced:
        force_reply(agent, state, FORCED_ANSWER)
    final = next((m for m in reversed(agent.messages) if m["role"] == "assistant"), {})
    return text_of(final), stop, forced


def conversation(agent: Agent, state: dict, request: InvocationRequest) -> tuple[str, str, bool]:
    """Upstream's Cumulative loop (complit/cumulative.py)."""
    client = sim.SimModel(request.config.get("simulator_model_id") or SIMULATOR_MODEL, "us-west-2")
    constraints = sim.canonical_constraints({"constraints": [dict(c) for c in request.constraints]})
    for c in constraints:
        c.setdefault("rendered_value", c.get("value"))
    n_turns = len(request.turns)
    sim_msgs, ledger, replies, stops, forced_any = [], [], [], [], False
    for turn, spec in enumerate(request.turns, 1):
        # 1. the researcher speaks: the prepared utterance, or the simulator's delivery of it
        picked = [constraints[i] for i in spec.revealed if 0 <= i < len(constraints)]
        ref = spec.utterance.strip()
        last_agent = sim_msgs[-1]["content"] if sim_msgs else ""
        if ref and not sim.ADAPTER_RESPONSE_RE.search(last_agent):
            user_msg, rev = ref, list(range(len(picked)))
        else:
            facts = [v for _t, v in ledger] + [sim.fact_line(c) for c in picked]
            turn_sys = sim.SIM_SYS_SCHEDULED_NEED.format(
                intention=request.intention,
                all_facts="\n".join(facts) or "(nothing specific yet -- only the broad context of your work)",
            )
            instruction = sim.adapter_instruction(ref, turn == n_turns)
            user_msg, rev = sim.naturalize_turn(client, turn_sys, sim_msgs, instruction, picked)
            user_msg = user_msg or ref  # a simulator that returns nothing sends the prepared text
        sim_msgs.append({"role": "assistant", "content": user_msg})
        ledger += [(turn, sim.fact_line(picked[i])) for i in rev]

        # 2. the agent replies: 10 tool calls and 15 turns; no text gets upstream's nudge, with no tool calls left
        start = len(agent.messages)
        state["tools_left"] = TOOLS_PER_RESEARCHER_TURN
        body = researcher_turn(user_msg, turn=turn, n_turns=n_turns, use_web_search=False)
        stops.append(run(agent, state, body, MAX_TURNS))
        reply = last_text(agent.messages[start:])
        if not reply and stops[-1] != "refusal" and "infra_error" not in state:
            state["tools_left"] = 0
            nudge = TURN_NUDGE + ("\n\n" + CONTRACT if turn == n_turns else "")
            force_reply(agent, state, nudge)
            reply = last_text(agent.messages[start:])
            forced_any |= bool(reply)
        if "infra_error" in state:
            break
        reply = reply or "<no reply>"
        sim_msgs.append({"role": "user", "content": reply})
        replies.append(reply)

        # the next researcher turn starts from the conversation only: no tool calls, tool results or nudges
        agent.messages[:] = merge_roles(
            [
                {**m, "content": [b for b in m["content"] if "text" in b and not b["text"].startswith(TURN_NUDGE)]}
                for m in agent.messages
                if not any("toolUse" in b for b in m["content"])
            ]
        )
    stop = "refusal" if "refusal" in stops else "completed"
    answer = next((r for r in reversed(replies) if parse_answer(r)), replies[-1] if replies else "")
    return answer, stop, forced_any


def merge_roles(messages: list) -> list:
    """Drop empty messages and merge consecutive messages of one role."""
    out = []
    for m in messages:
        if not m["content"]:
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {**out[-1], "content": out[-1]["content"] + m["content"]}
        else:
            out.append(m)
    return out


@app.entrypoint
def invoke_agent(payload: dict):
    request = InvocationRequest(**payload)
    state = {}
    agent = Agent(
        model=make_model(request.config),
        tools=make_tools(state),
        hooks=[_Guard(state)],
        conversation_manager=NullConversationManager(),
        callback_handler=None,
    )
    response_text, stop, forced = (conversation if request.turns else single_turn)(agent, state, request)
    if "infra_error" in state:
        raise RuntimeError(f"infrastructure error: {state['infra_error']}")

    reward = reward_fn(response_text=response_text, gold=request.gold, answerable=request.answerable)
    logger.info("stop=%s forced=%s reward=%s", stop, forced, reward)
    return {
        "reward": reward,
        "stop": stop,
        "forced": forced,
        "response": response_text,
        "turns": sum(m["role"] == "assistant" for m in agent.messages),
    }


if __name__ == "__main__":
    app.run()
