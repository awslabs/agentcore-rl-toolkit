"""Contract tests for ``user_simulator``: reset, step, describe, termination, and user tools.

The user model is a scripted Strands ``Model``, so these tests need tau2-bench but no
Bedrock or network access.
"""

import copy
import json
from pathlib import Path

import pytest
from constants import USER_MODEL_CONFIG
from strands.models.model import Model
from tau2.data_model.tasks import EnvFunctionCall
from tau2.domains.airline.environment import get_environment as airline_env
from tau2.domains.retail.environment import get_environment as retail_env
from tau2.domains.telecom.environment import get_environment_workflow_policy as telecom_env
from user_simulator import TauBenchUserSimulator

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"
ENVIRONMENTS = {"airline": airline_env, "retail": retail_env, "telecom": telecom_env}
GREETING = {"role": "assistant", "content": [{"text": "Hello! How can I help you?"}], "from": "assistant"}


def text(value: str) -> dict:
    return {"text": value}


def tool(tool_use_id: str, name: str, **tool_input) -> dict:
    return {"toolUse": {"toolUseId": tool_use_id, "name": name, "input": tool_input}}


class ScriptedModel(Model):
    """Reply with the next scripted content-block list on each call, recording the messages it was sent."""

    def __init__(self, replies: list[list[dict]]):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    def update_config(self, **model_config):
        pass

    def get_config(self) -> dict:
        return {}

    def structured_output(self, *args, **kwargs):
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        content = self.replies.pop(0)
        yield {"messageStart": {"role": "assistant"}}
        for block in content:
            if "text" in block:
                yield {"contentBlockDelta": {"delta": {"text": block["text"]}}}
            else:
                use = block["toolUse"]
                yield {
                    "contentBlockStart": {"start": {"toolUse": {"toolUseId": use["toolUseId"], "name": use["name"]}}}
                }
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(use["input"])}}}}
            yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use" if any("toolUse" in b for b in content) else "end_turn"}}
        yield {
            "metadata": {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}, "metrics": {"latencyMs": 0}}
        }


def reset_simulator(domain: str, replies: list[list[dict]]):
    """Build an episode from the domain's sample task and reset a scripted simulator onto it."""
    raw = json.loads((TASKS_DIR / f"{domain}_example.json").read_text())
    task = {"domain": domain, "user_scenario": raw["user_scenario"], "initial_state": raw.get("initial_state")}
    env = ENVIRONMENTS[domain]()
    for action in (task["initial_state"] or {}).get("initialization_actions") or []:
        env.run_env_function_call(EnvFunctionCall(**action))
    model = ScriptedModel(replies)
    simulator = TauBenchUserSimulator(model=model)
    simulator.reset(task, env)
    return simulator, model, env, task


def test_step_before_reset_raises():
    simulator = TauBenchUserSimulator(model=ScriptedModel([]))
    with pytest.raises(RuntimeError, match="reset"):
        simulator.step([GREETING])


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_reset_renders_the_scenario_and_binds_user_tools_only_for_telecom(domain):
    simulator, _, env, task = reset_simulator(domain, [])
    assert json.dumps(task["user_scenario"]) in simulator.system_prompt
    expected_tools = [t.name for t in env.get_user_tools()] if domain == "telecom" else []
    assert [t.tool_name for t in simulator.tools] == expected_tools


def test_step_returns_the_turn_without_mutating_the_conversation():
    simulator, _, _, _ = reset_simulator("airline", [[text("I need to cancel a reservation.")]])
    conversation = [GREETING]
    before = copy.deepcopy(conversation)
    turn = simulator.step(conversation)
    assert conversation == before
    assert [(m["role"], m["from"]) for m in turn.messages] == [("assistant", "user")]
    assert turn.text == "I need to cancel a reservation."
    assert not turn.terminated
    assert turn.termination_reason is None


def test_user_sees_assistant_text_but_not_its_tool_calls_or_reasoning():
    simulator, model, _, _ = reset_simulator("airline", [[text("It's olivia_gonzalez_2305.")]])
    conversation = [
        GREETING,
        {"role": "assistant", "content": [text("I need help with a trip.")], "from": "user"},
        {"role": "assistant", "content": [tool("a-1", "get_user_details", user_id="x")], "from": "assistant"},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "a-1", "status": "success", "content": [text("{}")]}}],
            "from": "assistant",
        },
        {
            "role": "assistant",
            "content": [text("<think>Ask for the id.</think>What is your user id?")],
            "from": "assistant",
        },
    ]
    simulator.step(conversation)
    assert model.requests == [
        [
            {"role": "user", "content": [text("Hello! How can I help you?")]},
            {"role": "assistant", "content": [text("I need help with a trip.")]},
            {"role": "user", "content": [text("What is your user id?")]},
        ]
    ]


@pytest.mark.parametrize("keyword", ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"])
def test_termination_keyword_ends_the_conversation(keyword):
    simulator, _, _, _ = reset_simulator("retail", [[text(f"Thanks, that's all. {keyword}")]])
    turn = simulator.step([GREETING])
    assert turn.terminated
    assert turn.termination_reason == keyword


def test_termination_counts_only_the_turns_final_message():
    replies = [[text("###STOP###"), tool("u-1", "check_status_bar")], [text("It shows airplane mode.")]]
    simulator, _, _, _ = reset_simulator("telecom", replies)
    turn = simulator.step([GREETING])
    assert len(turn.messages) == 3  # tool call, tool result, final reply
    assert not turn.terminated


def test_user_tools_act_on_the_episode_environment():
    replies = [[tool("u-1", "toggle_airplane_mode")], [text("I turned airplane mode off.")]]
    simulator, _, env, _ = reset_simulator("telecom", replies)
    assert "Airplane Mode" in env.use_user_tool("check_status_bar")
    turn = simulator.step([GREETING])
    assert [m["from"] for m in turn.messages] == ["user", "user", "user"]
    assert turn.messages[1]["content"][0]["toolResult"]["status"] == "success"
    assert "Airplane Mode" not in env.use_user_tool("check_status_bar")


def test_describe_reports_the_user_model_config():
    assert TauBenchUserSimulator(model=ScriptedModel([])).describe() == {
        "source": "bedrock",
        **USER_MODEL_CONFIG["bedrock"],
    }
    config = {**USER_MODEL_CONFIG["bedrock"], "model_id": "test-user-model"}
    described = TauBenchUserSimulator(config=config, model=ScriptedModel([])).describe()
    assert described == {"source": "bedrock", **config}
    assert json.loads(json.dumps(described)) == described
