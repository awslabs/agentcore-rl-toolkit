"""Characterization test: full rollouts against scripted models must reproduce recorded goldens.

Each scenario runs ``rl_app.invoke_agent`` end to end with the real tau2 environment, tools,
and reward. Only the two models are replaced: ``ScriptedModel`` replays a fixed reply per call
and records what it was shown. The observation that is compared contains:

- the kwargs each model was constructed with;
- per model call: the system prompt and tool specs (by digest), the tool names, and the full
  message list the model received;
- the dict ``invoke_agent`` returns (reward, reward_info, conversation, meta).

The goldens in ``golden/`` were recorded from the original, unmodularized code (commit
``90f9d25``, kept unchanged in ``../strands_taubench_agent``), before the user simulator was
extracted into ``user_simulator.py``. A passing run therefore shows the refactor left every
model input, the conversation, the reward, and the result unchanged.

Re-record only when a behavior change is intended, and review the golden diff:

    TAUBENCH_RECORD_GOLDEN=1 python -m pytest tests/rollout_parity_test.py
"""

import copy
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import strands.models.bedrock
import strands.models.openai
from rollout_harness import EXAMPLE_MODULES, build_payload, import_rl_app
from scripted_model import ScriptedModel

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
RECORD = os.environ.get("TAUBENCH_RECORD_GOLDEN") == "1"


def text(value: str) -> dict:
    return {"text": value}


def tool(tool_use_id: str, name: str, **tool_input) -> dict:
    return {"toolUse": {"toolUseId": tool_use_id, "name": name, "input": tool_input}}


def thought(value: str) -> dict:
    return {"reasoningContent": {"reasoningText": {"text": value}}}


@dataclass
class Scenario:
    name: str
    task: str
    user: list
    assistant: list
    orchestrator: dict = field(default_factory=dict)  # ORCHESTRATOR_CONFIG overrides
    sampling_params: dict | None = None


THINKING_SAMPLING_PARAMS = {"max_tokens": 4096, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}

SCENARIOS = [
    # Two assistant tool calls; the golden cancel_reservation action matches, so DB reward is 1.
    Scenario(
        name="airline_cancel_stop",
        task="airline_example.json",
        user=[
            [text("Hi, I'm Olivia Gonzalez, user id olivia_gonzalez_2305. I need help with my Texas trip.")],
            [text("Actually, please just cancel reservation Z7GOZK.")],
            [text("Thanks, that's all. ###STOP###")],
        ],
        assistant=[
            [
                text("<think>Look the user up first.</think>"),
                tool("a-1", "get_user_details", user_id="olivia_gonzalez_2305"),
            ],
            [text("<think>Found the profile.</think>I found your profile. Which reservation should I change?")],
            [tool("a-2", "cancel_reservation", reservation_id="Z7GOZK")],
            [text("Reservation Z7GOZK has been cancelled.")],
        ],
    ),
    # User replies carry reasoningContent; ends on TRANSFER; sampling params supplied by the payload.
    Scenario(
        name="retail_return_transfer",
        task="retail_example.json",
        user=[
            [
                thought("State my email."),
                text("Hi, I'm Fatima Wilson, fatima.wilson5721@example.com. I want a return."),
            ],
            [thought("Exclude the coffee machine."), text("Order #W5272531, all but the coffee machine, to my card.")],
            [text("###TRANSFER###")],
        ],
        assistant=[
            [tool("r-1", "find_user_id_by_email", email="fatima.wilson5721@example.com")],
            [text("<think>User found.</think>Thanks Fatima. Which order and items?")],
            [
                tool(
                    "r-2",
                    "return_delivered_order_items",
                    order_id="#W5272531",
                    item_ids=["7228247242", "2698416822", "8098621301", "3320557165"],
                    payment_method_id="credit_card_6824399",
                )
            ],
            [text("<think>Done.</think>Your return has been requested.")],
        ],
        sampling_params=THINKING_SAMPLING_PARAMS,
    ),
    # Same conversation with thinking kept in each agent's own history.
    Scenario(
        name="retail_return_keep_thinking",
        task="retail_example.json",
        user=[
            [
                thought("State my email."),
                text("Hi, I'm Fatima Wilson, fatima.wilson5721@example.com. I want a return."),
            ],
            [text("###TRANSFER###")],
        ],
        assistant=[
            [
                text("<think>Look up by email.</think>"),
                tool("r-1", "find_user_id_by_email", email="fatima.wilson5721@example.com"),
            ],
            [text("<think>User found.</think>Thanks Fatima. Which order?")],
        ],
        orchestrator={"strip_thinking_from_history": False},
    ),
    # Telecom: one user turn makes three user-tool calls against the shared env before replying.
    Scenario(
        name="telecom_user_tools_stop",
        task="telecom_example.json",
        user=[
            [text("My phone has shown No Service for hours. I'm John Smith, 555-123-2002.")],
            [text("Checking now."), tool("u-1", "check_status_bar")],
            [tool("u-2", "toggle_airplane_mode")],
            [tool("u-3", "check_status_bar")],
            [text("Airplane mode was on. I turned it off and now I have signal.")],
            [text("Great, thanks. ###STOP###")],
        ],
        assistant=[
            [tool("a-1", "get_customer_by_phone", phone_number="555-123-2002")],
            [text("<think>Check the device.</think>Please check your status bar and turn airplane mode off if on.")],
            [text("Glad it's fixed! Anything else?")],
        ],
    ),
    # A keyword in a non-final user message must not terminate; only the turn's last message counts.
    Scenario(
        name="telecom_keyword_only_in_last_message",
        task="telecom_example.json",
        user=[
            [text("No service on my phone. ###STOP###"), tool("u-1", "check_status_bar")],
            [text("My status bar shows no signal.")],
            [text("###OUT-OF-SCOPE###")],
        ],
        assistant=[
            [text("Please try toggling airplane mode.")],
        ],
    ),
    # A leaked <tool_call> tag forces reward 0 and ends the rollout.
    Scenario(
        name="airline_malformed_tool_call",
        task="airline_example.json",
        user=[
            [text("Hi, I'm Olivia Gonzalez. I need to change a flight.")],
        ],
        assistant=[
            [text('<think>Call a tool.</think><tool_call>{"name": "get_user_details"}</tool_call>')],
        ],
    ),
    # Neither side ever stops; the loop ends at max_turns and the reward is still computed.
    Scenario(
        name="retail_max_turns",
        task="retail_example.json",
        user=[
            [text("I want to return some things.")],
            [text("Still waiting on that return.")],
        ],
        assistant=[
            [text("Sure, what is your email?")],
            [text("I need your email to continue.")],
        ],
        orchestrator={"max_turns": 2},
    ),
    # An assistant-side exception yields the minimal {"rewards": 0.0} result.
    Scenario(
        name="airline_assistant_error",
        task="airline_example.json",
        user=[
            [text("Hi, I'm Olivia Gonzalez.")],
        ],
        assistant=[
            RuntimeError("inference server unavailable"),
        ],
    ),
    # A user-simulator exception, mid-turn after a user tool call, also yields {"rewards": 0.0}.
    Scenario(
        name="telecom_user_sim_error",
        task="telecom_example.json",
        user=[
            [tool("u-1", "check_status_bar")],
            RuntimeError("user simulator throttled"),
        ],
        assistant=[],
    ),
]


def _refuse_real_model(*args, **kwargs):
    raise AssertionError("a real model was called; a model class binding was not patched")


def run_scenario(scenario: Scenario, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run one rollout with scripted models and return everything the goldens pin."""
    rl_app = import_rl_app(monkeypatch)

    import constants

    for key, value in scenario.orchestrator.items():
        monkeypatch.setitem(constants.ORCHESTRATOR_CONFIG, key, value)

    created: dict[str, dict] = {}

    def factory(role: str, script: list):
        def make(**kwargs):
            assert role not in created, f"{role} model constructed twice"
            model = ScriptedModel(role, script)
            created[role] = {"init_kwargs": copy.deepcopy(kwargs), "model": model}
            return model

        return make

    monkeypatch.setattr(strands.models.bedrock.BedrockModel, "stream", _refuse_real_model)
    monkeypatch.setattr(strands.models.openai.OpenAIModel, "stream", _refuse_real_model)
    for name in EXAMPLE_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        if hasattr(module, "BedrockModel"):
            monkeypatch.setattr(module, "BedrockModel", factory("user", scenario.user))
        if hasattr(module, "OpenAIModel"):
            monkeypatch.setattr(module, "OpenAIModel", factory("assistant", scenario.assistant))

    result = rl_app.invoke_agent(build_payload(scenario.task, scenario.sampling_params))

    for role, script in (("user", scenario.user), ("assistant", scenario.assistant)):
        model = created[role]["model"]
        assert not model.overran, f"{role} model asked for more replies than scripted"
        assert len(model.calls) == len(script), f"{role} model used {len(model.calls)} of {len(script)} replies"

    observation = {
        "result": result,
        "models": {
            role: {"init_kwargs": entry["init_kwargs"], "calls": entry["model"].calls}
            for role, entry in sorted(created.items())
        },
    }
    # Round-trip through JSON: the S3 save path requires a JSON-serializable result anyway.
    return _mask_tracking_ids(json.loads(json.dumps(observation)))


def _mask_tracking_ids(value):
    """Replace the random per-message ``tracking_id`` Strands assigns, keeping the key itself."""
    if isinstance(value, dict):
        return {k: "<tracking_id>" if k == "tracking_id" else _mask_tracking_ids(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_tracking_ids(v) for v in value]
    return value


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_rollout_matches_golden(scenario: Scenario, monkeypatch: pytest.MonkeyPatch):
    observed = run_scenario(scenario, monkeypatch)
    golden_path = GOLDEN_DIR / f"{scenario.name}.json"

    if RECORD:
        # A golden is only worth pinning if the rollout is deterministic.
        with pytest.MonkeyPatch.context() as rerun_patch:
            assert run_scenario(scenario, rerun_patch) == observed, "rollout is not deterministic"
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")

    golden = json.loads(golden_path.read_text())
    assert observed["result"] == golden["result"]
    assert observed["models"].keys() == golden["models"].keys()
    for role, golden_model in golden["models"].items():
        observed_model = observed["models"][role]
        assert observed_model["init_kwargs"] == golden_model["init_kwargs"], f"{role} model constructed differently"
        assert len(observed_model["calls"]) == len(golden_model["calls"]), f"{role} call count differs"
        for index, (observed_call, golden_call) in enumerate(
            zip(observed_model["calls"], golden_model["calls"], strict=True)
        ):
            assert observed_call == golden_call, f"{role} model call {index} received different input"
