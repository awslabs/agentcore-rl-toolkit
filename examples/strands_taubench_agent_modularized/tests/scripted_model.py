"""A Strands ``Model`` that replays a fixed script and records every request it receives.

Stands in for both the vLLM-served assistant and the Bedrock user simulator, so a whole
rollout runs offline and deterministically while the tau2 environment, tools, and reward
stay real.
"""

import copy
import hashlib
import json
from typing import Any, AsyncIterable

from strands.models.model import Model


class ScriptExhaustedError(AssertionError):
    """Raised when the rollout asks a scripted model for more responses than it scripts."""


class ScriptedModel(Model):
    """Replay ``script`` one entry per model call.

    Each entry is either a Strands content-block list (the reply message's ``content``),
    or an exception instance to raise from that call. Every call's inputs are captured in
    ``calls`` so a test can compare exactly what each model was shown.
    """

    def __init__(self, role: str, script: list[Any]):
        self.role = role
        self.script = list(script)
        self.calls: list[dict] = []
        self.overran = False

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict:
        return {"model_id": f"scripted-{self.role}"}

    def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError("structured_output is not scripted")

    async def stream(
        self,
        messages: list[dict],
        tool_specs: list[dict] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> AsyncIterable[dict]:
        # System prompts and tool specs are large and identical across calls, so they are
        # pinned by digest; messages are small and are what a refactor is most likely to break.
        self.calls.append(
            {
                "system_prompt_sha256": _digest(system_prompt),
                "tool_names": [spec["name"] for spec in tool_specs or []],
                "tool_specs_sha256": _digest(tool_specs or []),
                "tool_choice": tool_choice,
                "messages": copy.deepcopy(messages),
            }
        )
        index = len(self.calls) - 1
        if index >= len(self.script):
            self.overran = True
            raise ScriptExhaustedError(f"{self.role} model called {index + 1} times; script has {len(self.script)}")
        entry = self.script[index]
        if isinstance(entry, BaseException):
            raise entry
        for event in _events_for(entry):
            yield event


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _events_for(content: list[dict]) -> list[dict]:
    """Render a reply message's content blocks as a Bedrock-style ConverseStream."""
    events: list[dict] = [{"messageStart": {"role": "assistant"}}]
    for block in content:
        if "text" in block:
            events.append({"contentBlockDelta": {"delta": {"text": block["text"]}}})
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            events.append(
                {
                    "contentBlockStart": {
                        "start": {"toolUse": {"toolUseId": tool_use["toolUseId"], "name": tool_use["name"]}}
                    }
                }
            )
            events.append({"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_use["input"])}}}})
        elif "reasoningContent" in block:
            reasoning = block["reasoningContent"]["reasoningText"]
            events.append({"contentBlockDelta": {"delta": {"reasoningContent": {"text": reasoning["text"]}}}})
            if "signature" in reasoning:
                events.append(
                    {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": reasoning["signature"]}}}}
                )
        else:
            raise ValueError(f"unsupported scripted block: {block}")
        events.append({"contentBlockStop": {}})
    stop_reason = "tool_use" if any("toolUse" in block for block in content) else "end_turn"
    events.append({"messageStop": {"stopReason": stop_reason}})
    events.append(
        {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }
    )
    return events
