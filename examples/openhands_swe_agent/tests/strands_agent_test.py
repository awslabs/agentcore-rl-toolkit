"""Unit tests for the Strands agent backend, with the LLM mocked so no vLLM endpoint is
needed: the non-streaming completion path, model construction, tool timing, a full
rollout, and backend dispatch in the app.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

from swe_agent_server import strands_agent
from swe_agent_server.strands_agent import (
    _build_model,
    _CapturingLiteLLMModel,
    _ToolTimingHooks,
)

from agentcore_rl_toolkit.rollout_session.wire import RolloutStartRequest


def _fake_response(content: str = "hello") -> SimpleNamespace:
    """A non-streaming LiteLLM completion response, shaped as the re-emit path
    (``_format_non_streaming_response``) consumes it.
    """
    message = SimpleNamespace(content=content, reasoning_content=None, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


class HandleNonStreamingResponseTest(IsolatedAsyncioTestCase):
    async def test_accumulates_latency_and_emits_stream_events(self):
        model = _build_model(dict(model="openai/test"))
        response = _fake_response(content="done")

        async def fake_acompletion(litellm_request):
            return response

        model._acompletion = fake_acompletion  # type: ignore[method-assign]

        # Two monotonic reads bracket the completion call: start=100, end=102.5.
        with mock.patch.object(strands_agent, "_monotonic", side_effect=[100.0, 102.5]):
            chunks = [chunk async for chunk in model._handle_non_streaming_response({})]

        self.assertEqual(model.llm_latency_sum, 2.5)
        # The assistant text is re-emitted so the Strands agent loop proceeds.
        text = "".join(
            chunk["contentBlockDelta"]["delta"].get("text", "") for chunk in chunks if "contentBlockDelta" in chunk
        )
        self.assertEqual(text, "done")


class BuildModelTest(unittest.TestCase):
    def test_splits_model_id_extra_body_and_client_args(self):
        model = _build_model(
            dict(
                model="openai/Qwen",
                base_url="http://host/v1",
                api_key="secret",
                litellm_extra_body={"return_token_ids": True, "logprobs": True},
            ),
        )

        self.assertIsInstance(model, _CapturingLiteLLMModel)
        self.assertEqual(model.get_config()["model_id"], "openai/Qwen")
        # Forced non-streaming so completion latency can be timed per turn.
        self.assertFalse(model.get_config()["stream"])
        self.assertEqual(
            model.get_config()["params"]["extra_body"],
            {"return_token_ids": True, "logprobs": True},
        )
        # Remaining keys become LiteLLM client args, not model config.
        self.assertEqual(model.client_args, {"base_url": "http://host/v1", "api_key": "secret"})


class ToolTimingHooksTest(unittest.TestCase):
    def test_accumulates_elapsed_time_per_tool_call(self):
        timing = _ToolTimingHooks()
        # monotonic reads: t1 start=10, end=12; t2 start=20, end=25.
        with mock.patch.object(strands_agent, "_monotonic", side_effect=[10.0, 12.0, 20.0, 25.0]):
            timing._on_before(SimpleNamespace(tool_use={"toolUseId": "t1"}))
            timing._on_after(SimpleNamespace(tool_use={"toolUseId": "t1"}))
            timing._on_before(SimpleNamespace(tool_use={"toolUseId": "t2"}))
            timing._on_after(SimpleNamespace(tool_use={"toolUseId": "t2"}))

        self.assertEqual(timing.num_tool_calls, 2)
        self.assertEqual(timing.total_time_s, 7.0)

    def test_unmatched_after_does_not_add_time(self):
        # An AfterToolCallEvent with no recorded start still counts the call.
        timing = _ToolTimingHooks()
        timing._on_after(SimpleNamespace(tool_use={"toolUseId": "orphan"}))

        self.assertEqual(timing.num_tool_calls, 1)
        self.assertEqual(timing.total_time_s, 0.0)


class RolloutTest(unittest.TestCase):
    def test_rollout_populates_dump_from_mocked_llm(self):
        # Single-turn LLM response (no tool calls, so the loop ends after one turn), with
        # git and eval stubbed out.
        response = _fake_response(content="all set")

        async def fake_acompletion(self, litellm_request):
            return response

        request = RolloutStartRequest(
            rollout_id="r",
            task_input=dict(
                repo_path="/tmp",
                base_commit="HEAD",
                problem_statement="fix it",
                llm=dict(model="openai/test", base_url="x", api_key="k"),
            ),
        )

        with (
            mock.patch.object(_CapturingLiteLLMModel, "_acompletion", fake_acompletion),
            mock.patch.object(strands_agent.subprocess, "check_output", return_value=b"diff --git a b"),
            mock.patch.object(strands_agent, "run_evaluation", return_value={"resolved": True}),
        ):
            dump = strands_agent.rollout(request)

        self.assertIsNone(dump.exception)
        self.assertEqual(dump.reward, 1.0)
        # Backend-agnostic metrics land in the generic metrics dict, not task_output.
        self.assertIn("llm_latency_sum", dump.metrics)
        assert dump.task_output is not None
        self.assertEqual(dump.task_output["git_diff"], "diff --git a b")


class BackendDispatchTest(unittest.TestCase):
    def _request(self, agent: str) -> RolloutStartRequest:
        return RolloutStartRequest(rollout_id="r", task_input={"agent": agent})

    def test_dispatches_to_strands_backend(self):
        from swe_agent_server import app

        sentinel = object()
        with mock.patch.object(strands_agent, "rollout", return_value=sentinel) as strands_rollout:
            result = app.run_rollout(self._request("strands"))

        strands_rollout.assert_called_once()
        self.assertIs(result, sentinel)

    def test_dispatches_to_openhands_backend(self):
        # openhands-sdk is optional and may be absent here, so stub the module rather
        # than importing it.
        from swe_agent_server import app

        sentinel = object()
        fake_module = types.ModuleType("swe_agent_server.open_hands_agent")
        fake_module.rollout = mock.Mock(return_value=sentinel)  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"swe_agent_server.open_hands_agent": fake_module}):
            result = app.run_rollout(self._request("openhands"))

        fake_module.rollout.assert_called_once()
        self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2)
