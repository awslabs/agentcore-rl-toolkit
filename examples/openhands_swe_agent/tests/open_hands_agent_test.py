"""Unit tests for the OpenHands agent backend's metric-extraction helpers and tool specs."""

import inspect
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from openhands.tools import FileEditorTool, TerminalTool
from openhands.tools.terminal.definition import TerminalAction
from openhands.tools.terminal.impl import TerminalExecutor
from swe_agent_server.open_hands_agent import (
    NO_PAGER_ENV,
    build_tools,
    compute_llm_latency_sum,
    compute_token_metrics,
)


def _state(**default_bucket):
    return {"stats": {"usage_to_metrics": {"default": default_bucket}}}


class ComputeLlmLatencySumTest(unittest.TestCase):
    def test_sums_default_bucket_response_latencies(self):
        state = {
            "stats": {
                "usage_to_metrics": {
                    "default": {
                        "response_latencies": [
                            {"latency": 1.5},
                            {"latency": 2.0},
                        ]
                    }
                }
            }
        }

        self.assertEqual(compute_llm_latency_sum(state), 3.5)

    def test_ignores_non_default_buckets(self):
        # Auxiliary LLMs (e.g. a condenser) get their own bucket.
        state = {
            "stats": {
                "usage_to_metrics": {
                    "default": {"response_latencies": [{"latency": 1.0}]},
                    "condenser": {"response_latencies": [{"latency": 9.0}]},
                }
            }
        }

        self.assertEqual(compute_llm_latency_sum(state), 1.0)

    def test_missing_stats_yield_zero(self):
        # A metric should never fail a rollout.
        self.assertEqual(compute_llm_latency_sum({}), 0.0)
        self.assertEqual(compute_llm_latency_sum({"stats": {"usage_to_metrics": {}}}), 0.0)


class ComputeTokenMetricsTest(unittest.TestCase):
    def test_forwards_accumulated_counts_under_openhands_prefix(self):
        state = _state(
            accumulated_token_usage={
                "model": "openai/qwen",
                "response_id": "",
                "prompt_tokens": 900,
                "completion_tokens": 50,
                "cache_read_tokens": 500,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
                "context_window": 0,
                "per_turn_token": 320,
            },
        )

        metrics = compute_token_metrics(state)

        # Prefixed so nothing collides with the loop's names; label fields dropped so
        # metrics stays numeric.
        self.assertEqual(metrics["openhands_prompt_tokens"], 900.0)
        self.assertEqual(metrics["openhands_completion_tokens"], 50.0)
        self.assertEqual(metrics["openhands_cache_read_tokens"], 500.0)
        self.assertNotIn("openhands_model", metrics)
        self.assertNotIn("openhands_response_id", metrics)
        self.assertTrue(all(isinstance(v, float) for v in metrics.values()))

    def test_per_turn_token_carries_end_of_rollout_context_length(self):
        # per_turn_token is overwritten rather than summed, so it is the final call's
        # prompt + completion; forwarded as-is.
        state = _state(accumulated_token_usage={"prompt_tokens": 400, "per_turn_token": 320})

        self.assertEqual(compute_token_metrics(state)["openhands_per_turn_token"], 320.0)
        self.assertNotIn("openhands_context_length", compute_token_metrics(state))

    def test_forwards_unknown_numeric_fields(self):
        # New SDK TokenUsage fields flow through without a code change here.
        state = _state(accumulated_token_usage={"some_future_tokens": 7})

        self.assertEqual(compute_token_metrics(state)["openhands_some_future_tokens"], 7.0)

    def test_missing_stats_report_nothing(self):
        # An unmeasured count must not be reported as 0.0 -- that drags the average down.
        self.assertEqual(compute_token_metrics({}), {})
        self.assertEqual(compute_token_metrics({"stats": {"usage_to_metrics": {}}}), {})
        self.assertEqual(compute_token_metrics(_state(accumulated_token_usage=None)), {})

    def test_ignores_non_default_buckets(self):
        state = {
            "stats": {
                "usage_to_metrics": {
                    "default": {"accumulated_token_usage": {"prompt_tokens": 10}},
                    "condenser": {"accumulated_token_usage": {"prompt_tokens": 900}},
                }
            }
        }

        self.assertEqual(compute_token_metrics(state)["openhands_prompt_tokens"], 10.0)


class BuildToolsTest(unittest.TestCase):
    def test_terminal_tool_carries_the_no_pager_env(self):
        specs = {tool.name: tool for tool in build_tools()}

        self.assertEqual(set(specs), {TerminalTool.name, FileEditorTool.name})
        self.assertEqual(specs[TerminalTool.name].params, {"env": NO_PAGER_ENV})
        self.assertEqual(NO_PAGER_ENV["GIT_PAGER"], "cat")
        self.assertEqual(NO_PAGER_ENV["PAGER"], "cat")

    def test_terminal_tool_still_takes_an_env_parameter(self):
        # params are passed to create() as keywords; a renamed/dropped `env` would silently un-disable pagers.
        self.assertIn("env", inspect.signature(TerminalTool.create).parameters)


def _repo_with_a_long_commit(directory: str) -> str:
    """A git repo whose HEAD is too long to fit one screen, so `git show` would page."""
    (Path(directory) / "big.txt").write_text("".join(f"line {i}\n" for i in range(500)))
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    for args in (["init", "-q"], ["add", "."], [*identity, "commit", "-qm", "long commit"]):
        subprocess.run(["git", *args], cwd=directory, check=True, capture_output=True)
    return directory


@unittest.skipUnless(shutil.which("tmux"), "the pooled tmux terminal needs tmux")
@unittest.skipUnless(shutil.which("less"), "without a pager installed there is nothing to wedge")
class TerminalPagerTest(unittest.TestCase):
    """End-to-end check that NO_PAGER_ENV reaches TmuxPanePool's panes, which skip the
    pager defence openhands-tools only applies in TmuxTerminal.initialize()."""

    def test_paging_command_completes_and_leaves_the_terminal_usable(self):
        directory = _repo_with_a_long_commit(self.enterContext(tempfile.TemporaryDirectory()))
        # Short timeout so a regression fails in seconds, not the 30s default.
        executor = TerminalExecutor(working_dir=directory, no_change_timeout_seconds=5, env=NO_PAGER_ENV)
        self.addCleanup(executor.close)

        paged = executor(TerminalAction(command="git show HEAD"))
        # A wedged pane returns -1: no prompt came back, so no exit code could be read.
        self.assertEqual(paged.exit_code, 0, msg=paged.text[:500])
        self.assertIn("long commit", paged.text)

        # And the next command is executed, rather than typed into a pager.
        after = executor(TerminalAction(command="echo still-alive"))
        self.assertEqual(after.exit_code, 0, msg=after.text[:500])
        self.assertIn("still-alive", after.text)


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2)
