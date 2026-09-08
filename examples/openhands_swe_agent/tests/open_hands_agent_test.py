"""Unit tests for the OpenHands agent backend helpers.

These cover the pure metric-extraction logic that the container agent loop relies
on but no longer knows about -- folding backend-specific parsing into the backend.
"""

import unittest

from swe_agent_server.open_hands_agent import (
    compute_llm_latency_sum,
    compute_token_metrics,
)


def _state(**default_bucket):
    return {"stats": {"usage_to_metrics": {"default": default_bucket}}}


class ComputeLlmLatencySumTest(unittest.TestCase):
    def test_sums_default_bucket_response_latencies(self):
        # OpenHands records one latency per model response under the "default"
        # usage bucket; the total is the backend's llm_latency_sum.
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
        # Auxiliary LLMs (e.g. a condenser) get their own bucket and must not
        # inflate the task-solving LLM's latency.
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
        # A metric should never fail a rollout, so partial/absent stats -> 0.0.
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

        # Every count is prefixed, so nothing collides with the loop's own names,
        # and the label fields are dropped so metrics stays numeric.
        self.assertEqual(metrics["openhands_prompt_tokens"], 900.0)
        self.assertEqual(metrics["openhands_completion_tokens"], 50.0)
        self.assertEqual(metrics["openhands_cache_read_tokens"], 500.0)
        self.assertNotIn("openhands_model", metrics)
        self.assertNotIn("openhands_response_id", metrics)
        self.assertTrue(all(isinstance(v, float) for v in metrics.values()))

    def test_per_turn_token_carries_end_of_rollout_context_length(self):
        # per_turn_token is overwritten rather than summed, so it is the final call's
        # prompt + completion -- the context length at the end of the rollout. Nothing
        # else needs deriving, so guard that it is forwarded as-is.
        state = _state(accumulated_token_usage={"prompt_tokens": 400, "per_turn_token": 320})

        self.assertEqual(compute_token_metrics(state)["openhands_per_turn_token"], 320.0)
        self.assertNotIn("openhands_context_length", compute_token_metrics(state))

    def test_forwards_unknown_numeric_fields(self):
        # New SDK TokenUsage fields should flow through without a code change here.
        state = _state(accumulated_token_usage={"some_future_tokens": 7})

        self.assertEqual(compute_token_metrics(state)["openhands_some_future_tokens"], 7.0)

    def test_missing_stats_report_nothing(self):
        # A metric should never fail a rollout, and an unmeasured count must not be
        # reported as 0.0 -- that would drag the trainer's average down.
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


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2)
