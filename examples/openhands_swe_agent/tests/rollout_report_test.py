#!/usr/bin/env python
"""Unit tests for the read side of a run: loading its records from the session table and
reducing them, including the ``Decimal`` conversion DynamoDB forces and picking which run
of an experiment name a report is about. No AWS: the query returns canned items.
"""

import asyncio
import unittest
from decimal import Decimal
from unittest import mock

# The recipe directory is on ``sys.path`` via its ``conftest.py``.
import rollout_report

MODULE = "rollout_report"
TABLE = "swe_agent_sessions"
REGION = "us-west-2"
EXPERIMENT = "eval_none_n1_gym"
RUN = "2026-09-03T17:44:48"


def rollout(task_id, n_idx=0, reward=1.0, aborted=False, start_at=RUN, **fields) -> dict:
    """One rollout record, as flat as the session item it stands for."""
    return {
        "session_id": f"eval_{task_id}_{n_idx}_{start_at}",
        "experiment_name": EXPERIMENT,
        "experiment_start_at": start_at,
        "task_id": str(task_id),
        "n_idx": n_idx,
        "reward": reward,
        "resolved": bool(reward),
        "aborted": aborted,
        **fields,
    }


def as_dynamodb(row: dict) -> dict:
    """The same record as boto3 hands it back: every number a ``Decimal``."""
    return {
        key: Decimal(str(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        for key, value in row.items()
    }


def fake_query(sessions: list[dict]):
    """Stand in for the index query, recording what it was asked for."""
    calls = []

    async def query(experiment_name, experiment_start_at="", *, table_name, region_name=None):
        calls.append((experiment_name, experiment_start_at, table_name, region_name))
        prefix = experiment_start_at or ""
        return [
            s
            for s in sessions
            if s["experiment_name"] == experiment_name and str(s["experiment_start_at"]).startswith(prefix)
        ]

    return query, calls


def load_runs(sessions, **kwargs) -> list[rollout_report.Run]:
    query, _ = fake_query(sessions)
    with mock.patch.multiple(MODULE, get_sessions_for_experiment_run=query):
        return asyncio.run(
            rollout_report.load_runs(
                EXPERIMENT,
                table_name=TABLE,
                region_name=REGION,
                **kwargs,
            )
        )


def load_run(sessions, **kwargs) -> rollout_report.Run:
    query, _ = fake_query(sessions)
    with mock.patch.multiple(MODULE, get_sessions_for_experiment_run=query):
        return asyncio.run(
            rollout_report.load_run(
                EXPERIMENT,
                table_name=TABLE,
                region_name=REGION,
                **kwargs,
            )
        )


class SummarizeTest(unittest.TestCase):
    def test_a_task_passes_if_any_of_its_samples_resolved(self):
        # pass@k is the only reason `summarize` knows any field name at all.
        rows = [
            rollout(1, 0, reward=0.0),
            rollout(1, 1, reward=1.0),
            rollout(2, 0, reward=0.0),
            rollout(2, 1, reward=0.0),
        ]
        report = rollout_report.summarize(rows)
        self.assertEqual(report["reward"]["tasks_passed"], 1)
        self.assertEqual(report["reward"]["pass_at_k"], 0.5)
        self.assertEqual(
            report["counts"],
            {
                "rollouts": 4,
                "rollouts_ok": 4,
                "rollouts_aborted": 0,
                "tasks": 2,
            },
        )

    def test_k_is_observed_from_the_rows_when_not_given(self):
        # The eval's config is not recorded anywhere, so k has to come off the rows.
        rows = [rollout(1, i) for i in range(4)] + [rollout(2, i) for i in range(4)]
        self.assertEqual(rollout_report.summarize(rows)["reward"]["pass_at_k_k"], 4)

    def test_a_given_k_wins_over_the_rows(self):
        # A run cut short still reports pass@n for the n it was asked for.
        rows = [rollout(1, 0), rollout(2, 0)]
        self.assertEqual(rollout_report.summarize(rows, k=4)["reward"]["pass_at_k_k"], 4)

    def test_an_aborted_rollout_counts_against_pass_at_k_but_not_in_the_stats(self):
        # A task the run failed to answer, but its timings are a container that never
        # ran and would drag every distribution.
        rows = [
            rollout(1, 0, reward=1.0, llm_latency_sum=30.0),
            rollout(2, 0, reward=None, aborted=True, llm_latency_sum=0.5),
        ]
        report = rollout_report.summarize(rows)
        self.assertEqual(report["counts"]["rollouts_aborted"], 1)
        self.assertEqual(report["reward"]["pass_at_k"], 0.5)
        self.assertEqual(report["metrics"]["llm_latency_sum"]["count"], 1)
        self.assertEqual(report["metrics"]["llm_latency_sum"]["mean"], 30.0)

    def test_every_numeric_field_is_summarized_and_nothing_else_is(self):
        # The row is the whole session meta, so the reduction walks past arns, ids and
        # timestamps -- and picks up a metric nobody declared here.
        rows = [rollout(1, 0, num_tool_calls=9, runtime_arn="arn:aws:...", eval_start_at="2026-09-03")]
        metrics = rollout_report.summarize(rows)["metrics"]
        self.assertEqual(metrics["num_tool_calls"]["mean"], 9.0)
        self.assertNotIn("runtime_arn", metrics)
        self.assertNotIn("eval_start_at", metrics)

    def test_a_field_only_some_rollouts_report_is_summarized_over_those(self):
        # Absent is not zero: counting the rest as 0 would halve the reported lengths.
        rows = [rollout(1, 0, num_tokens=1000), rollout(1, 1)]
        stats = rollout_report.summarize(rows)["metrics"]["num_tokens"]
        self.assertEqual((stats["count"], stats["mean"]), (1, 1000.0))

    def test_records_from_dynamodb_reduce_to_the_same_numbers(self):
        """Same rollouts, as the driver held them and as the table hands them back;
        without the ``Decimal`` conversion the second report has no metrics at all.
        """
        rows = [
            rollout(1, 0, reward=1.0, num_tool_calls=9, llm_latency_sum=30.5),
            rollout(1, 1, reward=0.0, num_tool_calls=3, llm_latency_sum=12.25),
            rollout(2, 0, reward=None, aborted=True),
        ]
        from_table = load_run([as_dynamodb(row) for row in rows]).rows
        self.assertEqual(rollout_report.summarize(from_table), rollout_report.summarize(rows))
        self.assertIn("llm_latency_sum", rollout_report.summarize(from_table)["metrics"])

    def test_an_integral_number_survives_as_an_int(self):
        # n_idx and token counts are counts: a float prints as `4.0` and keys differently.
        (row,) = load_run([as_dynamodb(rollout(1, 4, num_tokens=1000))]).rows
        self.assertEqual((row["n_idx"], row["num_tokens"]), (4, 1000))
        self.assertIsInstance(row["n_idx"], int)


class LoadRunsTest(unittest.TestCase):
    def test_runs_are_split_by_start_time_oldest_first(self):
        sessions = [rollout(1, 0, start_at="2026-09-03T17:44"), rollout(1, 0, start_at="2026-09-03T17:34")]
        runs = load_runs(sessions)
        self.assertEqual([run.experiment_start_at for run in runs], ["2026-09-03T17:34", "2026-09-03T17:44"])
        self.assertEqual([len(run.rows) for run in runs], [1, 1])

    def test_the_latest_run_is_the_one_reported_on(self):
        # The earlier runs of a name are the attempts that were re-run.
        sessions = [rollout(1, 0, start_at="2026-09-03T17:34"), rollout(2, 0, start_at="2026-09-03T17:44")]
        with self.assertLogs(MODULE, level="WARNING") as logs:
            run = load_run(sessions)
        self.assertEqual(run.experiment_start_at, "2026-09-03T17:44")
        # Loud, because analysing the wrong run is the failure that looks like a result.
        self.assertIn("2026-09-03T17:34", "\n".join(logs.output))

    def test_one_run_is_reported_without_a_warning(self):
        with mock.patch.object(rollout_report.logger, "warning") as warning:
            load_run([rollout(1, 0)])
        warning.assert_not_called()

    def test_a_start_at_prefix_pins_an_older_run(self):
        sessions = [rollout(1, 0, start_at="2026-09-03T17:34"), rollout(2, 0, start_at="2026-09-03T17:44")]
        run = load_run(sessions, experiment_start_at="2026-09-03T17:34")
        self.assertEqual(run.experiment_start_at, "2026-09-03T17:34")
        self.assertEqual(run.rows[0]["task_id"], "1")

    def test_a_name_with_no_records_is_an_error(self):
        # Not an empty report: zero rollouts is exactly how a typo would present.
        with self.assertRaises(RuntimeError) as caught:
            load_run([])
        self.assertIn(EXPERIMENT, str(caught.exception))
        self.assertIn(TABLE, str(caught.exception))

    def test_the_query_is_told_which_table_and_region_to_read(self):
        # A default here is how an analysis ends up reporting on somebody else's table.
        query, calls = fake_query([rollout(1, 0)])
        with mock.patch.multiple(MODULE, get_sessions_for_experiment_run=query):
            asyncio.run(
                rollout_report.load_run(
                    EXPERIMENT,
                    table_name=TABLE,
                    region_name=REGION,
                )
            )
        self.assertEqual(calls, [(EXPERIMENT, "", TABLE, REGION)])


if __name__ == "__main__":
    unittest.main()
