#!/usr/bin/env python
"""Unit tests for the eval harness's own seams: :class:`LateBoundLlmSession`, which
defers the perishable ``task["llm"]`` until inside the bounded run, and :func:`run_one`'s
bookkeeping. No AWS: the session store, the S3 upload and the rollout session are faked.
"""

import asyncio
import json
import unittest
from unittest import IsolatedAsyncioTestCase

# The recipe directory is on ``sys.path`` via its ``conftest.py``.
import rollout_batch as bae

from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister
from agentcore_rl_toolkit.rollout_session.lifecycle import RolloutSession
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse

GOOD = RolloutDumpResponse(
    task_output={"patch": "diff"},
    reward=1.0,
    exception=None,
    metrics={"num_tool_calls": 9.0, "llm_latency_sum": 30.0},
)
FAILED = RolloutDumpResponse(
    task_output={"patch": ""},
    reward=None,
    exception="Traceback: agent crashed",
    metrics={"num_tool_calls": 3.0, "llm_latency_sum": 12.5},
)
# Failed without saying why: a dump came back, but no trace was recorded.
SILENT = RolloutDumpResponse(
    task_output={"patch": ""},
    reward=None,
    exception=None,
    metrics={"num_tool_calls": 1.0},
)


class FakeSession:
    """A rollout session that records the task each phase saw."""

    def __init__(self, *args, rollout: RolloutDumpResponse = GOOD, **kwargs):
        self.rollout = rollout
        self.setup_task: dict | None = None
        self.run_task: dict | None = None
        self.entered = False
        self.torn_down = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc_info):
        await self.shutdown()

    async def setup(self, task):
        # a copy: run() mutates the same dict, and the question is what setup saw
        self.setup_task = dict(task)

    async def run(self, task):
        self.run_task = dict(task)
        return self.rollout

    async def shutdown(self):
        self.torn_down = True


class LateBoundLlmSessionTest(IsolatedAsyncioTestCase):
    def build(self, llm, session=None):
        self.built = 0

        def build_llm():
            self.built += 1
            return llm

        self.inner = session if session is not None else FakeSession()
        return bae.LateBoundLlmSession(self.inner, build_llm)

    async def test_it_satisfies_the_session_interface(self):
        self.assertIsInstance(self.build({"api_key": "k"}), RolloutSession)

    async def test_the_llm_is_absent_until_run(self):
        wrapper = self.build({"api_key": "token"})
        task = {"index": 1}
        async with wrapper:
            await wrapper.setup(task)
            self.assertEqual(self.built, 0, "built before it was needed")
            self.assertNotIn("llm", self.inner.setup_task)
            await wrapper.run(task)
        self.assertEqual(self.built, 1)
        self.assertEqual(self.inner.run_task["llm"], {"api_key": "token"})

    async def test_it_is_built_once_per_rollout(self):
        wrapper = self.build({"api_key": "token"})
        async with wrapper:
            await wrapper.run({"index": 1})
        self.assertEqual(self.built, 1)

    async def test_a_none_config_leaves_the_key_absent(self):
        # NullEndpoint has no inference; the key must stay unset rather than be None, so
        # a session that reads it fails loudly instead of talking to a stray model.
        wrapper = self.build(None)
        task = {"index": 1}
        async with wrapper:
            await wrapper.run(task)
        self.assertNotIn("llm", task)
        self.assertNotIn("llm", self.inner.run_task)

    async def test_the_rest_of_the_lifecycle_is_delegated_untouched(self):
        wrapper = self.build({"api_key": "k"})
        async with wrapper:
            self.assertTrue(self.inner.entered)
        self.assertTrue(self.inner.torn_down)

    async def test_the_null_endpoints_own_config_is_none(self):
        self.assertIsNone(bae.NullEndpoint().build_task_llm("s", 1.0, 1.0))


class LocalBoundsTest(unittest.TestCase):
    def test_the_config_knobs_land_on_the_matching_bounds_fields(self):
        config = eval_config(concurrency=8, rollout_concurrency=3, timeout=900, container_setup_timeout=600)
        bounds = bae.local_bounds(config)
        self.assertEqual(bounds.agent_run_timeout, 900)
        self.assertEqual(bounds.container_setup_timeout, 600)
        self.assertEqual(bounds.container_semaphore.value, 8)
        self.assertEqual(bounds.rollout_semaphore.value, 3)

    def test_rollout_concurrency_defaults_to_non_binding(self):
        # None means every container that exists may run.
        bounds = bae.local_bounds(eval_config(concurrency=8, rollout_concurrency=None))
        self.assertEqual(bounds.rollout_semaphore.value, 8)

    def test_the_assigners_are_separate_instances(self):
        # Container and rollout priorities are numbered independently.
        bounds = bae.local_bounds(eval_config())
        self.assertIsNot(bounds.container_priority_assigner, bounds.rollout_priority_assigner)


def eval_config(**overrides):
    kwargs = dict(
        experiment_name="unit-test",
        endpoint=bae.NullEndpoint(),
        dataset="unused.parquet",
        num_tasks=1,
        n=1,
        concurrency=1,
        session_create_rate=100.0,
        timeout=30,
        # Never written to: these tests drive `run_one` and `local_bounds`, not `run_eval`.
        report_dir="unused",
        container_setup_timeout=30,
    )
    kwargs.update(overrides)
    return bae.EvalConfig(**kwargs)


class RunOneTest(IsolatedAsyncioTestCase):
    """run_one driven end to end: what lands in the report row and in S3."""

    def setUp(self):
        self.uploads: dict[str, dict] = {}
        self.sessions: list[FakeSession] = []
        self._originals = {
            name: getattr(bae, name) for name in ("AgentCoreSession", "upload_object", "DynamoDBPersister")
        }

        async def fake_upload(s3_uri, region_name, data):
            self.uploads[s3_uri] = json.loads(data)

        bae.upload_object = fake_upload
        # In-memory session store: persistence is persistent_dict's own test's subject.
        bae.DynamoDBPersister = lambda *args, **kwargs: NullPersister()

    def tearDown(self):
        for name, original in self._originals.items():
            setattr(bae, name, original)

    async def run_one(self, rollout: RolloutDumpResponse, **config_overrides):
        def make_session(*args, **kwargs):
            session = FakeSession(rollout=rollout)
            self.sessions.append(session)
            return session

        bae.AgentCoreSession = make_session
        config = eval_config(**config_overrides)
        row = await bae.run_one(
            config,
            "arn:runtime",
            "arn:capacity",
            "2026-09-03T00:00:00",
            {"index": 7, "instance_id": "repo__proj-1"},
            0,
            bae.local_bounds(config),
            "s3://bucket/prefix",
            "sessions-table",
            "us-west-2",
        )
        ((uri, dumped),) = self.uploads.items()
        return row, dumped

    async def test_a_working_rollout_is_recorded_as_such(self):
        row, dumped = await self.run_one(GOOD)
        self.assertIs(row["aborted"], False)
        self.assertEqual(row["reward"], 1.0)
        self.assertIs(row["resolved"], True)
        self.assertIsNone(dumped["exception"])

    async def test_the_dumps_metrics_are_flattened_onto_the_row(self):
        # The row IS the session meta: the report reduces whatever is numeric on it.
        row, _ = await self.run_one(GOOD)
        self.assertEqual(row["num_tool_calls"], 9.0)
        self.assertEqual(row["llm_latency_sum"], 30.0)

    async def test_a_failed_dump_is_recorded_as_aborted(self):
        # The regression: it used to be recorded aborted=False, inflating rollouts_ok.
        row, _ = await self.run_one(FAILED)
        self.assertIs(row["aborted"], True)

    async def test_no_reward_is_read_off_a_failed_dump(self):
        # A null reward keeps the attempt in the pass@k denominator without counting as
        # a pass.
        row, _ = await self.run_one(FAILED)
        self.assertIsNone(row["reward"])
        self.assertIs(row["resolved"], False)

    async def test_a_failed_attempts_metrics_survive(self):
        # Aborting must not throw away the evidence a crash is diagnosed from.
        row, _ = await self.run_one(FAILED)
        self.assertEqual(row["num_tool_calls"], 3.0)
        self.assertEqual(row["llm_latency_sum"], 12.5)

    async def test_the_failed_dump_itself_is_kept_in_s3(self):
        _, dumped = await self.run_one(FAILED)
        self.assertIn("Traceback: agent crashed", dumped["exception"])
        self.assertEqual(dumped["rollout_dump_response"]["task_output"], {"patch": ""})
        self.assertEqual(dumped["rollout_dump_response"]["metrics"]["num_tool_calls"], 3.0)

    async def test_the_containers_own_trace_is_recorded_unwrapped(self):
        # The failure happened in the container, so keep that trace verbatim rather than
        # a local one that only points back at the harness.
        _, dumped = await self.run_one(FAILED)
        self.assertEqual(dumped["exception"], "Traceback: agent crashed")

    async def test_a_silent_failure_is_aborted_with_a_reason(self):
        # aborted with a null exception would be uncategorisable in the report.
        row, dumped = await self.run_one(SILENT)
        self.assertIs(row["aborted"], True)
        self.assertIsNone(row["reward"])
        self.assertIsInstance(dumped["exception"], str)

    async def test_run_one_never_raises(self):
        # Callers gather every rollout: one failure must not take the batch down.
        row, _ = await self.run_one(FAILED)
        self.assertIsInstance(row, dict)

    async def test_a_timed_out_rollout_is_also_recorded(self):
        class SlowSession(FakeSession):
            async def run(self, task):
                await asyncio.sleep(10)
                return GOOD

        def make_session(*args, **kwargs):
            return SlowSession()

        bae.AgentCoreSession = make_session
        config = eval_config(timeout=0.05)
        row = await bae.run_one(
            config,
            "arn:runtime",
            "arn:capacity",
            "2026-09-03T00:00:00",
            {"index": 7, "instance_id": "repo__proj-1"},
            0,
            bae.local_bounds(config),
            "s3://bucket/prefix",
            "sessions-table",
            "us-west-2",
        )
        self.assertIs(row["aborted"], True)
        self.assertEqual(row["agent_run_timeout_exceeded"], 1.0)
        # nothing came back at all, so there is no dump to keep
        ((_, dumped),) = self.uploads.items()
        self.assertIsNone(dumped["rollout_dump_response"])

    async def test_the_task_id_is_the_datasets_own_index(self):
        # So an eval task id names the same dataset row a training task id does.
        row, _ = await self.run_one(GOOD)
        self.assertEqual(row["task_id"], "7")

    async def test_the_s3_uri_is_recorded_on_the_row(self):
        row, _ = await self.run_one(GOOD)
        self.assertEqual(row["output_s3_uri"], "s3://bucket/prefix/" + row["session_id"])


if __name__ == "__main__":
    unittest.main()
