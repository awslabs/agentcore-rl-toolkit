#!/usr/bin/env python
"""Unit tests for :class:`AgentCoreS3Session` -- the translation between an
``AgentCoreRLApp`` result dict and a :class:`RolloutDumpResponse`, the invoke arguments
that make the trainer's gateway capture the right session, the setup-stage session warm-up,
and the teardown guarantees (idempotent, and reached through a timeout). Fake client,
futures and data-plane client: no AWS.
"""

import asyncio
import time
import unittest
from unittest import IsolatedAsyncioTestCase, mock

from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister, PersistentDict
from agentcore_rl_toolkit.client import ACRRateLimiter
from agentcore_rl_toolkit.concurrency.priority_assigner import LocalPriorityAssigner
from agentcore_rl_toolkit.concurrency.priority_semaphore import LocalPrioritySemaphore
from agentcore_rl_toolkit.rollout_session import agentcore_s3_session as mod
from agentcore_rl_toolkit.rollout_session.agentcore_s3_session import AgentCoreS3Session
from agentcore_rl_toolkit.rollout_session.factory import make_session
from agentcore_rl_toolkit.rollout_session.lifecycle import (
    RolloutSessionBounds,
    run_rollout_with_bounds,
)

RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/agent-abc"
SESSION_ID = "verl_" + "0" * 32  # 37 chars, what RolloutSessionAgentLoop generates
BUCKET = "results-bucket"

LLM = {
    "model": "openai/Qwen3-8B",
    "base_url": "http://10.0.0.1:8123/v1",
    "api_key": SESSION_ID,
    "temperature": 1.0,
    "top_p": 1.0,
}


def task(**overrides) -> dict:
    """One per-rollout task dict, as the agent loop builds it."""
    return {
        # The group of rollouts this one belongs to -- verl's per-prompt `uid`, mapped to the
        # session layer's own name. Deliberately unlike `task_id` below, so the tests can tell
        # which of the two the session reads.
        "group_id": "group-1",
        "task_id": "org/repo",
        "payload": {"prompt": "migrate this", "repo": "org/repo"},
        "sampling_params": {"temperature": 1.0, "top_p": 1.0},
        "llm": dict(LLM),
        **overrides,
    }


class FakeFuture:
    """A :class:`RolloutFuture` stand-in: awaiting it polls, and a fetched result stops
    the ACR session exactly as the real one does."""

    def __init__(self, result: dict, result_key: str = "exp/org/repo/s.json", delay: float = 0.0):
        self.result = result
        self.result_key = result_key
        self.delay = delay
        self.cancel_calls = 0  # every cancel_async, including the no-op ones
        self.stops = 0  # the ones that actually stopped a session
        self._cancelled = False

    async def _poll(self) -> dict:
        if self.delay:
            await asyncio.sleep(self.delay)
        await self.cancel_async()  # the real future cancels once it has the result
        return self.result

    def __await__(self):
        return self._poll().__await__()

    async def cancel_async(self) -> bool:
        self.cancel_calls += 1
        if self._cancelled:
            return False
        self._cancelled = True
        self.stops += 1
        return True


class FakeClient:
    """Records what the session asked ACR to do, and hands back a fake future."""

    def __init__(self, result: dict | None = None, delay: float = 0.0, invoke_error: Exception | None = None):
        self.agent_runtime_arn = RUNTIME_ARN
        self.s3_bucket = BUCKET
        self.exp_id = "exp-1"
        self.result = result if result is not None else {"status_code": 200, "rewards": 1.0}
        self.delay = delay
        self.invoke_error = invoke_error
        self.invocations: list[dict] = []
        self.futures: list[FakeFuture] = []

    async def invoke_async(self, payload, session_id=None, input_id=None, **overrides) -> FakeFuture:
        self.invocations.append({"payload": payload, "session_id": session_id, "input_id": input_id, **overrides})
        await asyncio.sleep(0)  # a real invoke suspends, so sessions genuinely interleave
        if self.invoke_error is not None:
            raise self.invoke_error
        future = FakeFuture(self.result, delay=self.delay)
        self.futures.append(future)
        return future


async def one_ok_chunk():
    """The ``invoke_agent_runtime_command`` stream of a session that started."""
    yield {"chunk": {"contentStop": {"exitCode": 0}}}


class FakeAcrClient:
    """The shared ``bedrock-agentcore`` client, recording the session calls made on it."""

    def __init__(self, start_error: Exception | None = None):
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.start_error = start_error

    async def invoke_agent_runtime_command(self, *, runtimeSessionId, **kwargs):  # noqa: N803 - boto3 spelling
        self.started.append(runtimeSessionId)
        if self.start_error is not None:
            raise self.start_error
        return {"stream": one_ok_chunk()}

    async def stop_runtime_session(self, *, agentRuntimeArn, runtimeSessionId):  # noqa: N803 - boto3 spelling
        assert agentRuntimeArn == RUNTIME_ARN
        self.stopped.append(runtimeSessionId)


class SessionCase(IsolatedAsyncioTestCase):
    """Base case: ``setup`` warms the ACR session over the process-wide client, faked here.

    ``RolloutClient`` (``FakeClient`` below) is the agent-invoke path only; the warm-up is a
    data-plane call on a client this session borrows rather than owns.
    """

    def setUp(self):
        self.acr = FakeAcrClient()
        self.shared_client = mock.AsyncMock(return_value=self.acr)
        patch = mock.patch.object(mod, "shared_agentcore_client", self.shared_client)
        patch.start()
        self.addCleanup(patch.stop)


def state(session_id: str = SESSION_ID) -> PersistentDict:
    return PersistentDict(data={"session_id": session_id}, persister=NullPersister())


def make(client: FakeClient, session_id: str = SESSION_ID) -> AgentCoreS3Session:
    return AgentCoreS3Session(session_id, state(session_id), client=client)  # type: ignore[arg-type]


async def run_rollout(client: FakeClient, task_dict: dict | None = None):
    """The lifecycle a bounded rollout drives: enter, setup, run, teardown on exit."""
    session = make(client)
    async with session:
        await session.setup(task_dict or task())
        return await session.run(task_dict or task())


class DumpTranslationTest(SessionCase):
    """A result dict becomes a dump: reward by convention, metrics narrowed to scalars."""

    async def test_a_scalar_reward_and_the_numeric_metrics_reach_the_dump(self):
        result = {
            "status_code": 200,
            "rewards": 0.75,
            "metrics": {"compiled": True, "n_turns": 3, "latency_ms": 12.5, "log": "text"},
        }
        dump = await run_rollout(FakeClient(result))

        self.assertTrue(dump.is_successful())
        self.assertEqual(dump.reward, 0.75)
        # Non-numeric entries are dropped from the scalar metrics but survive in the output.
        self.assertEqual(dump.metrics, {"compiled": 1.0, "n_turns": 3.0, "latency_ms": 12.5})
        self.assertEqual(dump.task_output, result)

    async def test_a_list_of_rewards_scores_by_its_last_element(self):
        dump = await run_rollout(FakeClient({"status_code": 200, "rewards": [0.0, 0.5, 1.0]}))
        self.assertEqual(dump.reward, 1.0)

    async def test_a_result_without_metrics_dumps_no_metrics(self):
        dump = await run_rollout(FakeClient({"status_code": 200, "rewards": 1.0}))
        self.assertEqual(dump.metrics, {})

    async def test_a_result_with_no_reward_is_not_a_usable_rollout(self):
        # Deliberately not a 0.0: a missing reward is a broken agent, and scoring it zero
        # would teach the model that this rollout was bad.
        dump = await run_rollout(FakeClient({"status_code": 200, "conversation": []}))
        self.assertIsNone(dump.reward)
        self.assertFalse(dump.is_successful())
        self.assertIn("no reward", dump.failure_reason())

    async def test_an_empty_reward_list_reports_no_reward(self):
        dump = await run_rollout(FakeClient({"status_code": 200, "rewards": []}))
        self.assertIsNone(dump.reward)

    async def test_a_non_numeric_reward_raises_rather_than_scoring_zero(self):
        with self.assertRaises(ValueError) as caught:
            await run_rollout(FakeClient({"status_code": 200, "rewards": "great"}))
        self.assertIn("non-numeric", str(caught.exception))

    async def test_a_missing_status_code_is_treated_as_success(self):
        # `status_code` is injected only when absent, so an agent may own the key itself.
        dump = await run_rollout(FakeClient({"rewards": 1.0}))
        self.assertTrue(dump.is_successful())


class AgentFailureTest(SessionCase):
    """A handler that raised still saves a result; it must not look like a rollout."""

    async def test_a_non_200_status_becomes_the_dumps_exception(self):
        result = {"status_code": 500, "stop_reason": "ValueError: no pom.xml"}
        dump = await run_rollout(FakeClient(result))

        self.assertFalse(dump.is_successful())
        self.assertIn("status_code=500", dump.exception)
        self.assertIn("no pom.xml", dump.exception)
        # The whole result is kept: it is the only record of what the container did.
        self.assertEqual(dump.task_output, result)

    async def test_a_failed_result_is_not_mined_for_a_reward(self):
        dump = await run_rollout(FakeClient({"status_code": 500, "stop_reason": "boom", "rewards": 1.0}))
        self.assertIsNone(dump.reward)

    async def test_a_failure_without_a_stop_reason_still_reports(self):
        dump = await run_rollout(FakeClient({"status_code": 424}))
        self.assertIn("unknown", dump.exception)


class InvokeArgumentsTest(SessionCase):
    """What the container is told, and what makes its trajectory findable afterwards."""

    async def test_the_session_id_is_both_the_acr_session_and_the_capture_key(self):
        client = FakeClient()
        await run_rollout(client)

        invocation = client.invocations[0]
        self.assertEqual(invocation["session_id"], SESSION_ID)
        # The gateway keys trajectory capture off the api-key slot, so the two must agree
        # or the trainer would drain an empty session.
        self.assertEqual(invocation["api_key"], SESSION_ID)

    async def test_the_group_id_is_the_input_id_so_one_groups_rollouts_land_together(self):
        # `input_id` is the middle segment of the result key, so it decides what a listing
        # of the bucket groups by: the rollout group, not the dataset's name for the task.
        client = FakeClient()
        await run_rollout(client)
        self.assertEqual(client.invocations[0]["input_id"], "group-1")

    async def test_the_litellm_model_prefix_is_stripped_for_the_agents_client(self):
        client = FakeClient()
        await run_rollout(client)
        self.assertEqual(client.invocations[0]["model_id"], "Qwen3-8B")
        self.assertEqual(client.invocations[0]["base_url"], LLM["base_url"])

    async def test_the_rows_payload_is_forwarded_verbatim(self):
        client = FakeClient()
        task_dict = task()
        await run_rollout(client, task_dict)
        # Exactly the payload column: no trainer plumbing fields, no _rollout block (the
        # client adds that itself), nothing selected or renamed.
        self.assertEqual(client.invocations[0]["payload"], task_dict["payload"])

    async def test_no_sampling_params_are_forwarded(self):
        # The gateway applies the session's own sampling defaults over whatever the
        # container asks for; a second copy could only disagree with it.
        client = FakeClient()
        await run_rollout(client)
        self.assertNotIn("sampling_params", client.invocations[0])

    async def test_setup_starts_the_acr_session_and_records_where_it_ran(self):
        client = FakeClient()
        session = make(client)
        async with session:
            await session.setup(task())
            # The agent is not invoked yet; setup only starts the session it will run in,
            # so the microVM cold start is charged to container_setup_timeout.
            self.assertEqual(client.invocations, [])
            self.assertEqual(self.acr.started, [SESSION_ID])
            self.assertEqual(session.session_state["runtime_arn"], RUNTIME_ARN)
            self.assertEqual(session.session_state["result_s3_bucket"], BUCKET)
            self.assertEqual(session.session_state["result_s3_prefix"], "exp-1")
            # The start is a measured span, so its cost shows up in the session record.
            self.assertGreaterEqual(session.session_state["agentcore_setup"], 0)

    async def test_the_warm_up_rides_the_shared_client_of_the_runtimes_region(self):
        # Not RolloutClient's own boto3 client: that one is synchronous, so hundreds of
        # concurrent setups would queue on the default thread pool.
        await run_rollout(FakeClient())
        self.shared_client.assert_awaited_once_with("us-west-2")

    async def test_the_result_key_is_recorded_before_the_wait(self):
        client = FakeClient(delay=0.01)
        session = make(client)
        async with session:
            running = asyncio.create_task(session.run(task()))
            while "result_key" not in session.session_state:
                await asyncio.sleep(0)
            self.assertEqual(session.session_state["result_key"], "exp/org/repo/s.json")
            await running


class TaskContractTest(SessionCase):
    """The three task fields this session cannot invent -- ``payload``, ``group_id``, ``llm``
    -- each failing loudly, and the one (``task_id``) it merely records."""

    async def test_a_task_without_a_payload_dict_is_a_config_error(self):
        with self.assertRaises(ValueError) as caught:
            await run_rollout(FakeClient(), task(payload="migrate this"))
        self.assertIn("`payload`", str(caught.exception))

    async def test_a_task_without_a_group_id_is_a_config_error(self):
        # There is deliberately no fallback: `group_id` keys the result object, and inventing
        # one (the row index, the session id) would scatter a group's results rather than
        # group them.
        broken = task()
        del broken["group_id"]
        with self.assertRaises(ValueError) as caught:
            await run_rollout(FakeClient(), broken)
        self.assertIn("`group_id`", str(caught.exception))

    async def test_a_blank_group_id_is_rejected_like_a_missing_one(self):
        for blank in (None, "", "  "):
            with self.assertRaises(ValueError):
                await run_rollout(FakeClient(), task(group_id=blank))

    async def test_a_task_without_a_task_id_still_runs(self):
        # `task_id` is a recorded coordinate, not something this session reads: it stopped
        # being the result key when `group_id` took over as the input_id.
        broken = task()
        del broken["task_id"]
        dump = await run_rollout(FakeClient(), broken)
        self.assertTrue(dump.is_successful())

    async def test_an_llm_block_without_an_api_key_is_a_config_error(self):
        # The nastiest misconfiguration to debug: the rollout would run to completion and
        # capture nothing, so it is rejected before the invoke.
        with self.assertRaises(ValueError) as caught:
            await run_rollout(FakeClient(), task(llm={**LLM, "api_key": ""}))
        self.assertIn("api_key", str(caught.exception))

    async def test_a_task_without_an_llm_block_is_a_config_error(self):
        broken = task()
        del broken["llm"]
        with self.assertRaises(ValueError) as caught:
            await run_rollout(FakeClient(), broken)
        self.assertIn("`llm`", str(caught.exception))

    def test_a_session_id_shorter_than_acr_accepts_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as caught:
            make(FakeClient(), session_id="short")
        self.assertIn("33", str(caught.exception))


class ShutdownTest(SessionCase):
    """Teardown stops the ACR session once, whether the rollout finished or not."""

    async def test_a_finished_rollout_is_stopped_exactly_once(self):
        client = FakeClient()
        session = make(client)
        async with session:
            await session.run(task())
        # The result fetch stops the session; the exit's cancel is the idempotent no-op.
        future = client.futures[0]
        self.assertEqual(future.stops, 1)
        self.assertEqual(future.cancel_calls, 2)
        # And no second stop beside the future's, which owns this one.
        self.assertEqual(self.acr.stopped, [])

    async def test_shutdown_is_idempotent(self):
        client = FakeClient()
        session = make(client)
        async with session:
            await session.run(task())
        for _ in range(3):
            await session.shutdown()
        self.assertEqual(client.futures[0].stops, 1)
        self.assertEqual(self.acr.stopped, [])

    async def test_a_session_only_set_up_is_stopped_rather_than_left_warm(self):
        # No invoke means no future to ride, but setup started a microVM: unstopped it holds
        # an ACR session slot until the idle reaper takes it.
        client = FakeClient()
        session = make(client)
        async with session:
            await session.setup(task())
        self.assertEqual(client.futures, [])
        self.assertEqual(self.acr.stopped, [SESSION_ID])

    async def test_a_setup_that_failed_still_stops_the_session_it_may_have_created(self):
        # The dominant real failure: the warm-up 500s because command dispatch loses a race
        # with the container's start-up, and the microVM exists all the same.
        self.acr.start_error = RuntimeError("Received error (500) from runtime")
        session = make(FakeClient())
        with self.assertRaises(RuntimeError):
            async with session:
                await session.setup(task())
        self.assertEqual(self.acr.stopped, [SESSION_ID])

    async def test_stopping_a_started_session_twice_is_a_no_op(self):
        session = make(FakeClient())
        async with session:
            await session.setup(task())
        for _ in range(3):
            await session.shutdown()
        self.assertEqual(self.acr.stopped, [SESSION_ID])

    async def test_a_failing_stop_does_not_replace_the_error_that_ended_the_rollout(self):
        # Teardown runs while the rollout's own exception is in flight; a stop that raised
        # would take its place and cost the diagnosis.
        self.acr.start_error = RuntimeError("the real failure")
        self.acr.stop_runtime_session = mock.AsyncMock(side_effect=RuntimeError("stop denied"))
        session = make(FakeClient())
        with self.assertRaises(RuntimeError) as caught:
            async with session:
                await session.setup(task())
        self.assertIn("the real failure", str(caught.exception))

    async def test_a_failed_invoke_still_leaves_the_session_shut_down(self):
        # ACR may have accepted the invoke before it raised, so this stops the session too.
        client = FakeClient(invoke_error=RuntimeError("throttled"))
        session = make(client)
        with self.assertRaises(RuntimeError):
            async with session:
                await session.run(task())
        self.assertEqual(client.futures, [])
        self.assertEqual(self.acr.stopped, [SESSION_ID])

    async def test_a_session_that_never_started_is_not_stopped(self):
        # Nothing to tear down before setup, and a stop call would only spend ACR budget.
        session = make(FakeClient())
        async with session:
            pass
        self.assertEqual(self.acr.stopped, [])


class TimeoutTest(SessionCase):
    """A rollout that outlives ``agent_run_timeout`` must not leave a session running."""

    @staticmethod
    def bounds(agent_run_timeout: float) -> RolloutSessionBounds:
        return RolloutSessionBounds(
            container_semaphore=LocalPrioritySemaphore(4),
            rollout_semaphore=LocalPrioritySemaphore(4),
            container_priority_assigner=LocalPriorityAssigner(),
            rollout_priority_assigner=LocalPriorityAssigner(),
            container_setup_timeout=10.0,
            agent_run_timeout=agent_run_timeout,
        )

    async def test_the_timeout_fires_and_the_session_is_stopped_on_the_way_out(self):
        client = FakeClient(delay=10.0)
        session = make(client)
        with self.assertRaises(TimeoutError):
            await run_rollout_with_bounds(session.session_state, self.bounds(0.01), "k", session, task())

        # The stop rides the future the invoke returned, even though nobody ever read it.
        self.assertEqual(client.futures[0].stops, 1)
        self.assertEqual(session.session_state["agent_run_timeout_exceeded"], 1.0)

    async def test_a_rollout_inside_the_timeout_returns_its_dump(self):
        client = FakeClient(delay=0.01)
        session = make(client)
        dump = await run_rollout_with_bounds(session.session_state, self.bounds(10.0), "k", session, task())

        self.assertEqual(dump.reward, 1.0)
        self.assertEqual(session.session_state["container_setup_timeout_exceeded"], 0.0)
        self.assertEqual(client.futures[0].stops, 1)


class ResultLocationTest(unittest.TestCase):
    """``rollout_output_s3`` + ``experiment_name`` -> the bucket and prefix the agent writes
    under. The agent SDK's key layout has no prefix of its own, so one given here has to
    ride in the exp_id."""

    def test_a_bare_bucket_name_prefixes_results_with_the_experiment(self):
        self.assertEqual(mod.result_location("results-bucket", "exp-1"), ("results-bucket", "exp-1"))

    def test_the_s3_scheme_is_accepted(self):
        self.assertEqual(mod.result_location("s3://results-bucket", "exp-1"), ("results-bucket", "exp-1"))

    def test_a_prefix_is_folded_into_the_key_rather_than_dropped(self):
        # s3://bucket/runs + exp-1 -> s3://bucket/runs/exp-1/<uid>/<session>.json
        self.assertEqual(mod.result_location("s3://results-bucket/runs/", "exp-1"), ("results-bucket", "runs/exp-1"))

    def test_a_multi_segment_prefix_survives_whole(self):
        self.assertEqual(mod.result_location("s3://b/team/runs", "exp-1"), ("b", "team/runs/exp-1"))

    def test_a_location_naming_no_bucket_is_a_config_error(self):
        for bad in ("", "s3://", "  "):
            with self.assertRaises(ValueError) as caught:
                mod.result_location(bad, "exp-1")
            self.assertIn("bucket", str(caught.exception))

    def test_an_empty_experiment_name_is_a_config_error(self):
        with self.assertRaises(ValueError) as caught:
            mod.result_location(BUCKET, "  ")
        self.assertIn("experiment_name", str(caught.exception))


class ClientCacheTest(unittest.TestCase):
    """One client per config, because it owns the boto3 clients and their pools."""

    def setUp(self):
        mod.reset_client_cache()
        self.addCleanup(mod.reset_client_cache)
        patch = mock.patch.object(mod, "RolloutClient", lambda **kwargs: mock.Mock(kwargs=kwargs))
        patch.start()
        self.addCleanup(patch.stop)

    def client(self, **overrides):
        config = {
            "agentcore_runtime_arn": RUNTIME_ARN,
            "rollout_output_s3": f"s3://{BUCKET}",
            "experiment_name": "exp-1",
            **overrides,
        }
        return mod.get_or_create_rollout_client(**config)

    def test_the_same_config_shares_one_client(self):
        self.assertIs(self.client(), self.client())

    def test_two_spellings_of_one_location_still_share_a_client(self):
        # The cache keys off the resolved bucket/prefix, not the string it came from.
        self.assertIs(self.client(), self.client(rollout_output_s3=BUCKET))

    def test_a_different_runtime_gets_its_own_client(self):
        other = RUNTIME_ARN.replace("agent-abc", "agent-xyz")
        self.assertIsNot(self.client(), self.client(agentcore_runtime_arn=other))

    def test_a_different_experiment_gets_its_own_client(self):
        self.assertIsNot(self.client(), self.client(experiment_name="exp-2"))

    def test_the_pool_is_sized_for_concurrent_rollouts_not_boto3s_default_ten(self):
        self.assertEqual(self.client().kwargs["max_pool_connections"], 100)

    def test_the_clients_own_tps_limiter_is_effectively_off(self):
        # Throttling belongs to bounds.session_rate_limiter: it is cluster-wide, where the
        # client's limiter is per process and would admit 25 TPS per worker.
        self.assertEqual(self.client().kwargs["tps_limit"], mod.UNTHROTTLED_TPS)

    def test_the_unthrottled_limiter_costs_no_wait(self):
        # The whole point of the constant: at the client's default of 25 TPS these calls
        # would take 8 seconds between them.
        limiter = ACRRateLimiter(mod.UNTHROTTLED_TPS)
        start = time.perf_counter()
        for _ in range(200):
            limiter.wait_sync()
        self.assertLess(time.perf_counter() - start, 0.5)


class FactoryTest(unittest.TestCase):
    """``backend: agentcore_s3`` reaches this session, with its config applied."""

    def setUp(self):
        mod.reset_client_cache()
        self.addCleanup(mod.reset_client_cache)
        patch = mock.patch.object(mod, "RolloutClient", lambda **kwargs: mock.Mock(kwargs=kwargs))
        patch.start()
        self.addCleanup(patch.stop)

    def session(self, **overrides):
        config = {
            "backend": "agentcore_s3",
            "agentcore_runtime_arn": RUNTIME_ARN,
            "rollout_output_s3": f"s3://{BUCKET}/runs",
            "experiment_name": "exp-1",
            **overrides,
        }
        return make_session(SESSION_ID, config, state())  # type: ignore[arg-type]

    def test_the_backend_name_selects_this_session(self):
        session = self.session()
        self.assertIsInstance(session, AgentCoreS3Session)
        self.assertEqual(session._client.kwargs["s3_bucket"], BUCKET)
        self.assertEqual(session._client.kwargs["exp_id"], "runs/exp-1")

    def test_the_runtime_arn_key_is_the_one_the_agentcore_backend_uses(self):
        self.assertEqual(self.session()._client.kwargs["agent_runtime_arn"], RUNTIME_ARN)

    def test_sessions_of_one_run_share_the_process_client(self):
        self.assertIs(self.session()._client, self.session()._client)

    def test_the_connection_pool_is_configurable(self):
        session = self.session(max_pool_connections=512)
        self.assertEqual(session._client.kwargs["max_pool_connections"], 512)

    def test_an_unknown_backend_names_the_ones_that_exist(self):
        with self.assertRaises(ValueError) as caught:
            self.session(backend="nope")
        self.assertIn("agentcore_s3", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
