"""Unit tests for ``SageMakerSdkBackend`` — the SageMaker sampling seam.

The SDK itself is faked (see ``_install_sagemaker_stub`` below): what matters here is
the translation the backend owns — canonical sampling params to SDK ``SamplingParams``,
SDK sequence to ``TurnRecord`` — plus the two properties that are easy to regress and
expensive to notice: the blocking ``result()`` poll must not stall the event loop, and
``set_sampling_client()`` must swap weights without disturbing in-flight generations.
"""

import asyncio
import dataclasses
import enum
import sys
import threading
import types
from types import SimpleNamespace

import pytest


def _install_sagemaker_stub() -> None:
    """Register a stand-in for ``sagemaker.train.training_session`` when the SDK is absent.

    ``sagemaker_sdk.py`` imports that module at module scope, and the SDK is not a declared
    dependency of this package (not in the ``[gateway]`` extra). Stubbing the two symbols the
    backend touches keeps these tests running everywhere instead of skipping. A real
    installation always wins, so the same assertions also run against the actual SDK types.
    """
    try:
        import sagemaker.train.training_session  # noqa: F401
    except Exception:
        pass
    else:
        return

    class StopReason(enum.Enum):
        STOP = "stop"
        LENGTH = "length"

    @dataclasses.dataclass
    class SamplingParams:
        max_tokens: int = 4096
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int | None = None
        stop: list = dataclasses.field(default_factory=list)

    training_session = types.ModuleType("sagemaker.train.training_session")
    training_session.SamplingParams = SamplingParams
    training_session.StopReason = StopReason

    train = types.ModuleType("sagemaker.train")
    train.training_session = training_session

    sagemaker = types.ModuleType("sagemaker")
    sagemaker.train = train

    sys.modules["sagemaker"] = sagemaker
    sys.modules["sagemaker.train"] = train
    sys.modules["sagemaker.train.training_session"] = training_session


_install_sagemaker_stub()

# E402: these imports must follow the stub install above — importing the backend
# (directly or via the SDK symbols it shares) requires the SDK module to exist.
from sagemaker.train.training_session import SamplingParams, StopReason  # noqa: E402

from agentcore_rl_toolkit.rollout_gateway.sampling_backends.base import SamplingBackend  # noqa: E402
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sagemaker_sdk import SageMakerSdkBackend  # noqa: E402
from agentcore_rl_toolkit.rollout_gateway.trajectory import TurnRecord  # noqa: E402

_ABSENT = object()


def make_sequence(tokens, logprobs=_ABSENT, stop_reason=StopReason.STOP):
    """One SDK ``sequences[0]``. Omit ``logprobs`` to model a backend that returns none."""
    seq = SimpleNamespace(tokens=list(tokens), stop_reason=stop_reason)
    if logprobs is not _ABSENT:
        seq.logprobs = logprobs
    return seq


class FakeOperation:
    """The SDK's ``APIFuture``: ``result(timeout=...)`` blocks until the op completes."""

    def __init__(self, result_value, *, gate=None):
        self._result = result_value
        self._gate = gate
        self.timeouts = []

    def result(self, timeout=None):
        self.timeouts.append(timeout)
        if self._gate is not None and not self._gate.wait(5):
            raise AssertionError("gate never released — result() likely blocked the event loop")
        return self._result


class FakeSamplingClient:
    """Records every ``sample()`` call and hands back a scripted sequence."""

    def __init__(self, sequence=None, *, gate=None):
        self.sequence = sequence if sequence is not None else make_sequence([0])
        self.gate = gate
        self.calls = []
        self.ops = []

    def sample(self, *, prompt, num_samples, sampling_params):
        self.calls.append(SimpleNamespace(prompt=prompt, num_samples=num_samples, sampling_params=sampling_params))
        op = FakeOperation(SimpleNamespace(sequences=[self.sequence]), gate=self.gate)
        self.ops.append(op)
        return op


# ---------------------------------------------------------------------------
# TurnRecord translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_turn_record():
    client = FakeSamplingClient(make_sequence([4, 5], logprobs=[-0.1, -0.2]))
    backend = SageMakerSdkBackend(client)

    record = await backend.generate(prompt_ids=[1, 2, 3], sampling_params={})

    assert isinstance(record, TurnRecord)
    assert record.prompt_ids == [1, 2, 3]
    assert record.output_ids == [4, 5]
    assert record.output_log_probs == [-0.1, -0.2]
    assert record.finish_reason == "stop"
    assert record.ill_formed is False


@pytest.mark.asyncio
async def test_length_stop_reason_maps_to_length():
    backend = SageMakerSdkBackend(FakeSamplingClient(make_sequence([4], stop_reason=StopReason.LENGTH)))

    record = await backend.generate(prompt_ids=[1], sampling_params={})

    assert record.finish_reason == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize("logprobs", [_ABSENT, None], ids=["attribute-absent", "attribute-none"])
async def test_missing_logprobs_becomes_empty_list(logprobs):
    backend = SageMakerSdkBackend(FakeSamplingClient(make_sequence([4, 5], logprobs=logprobs)))

    record = await backend.generate(prompt_ids=[1], sampling_params={})

    assert record.output_log_probs == []


@pytest.mark.asyncio
async def test_prompt_ids_are_copied_not_aliased():
    """The caller's list must not be able to mutate a recorded trajectory."""
    client = FakeSamplingClient()
    backend = SageMakerSdkBackend(client)
    prompt_ids = [1, 2, 3]

    record = await backend.generate(prompt_ids=prompt_ids, sampling_params={})
    prompt_ids.append(999)

    assert record.prompt_ids == [1, 2, 3]
    assert client.calls[0].prompt == [1, 2, 3]


@pytest.mark.asyncio
async def test_multimodal_kwargs_accepted():
    """The protocol passes image/video data; this backend takes and ignores them."""
    backend = SageMakerSdkBackend(FakeSamplingClient(make_sequence([4])))

    record = await backend.generate(
        prompt_ids=[1], sampling_params={}, session_id="sid-1", image_data=b"png", video_data=None
    )

    assert record.output_ids == [4]


def test_satisfies_sampling_backend_protocol():
    assert isinstance(SageMakerSdkBackend(FakeSamplingClient()), SamplingBackend)


# ---------------------------------------------------------------------------
# Sampling param translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sampling_params_are_translated():
    client = FakeSamplingClient()
    backend = SageMakerSdkBackend(client)

    await backend.generate(
        prompt_ids=[1],
        sampling_params={
            "max_new_tokens": 128,
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 20,
            "stop": ["</s>"],
        },
    )

    sp = client.calls[0].sampling_params
    assert isinstance(sp, SamplingParams)
    assert sp.max_tokens == 128
    assert sp.temperature == 0.3
    assert sp.top_p == 0.8
    assert sp.top_k == 20
    assert sp.stop == ["</s>"]


@pytest.mark.asyncio
async def test_sampling_param_defaults():
    client = FakeSamplingClient()
    backend = SageMakerSdkBackend(client)

    await backend.generate(prompt_ids=[1], sampling_params={})

    sp = client.calls[0].sampling_params
    assert sp.max_tokens == 4096
    assert sp.temperature == 1.0
    assert sp.top_p == 1.0
    assert sp.top_k is None
    assert sp.stop == []


@pytest.mark.asyncio
async def test_explicit_none_stop_becomes_empty_list():
    """The gateway may pass ``stop: None``; the SDK wants a list."""
    client = FakeSamplingClient()
    backend = SageMakerSdkBackend(client)

    await backend.generate(prompt_ids=[1], sampling_params={"stop": None})

    assert client.calls[0].sampling_params.stop == []


@pytest.mark.asyncio
async def test_max_new_tokens_is_coerced_to_int():
    """max_new_tokens can arrive as a string from a JSON request body."""
    client = FakeSamplingClient()
    backend = SageMakerSdkBackend(client)

    await backend.generate(prompt_ids=[1], sampling_params={"max_new_tokens": "256"})

    assert client.calls[0].sampling_params.max_tokens == 256


@pytest.mark.asyncio
async def test_sample_requests_exactly_one_sequence():
    client = FakeSamplingClient()
    backend = SageMakerSdkBackend(client)

    await backend.generate(prompt_ids=[1, 2], sampling_params={})

    assert client.calls[0].num_samples == 1
    assert client.calls[0].prompt == [1, 2]


# ---------------------------------------------------------------------------
# Poll timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_poll_timeout():
    client = FakeSamplingClient()
    backend = SageMakerSdkBackend(client)

    await backend.generate(prompt_ids=[1], sampling_params={})

    assert client.ops[0].timeouts == [900.0]


@pytest.mark.asyncio
async def test_poll_timeout_override():
    client = FakeSamplingClient()
    backend = SageMakerSdkBackend(client)

    await backend.generate(prompt_ids=[1], sampling_params={"timeout": 12.5})

    assert client.ops[0].timeouts == [12.5]


# ---------------------------------------------------------------------------
# Weight rebinding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_sampling_client_rebinds_for_next_generate():
    old = FakeSamplingClient(make_sequence([1, 1]))
    new = FakeSamplingClient(make_sequence([2, 2]))
    backend = SageMakerSdkBackend(old)

    assert (await backend.generate(prompt_ids=[7], sampling_params={})).output_ids == [1, 1]
    backend.set_sampling_client(new)
    assert (await backend.generate(prompt_ids=[7], sampling_params={})).output_ids == [2, 2]

    assert len(old.calls) == 1
    assert len(new.calls) == 1


@pytest.mark.asyncio
async def test_in_flight_generate_finishes_against_the_old_client():
    """Rebinding mid-generate must not retarget or corrupt the running request."""
    gate = threading.Event()
    old = FakeSamplingClient(make_sequence([1, 1]), gate=gate)
    new = FakeSamplingClient(make_sequence([2, 2]))
    backend = SageMakerSdkBackend(old)

    task = asyncio.create_task(backend.generate(prompt_ids=[7], sampling_params={}))
    await asyncio.sleep(0.05)  # let the request reach the (blocked) old client
    assert not task.done()

    backend.set_sampling_client(new)
    gate.set()

    assert (await asyncio.wait_for(task, timeout=5)).output_ids == [1, 1]
    assert (await backend.generate(prompt_ids=[8], sampling_params={})).output_ids == [2, 2]
    assert len(old.calls) == 1
    assert len(new.calls) == 1


@pytest.mark.asyncio
async def test_blocking_poll_does_not_stall_the_event_loop():
    """``result()`` blocks; it must run in a thread or the gateway serves one rollout at a time."""
    gate = threading.Event()
    backend = SageMakerSdkBackend(FakeSamplingClient(make_sequence([9]), gate=gate))

    task = asyncio.create_task(backend.generate(prompt_ids=[1], sampling_params={}))
    await asyncio.sleep(0.05)
    assert not task.done()

    gate.set()  # released from the event loop thread: only reachable if the loop kept running

    assert (await asyncio.wait_for(task, timeout=5)).output_ids == [9]


@pytest.mark.asyncio
async def test_concurrent_generates_overlap():
    """Two rollouts sampling at once must not serialize behind each other."""
    gate = threading.Event()
    backend = SageMakerSdkBackend(FakeSamplingClient(make_sequence([9]), gate=gate))

    tasks = [asyncio.create_task(backend.generate(prompt_ids=[i], sampling_params={})) for i in range(2)]
    await asyncio.sleep(0.05)
    gate.set()

    records = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    assert [r.output_ids for r in records] == [[9], [9]]
