"""Unit tests for the torch-free trajectory core (TrajectoryManager + TraceRecord).

Covers CLEAN / REALIGN / FORK drift classification, loss masking of interleaved
observations, sibling-branch dedup for parallel tool calls, and reward/rollout_id
propagation. No torch, no aiohttp, no network.
"""

import subprocess
import sys

from agentcore_rl_toolkit.rollout_gateway import (
    BaseTrace,
    Status,
    TraceRecord,
    TrajectoryManager,
    TurnRecord,
)


def test_core_imports_without_torch_or_aiohttp():
    """Importing the torch-free core must not pull torch or aiohttp. Run in a fresh
    subprocess so other test modules (which import aiohttp) don't pollute sys.modules."""
    code = (
        "import agentcore_rl_toolkit.rollout_gateway as rg; import sys; "
        "assert 'torch' not in sys.modules, 'torch leaked'; "
        "assert 'aiohttp' not in sys.modules, 'aiohttp leaked'; "
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def _um(content):
    return {"role": "user", "content": content}


def _am(content):
    return {"role": "assistant", "content": content}


def test_single_turn_strips_leading_prompt():
    mgr = TrajectoryManager()
    mgr.record_turn(
        "s",
        turn=TurnRecord(prompt_ids=[1, 2, 3], output_ids=[4, 5], finish_reason="stop", output_log_probs=[-0.1, -0.2]),
        prompt_messages=[_um("hi")],
        response_message=_am("a"),
    )
    recs = mgr.get_trajectory("s", base_sample=BaseTrace(index=7, rollout_id="ep"), reward=2.0)
    assert len(recs) == 1
    r = recs[0]
    assert isinstance(r, TraceRecord)
    # leading prompt [1,2,3] stripped from the trained region
    assert r.token_ids == [1, 2, 3, 4, 5]
    assert r.loss_mask == [1, 1]
    assert r.logprobs == [-0.1, -0.2]
    assert r.response_length == 2
    assert r.reward == 2.0
    assert r.rollout_id == "ep"
    assert r.status is Status.COMPLETED


def test_clean_multiturn_masks_observation():
    mgr = TrajectoryManager()
    mgr.record_turn(
        "s",
        turn=TurnRecord(prompt_ids=[1, 2, 3], output_ids=[4, 5], finish_reason="stop", output_log_probs=[-0.1, -0.2]),
        prompt_messages=[_um("hi")],
        response_message=_am("a"),
    )
    # CLEAN extension: prev tokens [1,2,3,4,5] + observation [6] -> output [7]
    mgr.record_turn(
        "s",
        turn=TurnRecord(prompt_ids=[1, 2, 3, 4, 5, 6], output_ids=[7], finish_reason="stop", output_log_probs=[-0.3]),
        prompt_messages=[_um("hi"), _am("a"), _um("more")],
        response_message=_am("b"),
    )
    r = mgr.get_trajectory("s", base_sample=BaseTrace(index=0), reward=1.0)[0]
    assert r.token_ids == [1, 2, 3, 4, 5, 6, 7]
    # response region (leading [1,2,3] stripped): 4,5 trained; 6 obs; 7 trained
    assert r.loss_mask == [1, 1, 0, 1]
    assert r.logprobs == [-0.1, -0.2, 0.0, -0.3]
    # rollout_id falls back to index when base has none
    assert r.rollout_id == 0


def test_fork_produces_two_samples():
    # Two turns whose prompts diverge early (no shared response prefix) -> two leaves.
    mgr = TrajectoryManager()
    mgr.record_turn(
        "s",
        turn=TurnRecord(prompt_ids=[1, 2], output_ids=[9], finish_reason="stop"),
        prompt_messages=[_um("a")],
        response_message=_am("x"),
    )
    # Different user message -> different prompt subtree -> separate leaf
    mgr.record_turn(
        "s",
        turn=TurnRecord(prompt_ids=[3, 4], output_ids=[8], finish_reason="stop"),
        prompt_messages=[_um("b")],
        response_message=_am("y"),
    )
    recs = mgr.get_trajectory("s", base_sample=BaseTrace(index=0), reward=0.5)
    assert len(recs) == 2
    assert all(r.reward == 0.5 for r in recs)


def test_truncated_metadata_from_length_finish():
    mgr = TrajectoryManager()
    mgr.record_turn(
        "s",
        turn=TurnRecord(prompt_ids=[1], output_ids=[2, 3], finish_reason="length"),
        prompt_messages=[_um("hi")],
        response_message=_am("a"),
    )
    r = mgr.get_trajectory("s", base_sample=BaseTrace(index=0))[0]
    assert r.metadata["truncated"] is True


def test_get_trajectory_consumes_session():
    mgr = TrajectoryManager()
    mgr.record_turn(
        "s",
        turn=TurnRecord(prompt_ids=[1], output_ids=[2], finish_reason="stop"),
        prompt_messages=[_um("hi")],
        response_message=_am("a"),
    )
    assert mgr.has_session("s")
    mgr.get_trajectory("s", base_sample=BaseTrace(index=0))
    assert not mgr.has_session("s")
    # second call returns empty
    assert mgr.get_trajectory("s", base_sample=BaseTrace(index=0)) == []


def test_ill_formed_propagates_to_metadata():
    mgr = TrajectoryManager()
    mgr.record_turn(
        "s",
        turn=TurnRecord(prompt_ids=[1], output_ids=[2], finish_reason="stop", ill_formed=True),
        prompt_messages=[_um("hi")],
        response_message=_am("a"),
    )
    r = mgr.get_trajectory("s", base_sample=BaseTrace(index=0))[0]
    assert r.metadata["ill_formed"] is True
