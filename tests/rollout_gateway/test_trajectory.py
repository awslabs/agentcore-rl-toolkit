"""Unit tests for the torch-free trajectory core (TrajectoryManager + TraceRecord).

Covers CLEAN / FORK drift classification, loss masking of interleaved
observations, sibling-branch dedup for parallel tool calls, and reward/rollout_id
propagation. No torch, no aiohttp, no network.
"""

import subprocess
import sys

import pytest

from agentcore_rl_toolkit.rollout_gateway import (
    AuxiliaryTokenSamples,
    BaseTrace,
    MessageNode,
    Status,
    TraceRecord,
    TrajectoryManager,
    TurnRecord,
)


def test_leaves_handles_chain_deeper_than_recursion_limit():
    """leaves() drains a near-linear tree (one node per message, as long rollouts
    produce) that is far deeper than the interpreter's recursion limit, bounding
    stack use by the tree's breadth rather than its depth."""
    depth = sys.getrecursionlimit() * 3
    root = MessageNode()
    node = root
    for _ in range(depth):
        node = node.add_child(MessageNode(role="assistant"))
    leaves = list(root.leaves())
    assert len(leaves) == 1
    assert leaves[0] is node


def test_leaves_preserves_left_to_right_dfs_order():
    """leaves() yields in left-to-right DFS order: two branches off the root, each
    two nodes deep, yield the left leaf before the right."""
    root = MessageNode()
    left = root.add_child(MessageNode(role="user", message={"content": "L"}))
    left_leaf = left.add_child(MessageNode(role="assistant", message={"content": "L2"}))
    right = root.add_child(MessageNode(role="user", message={"content": "R"}))
    right_leaf = right.add_child(MessageNode(role="assistant", message={"content": "R2"}))
    assert list(root.leaves()) == [left_leaf, right_leaf]


def test_trace_record_rejects_loss_mask_longer_than_tokens():
    with pytest.raises(ValueError, match="loss_mask has 3 entries.*token_ids has only 2"):
        TraceRecord(token_ids=[1, 2], loss_mask=[1, 0, 1], logprobs=[-0.1, 0.0, -0.2])


def test_trace_record_rejects_misaligned_logprobs():
    with pytest.raises(ValueError, match="logprobs has 1 entries.*loss_mask has 2"):
        TraceRecord(token_ids=[1, 2, 3], loss_mask=[1, 1], logprobs=[-0.1])


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


@pytest.mark.parametrize("rewrite", [False, True], ids=["token-drift", "message-rewrite"])
@pytest.mark.parametrize("first_response_length", [2, 1500])
def test_drift_preserves_both_generations(rewrite, first_response_length):
    mgr = TrajectoryManager()
    first = TurnRecord(
        prompt_ids=[1, 2],
        output_ids=[9] * first_response_length,
        finish_reason="stop",
        output_log_probs=[-0.1] * first_response_length,
    )
    second = TurnRecord(prompt_ids=[1, 2, 8, 3], output_ids=[7], finish_reason="stop", output_log_probs=[-0.2])
    mgr.record_turn("s", turn=first, prompt_messages=[_um("q")], response_message=_am("a"))
    mgr.record_turn(
        "s",
        turn=second,
        prompt_messages=[_um("q"), _am("edited" if rewrite else "a"), _um("more")],
        response_message=_am("b"),
    )
    recs = mgr.get_trajectory("s", base_sample=BaseTrace(rollout_id="ep"), reward=1.0)
    assert len(recs) == 2
    for rec, turn in zip(recs, [first, second], strict=True):
        assert rec.token_ids == turn.prompt_ids + turn.output_ids
        assert rec.loss_mask == [1] * len(turn.output_ids)
        assert rec.logprobs == turn.output_log_probs
        assert rec.reward == 1.0 and rec.rollout_id == "ep"


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


def _samples(*tokens):
    """Distinguishable 2-sample rows per output token: row i = [token, token + 100]."""
    return AuxiliaryTokenSamples(ids=[[t, t + 100] for t in tokens], versions=[t % 3 for t in tokens])


def test_token_samples_follow_trained_responses():
    mgr = TrajectoryManager()
    mgr.record_turn(
        "s",
        turn=TurnRecord(
            prompt_ids=[1, 2, 3], output_ids=[4, 5], finish_reason="stop", output_auxiliary_token_samples=_samples(4, 5)
        ),
        prompt_messages=[_um("hi")],
        response_message=_am("a"),
    )
    mgr.record_turn(
        "s",
        turn=TurnRecord(
            prompt_ids=[1, 2, 3, 4, 5, 6],
            output_ids=[7],
            finish_reason="stop",
            output_auxiliary_token_samples=_samples(7),
        ),
        prompt_messages=[_um("hi"), _am("a"), _um("more")],
        response_message=_am("b"),
    )
    r = mgr.get_trajectory("s", base_sample=BaseTrace(index=0))[0]
    # response region [4, 5, 6, 7]: the observation 6 has no samples
    assert r.loss_mask == [1, 1, 0, 1]
    assert [(offset, s.ids) for offset, s in r.auxiliary_token_samples] == [(0, [[4, 104], [5, 105]]), (3, [[7, 107]])]


def test_token_samples_omitted_for_responses_reemitted_as_context():
    mgr = TrajectoryManager()
    mgr.record_turn(
        "s",
        turn=TurnRecord(
            prompt_ids=[1, 2, 3], output_ids=[4, 5], finish_reason="stop", output_auxiliary_token_samples=_samples(4, 5)
        ),
        prompt_messages=[_um("hi")],
        response_message=_am("a"),
    )
    for obs, out in ((6, 7), (8, 9)):
        mgr.record_turn(
            "s",
            turn=TurnRecord(
                prompt_ids=[1, 2, 3, 4, 5, obs],
                output_ids=[out],
                finish_reason="stop",
                output_auxiliary_token_samples=_samples(out),
            ),
            prompt_messages=[_um("hi"), _am("a"), _um(f"tool {obs}")],
            response_message=_am("b"),
        )
    first, second = mgr.get_trajectory("s", base_sample=BaseTrace(index=0))
    assert [offset for offset, _ in first.auxiliary_token_samples] == [0, 3]
    # the shared response trains on the first leaf only, so only the second leaf's own samples remain
    assert second.loss_mask == [0, 0, 0, 1]
    assert [(offset, s.ids) for offset, s in second.auxiliary_token_samples] == [(3, [[9, 109]])]


def test_token_samples_truncated_with_max_sample_tokens():
    mgr = TrajectoryManager()
    mgr.record_turn(
        "s",
        turn=TurnRecord(
            prompt_ids=[1, 2],
            output_ids=[4, 5, 6],
            finish_reason="stop",
            output_auxiliary_token_samples=_samples(4, 5, 6),
        ),
        prompt_messages=[_um("hi")],
        response_message=_am("a"),
    )
    r = mgr.get_trajectory("s", base_sample=BaseTrace(index=0), max_sample_tokens=4)[0]
    assert r.loss_mask == [1, 1]
    ((offset, samples),) = r.auxiliary_token_samples
    assert offset == 0
    assert samples.ids == [[4, 104], [5, 105]]
    assert samples.versions == [1, 2]
