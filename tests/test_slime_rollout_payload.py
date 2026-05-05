"""Tests for the slime backend's agent-payload conversion.

``_sample_to_payload`` is the contract between slime's ``Sample`` object
and the agent's ``@rollout_entrypoint`` payload. The rule: the agent
receives ``Sample.metadata`` verbatim (shallow-copied).
"""

from types import SimpleNamespace

from agentcore_rl_toolkit.backends.slime.integration.rollout import _sample_to_payload


def test_metadata_is_returned_verbatim():
    """Core contract: the agent payload is Sample.metadata as-is.

    Fields on Sample outside metadata (prompt, label) are slime's own
    concern and must not leak into the payload.
    """
    sample = SimpleNamespace(
        prompt="slime-side prompt",
        label="slime-side label",
        metadata={"task_id": "t1", "answer": "42"},
    )
    assert _sample_to_payload(sample) == {"task_id": "t1", "answer": "42"}


def test_returned_dict_is_a_shallow_copy():
    """Mutations to the payload must not leak back into Sample.metadata.

    ``_process_one_episode`` later injects keys into ``Sample.metadata``
    (e.g. ``task_metadata``); the agent's view must stay stable.
    """
    metadata = {"prompt": "hi"}
    sample = SimpleNamespace(metadata=metadata)

    payload = _sample_to_payload(sample)
    payload["injected"] = True

    assert "injected" not in metadata


def test_missing_or_invalid_metadata_returns_empty_dict():
    """Defensive fallback when metadata is absent or not a dict."""
    for sample in [
        SimpleNamespace(),  # attribute absent
        SimpleNamespace(metadata=None),  # explicit None
        SimpleNamespace(metadata="not a dict"),  # wrong type
    ]:
        assert _sample_to_payload(sample) == {}
