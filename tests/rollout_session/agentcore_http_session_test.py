"""Rollout result validation for HTTP sessions."""

import pytest
from pydantic import ValidationError

from agentcore_rl_toolkit.rollout_session.agentcore_http_session import to_dump


def test_result_mapping_preserves_application_data_without_inventing_a_reward():
    result = {"answer": "42", "metrics": {"turns": 2}}
    dump = to_dump(result)
    assert dump.task_output == result
    assert dump.reward is None
    assert dump.metrics == {"turns": 2}
    with pytest.raises(ValidationError):
        to_dump({"metrics": {"description": "done"}})
