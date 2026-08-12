"""Shared fixtures for the experimental slime backend tests.

Slime is not a declared dev dependency (it requires a full CUDA-13 training
stack), so the rollout module is tested by injecting fake slime modules into
sys.modules before import.  The rewards module has no slime dependency and
imports cleanly in any environment.
"""

from __future__ import annotations

import enum
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Inject fake slime modules at collection time so test files can import
# rollout.py (which has top-level `from slime...` imports) without the full
# training stack installed.  Done at module level so it runs before pytest
# imports any test file in this directory.
# ---------------------------------------------------------------------------

if "slime" not in sys.modules:
    _http = ModuleType("slime.utils.http_utils")
    _http.get_host_info = MagicMock(return_value=("hostname", "127.0.0.1"))

    _misc = ModuleType("slime.utils.misc")
    _misc.SingletonMeta = type  # AgentCoreRLService is never instantiated in unit tests

    _proc = ModuleType("slime.utils.processing_utils")
    _proc.load_tokenizer = MagicMock(return_value=MagicMock())

    _types = ModuleType("slime.utils.types")
    # _types.Sample filled in after FakeSample is defined below

    _utils = ModuleType("slime.utils")
    _slime = ModuleType("slime")
    _slime.utils = _utils

    sys.modules.update(
        {
            "slime": _slime,
            "slime.utils": _utils,
            "slime.utils.http_utils": _http,
            "slime.utils.misc": _misc,
            "slime.utils.processing_utils": _proc,
            "slime.utils.types": _types,
        }
    )
else:
    _types = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fake slime.utils.types.Sample
# ---------------------------------------------------------------------------


class FakeStatus(enum.Enum):
    COMPLETED = "completed"
    ABORTED = "ABORTED"


class FakeSample:
    """Minimal duck-type of slime's Sample for unit testing."""

    Status = FakeStatus

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        for attr in (
            "index",
            "group_index",
            "rollout_id",
            "prompt",
            "label",
            "session_id",
            "tokens",
            "response",
            "response_length",
            "loss_mask",
            "rollout_log_probs",
            "reward",
            "status",
            "metadata",
            "remove_sample",
        ):
            if not hasattr(self, attr):
                setattr(self, attr, None)


if _types is not None:
    _types.Sample = FakeSample


@pytest.fixture
def make_sample():
    """Factory for FakeSample instances."""

    def _make(index=0, group_index=0, rollout_id=None, metadata=None, **kwargs):
        return FakeSample(
            index=index,
            group_index=group_index,
            rollout_id=rollout_id if rollout_id is not None else index,
            metadata=metadata or {},
            **kwargs,
        )

    return _make
