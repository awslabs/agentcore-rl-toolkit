"""Shared helpers for running ``rl_app`` rollouts in-process under test."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from agentcore_rl_toolkit import AgentCoreRLApp

EXAMPLE_DIR = Path(__file__).resolve().parents[1]

# Example modules that may bind the model classes by name.
EXAMPLE_MODULES = ("rl_app", "user_simulator")


def import_rl_app(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Freshly import ``rl_app`` with ``@rollout_entrypoint`` made transparent.

    ``invoke_agent`` is then the plain handler: no background task and no S3 save.
    """
    monkeypatch.setattr(AgentCoreRLApp, "rollout_entrypoint", lambda self, func: func)
    for name in EXAMPLE_MODULES:
        sys.modules.pop(name, None)
    return importlib.import_module("rl_app")


def build_payload(task_file: str, sampling_params: dict | None = None) -> dict:
    """Build the invocation payload the same way test_local.py does, minus the S3 fields."""
    task = json.loads((EXAMPLE_DIR / "tasks" / task_file).read_text())
    rollout = {"base_url": "http://assistant.test:4000/v1", "model_id": "test/assistant"}
    if sampling_params is not None:
        rollout["sampling_params"] = copy.deepcopy(sampling_params)
    return {
        "_task": {
            "domain": task["user_scenario"]["instructions"]["domain"],
            "user_scenario": task["user_scenario"],
            "initial_state": task.get("initial_state"),
            "evaluation_criteria": task.get("evaluation_criteria"),
        },
        "_rollout": rollout,
    }
