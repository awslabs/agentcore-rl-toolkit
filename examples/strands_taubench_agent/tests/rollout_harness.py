"""Shared helpers for running ``rl_app`` rollouts in-process under test."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

EXAMPLE_DIR = Path(__file__).resolve().parents[1]

# Example modules that may bind the model classes by name.
EXAMPLE_MODULES = ("rl_app", "user_simulator")


def import_rl_app() -> ModuleType:
    """Freshly import ``rl_app``.

    ``@app.entrypoint`` registers ``invoke_agent`` and returns it unchanged, so tests call the
    plain handler, without the HTTP server or the start/get invocation lifecycle.
    """
    for name in EXAMPLE_MODULES:
        sys.modules.pop(name, None)
    return importlib.import_module("rl_app")


def build_payload(task_file: str, sampling_params: dict | None = None) -> dict:
    """Build the invocation payload the same way test_local.py does."""
    task = json.loads((EXAMPLE_DIR / "tasks" / task_file).read_text())
    config = {"base_url": "http://assistant.test:4000/v1", "model_id": "test/assistant"}
    if sampling_params is not None:
        config["sampling_params"] = copy.deepcopy(sampling_params)
    return {
        "_task": {
            "domain": task["user_scenario"]["instructions"]["domain"],
            "user_scenario": task["user_scenario"],
            "initial_state": task.get("initial_state"),
            "evaluation_criteria": task.get("evaluation_criteria"),
        },
        "_config": config,
    }
