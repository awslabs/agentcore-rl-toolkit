"""The existing launcher adds an external-sampler mode without a second trainer."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["default", "external", "s3", "api-key"])
def test_existing_launcher_keeps_default_and_external_modes(tmp_path, mode):
    external = mode != "default"
    binary = tmp_path / "bin"
    binary.mkdir()
    capture = tmp_path / "argv.json"
    uv = binary / "uv"
    uv.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        f'open({str(capture)!r}, "w").write(json.dumps('
        '{"args": sys.argv[1:]}))\n'
    )
    uv.chmod(0o755)
    lock = binary / "flock"
    lock.write_text("#!/bin/sh\nexit 0\n")
    lock.chmod(0o755)
    backend = tmp_path / "backend config.json"
    backend.write_text('{"nested": {"value": "with spaces"}}')
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "SKYRL_DIR": str(tmp_path),
        "MODEL_DIR": str(tmp_path / "model"),
        "SKYRL_STATE_DIR": str(tmp_path / "state"),
        "SKYRL_BACKEND_CONFIG": str(backend),
        "TMPDIR": str(tmp_path / "tmp"),
    }
    for key in (
        "SKYRL_EXTERNAL_INFERENCE_URL",
        "SKYRL_CHECKPOINTS_BASE",
        "SKYRL_TRAINING_BACKEND",
        "TINKER_BIND_HOST",
        "SKYRL_EXTERNAL_INFERENCE_API_KEY",
    ):
        env.pop(key, None)
    if external:
        env["SKYRL_EXTERNAL_INFERENCE_URL"] = "http://127.0.0.1:8001"
    if mode == "s3":
        env["SKYRL_CHECKPOINTS_BASE"] = "s3://bucket/skyrl-checkpoints"
    if mode == "api-key":
        env["SKYRL_EXTERNAL_INFERENCE_API_KEY"] = "secret"
    source = Path(__file__).resolve().parents[2] / "src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serve.sh"
    result = subprocess.run(["bash", str(source)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    captured = json.loads(capture.read_text())
    args = captured["args"]
    assert args[args.index("--backend-config") + 1] == backend.read_text()
    assert args[args.index("--backend") + 1] == ("fsdp" if external else "megatron")
    assert "skyrl.tinker.api" in args
    assert "--external-inference-lora-base" in args
    expected_base = "s3://bucket/skyrl-checkpoints" if mode == "s3" else str(tmp_path / "state" / "checkpoints")
    assert args[args.index("--checkpoints-base") + 1] == expected_base
    if external:
        assert args[args.index("--runtime-role") + 1] == "trainer"
        assert args[args.index("--external-inference-url") + 1] == "http://127.0.0.1:8001"
        assert args[args.index("--host") + 1] == "127.0.0.1"
        if mode == "api-key":
            assert args[args.index("--external-inference-api-key") + 1] == "secret"
        else:
            assert "--external-inference-api-key" not in args
    else:
        assert "--runtime-role" not in args and "--external-inference-url" not in args
        assert args[args.index("--host") + 1] == "0.0.0.0"
