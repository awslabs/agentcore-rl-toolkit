"""serve.sh starts stock vllm serve with both refresh plugins and the selected profile."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SERVING = Path(__file__).resolve().parents[2] / "src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/vllm_neuron"
CAPTURED_ENV = (
    "PYTHONPATH",
    "VLLM_API_KEY",
    "ART_WEIGHT_REFRESH_BASE",
    "ART_WEIGHT_REFRESH_ROOT",
    "ART_WEIGHT_REFRESH_CHECKPOINTS",
    "ART_WEIGHT_REFRESH_S3_REGION",
    "NEURON_VISIBLE_DEVICES",
    "NEURON_PLATFORM_TARGET_OVERRIDE",
    "NEURON_LIBTORCH_CACHE_ROOT",
    "NEURON_RT_VISIBLE_CORES",
)


def launch(tmp_path, *args, **overrides):
    binary = tmp_path / "bin"
    binary.mkdir(exist_ok=True)
    capture = tmp_path / "vllm.json"
    fake = binary / "vllm"
    fake.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"json.dump({{'argv': sys.argv[1:], 'env': {{k: os.environ.get(k) for k in {CAPTURED_ENV!r}}}}}, "
        f"open({str(capture)!r}, 'w'))\n"
    )
    fake.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ART_", "NEURON_", "VLLM_NEURON_")) and key != "PYTHONPATH"
    }
    env.update(
        {
            "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(tmp_path),
            "ART_CHECKPOINT_DIR": "/models/Qwen3.6-27B",
            "ART_WEIGHT_REFRESH_ROOT": "/nvme/merged",
            "ART_WEIGHT_REFRESH_CHECKPOINTS": "s3://bucket/skyrl-checkpoints",
            "NEURON_RT_VISIBLE_CORES": "0-3",
        }
    )
    env.update(overrides)
    env = {key: value for key, value in env.items() if value is not None}
    result = subprocess.run(["bash", str(SERVING / "serve.sh"), *args], env=env, capture_output=True, text=True)
    return result, (json.loads(capture.read_text()) if result.returncode == 0 else None)


def test_launches_vllm_serve_with_both_plugins_and_profile(tmp_path):
    result, captured = launch(tmp_path, "--max-num-seqs", "4")
    assert result.returncode == 0, result.stderr
    argv, env = captured["argv"], captured["env"]
    assert argv[:2] == ["serve", "/models/Qwen3.6-27B"]
    assert argv[argv.index("--config") + 1] == str(SERVING / "config/trn2.48xlarge/qwen3.6-27b.yaml")
    assert argv[argv.index("--middleware") + 1] == "server.ModelResolver"
    assert argv[argv.index("--worker-extension-cls") + 1] == "server.WeightRefreshWorker"
    assert argv[argv.index("--api-server-count") + 1] == "1"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--served-model-name") + 1] == "initial"
    assert argv[-2:] == ["--max-num-seqs", "4"]
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(SERVING)
    assert env["ART_WEIGHT_REFRESH_BASE"] == "/models/Qwen3.6-27B"
    assert env["ART_WEIGHT_REFRESH_ROOT"] == "/nvme/merged"
    assert env["NEURON_PLATFORM_TARGET_OVERRIDE"] == "trn2" and env["NEURON_VISIBLE_DEVICES"] == "0-63"
    assert env["NEURON_LIBTORCH_CACHE_ROOT"] == str(tmp_path / ".cache/art-inference/trn2.48xlarge/qwen3.6-27b")
    assert env["NEURON_RT_VISIBLE_CORES"] is None


def test_environment_overrides_profile_defaults(tmp_path):
    result, captured = launch(tmp_path, NEURON_VISIBLE_DEVICES="0-31", NEURON_LIBTORCH_CACHE_ROOT="/cache/27b")
    assert result.returncode == 0, result.stderr
    assert captured["env"]["NEURON_VISIBLE_DEVICES"] == "0-31"
    assert captured["env"]["NEURON_LIBTORCH_CACHE_ROOT"] == "/cache/27b"


@pytest.mark.parametrize("missing", ["ART_CHECKPOINT_DIR", "ART_WEIGHT_REFRESH_ROOT", "ART_WEIGHT_REFRESH_CHECKPOINTS"])
def test_required_settings_fail_before_starting_vllm(tmp_path, missing):
    result, _ = launch(tmp_path, **{missing: None})
    assert result.returncode != 0 and missing in result.stderr
    assert not (tmp_path / "vllm.json").exists()


def test_unknown_profile_fails_before_starting_vllm(tmp_path):
    result, _ = launch(tmp_path, ART_SERVING_PROFILE="qwen3.5-4b")
    assert result.returncode != 0 and "No serving profile" in result.stderr
    assert not (tmp_path / "vllm.json").exists()


def test_private_bind_requires_an_api_key(tmp_path):
    result, _ = launch(tmp_path, VLLM_NEURON_BIND_HOST="10.0.1.5")
    assert result.returncode != 0 and "VLLM_API_KEY" in result.stderr
    assert not (tmp_path / "vllm.json").exists()
    result, captured = launch(tmp_path, VLLM_NEURON_BIND_HOST="10.0.1.5", VLLM_API_KEY="secret")
    assert result.returncode == 0, result.stderr
    argv = captured["argv"]
    assert argv[argv.index("--host") + 1] == "10.0.1.5"
    # vLLM reads the key from the environment; it is not placed on the command line.
    assert captured["env"]["VLLM_API_KEY"] == "secret" and "secret" not in argv
