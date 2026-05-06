"""Tests for SlimeRunner — the Python entry point around train.sh."""

from pathlib import Path
from unittest.mock import patch

import yaml

from agentcore_rl_toolkit.backends.slime import SlimeRunner

REQUIRED_KWARGS = dict(
    exp_id="exp-1",
    agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/foo",
    s3_bucket="my-bucket",
    model_dir="/models/Qwen2.5-3B-Instruct",
    data_path="/data/gsm8k.jsonl",
    model_type="qwen2.5-3B",
)


def test_build_runtime_env_required_keys_and_wandb_opt_in():
    """PYTHONPATH + CUDA_DEVICE_MAX_CONNECTIONS are always present; wandb keys
    are opt-in, forwarded only when set in the parent env."""
    runner = SlimeRunner(**REQUIRED_KWARGS, megatron_dir="/opt/megatron")

    with patch.dict("os.environ", {}, clear=True):
        env_no_wandb = runner._build_runtime_env()
    assert env_no_wandb == {
        "env_vars": {
            "PYTHONPATH": "/opt/megatron",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
    }

    with patch.dict("os.environ", {"WANDB_API_KEY": "abc", "WANDB_ENTITY": "me"}, clear=True):
        env_with_wandb = runner._build_runtime_env()
    assert env_with_wandb["env_vars"]["WANDB_API_KEY"] == "abc"
    assert env_with_wandb["env_vars"]["WANDB_ENTITY"] == "me"


def test_from_yaml_round_trips(tmp_path: Path):
    """from_yaml should accept the same keys the dataclass does."""
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({**REQUIRED_KWARGS, "num_gpus": 4, "lr": 5e-7}))

    runner = SlimeRunner.from_yaml(config)

    assert runner.exp_id == "exp-1"
    assert runner.num_gpus == 4
    assert runner.lr == 5e-7


def test_slime_flags_pass_through_key_kwargs():
    """Flags the user cares about most (num_rollout, tp_size, lr, rollout_batch_size,
    rollout-function-path) must reach the slime CLI verbatim."""
    runner = SlimeRunner(**REQUIRED_KWARGS, tp_size=4, lr=5e-7, rollout_batch_size=16)
    flags = runner._build_slime_flags(num_rollout=10, model_args=["--fake-model-arg"], config_path="/tmp/cfg.yaml")

    # --fake-model-arg comes from the (mocked) model script and must be first
    assert flags[0] == "--fake-model-arg"
    # Key kwargs flow through as --k v pairs
    assert "--num-rollout" in flags and flags[flags.index("--num-rollout") + 1] == "10"
    assert "--tensor-model-parallel-size" in flags and flags[flags.index("--tensor-model-parallel-size") + 1] == "4"
    assert "--lr" in flags and flags[flags.index("--lr") + 1] == "5e-07"
    assert "--rollout-batch-size" in flags and flags[flags.index("--rollout-batch-size") + 1] == "16"
    # Integration hooks (load-bearing dotted paths)
    assert "agentcore_rl_toolkit.backends.slime.integration.rollout.generate_rollout" in flags
    assert "agentcore_rl_toolkit.backends.slime.integration.rewards.normalize_episode_rewards" in flags
    assert "/tmp/cfg.yaml" in flags


def test_extra_flags_are_appended_to_slime_cli():
    """The escape hatch: extra_flags end up at the end of the CLI unmodified."""
    runner = SlimeRunner(**REQUIRED_KWARGS, extra_flags=["--num-epoch", "3", "--use-rollout-routing-replay"])
    flags = runner._build_slime_flags(num_rollout=1, model_args=[], config_path="/tmp/cfg.yaml")

    assert flags[-3:] == ["--num-epoch", "3", "--use-rollout-routing-replay"]


def test_toolkit_config_yaml_includes_acr_pointers():
    """The temp yaml written for --custom-config-path must carry the ACR fields
    the rollout integration reads via SlimeArtConfig.from_args."""
    runner = SlimeRunner(**REQUIRED_KWARGS, model_id="qwen-served")

    with runner._write_toolkit_config() as path:
        data = yaml.safe_load(Path(path).read_text())

    assert data["agent_runtime_arn"] == REQUIRED_KWARGS["agent_runtime_arn"]
    assert data["s3_bucket"] == REQUIRED_KWARGS["s3_bucket"]
    assert data["exp_id"] == REQUIRED_KWARGS["exp_id"]
    assert data["model_id"] == "qwen-served"
