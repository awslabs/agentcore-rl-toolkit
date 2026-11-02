"""``load_config`` unit tests: unknown keys are ignored, and ``config_cls`` lets a recipe load
the same YAML into a ``Config`` subclass that adds its own fields."""

from dataclasses import dataclass

from agentcore_rl_toolkit.backends.experimental.sagemaker.config import Config, load_config


@dataclass
class _RecipeConfig(Config):
    extra_threshold: float = 0.5
    extra_count: int = 3


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return str(path)


def test_default_class_is_config(tmp_path):
    config = load_config(_write(tmp_path, "batch_size: 8\n"))

    assert type(config) is Config
    assert config.batch_size == 8


def test_empty_file_gives_defaults(tmp_path):
    assert load_config(_write(tmp_path, "")) == Config()


def test_unknown_keys_are_ignored(tmp_path):
    config = load_config(_write(tmp_path, "batch_size: 8\nnot_a_field: 1\n"))

    assert config.batch_size == 8
    assert not hasattr(config, "not_a_field")


def test_config_cls_reads_subclass_fields(tmp_path):
    config = load_config(_write(tmp_path, "batch_size: 8\nextra_threshold: 0.9\n"), _RecipeConfig)

    assert type(config) is _RecipeConfig
    assert config.batch_size == 8  # base field
    assert config.extra_threshold == 0.9  # subclass field from YAML
    assert config.extra_count == 3  # subclass field not in YAML keeps its default


def test_base_config_ignores_subclass_fields(tmp_path):
    # One YAML for two recipes: the base loader skips fields only the subclass defines.
    path = _write(tmp_path, "batch_size: 8\nextra_threshold: 0.9\nextra_count: 7\n")

    assert load_config(path) == Config(batch_size=8)
    assert load_config(path, _RecipeConfig) == _RecipeConfig(batch_size=8, extra_threshold=0.9, extra_count=7)
