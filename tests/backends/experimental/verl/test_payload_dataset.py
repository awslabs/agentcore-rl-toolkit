"""PayloadDataset tests through verl's RLHFDataset load path: the class under
test subclasses RLHFDataset, and the value being tested is compatibility with
its parquet -> filter -> __getitem__ machinery (with an actual HF tokenizer,
since length filtering renders the synthesized prompt)."""

import json

import pytest
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from agentcore_rl_toolkit.backends.experimental.verl.dataset import PayloadDataset


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    # gpt2 ships no chat template; length filtering renders the synthesized
    # prompt through apply_chat_template, so give it a minimal one.
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return tok


def _write_parquet(tmp_path, rows):
    import pandas as pd

    path = str(tmp_path / "data.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _make_config(**overrides):
    return OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 512,
            "filter_overlong_prompts": False,
            "return_raw_chat": True,
            **overrides,
        }
    )


def test_synthesizes_prompt_from_payload(tokenizer, tmp_path):
    path = _write_parquet(
        tmp_path,
        [
            {"payload": {"prompt": "What is 2+2?", "answer": "4"}},
            {"payload": {"prompt": "What is 3+3?", "answer": "6"}},
        ],
    )
    ds = PayloadDataset(data_files=[path], tokenizer=tokenizer, config=_make_config(), processor=None)

    assert len(ds) == 2
    row = ds[0]
    # verl machinery got its chat-format raw_prompt...
    assert list(row["raw_prompt"]) == [{"role": "user", "content": "What is 2+2?"}]
    # ...and the agent's payload rides through untouched
    assert row["payload"]["prompt"] == "What is 2+2?"
    assert row["payload"]["answer"] == "4"


def test_custom_payload_prompt_field(tokenizer, tmp_path):
    path = _write_parquet(tmp_path, [{"payload": {"question": "Q1?", "answer": "A1"}}])
    ds = PayloadDataset(
        data_files=[path],
        tokenizer=tokenizer,
        config=_make_config(payload_prompt_field="question"),
        processor=None,
    )
    assert list(ds[0]["raw_prompt"]) == [{"role": "user", "content": "Q1?"}]


def test_explicit_prompt_column_wins(tokenizer, tmp_path):
    path = _write_parquet(
        tmp_path,
        [
            {
                "prompt": [{"role": "user", "content": "explicit"}],
                "payload": {"prompt": "from-payload"},
            }
        ],
    )
    ds = PayloadDataset(data_files=[path], tokenizer=tokenizer, config=_make_config(), processor=None)
    assert list(ds[0]["raw_prompt"]) == [{"role": "user", "content": "explicit"}]


def test_non_string_prompt_field_raises(tokenizer, tmp_path):
    path = _write_parquet(tmp_path, [{"payload": {"prompt": {"nested": "dict"}}}])
    with pytest.raises((ValueError, Exception), match="payload_prompt_field|must be a string|datasets.map"):
        PayloadDataset(data_files=[path], tokenizer=tokenizer, config=_make_config(), processor=None)


def test_length_filtering_works_on_synthesized_prompt(tokenizer, tmp_path):
    long_text = "word " * 2000
    path = _write_parquet(
        tmp_path,
        [
            {"payload": {"prompt": "short question"}},
            {"payload": {"prompt": long_text}},
        ],
    )
    ds = PayloadDataset(
        data_files=[path],
        tokenizer=tokenizer,
        config=_make_config(filter_overlong_prompts=True, max_prompt_length=128),
        processor=None,
    )
    assert len(ds) == 1  # the overlong row was filtered via the synthesized prompt


def test_payload_survives_json_roundtrip(tokenizer, tmp_path):
    """The payload column must stay JSON-serializable through the dataset (it
    becomes the ACR invoke body)."""
    path = _write_parquet(tmp_path, [{"payload": {"prompt": "q", "nested": {"a": [1, 2]}}}])
    ds = PayloadDataset(data_files=[path], tokenizer=tokenizer, config=_make_config(), processor=None)
    json.dumps(dict(ds[0]["payload"]))  # must not raise
