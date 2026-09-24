"""Check that Invoke failures do not stop later training batches or evaluations."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from botocore.exceptions import ClientError

from agentcore_rl_toolkit.backends.tinker_api import train as recipe
from agentcore_rl_toolkit.rollout_gateway.trace import TraceRecord


@pytest.mark.asyncio
async def test_existing_run_is_not_overwritten(tmp_path):
    original = {"config.json": '{"exp_id":"previous-run"}\n', "metrics.jsonl": '{"step":1}\n'}
    for name, contents in original.items():
        (tmp_path / name).write_text(contents)
    config = recipe.Config(
        endpoint="endpoint",
        base_model="model",
        tokenizer="tokenizer",
        dataset="unused",
        agent_runtime_arn="runtime",
        s3_bucket="bucket",
        gateway_host="127.0.0.1",
        output_dir=str(tmp_path),
    )
    with pytest.raises(FileExistsError, match="config.json"):
        await recipe.train(config)
    assert {name: (tmp_path / name).read_text() for name in original} == original


@pytest.mark.parametrize(
    ("empty_first_training_batch", "failed_evaluation_inputs"),
    [
        pytest.param(False, {1}, id="partial-failures"),
        pytest.param(True, {0, 1, 2}, id="empty-batch-and-evaluation"),
    ],
)
@pytest.mark.asyncio
async def test_invoke_failures_do_not_interrupt_subsequent_updates(
    tmp_path, monkeypatch, failed_evaluation_inputs, empty_first_training_batch
):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text('{"kind":"train"}\n' * 2)
    eval_path.write_text("".join(json.dumps({"kind": "eval", "index": i}) + "\n" for i in range(3)))
    output = tmp_path / "run"
    sessions = {}
    calls = {"train": 0, "eval": 0}

    class Gateway:
        def __init__(self, **kwargs):
            self.app = recipe.web.Application()

        def create_session(self, sid, **kwargs):
            sessions[sid] = []

        async def finish_session(self, sid, *, base_sample, reward):
            for record in sessions[sid]:
                record.rollout_id = base_sample.rollout_id
            return sessions[sid]

        async def drop_session(self, sid):
            del sessions[sid]

    async def invoke(payload, *, session_id, **kwargs):
        kind = payload["kind"]
        index = calls[kind]
        calls[kind] += 1
        if kind == "eval":
            fail = index // 3 == 1 and payload["index"] in failed_evaluation_inputs
            reward = 1
        else:
            fail = index < 4 and (empty_first_training_batch or index == 3)
            reward = [1, 0, 1, 0][index % 4]
        if fail:
            raise ClientError(
                {"Error": {"Code": "RuntimeClientError", "Message": "Received error (502) from runtime."}},
                "InvokeAgentRuntime",
            )
        sessions[session_id] = [TraceRecord(token_ids=[10, 11], loss_mask=[1], logprobs=[-0.3])]
        return SimpleNamespace(result_async=AsyncMock(return_value={"rewards": reward}))

    training_client = SimpleNamespace(
        save_weights_and_get_sampling_client_async=AsyncMock(return_value=object()),
    )
    service = SimpleNamespace(create_lora_training_client_async=AsyncMock(return_value=training_client))
    update = AsyncMock(return_value=(SimpleNamespace(metrics={}), SimpleNamespace(metrics={})))
    monkeypatch.setattr(recipe.tinker, "ServiceClient", lambda **kwargs: service)
    monkeypatch.setattr(recipe.AutoTokenizer, "from_pretrained", Mock())
    monkeypatch.setattr(recipe, "HfTemplateRenderer", Mock())
    monkeypatch.setattr(recipe, "RolloutGateway", Gateway)
    monkeypatch.setattr(recipe, "RolloutClient", lambda **kwargs: SimpleNamespace(invoke_async=invoke))
    monkeypatch.setattr(recipe, "update_policy", update)
    monkeypatch.setattr(recipe, "save_checkpoint", AsyncMock(return_value="tinker://checkpoint"))
    config = recipe.Config(
        endpoint="endpoint",
        base_model="model",
        tokenizer="tokenizer",
        dataset=str(train_path),
        agent_runtime_arn="runtime",
        s3_bucket="bucket",
        gateway_host="127.0.0.1",
        output_dir=str(output),
        steps=2,
        batch_size=1,
        group_size=4,
        evaluation_dataset=str(eval_path),
        evaluation_batch_size=3,
        evaluation_interval=1,
    )
    await recipe.train(config)

    expected_updates = [0, 1] if empty_first_training_batch else [1, 2]
    assert update.await_count == expected_updates[-1]
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in metrics] == [1, 2]
    assert [row["train/optimizer_steps"] for row in metrics] == expected_updates
    aggregate = json.loads((output / "evaluation-0001.json").read_text())
    assert aggregate["eval/reward_mean"] == (None if len(failed_evaluation_inputs) == 3 else 1.0)
    assert aggregate["eval/failed_episodes"] == len(failed_evaluation_inputs)
    assert json.loads((output / "evaluation-0002.json").read_text())["eval/scored_episodes"] == 3
    assert not sessions
