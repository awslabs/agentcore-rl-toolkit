"""A synchronous, group-relative agent training loop over the Tinker API.

Run with ``python -m agentcore_rl_toolkit.backends.tinker_api.train config.json``.
Rollouts within a batch are concurrent; all finish before the policy is updated.
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import math
import os
import time
import uuid
from pathlib import Path

import tinker
from aiohttp import web
from transformers import AutoTokenizer

from agentcore_rl_toolkit.client import RolloutClient
from agentcore_rl_toolkit.rollout_gateway import BaseTrace, HfTemplateRenderer
from agentcore_rl_toolkit.rollout_gateway.gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.tinker_sdk import TinkerSdkBackend
from agentcore_rl_toolkit.rollout_gateway.server import FilteredAccessLogger

from .data import Episode, training_data

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Config:
    endpoint: str
    base_model: str
    tokenizer: str
    dataset: str
    agent_runtime_arn: str
    s3_bucket: str
    gateway_host: str
    output_dir: str
    tokenizer_revision: str | None = None
    chat_template_kwargs: dict = dataclasses.field(default_factory=dict)
    gateway_port: int = 0
    history_mode: str = "tree"
    steps: int = 3
    batch_size: int = 4
    group_size: int = 4
    lora_rank: int = 32
    learning_rate: float = 1e-5
    max_new_tokens: int = 2048
    max_context_tokens: int = 16384
    rollout_timeout: float = 180.0
    tps_limit: int = 4
    evaluation_dataset: str | None = None
    evaluation_batch_size: int = 256
    evaluation_temperature: float = 0.6
    evaluation_interval: int = 10
    checkpoint_interval: int = 20
    wandb_project: str | None = None
    wandb_entity: str | None = None
    exp_id: str = dataclasses.field(default_factory=lambda: f"tinker-{uuid.uuid4().hex[:12]}")

    def __post_init__(self):
        if self.steps < 1 or self.batch_size < 1 or self.group_size < 2:
            raise ValueError("steps and batch_size must be positive; group_size must be >= 2")
        if self.lora_rank < 1 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("This recipe requires a positive LoRA rank and learning rate")
        if not 0 < self.max_new_tokens < self.max_context_tokens:
            raise ValueError("Require 0 < max_new_tokens < max_context_tokens")
        if self.tps_limit < 1:
            raise ValueError("tps_limit must be positive")
        if self.history_mode not in ("tree", "linear"):
            raise ValueError("history_mode must be 'tree' or 'linear'")
        if min(self.evaluation_batch_size, self.evaluation_interval, self.checkpoint_interval) < 1:
            raise ValueError("Evaluation batch size and evaluation/checkpoint intervals must be positive")
        if not math.isfinite(self.evaluation_temperature) or self.evaluation_temperature < 0:
            raise ValueError("Evaluation temperature must be finite and nonnegative")


def agent_reward(result: dict) -> float:
    """Validate the reward contract of a successful invocation."""
    reward = result.get("rewards")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError(f"Expected a finite scalar agent reward, got {reward!r}")
    return float(reward)


async def rollout(
    config: Config,
    gateway: RolloutGateway,
    client: RolloutClient,
    payload: dict,
    group_index: int,
    sample_index: int,
    *,
    temperature: float = 1.0,
) -> Episode:
    sid = str(uuid.uuid4())
    gateway.create_session(
        sid,
        sampling_defaults={"max_new_tokens": config.max_new_tokens, "temperature": temperature, "top_p": 1.0},
        max_context_tokens=config.max_context_tokens,
    )
    try:
        result = {}
        error = None
        try:
            future = await client.invoke_async(
                payload,
                session_id=sid,
                input_id=sid,
                api_key=sid,
                sampling_params={
                    "max_completion_tokens": config.max_new_tokens,
                    "temperature": temperature,
                    "top_p": 1.0,
                },
            )
            result = await future.result_async(timeout=config.rollout_timeout)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if result.get("status_code", 200) != 200:
            error = f"agent returned status_code={result['status_code']}: {result.get('stop_reason', 'unknown')}"
        if error:
            logger.warning("rollout=%s invocation failed: %s", sid, error)
        records = await gateway.finish_session(
            sid,
            base_sample=BaseTrace(index=sample_index, group_index=group_index, rollout_id=sid),
            reward=0.0,
        )
        if not records or not any(any(record.loss_mask) for record in records):
            error = error or "no trainable tokens captured"
            logger.warning("rollout=%s excluded from scoring: %s", sid, error)
            return Episode(sid, None, records, error=error)
        reward = 0.0 if error else agent_reward(result)
        for record in records:
            record.reward = reward
        logger.debug("rollout=%s reward=%g records=%d", sid, reward, len(records))
        return Episode(sid, reward, records, error=error)
    finally:
        await gateway.drop_session(sid)


async def drain_rollouts(coroutines):
    # Wait for all sibling rollout tasks before advancing training.
    # Invocation failures are Episode data;
    # unexpected processing errors and cancellation still propagate after draining.
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise failures[0]
    return results


async def collect_batch(config: Config, gateway: RolloutGateway, client: RolloutClient, payloads: list[dict]):
    results = await drain_rollouts(
        [
            rollout(config, gateway, client, payload, group_index, group_index * config.group_size + i)
            for group_index, payload in enumerate(payloads)
            for i in range(config.group_size)
        ]
    )
    return [results[i : i + config.group_size] for i in range(0, len(results), config.group_size)]


async def evaluate(config: Config, gateway: RolloutGateway, client: RolloutClient, payloads: list[dict], step: int):
    """Evaluate each held-out prompt once without constructing training data."""
    start = time.monotonic()
    summaries = []
    path = Path(config.output_dir) / f"evaluation-{step:04d}.jsonl"
    with path.open("w") as stream:

        async def evaluate_one(index, payload):
            episode = await rollout(
                config, gateway, client, payload, index, index, temperature=config.evaluation_temperature
            )
            summary = {
                "input_index": index,
                "rollout_id": episode.rollout_id,
                "reward": episode.reward,
                "truncated": any(record.metadata.get("truncated", False) for record in episode.records),
                "error": episode.error,
            }
            summaries.append(summary)
            stream.write(json.dumps(summary) + "\n")
            stream.flush()

        for offset in range(0, len(payloads), config.evaluation_batch_size):
            await drain_rollouts(
                [
                    evaluate_one(offset + i, payload)
                    for i, payload in enumerate(payloads[offset : offset + config.evaluation_batch_size])
                ]
            )
            logger.info("Evaluation step=%d completed=%d/%d", step, len(summaries), len(payloads))
    rewards = [summary["reward"] for summary in summaries if summary["reward"] is not None]
    failed_episodes = sum(summary["error"] is not None for summary in summaries)
    metrics = {
        "eval/reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "eval/episodes": len(summaries),
        "eval/scored_episodes": len(rewards),
        "eval/empty_episodes": len(summaries) - len(rewards),
        "eval/coverage": len(rewards) / len(summaries) if summaries else 0.0,
        "eval/truncated_episodes": sum(summary["truncated"] for summary in summaries),
        "eval/failed_episodes": failed_episodes,
        "eval/failure_rate": failed_episodes / len(summaries) if summaries else 0.0,
        "eval/seconds": time.monotonic() - start,
    }
    path.with_suffix(".json").write_text(json.dumps({"step": step, **metrics}) + "\n")
    return metrics


async def save_checkpoint(training_client, output: Path, name: str, step: int, updates: int):
    future = await training_client.save_state_async(name)
    checkpoint = await future.result_async()
    metadata = {"path": checkpoint.path, "step": step, "optimizer_steps": updates}
    contents = json.dumps(metadata) + "\n"
    (output / f"checkpoint-{step:04d}.json").write_text(contents)
    (output / "checkpoint.json").write_text(contents)
    logger.info("Saved %s after %d updates", checkpoint.path, updates)
    return checkpoint.path


def read_payloads(path: str) -> list[dict]:
    payloads = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not payloads or not all(isinstance(payload, dict) for payload in payloads):
        raise ValueError("Dataset must contain JSON objects representing agent invocation payloads")
    return payloads


async def update_policy(training_client, datums, learning_rate: float):
    """Enqueue ordered training operations and consume both results before sampling."""
    backward = await training_client.forward_backward_async(datums, loss_fn="importance_sampling")
    optimizer = await training_client.optim_step_async(
        tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    )
    backward_result = await backward.result_async()
    optimizer_result = await optimizer.result_async()
    return backward_result, optimizer_result


async def train(config: Config):
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "config.json").open("x") as stream:
        stream.write(json.dumps(dataclasses.asdict(config), indent=2) + "\n")
    payloads = read_payloads(config.dataset)
    if len(payloads) < config.steps * config.batch_size:
        raise ValueError("Dataset needs at least steps * batch_size payload rows")
    evaluation_payloads = read_payloads(config.evaluation_dataset) if config.evaluation_dataset else None

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, revision=config.tokenizer_revision)
    renderer = HfTemplateRenderer(tokenizer, chat_template_kwargs=config.chat_template_kwargs)
    service = tinker.ServiceClient(base_url=config.endpoint, api_key=os.environ.get("TINKER_API_KEY") or "tml-dummy")
    logger.info("Creating LoRA training client for %s", config.base_model)
    training_client = await service.create_lora_training_client_async(
        base_model=config.base_model, rank=config.lora_rank, seed=42
    )
    backend = TinkerSdkBackend(await training_client.save_weights_and_get_sampling_client_async())
    gateway = RolloutGateway(backend=backend, renderer=renderer, tokenizer=tokenizer, history_mode=config.history_mode)
    runner = web.AppRunner(gateway.app, handler_cancellation=True, access_log_class=FilteredAccessLogger)
    await runner.setup()
    wandb_run = None
    updates = 0
    try:
        await web.TCPSite(runner, host=config.gateway_host, port=config.gateway_port).start()
        bound_port = runner.addresses[0][1]
        gateway_url = f"http://{config.gateway_host}:{bound_port}/v1"
        logger.info("Gateway listening on %s", gateway_url)
        client = RolloutClient(
            agent_runtime_arn=config.agent_runtime_arn,
            s3_bucket=config.s3_bucket,
            exp_id=config.exp_id,
            base_url=gateway_url,
            model_id=config.base_model,
            tps_limit=config.tps_limit,
            max_pool_connections=max(10, config.batch_size * config.group_size, config.evaluation_batch_size),
        )
        if config.wandb_project:
            import wandb

            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.exp_id,
                config=dataclasses.asdict(config),
                dir=str(output),
            )
            # Training and evaluation publish independently, using the same
            # training-step axis rather than reusing an already committed _step.
            wandb_run.define_metric("step")
            wandb_run.define_metric("*", step_metric="step")
            (output / "wandb-url.txt").write_text(wandb_run.url + "\n")

        if evaluation_payloads:
            metrics = await evaluate(config, gateway, client, evaluation_payloads, step=0)
            (output / "evaluation-before.json").write_text(json.dumps(metrics) + "\n")
            logger.info("Initial evaluation: %s", json.dumps(metrics))
            if wandb_run:
                wandb_run.log({"step": 0, **metrics})

        for step in range(config.steps):
            start = time.monotonic()
            batch = payloads[step * config.batch_size : (step + 1) * config.batch_size]
            groups = await collect_batch(config, gateway, client, batch)
            # Persist raw IDs/masks for auditing the actual sampled/trained data.
            (output / f"rollouts-{step:04d}.json").write_text(
                json.dumps(
                    [[dataclasses.asdict(episode) for episode in group] for group in groups],
                    default=lambda value: value.value,
                )
                + "\n"
            )
            datums, metrics = training_data(groups)
            metrics["time/rollout_seconds"] = time.monotonic() - start
            if datums:
                backward, optimizer = await update_policy(training_client, datums, config.learning_rate)
                for prefix, result in (("backward", backward), ("optimizer", optimizer)):
                    for key, value in (result.metrics or {}).items():
                        if isinstance(value, (int, float)):
                            if not math.isfinite(value):
                                raise ValueError(f"Non-finite {prefix} metric: {key}={value}")
                            metrics[f"{prefix}/{key}"] = value
                updates += 1
                backend.sampling_client = await training_client.save_weights_and_get_sampling_client_async()
            metrics["train/optimizer_steps"] = updates
            metrics["time/batch_seconds"] = time.monotonic() - start
            completed = step + 1
            metrics["step"] = completed
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            logger.info("Batch metrics: %s", json.dumps(metrics))
            if wandb_run:
                wandb_run.log(metrics)
            if completed % config.checkpoint_interval == 0 or completed == config.steps:
                suffix = "final" if completed == config.steps else f"step-{completed:04d}"
                checkpoint_path = await save_checkpoint(
                    training_client, output, f"{config.exp_id}-{suffix}", completed, updates
                )
                if wandb_run:
                    wandb_run.summary["checkpoint_path"] = checkpoint_path
                    wandb_run.summary["optimizer_steps"] = updates
            if evaluation_payloads and (completed % config.evaluation_interval == 0 or completed == config.steps):
                evaluation_metrics = await evaluate(config, gateway, client, evaluation_payloads, completed)
                if wandb_run:
                    wandb_run.log({"step": completed, **evaluation_metrics})

    except BaseException:
        if wandb_run:
            wandb_run.finish(exit_code=1)
            wandb_run = None
        raise
    finally:
        await runner.cleanup()
        if wandb_run:
            wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON training configuration")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "agentcore_rl_toolkit.client", "agentcore_rl_toolkit.rollout_gateway"):
        logging.getLogger(name).setLevel(logging.DEBUG if args.log_level == "DEBUG" else logging.WARNING)
    asyncio.run(train(Config(**json.loads(args.config.read_text()))))


if __name__ == "__main__":
    main()
