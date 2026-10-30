"""DAPO training with the rollout gateway + SageMaker Training Sessions.

Same loop as ``train_grpo.py`` with two DAPO changes:

- Dynamic sampling: each step rolls out generation batches of ``batch_size`` prompts and keeps
  only groups whose rewards differ (an all-equal group has zero advantage and no gradient),
  until ``batch_size`` groups are kept.
- Clip-Higher: the PPO ratio bounds ``clip_low_threshold`` / ``clip_high_threshold`` are sent
  to the service as ``loss_fn_config``.

Reads the same config.yaml as ``train_grpo.py``; the DAPO-only fields in ``DapoConfig`` have
defaults, so they are optional in the YAML.

Usage:
    python -m agentcore_rl_toolkit.backends.experimental.sagemaker.train_dapo --config path/to/config.yaml
"""

import argparse
import asyncio
import logging
import signal
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from sagemaker.train.training_session import AdamParams, ServiceClient

from agentcore_rl_toolkit.backends.experimental.sagemaker.config import Config, load_config
from agentcore_rl_toolkit.backends.experimental.sagemaker.datum import trace_record_to_datum
from agentcore_rl_toolkit.backends.experimental.sagemaker.rollout import load_dataset, local_ip, run_one_rollout
from agentcore_rl_toolkit.backends.experimental.sagemaker.train_grpo import _init_wandb, compute_grpo_advantages
from agentcore_rl_toolkit.concurrency.rate_limiter import LocalRateLimiter
from agentcore_rl_toolkit.rollout_gateway import HfTemplateRenderer, TraceRecord
from agentcore_rl_toolkit.rollout_gateway.gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sagemaker_sdk import SageMakerSdkBackend
from agentcore_rl_toolkit.rollout_gateway.server import ThreadedGatewayServer
from agentcore_rl_toolkit.runtime import AgentCoreHttpClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

# One prompt's K rollouts: [(records, reward), ...]
Group = list[tuple[list[TraceRecord], float]]


@dataclass
class DapoConfig(Config):
    # Clip-Higher: PPO ratio bounds sent to the service, i.e. 1 - eps_low and 1 + eps_high.
    clip_low_threshold: float = 0.8
    clip_high_threshold: float = 1.28
    # Dynamic sampling: max generation batches per step to fill batch_size informative groups (0 = no cap).
    max_num_gen_batches: int = 10


class InsufficientGroupsError(RuntimeError):
    """Dynamic sampling hit max_num_gen_batches before keeping batch_size informative groups."""


def is_informative(rewards: list[float]) -> bool:
    """Whether a group can train: its rewards must differ, or every advantage is zero."""
    return len(rewards) >= 2 and max(rewards) > min(rewards)


async def sample_informative_groups(
    example_queue: list[dict],
    batch_size: int,
    max_num_gen_batches: int,
    collect_group: Callable[[dict], Awaitable[Group]],
) -> tuple[list[Group] | None, list[float], dict[str, int]]:
    """Dynamic sampling for one step.

    Pops ``batch_size`` prompts at a time from ``example_queue`` (in place), rolls them out with
    ``collect_group`` and keeps the informative groups, until ``batch_size`` groups are kept.
    Only one generation batch is in flight at a time, so at most ``batch_size * K`` rollouts run
    concurrently, as in ``train_grpo.py``.

    Returns ``(groups, rewards, stats)``:
        groups: the first ``batch_size`` kept groups (extras are dropped), or None if the epoch
            ran out of prompts first.
        rewards: every rollout reward of this step, before filtering.
        stats: num_gen_batches, num_groups_seen, num_groups_filtered, num_kept, num_empty_rollouts.

    Raises:
        InsufficientGroupsError: ``max_num_gen_batches`` generation batches ran (0 = no cap)
            without keeping ``batch_size`` groups.
    """
    kept: list[Group] = []
    rewards: list[float] = []
    stats = {
        "num_gen_batches": 0,
        "num_groups_seen": 0,
        "num_groups_filtered": 0,
        "num_kept": 0,
        "num_empty_rollouts": 0,
    }
    while len(kept) < batch_size:
        # Check the cap before the queue so hitting it raises instead of quietly ending the epoch.
        if max_num_gen_batches > 0 and stats["num_gen_batches"] >= max_num_gen_batches:
            raise InsufficientGroupsError(
                f"kept {len(kept)} of {batch_size} informative groups "
                f"after {stats['num_gen_batches']} generation batches"
            )
        if len(example_queue) < batch_size:
            return None, rewards, stats
        prompts = [example_queue.pop(0) for _ in range(batch_size)]
        groups = await asyncio.gather(*[collect_group(p) for p in prompts])
        stats["num_gen_batches"] += 1
        for group in groups:
            group_rewards = [r for _, r in group]
            rewards.extend(group_rewards)
            stats["num_groups_seen"] += 1
            stats["num_empty_rollouts"] += sum(1 for records, _ in group if not records)
            if is_informative(group_rewards):
                kept.append(group)
            else:
                stats["num_groups_filtered"] += 1
        stats["num_kept"] = len(kept)
    return kept[:batch_size], rewards, stats


async def train(config: DapoConfig) -> None:
    if not config.dataset_path:
        raise ValueError("dataset_path is required")

    # Load data before creating the session: bad data should not start billing.
    dataset = load_dataset(config.dataset_path)
    logger.info("Loaded %d examples from %s", len(dataset), config.dataset_path)
    if config.max_prompts > 0:
        dataset = dataset[: config.max_prompts]
    if len(dataset) < config.batch_size:
        raise ValueError(
            f"dataset has {len(dataset)} prompts, fewer than batch_size={config.batch_size}; no step would run"
        )

    eval_dataset: list[dict] = []
    if config.eval_dataset_path:
        eval_dataset = load_dataset(config.eval_dataset_path)
        if config.eval_max_prompts > 0:
            eval_dataset = eval_dataset[: config.eval_max_prompts]
        logger.info("Loaded %d eval examples", len(eval_dataset))

    # Clip-Higher. The SDK takes ratio bounds, not eps.
    loss_fn_config = {
        "clip_low_threshold": config.clip_low_threshold,
        "clip_high_threshold": config.clip_high_threshold,
    }
    logger.info("loss_fn=%s loss_fn_config=%s", config.loss, loss_fn_config)

    # Create SageMaker training session client
    svc = ServiceClient(
        region=config.region,
        role_arn=config.role_arn,
        model_package_group_arn=config.model_package_group_arn or None,
        s3_output_path=config.s3_output_path or None,
    )

    if config.resume_from_model_package:
        logger.info("Resuming from checkpoint: %s", config.resume_from_model_package)
        tc = svc.create_training_client_from_state_with_optimizer(
            model_package_arn=config.resume_from_model_package,
            base_model=config.base_model_arn,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
        )
    else:
        logger.info("Creating LoRA training session (base_model=%s)", config.base_model_arn)
        tc = svc.create_lora_training_client(
            base_model=config.base_model_arn,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
        )
    logger.info("Training session ready: %s", tc.training_session_name)

    # The session bills until stopped, so every exit path must reach the finally below.
    # SIGTERM would otherwise kill the interpreter before it runs; cancellation unwinds
    # through it instead. (SIGINT already does, via KeyboardInterrupt.)
    task = asyncio.current_task()
    assert task is not None  # always set inside a running coroutine
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)

    gateway: ThreadedGatewayServer | None = None
    rollout_client: AgentCoreHttpClient | None = None
    wandb_run: Any = None
    try:
        wandb_run = _init_wandb(config, tc.training_session_name)

        tokenizer = tc.get_tokenizer()
        backend = SageMakerSdkBackend(tc.create_sampling_client())
        _gw = RolloutGateway(
            backend=backend,
            renderer=HfTemplateRenderer(tokenizer, chat_template_kwargs=config.chat_template_kwargs),
            tokenizer=tokenizer,
            adapters=["openai", "anthropic"],
            history_mode=config.history_mode,
        )
        gateway = ThreadedGatewayServer(_gw, host=local_ip(), port=config.gateway_port)
        gateway.start()
        logger.info("Gateway base_url: %s", gateway.base_url)

        rollout_client = AgentCoreHttpClient(
            config.agent_runtime_arn,
            max_pool_connections=config.batch_size * config.responses_per_prompt + 10,
            request_rate_limiter=LocalRateLimiter(config.acr_tps_limit),
        )

        sampling_defaults = {"max_new_tokens": config.max_new_tokens, "temperature": config.temperature}
        max_context_tokens = config.max_context_tokens

        async def collect_group(payload: dict) -> Group:
            results = await asyncio.gather(
                *[
                    run_one_rollout(
                        client=rollout_client,
                        gateway=_gw,
                        payload=payload,
                        base_url=gateway.base_url,
                        model_id=config.base_model_arn,
                        max_rollout_time=config.max_rollout_time,
                        sampling_defaults=sampling_defaults,
                        max_context_tokens=max_context_tokens,
                    )
                    for _ in range(config.responses_per_prompt)
                ]
            )
            return [(records, reward) for records, reward, _ in results]

        async def run_eval(step: int) -> None:
            if not eval_dataset:
                return
            eval_sd = {**sampling_defaults, "temperature": config.eval_temperature}
            logger.info("  Evaluating %d held-out examples…", len(eval_dataset))
            results = await asyncio.gather(
                *[
                    run_one_rollout(
                        client=rollout_client,
                        gateway=_gw,
                        payload=p,
                        base_url=gateway.base_url,
                        model_id=config.base_model_arn,
                        max_rollout_time=config.max_rollout_time,
                        sampling_defaults=eval_sd,
                        max_context_tokens=max_context_tokens,
                    )
                    for p in eval_dataset
                ]
            )
            scores = [reward for _, reward, _ in results]
            acc = float(np.mean(scores))
            logger.info("  eval/accuracy=%.4f (n=%d, step=%d)", acc, len(scores), step)
            if wandb_run:
                wandb_run.log({"eval/accuracy": acc, "eval/n": len(scores)}, step=step)

        global_step = 0
        last_eval_step: int | None = None  # step of the latest eval; decides whether a final eval is needed

        for epoch in range(config.epochs):
            example_queue = list(dataset)
            logger.info(
                "Epoch %d: %d examples, batch=%d, K=%d",
                epoch,
                len(example_queue),
                config.batch_size,
                config.responses_per_prompt,
            )

            while len(example_queue) >= config.batch_size:
                logger.info("=== Epoch %d | Step %d [%s] ===", epoch, global_step, datetime.now().strftime("%H:%M:%S"))
                metrics: dict[str, Any] = {"step": global_step, "epoch": epoch}

                if config.save_every > 0 and global_step > 0 and global_step % config.save_every == 0:
                    ckpt = await asyncio.to_thread(lambda: tc.save_state().result(timeout=config.train_op_timeout))
                    logger.info("Checkpoint saved: %s", ckpt.model_package_arn)

                st = time.monotonic()
                try:
                    # Sample prompts, roll out, filter; repeat until batch_size informative groups are kept
                    all_groups, all_rewards, ds_stats = await sample_informative_groups(
                        example_queue, config.batch_size, config.max_num_gen_batches, collect_group
                    )
                except InsufficientGroupsError:
                    # The session is healthy here, so keep the progress before stopping.
                    logger.exception("Dynamic sampling exhausted at step %d; saving before abort", global_step)
                    ckpt = await asyncio.to_thread(lambda: tc.save_state().result(timeout=config.train_op_timeout))
                    logger.info("Abort checkpoint: %s", ckpt.model_package_arn)
                    raise
                metrics["time/rollout"] = time.monotonic() - st
                metrics["sampler/mean_reward"] = float(np.mean(all_rewards)) if all_rewards else 0.0
                metrics.update({f"dynamic_sampling/{k}": v for k, v in ds_stats.items()})
                seen = ds_stats["num_groups_seen"]
                filtered_frac = ds_stats["num_groups_filtered"] / seen if seen else 0.0
                metrics["dynamic_sampling/filtered_frac"] = filtered_frac
                logger.info(
                    "  sampling: gen_batches=%d seen=%d filtered=%d (%.0f%%) kept=%d empty_rollouts=%d "
                    "reward=%.4f rollout_time=%.0fs",
                    ds_stats["num_gen_batches"],
                    seen,
                    ds_stats["num_groups_filtered"],
                    filtered_frac * 100,
                    ds_stats["num_kept"],
                    ds_stats["num_empty_rollouts"],
                    metrics["sampler/mean_reward"],
                    metrics["time/rollout"],
                )

                if all_groups is None:
                    logger.info("  Epoch %d ran out of prompts during dynamic sampling.", epoch)
                    if wandb_run:
                        wandb_run.log(
                            {k: v for k, v in metrics.items() if k not in ("step", "epoch")}, step=global_step
                        )
                    break

                # --- advantage + datums ---
                training_datums: list[dict] = []
                for group in all_groups:
                    # GRPO-normalized advantage within each group of K rollouts
                    advantages = compute_grpo_advantages([r for _, r in group])
                    for (records, _), adv in zip(group, advantages, strict=True):
                        for record in records:
                            datum = trace_record_to_datum(record, adv)
                            if datum is not None:
                                training_datums.append(datum)

                metrics["sampler/num_datums"] = len(training_datums)

                if not training_datums:
                    logger.warning("Step %d: no valid datums, skipping update.", global_step)
                    global_step += 1
                    continue

                logger.info("  Training on %d datums…", len(training_datums))
                st = time.monotonic()
                # Server side: forward_backward + optim_step
                fwd_bwd_op = tc.forward_backward(training_datums, loss_fn=config.loss, loss_fn_config=loss_fn_config)
                optim_op = tc.optim_step(AdamParams(learning_rate=config.learning_rate))
                fwd_bwd_result = await asyncio.to_thread(fwd_bwd_op.result, timeout=config.train_op_timeout)
                await asyncio.to_thread(optim_op.result, timeout=config.train_op_timeout)

                metrics["time/train"] = time.monotonic() - st
                metrics["timestamp"] = datetime.now().isoformat()
                metrics.update({f"train/{k}": v for k, v in fwd_bwd_result.metrics.items()})
                logger.info("  step=%d  reward=%.4f", global_step, metrics["sampler/mean_reward"])
                if wandb_run:
                    wandb_run.log(
                        {k: v for k, v in metrics.items() if k not in ("step", "epoch", "timestamp")}, step=global_step
                    )

                # rebind to updated weights before next rollout
                new_sc = await asyncio.to_thread(tc.save_weights_and_get_sampling_client)
                backend.set_sampling_client(new_sc)

                global_step += 1
                if config.eval_every > 0 and global_step % config.eval_every == 0:
                    # Eval every eval_every steps
                    await run_eval(global_step)
                    last_eval_step = global_step

            logger.info("Epoch %d complete (%d steps)", epoch, global_step)

        # Compare with the last eval rather than global_step % eval_every: that check skips the
        # final eval after 0 steps, or when the last step was skipped for having no datums.
        if eval_dataset and last_eval_step != global_step:
            await run_eval(global_step)

        logger.info("Training complete. Saving final checkpoint…")
        final_ckpt = await asyncio.to_thread(lambda: tc.save_state().result(timeout=config.train_op_timeout))
        logger.info("Final checkpoint: %s", final_ckpt.model_package_arn)

        if config.save_weights_at_end:
            w = await asyncio.to_thread(
                lambda: tc.save_weights_for_sampler(description="final weights").result(timeout=config.train_op_timeout)
            )
            logger.info("Inference model package: %s", w.model_package_arn)
    finally:
        tc.stop()  # first: nothing may preempt stopping the billing session
        if gateway is not None:
            gateway.shutdown()
        if rollout_client is not None:
            await rollout_client.aclose()
        if wandb_run:
            wandb_run.finish()

    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DAPO training with rollout gateway + SageMaker Training Sessions")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config, DapoConfig)
    if not isinstance(config, DapoConfig):
        raise TypeError(f"expected DapoConfig, got {type(config).__name__}")
    try:
        asyncio.run(train(config))
    except asyncio.CancelledError:
        # Raised by the SIGTERM/SIGINT handler after teardown has already run.
        logger.warning("Training cancelled.")


if __name__ == "__main__":
    main()
