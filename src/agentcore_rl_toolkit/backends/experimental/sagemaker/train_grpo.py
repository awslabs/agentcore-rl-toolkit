"""GRPO training with the rollout gateway + SageMaker Training Sessions.

Usage:
    python -m agentcore_rl_toolkit.backends.experimental.sagemaker.train_grpo --config path/to/config.yaml
"""

import argparse
import asyncio
import dataclasses
import logging
import signal
import time
from datetime import datetime
from typing import Any

import numpy as np
from sagemaker.train.training_session import AdamParams, ServiceClient

from agentcore_rl_toolkit.backends.experimental.sagemaker.config import Config, load_config
from agentcore_rl_toolkit.backends.experimental.sagemaker.datum import trace_record_to_datum
from agentcore_rl_toolkit.backends.experimental.sagemaker.rollout import load_dataset, local_ip, run_one_rollout
from agentcore_rl_toolkit.client import RolloutClient
from agentcore_rl_toolkit.rollout_gateway import HfTemplateRenderer, TraceRecord
from agentcore_rl_toolkit.rollout_gateway.gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sagemaker_sdk import SageMakerSdkBackend
from agentcore_rl_toolkit.rollout_gateway.server import ThreadedGatewayServer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")


def _init_wandb(config: Config, run_name: str):
    if not config.wandb_project:
        return None
    try:
        import wandb

        return wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or run_name,
            config=dataclasses.asdict(config),
        )
    except ImportError:
        logger.warning("wandb not installed; skipping W&B logging (pip install wandb)")
        return None


def compute_grpo_advantages(group_rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Center and normalize rewards within a GRPO group."""
    arr = np.array(group_rewards, dtype=np.float64)
    centered = arr - arr.mean()
    std = arr.std()
    if std > eps:
        centered = centered / (std + eps)
    return centered.tolist()


async def train(config: Config) -> None:
    if not config.dataset_path:
        raise ValueError("dataset_path is required")

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
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)

    gateway: ThreadedGatewayServer | None = None
    wandb_run: Any = None
    try:
        wandb_run = _init_wandb(config, tc.training_session_name)

        tokenizer = tc.get_tokenizer()
        backend = SageMakerSdkBackend(tc.create_sampling_client())
        _gw = RolloutGateway(
            backend=backend,
            renderer=HfTemplateRenderer(tokenizer),
            tokenizer=tokenizer,
            adapters=["openai", "anthropic"],
        )
        gateway = ThreadedGatewayServer(_gw, host=local_ip(), port=config.gateway_port)
        gateway.start()
        logger.info("Gateway base_url: %s", gateway.base_url)

        exp_id = config.exp_id or f"sagemaker-train-{int(time.time())}"
        rollout_client = RolloutClient(
            agent_runtime_arn=config.agent_runtime_arn,
            s3_bucket=config.s3_bucket,
            exp_id=exp_id,
            tps_limit=config.acr_tps_limit,
            max_pool_connections=config.batch_size * config.responses_per_prompt + 10,
        )

        dataset = load_dataset(config.dataset_path)
        logger.info("Loaded %d examples from %s", len(dataset), config.dataset_path)
        if config.max_prompts > 0:
            dataset = dataset[: config.max_prompts]

        eval_dataset: list[dict] = []
        if config.eval_dataset_path:
            eval_dataset = load_dataset(config.eval_dataset_path)
            if config.eval_max_prompts > 0:
                eval_dataset = eval_dataset[: config.eval_max_prompts]
            logger.info("Loaded %d eval examples", len(eval_dataset))

        sampling_defaults = {"max_new_tokens": config.max_new_tokens, "temperature": config.temperature}
        max_context_tokens = config.max_context_tokens

        async def collect_group(payload: dict) -> list[tuple[list[TraceRecord], float]]:
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
                batch = [example_queue.pop(0) for _ in range(config.batch_size)]
                logger.info("=== Epoch %d | Step %d [%s] ===", epoch, global_step, datetime.now().strftime("%H:%M:%S"))
                metrics: dict[str, Any] = {"step": global_step, "epoch": epoch}

                if config.save_every > 0 and global_step > 0 and global_step % config.save_every == 0:
                    ckpt = await asyncio.to_thread(lambda: tc.save_state().result(timeout=600))
                    logger.info("Checkpoint saved: %s", ckpt.model_package_arn)

                st = time.monotonic()
                all_groups = list(await asyncio.gather(*[collect_group(p) for p in batch]))
                metrics["time/rollout"] = time.monotonic() - st

                all_rewards = [r for group in all_groups for _, r in group]
                metrics["sampler/mean_reward"] = float(np.mean(all_rewards)) if all_rewards else 0.0

                training_datums: list[dict] = []
                for group in all_groups:
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
                fwd_bwd_op = tc.forward_backward(training_datums, loss_fn=config.loss)
                optim_op = tc.optim_step(AdamParams(learning_rate=config.learning_rate))
                fwd_bwd_result = await asyncio.to_thread(fwd_bwd_op.result, timeout=600)
                await asyncio.to_thread(optim_op.result, timeout=600)

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
                    await run_eval(global_step)

            logger.info("Epoch %d complete (%d steps)", epoch, global_step)

        if eval_dataset and not (config.eval_every > 0 and global_step % config.eval_every == 0):
            await run_eval(global_step)

        logger.info("Training complete. Saving final checkpoint…")
        final_ckpt = await asyncio.to_thread(lambda: tc.save_state().result(timeout=600))
        logger.info("Final checkpoint: %s", final_ckpt.model_package_arn)

        if config.save_weights_at_end:
            w = await asyncio.to_thread(
                lambda: tc.save_weights_for_sampler(description="final weights").result(timeout=600)
            )
            logger.info("Inference model package: %s", w.model_package_arn)
    finally:
        tc.stop()  # first: nothing may preempt stopping the billing session
        if gateway is not None:
            gateway.shutdown()
        if wandb_run:
            wandb_run.finish()

    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training with rollout gateway + SageMaker Training Sessions")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    try:
        asyncio.run(train(load_config(args.config)))
    except asyncio.CancelledError:
        # Raised by the SIGTERM/SIGINT handler after teardown has already run.
        logger.warning("Training cancelled.")


if __name__ == "__main__":
    main()
