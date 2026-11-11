"""Train the SWE-Grep agent with the AgentCore RL toolkit's Tinker loop, plus the agent's metrics and resume.

  EXP=<name> .venv/bin/python swegrep_train.py [recipe] [--test-run]

Writes runs/EXP. If runs/EXP already holds an unfinished run, it is moved to runs/EXP-to-step-N and
training continues from its last checkpoint; a finished run is left alone.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

from agentcore_rl_toolkit.backends.tinker_api import train as base
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession

logger = logging.getLogger("swegrep_train")

ROOT = Path(__file__).resolve().parent
BASE_MODEL = "hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

# One step end to end: 2 tasks x 4 rollouts, a policy update, a sampler weight sync and a checkpoint.
TEST_RUN = {
    "steps": 1,
    "batch_size": 2,
    "group_size": 4,
    "checkpoint_interval": 1,
    "evaluation_dataset": None,
    "wandb_project": None,
    "rollout_timeout": 2400,
}


def load_config(recipe: Path, exp: str, test_run: bool) -> tuple[base.Config, float]:
    """The recipe plus this host's settings as the toolkit's Config, and the recipe's reward beta."""
    raw = json.loads(recipe.read_text())
    raw.pop("_comment", None)
    beta = raw.pop("beta")
    if test_run:
        raw |= TEST_RUN
    hf_home = os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")
    model = os.environ.get("MODEL_DIR") or f"{hf_home}/{BASE_MODEL}"
    config = base.Config(
        **raw,
        endpoint=f"http://127.0.0.1:{os.environ.get('TINKER_PORT', '18080')}",
        base_model=model,
        tokenizer=model,
        agent_runtime_arn=os.environ["AGENTCORE_AGENT_ARN"],
        # The runtime reaches the gateway on this host's private IP.
        gateway_host=subprocess.check_output(["hostname", "-I"], text=True).split()[0],
        output_dir=f"runs/{exp}",
        exp_id=exp,
    )
    if config.wandb_project and not os.environ.get("WANDB_API_KEY"):
        logger.warning("W&B off: WANDB_API_KEY is not set")
        config.wandb_project = None
    return config, beta


def check_runtime_beta(arn: str, beta: float, recipe: Path) -> None:
    control = boto3.client("bedrock-agentcore-control", region_name=arn.split(":")[3])
    runtime = control.get_agent_runtime(agentRuntimeId=arn.split("/")[-1])
    deployed = runtime["environmentVariables"].get("SWEGREP_BETA")
    if deployed is None or float(deployed) != float(beta):
        raise SystemExit(
            f"the runtime has beta {deployed}, {recipe} has {beta}: run bash cdk/deploy.sh -c recipe={recipe}"
        )


def progress(run: Path) -> tuple[int, str | None]:
    """Steps trained so far by a run directory, and its last checkpoint."""
    if not (run / "checkpoint.json").exists():
        return 0, None
    checkpoint = json.loads((run / "checkpoint.json").read_text())
    resumed_at = json.loads((run / "swegrep.json").read_text())["resume_step"]
    return resumed_at + checkpoint["step"], checkpoint["path"]


class Hooks:
    """Additions to the toolkit's loop, which calls its helpers by module name, so install() swaps them.

    Agent metrics: the loop keeps only each rollout's reward. The session's run() also sees the agent's
    metrics; they are written per rollout to rollouts.jsonl, and their means are added to the step's
    metrics (swegrep/...) and to each evaluation (eval/swegrep/...).

    Resume: the LoRA client loads the checkpoint's weights and optimizer state as its first request,
    the already-trained batches are skipped, and so is the step-0 evaluation.
    """

    def __init__(self, config: base.Config, resume_from: str | None, resume_step: int):
        self.config = config
        self.resume_from = resume_from
        self.resume_step = resume_step
        self.rows: dict[str, dict] = {}  # rollout id -> its rollouts.jsonl row
        self.agent_metrics: dict[str, dict] = {}  # session id -> agent metrics, until its rollout ends
        self.phase = "train"

    def install(self) -> None:
        self.session_run = AgentCoreHttpSession.run
        self.rollout = base.rollout
        self.evaluate = base.evaluate
        self.training_data = base.training_data

        async def session_run(session, task):
            dump = await self.session_run(session, task)
            self.agent_metrics[session.session_id] = (dump.task_output or {}).get("metrics") or {}
            return dump

        AgentCoreHttpSession.run = session_run
        base.rollout = self.rollout_with_metrics
        base.evaluate = self.evaluate_with_metrics
        base.training_data = self.training_data_with_metrics
        if self.resume_from:
            create_client = base.tinker.ServiceClient.create_lora_training_client_async
            read_payloads = base.read_payloads

            async def create_client_from_checkpoint(service, *args, **kwargs):
                client = await create_client(service, *args, **kwargs)
                await (await client.load_state_with_optimizer_async(self.resume_from)).result_async()
                return client

            def skip_trained_batches(path):
                rows = read_payloads(path)
                return rows[self.resume_step * self.config.batch_size :] if path == self.config.dataset else rows

            base.tinker.ServiceClient.create_lora_training_client_async = create_client_from_checkpoint
            base.read_payloads = skip_trained_batches

    async def rollout_with_metrics(self, config, gateway, client, payload, *args, **kwargs):
        start = time.monotonic()
        episode = await self.rollout(config, gateway, client, payload, *args, **kwargs)
        row = {
            "phase": self.phase,
            "rollout_id": episode.rollout_id,
            "instance_id": payload["instance_id"],
            "reward": episode.reward,
            "error": episode.error,
            "metrics": self.agent_metrics.pop(episode.rollout_id, {}),
            "trained_tokens": sum(sum(record.loss_mask) for record in episode.records),
            "seconds": time.monotonic() - start,
        }
        self.rows[episode.rollout_id] = row
        with (Path(config.output_dir) / "rollouts.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        return episode

    def training_data_with_metrics(self, groups):
        datums, metrics = self.training_data(groups)
        metrics |= self.means([episode.rollout_id for group in groups for episode in group], "swegrep/")
        return datums, metrics

    async def evaluate_with_metrics(self, config, gateway, client, payloads, step, **kwargs):
        if step == 0 and self.resume_from:
            return {}
        before = set(self.rows)
        self.phase = "eval"
        try:
            metrics = await self.evaluate(config, gateway, client, payloads, step, **kwargs)
        finally:
            self.phase = "train"
        metrics |= self.means(set(self.rows) - before, "eval/swegrep/")
        (Path(config.output_dir) / f"evaluation-{step:04d}.json").write_text(
            json.dumps({"step": step, **metrics}) + "\n"
        )
        return metrics

    def means(self, rollout_ids, prefix: str) -> dict[str, float]:
        """Means of the agent's metrics over rollouts that were scored."""
        scored = [self.rows[i]["metrics"] for i in rollout_ids if i in self.rows and self.rows[i]["reward"] is not None]
        keys = {key for metrics in scored for key in metrics}
        return {prefix + key: sum(m[key] for m in scored if key in m) / sum(key in m for m in scored) for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path, nargs="?", default=ROOT / "recipes/final.json")
    parser.add_argument("--test-run", action="store_true", help="one training step with 2 x 4 rollouts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "agentcore_rl_toolkit.client", "agentcore_rl_toolkit.rollout_gateway"):
        logging.getLogger(name).setLevel(logging.WARNING)

    exp = os.environ.get("EXP", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", exp):  # SkyRL rejects other names, but only at the first checkpoint
        raise SystemExit(f"set EXP to a run name matching [A-Za-z0-9_-]+, not {exp!r}")
    recipe = args.recipe.resolve()
    os.chdir(ROOT)
    load_dotenv(".env")
    config, beta = load_config(recipe, exp, args.test_run)
    check_runtime_beta(config.agent_runtime_arn, beta, recipe)

    # Resume from the furthest checkpoint of this run or of earlier attempts moved aside.
    run = Path(config.output_dir)
    attempts = [run, *Path("runs").glob(f"{exp}-to-step-*")]
    done, checkpoint = max((progress(a) for a in attempts if a.exists()), key=lambda p: p[0], default=(0, None))
    if done >= config.steps:
        logger.info("%s already trained %d steps", run, done)
        return
    if run.exists():
        run.rename(f"{run}-to-step-{done}-{time.strftime('%H%M%S')}")
    if checkpoint:
        logger.info("resuming from %s at step %d", checkpoint, done)
        config.steps -= done
    run.mkdir(parents=True)
    (run / "swegrep.json").write_text(json.dumps({"beta": beta, "resume_from": checkpoint, "resume_step": done}) + "\n")
    Hooks(config, checkpoint, done).install()
    asyncio.run(base.train(config))


if __name__ == "__main__":
    main()
