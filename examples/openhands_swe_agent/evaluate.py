#!/usr/bin/env python

"""What to evaluate: the dataset slices and the config grid, plus the entrypoint.

The harness itself lives in :mod:`rollout_batch`; this file holds only the run-specific
part (which parquet, which endpoint, how many samples, how much concurrency). Everything
else comes from ``config.toml``. Needs ambient AWS creds, and -- for a gateway endpoint --
a reachable vLLM server plus the model's HF tokenizer.
"""

import asyncio
import logging
import pprint

from config import (
    LOCAL_DIR,
    dataset_parquet,
    load_config,
    resolve_env,
    task_image_namespace,
)

# All flavors imported so switching a run over is just uncommenting a line in the grid;
# the ones left commented out are unused imports by definition, hence the noqa.
from rollout_batch import (  # noqa: F401
    BedrockEndpoint,
    EvalConfig,
    NullEndpoint,
    VllmGatewayEndpoint,
    run_eval,
)

CONFIG = load_config()

# The pull through cache the deploy brought up, assembled from the config's parts so an
# eval cannot pull through a cache the deploy did not create.
TASK_IMAGE_NAMESPACE = task_image_namespace(CONFIG)

REPORT_DIR = str(LOCAL_DIR)

# --- dataset slices -----------------------------------------------------------
# Built parquets by dataset key, so a rebuilt dataset is the one the next eval reads. A
# filtered cut lives under its own name and is the case for stating a path.
SWE_GYM = str(dataset_parquet("swegym"))
# SWE_BENCH = str(dataset_parquet("swebench"))


def implies(a, b):
    return b if a else True


# Task matrix: one EvalConfig per experiment run (edit and re-run).
EVAL_CONFIGS = [
    EvalConfig(
        experiment_name=f"eval_gym_{endpoint.model.split('/')[-1]}_n{n}_{agent}_r1",
        endpoint=endpoint,
        dataset=SWE_GYM,
        num_tasks=None,
        n=n,
        concurrency=128,
        session_create_rate=1,
        timeout=1800,
        report_dir=REPORT_DIR,
        task_kwargs=dict(
            agent=agent,
            docker_image_namespace=TASK_IMAGE_NAMESPACE,
        ),
    )
    for endpoint in [
        NullEndpoint(),
        BedrockEndpoint(model="openai/qwen.qwen3-coder-30b-a3b-instruct", region="us-west-2"),
        # VllmGatewayEndpoint(
        #     model="qwen.qwen3-coder-30b-a3b-instruct",
        #     vllm_url="http://INFERENCE_SERVER:8000",
        #     tokenizer_path="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        # ),
    ]
    for n in [1, 4]
    for agent in ["oracle", "noop", "openhands"]
    if (n == 1) == (endpoint.kind == "null") == (agent in ["oracle", "noop"])
]


async def main():
    env = await resolve_env(CONFIG)
    pprint.pprint(EVAL_CONFIGS)
    for config in EVAL_CONFIGS:
        await run_eval(config, env)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    asyncio.run(main())
