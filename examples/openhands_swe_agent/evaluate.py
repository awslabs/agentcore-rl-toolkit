#!/usr/bin/env python

"""What to evaluate: the dataset slices and the config grid, plus the entrypoint.

The harness itself -- the endpoints, :class:`EvalConfig`, the bounded per-rollout
driver and the report -- is :mod:`rollout_batch` and does not change between runs.
This file holds only the run-specific part: which parquet, which endpoint, how many
samples, how much concurrency.

Everything else comes from this recipe's ``config.toml`` (see :mod:`config`),
including the AgentCore ARNs, which are resolved from the names it states rather
than pasted in -- so an eval necessarily hits the runtime ``./deploy.py`` last
deployed. Also needs ambient AWS creds for Bedrock token minting, and -- for a
gateway endpoint -- a reachable vLLM server plus the model's HF tokenizer. See
``rollout_batch``'s module docstring for what each endpoint flavor costs and
captures.
"""

import asyncio
import logging
import pprint

# This recipe's own modules, imported as top-level names: these scripts are
# entrypoints run from this directory (``./evaluate.py``), and ``conftest.py`` puts the
# same directory on ``sys.path`` so the tests resolve them identically.
from config import (
    LOCAL_DIR,
    dataset_parquet,
    load_config,
    resolve_env,
    task_image_namespace,
)

# All three endpoint flavors are imported, not only the one the grid below has
# enabled: switching a run over is meant to be uncommenting a line, not also
# remembering an import. Hence the noqa -- whichever flavors the grid has commented
# out are unused imports by definition, and that is the point.
from rollout_batch import (  # noqa: F401
    BedrockEndpoint,
    EvalConfig,
    NullEndpoint,
    VllmGatewayEndpoint,
    run_eval,
)

CONFIG = load_config()

# Where task images are pulled from: the pull through cache the deploy brought up,
# assembled from the config's parts rather than spelled out, so an eval cannot pull
# through a cache the deploy did not create -- see ``config.task_image_namespace``.
TASK_IMAGE_NAMESPACE = task_image_namespace(CONFIG)

# Run reports go next to this script rather than under the repo's top-level ``local``:
# they are artifacts of this recipe, so every script here finds them relative to its own
# location instead of reaching for the repo root. Absolute, so an eval writes to the
# same place whatever directory it is launched from.
REPORT_DIR = str(LOCAL_DIR)

# --- dataset slices -----------------------------------------------------------
# The built parquets by dataset key, rather than paths spelled out here: writer and
# reader are the two halves of one convention (``config.dataset_parquet``), so a rebuilt
# dataset is the one the next eval reads without either side being edited. A filtered cut
# lives under its own name and is the case for stating a path --
# ``str(dataset_parquet("swegym").with_name(...))``.
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
