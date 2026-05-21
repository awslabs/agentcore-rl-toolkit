---
title: Strands OfficeBench Agent
description: An agent evaluating LLMs on OfficeBench — 300 office-automation tasks spanning calendar, email, Excel, Word, PDF, OCR.
---

An agent that evaluates LLMs on the
[OfficeBench](https://github.com/zlwang-cs/OfficeBench) benchmark —
300 office-automation tasks spanning calendar, email, Excel, Word,
PDF, and OCR operations. It deploys to Bedrock AgentCore Runtime
for parallel evaluation at scale.

Built on 20 Strands `@tool` functions (in `tools.py`) that wrap
OfficeBench's original app scripts via subprocess. All 20 tools
are available simultaneously — no explicit app switching, no
single-action-per-turn JSON dispatch.

## Benchmark results

*Averaged over 5 runs (mean ± std).*

| Model | Single App (93) | Two Apps (95) | Three Apps (112) | Overall (300) |
|---|---|---|---|---|
| Claude Sonnet 4.5 — non-thinking | 51.61 ± 1.08 | 64.63 ± 1.41 | 50.18 ± 0.74 | **55.20 ± 0.38** |
| Claude Sonnet 4.5 — thinking (budget=10000) | 48.82 ± 2.10 | 65.90 ± 1.91 | 47.86 ± 1.85 | **53.87 ± 1.07** |

Results are **not directly comparable** with the original
OfficeBench leaderboard — this agent uses Strands native tool-use
instead of JSON action dispatch, has all tools available
simultaneously, and has no iteration cap. See the full README for
the complete list of divergences.

## Quickstart

```bash
cd examples/strands_officebench_agent
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -e .

# Local testing (no ACR)
python test_local.py
```

Deploy + run the full benchmark:

```bash
python deploy.py                         # build, push, deploy to ACR
python benchmark.py --limit 300          # full 300-task run
python benchmark.py --limit 1            # smoke test
```

## What's in the example

- **`rl_app.py`** — the rollout entrypoint; stages files in
  `/testbed/`, runs the agent with all 20 tools, evaluates
  file-state at the end.
- **`reward.py`** — `OfficeBenchReward`: file-state comparison on
  the testbed directory after the agent finishes.
- **`tools.py`** — 20 Strands `@tool` functions wrapping
  OfficeBench app scripts (calendar, email, Excel, Word, PDF, OCR).
- **`benchmark.py`** — parallel evaluation over the 300-task
  benchmark via `RolloutClient`.
- **`run_local_eval.py`** / **`test_local.py`** — local
  (no-ACR) evaluation and smoke tests.
- **`Dockerfile`** + **`deploy.py`** + **`config.toml`** —
  container build + programmatic ACR deploy.
- **`preprocess.py`**, **`models.py`**, **`utils.py`** — shared
  helpers (S3 task fetch, testbed setup, file readers).

For the full list of divergences from the original OfficeBench,
Docker + ECR setup, benchmark config, and local-testing workflow,
see the
[full README on GitHub](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/examples/strands_officebench_agent/README.md).
