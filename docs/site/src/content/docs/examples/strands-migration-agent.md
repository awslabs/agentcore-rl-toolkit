---
title: Strands Migration Agent
description: An RL-trainable code-migration agent that upgrades Java 8 projects to Java 17 using shell + editor tools.
---

An RL-trainable code-migration agent that upgrades Java 8 projects
to Java 17, as introduced in
[MigrationBench](https://github.com/amazon-science/MigrationBench).
Given a repo URI, the agent iteratively edits `pom.xml` and source
files with `shell` + `editor` tools until `mvn clean verify` passes
on Java 17 and the test suite is preserved.

Built on the official
[JavaMigrationAgent](https://github.com/amazon-science/JavaMigration/tree/main/java_migration_agent)
baseline, rewired to use open-source LLMs through Strands.

## Quickstart

```bash
# Prerequisite: Java 17 + Maven 3.9.6 installed (see full README)
cd examples/strands_migration_agent
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -e .
uv pip install -e ../../ --force-reinstall --no-deps

# Preprocess 2 repos as a smoke test
python preprocess.py --s3-bucket-name my-migration-bench-data \
    --max-repos-per-split 2 --skip-s3-sync

# Start the RL app (needs a local vLLM server running on :4000)
uvicorn rl_app:app --port 8080 --reload --reload-dir ../..
```

Deploy via the
[Prepare agent for RL → Deploy](/agentcore-rl-toolkit/guides/agent-adaptation/)
flow, then evaluate with `evaluate.py` / `evaluate_async.py`.

## What's in the example

- **`rl_app.py`** — the rollout entrypoint; loads the target repo
  from S3, sets up the Maven env, runs the agent with dynamically
  composed tools.
- **`reward.py`** — `MigrationReward`: build success + test-case
  preservation vs. the original commit.
- **`preprocess.py`** — convert the MigrationBench dataset into
  per-repo tarballs + metadata + JSONL rollout requests.
- **`evaluate.py`** / **`evaluate_async.py`** — batch / async
  evaluation via `RolloutClient`.
- **`Dockerfile`** + **`deploy.py`** + **`config.toml`** —
  container build + programmatic ACR deploy.
- **`models.py`**, **`utils.py`**, **`eval_utils.py`** — shared
  helpers (S3 fetch, Maven setup, build verification).

For Java/Maven install, Docker + ECR setup, CodeArtifact maven
mirror (avoids Maven Central rate limits), and full evaluation
config, see the
[full README on GitHub](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/examples/strands_migration_agent/README.md).
