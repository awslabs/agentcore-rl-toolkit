---
title: rllm backend setup
description: Runbook for training via rllm (managed LoRA fine-tuning).
---

:::caution[Preview]
The rllm backend integration is under active development. This guide
will be finalized once `src/agentcore_rl_toolkit/backends/rllm/` lands.
Track progress in the
[docs roadmap](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/docs/plan/roadmap/astro-starlight-docs-setup.md).
:::

8-section outline (from roadmap §6):

1. When to choose rllm.
2. Prerequisites (`model-gateway`).
3. Configure access (env vars, LoRA knobs).
4. How the data plane works (gateway between agent and inference servers).
5. Deploy the agent to ACR (cross-link to slime guide).
6. Run training.
7. Run evaluation.
8. Troubleshooting.
