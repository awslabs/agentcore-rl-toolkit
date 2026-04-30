---
title: rllm backend setup
description: Runbook for training via rllm + Thinking Machines Tinker (managed LoRA fine-tuning).
---

:::caution[Preview]
The rllm backend integration is under active development. This guide
will be finalized once `src/agentcore_rl_toolkit/backends/rllm/` lands.
Track progress in the
[docs roadmap](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/docs/plan/roadmap/astro-starlight-docs-setup.md).
:::

8-section outline (from roadmap §6):

1. When to choose rllm + Tinker.
2. Prerequisites (`rllm-model-gateway`, Tinker API key).
3. Configure Tinker access (env vars, LoRA knobs).
4. How the data plane works (gateway between agent and Tinker).
5. Deploy the agent to ACR (cross-link to slime guide).
6. Run training.
7. Run evaluation.
8. Troubleshooting.
