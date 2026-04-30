---
title: verl backend setup
description: Runbook for training with verl (self-hosted FSDP + vLLM/SGLang).
---

:::caution[Preview]
The verl backend integration is under active development. This guide
will be finalized once `src/agentcore_rl_toolkit/backends/verl/` lands.
Track progress in the
[docs roadmap](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/docs/plan/roadmap/astro-starlight-docs-setup.md).
:::

8-section outline (from roadmap §6):

1. When to choose verl (vs slime, vs rllm).
2. Prerequisites (GPU, CUDA 12.x, Python 3.10, `rllm-model-gateway`).
3. Install verl.
4. Install agentcore-rl-toolkit into the verl venv.
5. Deploy the agent to ACR (cross-link to slime guide's VPC section).
6. Prepare data and model.
7. Run training.
8. Troubleshooting (FSDP OOM, version skew, rollout imbalance).
