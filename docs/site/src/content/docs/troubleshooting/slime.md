---
title: slime troubleshooting
description: Known issues and fixes when training with the slime backend.
---

Known gotchas when training with the
[slime](/agentcore-rl-toolkit/guides/slime-backend-setup/) backend.

## Tested versions

For reproducibility, here's the exact environment this integration
was validated against:

| Component | Version / SHA |
|---|---|
| Instance type | 8 × NVIDIA H100 80GB HBM3 |
| CUDA | `13.0` |
| PyTorch | `2.11.0+cu130` |
| slime | commit `fa3c990af6f18efd3fd9922698bf4bf4048d1263` |
| SGLang | `0.5.13` |
| Megatron-LM | commit `1dcf0dafa884ad52ffb243625717a3471643e087` |

## `--norm-epsilon` mismatch on Qwen2.5-32B-Instruct

**Cause:** slime's `scripts/models/qwen2.5-32B.sh` hardcodes
`--norm-epsilon 1e-5` (matching Qwen2.5-32B **base**), but
**Qwen2.5-32B-Instruct** uses `1e-6`.

**Fix:** Edit the slime model script to `--norm-epsilon 1e-6`, or
pass an override through `SlimeRunner(extra_flags=["--norm-epsilon", "1e-6"])`.
Qwen2.5-0.5B / 1.5B / 3B / 7B Instruct variants match their base-model
norm epsilons, so this only affects Qwen2.5-32B.
