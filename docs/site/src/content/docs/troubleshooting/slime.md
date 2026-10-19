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
| Instance type | 8 × NVIDIA B200 |
| CUDA | `13.0` |
| PyTorch | `2.11.0+cu130` |
| slime | commit `a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e` |
| SGLang | `0.5.15.post1` |
| Megatron-LM | commit `1dcf0dafa884ad52ffb243625717a3471643e087` |

## `--norm-epsilon` mismatch on Qwen2.5-32B-Instruct

**Cause:** slime's `scripts/models/qwen2.5-32B.sh` hardcodes
`--norm-epsilon 1e-5` (matching Qwen2.5-32B **base**), but
**Qwen2.5-32B-Instruct** uses `1e-6`.

**Fix:** Edit the slime model script to `--norm-epsilon 1e-6`, or
append `--norm-epsilon 1e-6` to the `train.py` arguments in your
`train.sh` (later flags override `MODEL_ARGS`).
Qwen2.5-0.5B / 1.5B / 3B / 7B Instruct variants match their base-model
norm epsilons, so this only affects Qwen2.5-32B.
