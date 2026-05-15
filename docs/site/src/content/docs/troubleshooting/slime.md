---
title: slime troubleshooting
description: Known issues and fixes when training with the slime backend.
---

Known gotchas when training with the
[slime](/agentcore-rl-toolkit/guides/slime-backend-setup/) backend.

## `LinearCrossEntropyModule` parallelism error on 32B (or any model with untied embeddings)

**Symptom:** During 32B training, the Megatron actor crashes with:

```
ValueError: Cannot determine parallelism type for module 'LinearCrossEntropyModule'
            at weight 'output_layer.weight'.
```

**Cause:** The Megatron-LM bundled in `slimerl/slime:latest` is
several hundred commits ahead of the sha pinned in slime's docker
README (`3714d81d`). Specifically, Megatron PR **#3226 "Reapply fix
Linear CE Fusion"** (2026-02-04) replaced `ColumnParallelLinear`
with a new `LinearCrossEntropyModule` that megatron-bridge's
`AutoMapping` doesn't recognize. Models with tied embeddings (0.5B,
3B, 7B) skip this code path; models with
`--untie-embeddings-and-output-weights` (32B and up) hit it.

**Fix:** Inside the container, pin `/root/Megatron-LM` to the stable
sha:

```bash
cd /root/Megatron-LM
# Stash any image-local patches first (can be restored later with `git stash pop`)
git stash -u -m "slime local patches"
git checkout 3714d81d418c9f1bca4594fc35f9e8289f652862
# Clear pyc caches that reference the old code
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
```

## `--norm-epsilon` mismatch on Qwen2.5-32B-Instruct

**Cause:** slime's `scripts/models/qwen2.5-32B.sh` hardcodes
`--norm-epsilon 1e-5` (matching Qwen2.5-32B **base**), but
**Qwen2.5-32B-Instruct** uses `1e-6`.

**Fix:** Edit the slime model script to `--norm-epsilon 1e-6`, or
pass an override through `SlimeRunner(extra_flags=["--norm-epsilon", "1e-6"])`.
0.5B/3B/7B Instruct variants match their base-model norm epsilons,
so this only affects 32B.
