#!/bin/bash
# Re-apply the megatron-bridge context-parallel chunk-one clamp after any `uv sync`.
#
# uv prunes and reinstalls packages, so a hand edit in .venv does not survive. Run this
# after `uv sync --extra verl --group verl-megatron` whenever CP>1 training is used.
#
# What it fixes: preprocess_packed_seqs slices chunk one of the zigzag-CP split using
# positions in the *padded* sequence while reading from a buffer of only real tokens,
# and (unlike chunk two) does not clamp. Any row shorter than tp*cp then raises
#   RuntimeError: The expanded size of the tensor (N) must match the existing size (M)
# from compute_log_prob -- after a full rollout has been paid for. verl itself injects
# such rows (trainer/ppo/padding_utils.py synthesizes prompt_len=1/response_len=1
# samples to make the batch divisible), so it cannot be avoided from config.
set -euo pipefail

VENV=${VENV:-.venv}
TARGET=$VENV/lib/python3.12/site-packages/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/utils.py
PATCH=$(dirname "$0")/megatron-bridge-cp-chunk1-clamp.diff

[ -f "$TARGET" ] || { echo "target not found: $TARGET (is the verl extra synced?)" >&2; exit 1; }

if grep -q "LOCAL PATCH (agentcore-rl-toolkit)" "$TARGET"; then
    echo "clamp already applied: $TARGET"
    exit 0
fi

cp "$TARGET" "$TARGET.orig-preclamp"
patch -p0 --input="$PATCH" "$TARGET"
grep -q "LOCAL PATCH (agentcore-rl-toolkit)" "$TARGET" && echo "clamp applied to $TARGET"
