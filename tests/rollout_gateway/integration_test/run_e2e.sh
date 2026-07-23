#!/usr/bin/env bash
# Rollout gateway integration tests. Each pytest module launches (and tears down) its
# own inference server
set -e
cd "$(dirname "$0")"

cd vllm && uv sync && uv run pytest test_e2e_qwen_vllm.py -v
cd ../sglang && uv sync && uv run pytest test_e2e_qwen_sglang.py -v
