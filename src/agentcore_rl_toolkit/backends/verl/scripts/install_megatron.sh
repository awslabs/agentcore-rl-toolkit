#!/bin/bash
# Install Megatron-related dependencies for verl.
#
# These packages require special build flags that cannot be expressed
# in pyproject.toml. Run this AFTER installing the verl backend:
#   uv pip install -e ".[verl]" --torch-backend=<cu128|cu129|cu130|...>
#
# Usage:
#   bash src/agentcore_rl_toolkit/backends/verl/scripts/install_megatron.sh <cu128|cu129|cu130|...>

set -euo pipefail

TORCH_BACKEND="${1:?Usage: bash scripts/install_megatron.sh <torch-backend, e.g. cu128 or cu129>}"
echo "=== Megatron dependency installer for rLLM ==="
echo "TORCH_BACKEND=${TORCH_BACKEND}"

CUDA_MAJOR="${TORCH_BACKEND#cu}"; CUDA_MAJOR="${CUDA_MAJOR:0:2}"  # e.g. cu130 -> 13

echo "[1/5] Installing nvidia-modelopt..."
uv pip install 'nvidia-modelopt>=0.37.0'

echo "[2/5] Installing transformer-engine (this may take a while)..."
if [ "${CUDA_MAJOR}" != "12" ]; then
    # If CUDA version is not cu12, explicitly exclude transformer-engine-cu12
    # to avoid "Multiple libcudart libraries found" errors.
    echo "transformer-engine-cu12 ; sys_platform == 'never'" | \
    MAX_JOBS=128 uv pip install --no-cache --no-build-isolation \
        --overrides - \
        "transformer_engine[pytorch,core-cu${CUDA_MAJOR}]==2.11"
else
    MAX_JOBS=128 uv pip install --no-cache --no-build-isolation \
        "transformer_engine[pytorch,core-cu${CUDA_MAJOR}]==2.11"
fi

# megatron-core > 0.15.0 required for numpy>=2.0.0 compatibility
echo "[3/5] Installing megatron-core..."
uv pip install --no-deps megatron-core==0.17.1

echo "[4/5] Installing megatron-bridge..."
# Pinned to 691a377f (2026-05-19): "Add external trainer integration helpers (#3813)"
# verl 0.8.0 requires LinearForLastLayer which was added in this commit (unreleased, post-v0.4.2).
uv pip install --no-deps git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@691a377f

echo "[5/5] Installing NVIDIA Apex (required for gradient accumulation fusion)..."
APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    uv pip install -v --no-cache --no-build-isolation \
    git+https://github.com/NVIDIA/apex.git --torch-backend="${TORCH_BACKEND}"

echo ""
echo "=== Megatron dependencies installed successfully ==="
