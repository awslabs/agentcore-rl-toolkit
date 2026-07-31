#!/usr/bin/env bash
set -euo pipefail

# Install the slime training backend (CUDA 13 only — cu12 is not supported).
#
# Usage:
#   bash src/agentcore_rl_toolkit/backends/experimental/slime/scripts/install_slime.sh

TORCH_BACKEND=cu130               # uv --torch-backend for the PyTorch ecosystem
: "${CUDA_HOME:=/usr/local/cuda-13.0}"
export CUDA_HOME

# The source builds below (flash-attn, transformer-engine, apex) write tens of GB
# of nvcc scratch files to $TMPDIR and can fill up a small root volume
# ("No space left on device" from cc1plus/nvcc). Prefer the instance's large
# ephemeral NVMe volume (present on AWS DLAMI/HyperPod nodes) unless the caller
# already set TMPDIR.
if [ -z "${TMPDIR:-}" ] && [ -d /opt/dlami/nvme ]; then
  TMPDIR="/opt/dlami/nvme/${USER}/tmp"
  mkdir -p "$TMPDIR"
fi
export TMPDIR="${TMPDIR:-/tmp}"

echo "=== slime installer (cu13): TORCH_BACKEND=$TORCH_BACKEND CUDA_HOME=$CUDA_HOME TMPDIR=$TMPDIR ==="

# Assumes your python environment is already activated.

uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --torch-backend="$TORCH_BACKEND"
uv pip install cmake ninja pybind11 "packaging>=24.2" wheel

MAX_JOBS=64 uv pip install "flash-attn==2.8.3" \
  --no-binary flash-attn --no-build-isolation --no-cache-dir --torch-backend="$TORCH_BACKEND"

uv pip install "git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" --no-deps
uv pip install "flash-linear-attention" --torch-backend="$TORCH_BACKEND"

uv pip install tilelang

# Explicitly exclude transformer-engine-cu12 to avoid "Multiple libcudart
# libraries found" errors.
echo "transformer-engine-cu12 ; sys_platform == 'never'" | \
MAX_JOBS=128 uv pip install --no-cache --no-build-isolation \
    --overrides - \
    "transformer_engine[pytorch,core-cu13]==2.11"

NVCC_APPEND_FLAGS="--threads 4" \
  APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_PARALLEL_BUILD=8 \
  uv pip install -v --no-build-isolation --no-cache-dir \
  "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"

# torch_memory_saver's TMS_CUDA_MAJOR sets the compiled .so suffix (_cu13) and
# must match what its runtime detector reads from torch.version.cuda, so we
# derive it from torch rather than hardcoding.
export TMS_CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')"
uv pip install -v "git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926" \
  --no-cache-dir --force-reinstall --no-build-isolation

uv pip install "git+https://github.com/radixark/Megatron-Bridge.git@6fde1c8538ea4ad966c7fba5f759be54f943b598" --no-deps --no-build-isolation
uv pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# sglang's default kernel + deep-gemm builds already target cu13, so no
# wheel-index reinstall is needed here.
uv pip install --prerelease=allow "sglang==0.5.13" --torch-backend="$TORCH_BACKEND"

# We have to git clone and install from local because wheel file does not expose megatron.training that is required by slime
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout "1dcf0dafa884ad52ffb243625717a3471643e087"
uv pip install -e . --no-build-isolation --config-settings editable_mode=compat
cd ..

uv pip install --reinstall-package nvidia-cutlass-dsl-libs-base --no-deps \
  "nvidia-cutlass-dsl-libs-base==4.5.2"

uv pip install --reinstall-package pyjwt PyJWT

# Install slime
git clone https://github.com/THUDM/slime.git
cd slime
git checkout "fa3c990af6f18efd3fd9922698bf4bf4048d1263"
uv pip install -r "requirements.txt"
uv pip install -e . --no-deps
cd ..

uv pip install "https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-1117d05/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl" --force-reinstall

# numpy<2 for Megatron; scipy<1.14 because scipy>=1.14 requires numpy>=2.
# --no-config so this isn't silently overridden.
uv pip install --no-config "numpy<2" "scipy<1.14"

# Apply slime's official patches to megatron + sglang.
SLIME_PATCH_DIR="$(cd slime/docker/patch/latest && pwd)"
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

# Megatron patch: apply against the repo without leaving cwd (git -C).
git -C Megatron-LM update-index --refresh >/dev/null 2>&1 || true
git -C Megatron-LM apply --3way "$SLIME_PATCH_DIR/megatron.patch"

# sglang patches: apply into site-packages without leaving cwd (patch -d).
patch -d "$SITE_PACKAGES" -p2 -F0 -N < "$SLIME_PATCH_DIR/sglang.patch"
patch -d "$SITE_PACKAGES" -p2 -F0 -N < "$SLIME_PATCH_DIR/sglang-top_p.patch"
