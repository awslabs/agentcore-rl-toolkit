#!/usr/bin/env bash
set -euo pipefail

# Install the slime training backend for either CUDA 12 or CUDA 13.
#
# Usage:
#   bash src/agentcore_rl_toolkit/backends/slime/scripts/install_slime.sh <cu12|cu13>
#
# Defaults to cu13 if no argument is given (backward compatible).

CUDA_VARIANT="${1:-cu13}"
case "$CUDA_VARIANT" in
  cu13)
    TORCH_BACKEND=cu130            # uv --torch-backend for the PyTorch ecosystem
    CUDA_MAJOR=13
    SGLANG_WHL_CU=cu130            # sglang wheel index CUDA tag
    : "${CUDA_HOME:=/usr/local/cuda-13.0}"
    ;;
  cu12)
    TORCH_BACKEND=cu129
    CUDA_MAJOR=12
    SGLANG_WHL_CU=cu129
    : "${CUDA_HOME:=/usr/local/cuda-12.9}"
    ;;
  *)
    echo "Usage: $0 <cu12|cu13>" >&2
    exit 1
    ;;
esac
export CUDA_HOME
echo "=== slime installer: CUDA_VARIANT=$CUDA_VARIANT TORCH_BACKEND=$TORCH_BACKEND CUDA_HOME=$CUDA_HOME ==="

# Assumes your python environment is already activated.

uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --torch-backend="$TORCH_BACKEND"
uv pip install cmake ninja pybind11 "packaging>=24.2" wheel

MAX_JOBS=64 uv pip install "flash-attn==2.8.3" \
  --no-binary flash-attn --no-build-isolation --no-cache-dir --torch-backend="$TORCH_BACKEND"

uv pip install "git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" --no-deps
uv pip install "flash-linear-attention" --torch-backend="$TORCH_BACKEND"

uv pip install tilelang

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

NVCC_APPEND_FLAGS="--threads 4" \
  APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_PARALLEL_BUILD=8 \
  uv pip install -v --no-build-isolation --no-cache-dir \
  "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"

# torch_memory_saver's TMS_CUDA_MAJOR sets the compiled .so suffix (_cu12/_cu13)
# and must match what its runtime detector reads from torch.version.cuda, so we
# derive it from torch rather than hardcoding.
export TMS_CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')"
uv pip install -v "git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926" \
  --no-cache-dir --force-reinstall --no-build-isolation

uv pip install "git+https://github.com/radixark/Megatron-Bridge.git@6fde1c8538ea4ad966c7fba5f759be54f943b598" --no-deps --no-build-isolation
uv pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

uv pip install --prerelease=allow "sglang==0.5.13" --torch-backend="$TORCH_BACKEND"

# On cu12, sglang's default kernels target cu13; reinstall the cu12-matched
# kernel + deep-gemm builds from sglang's own wheel index. Not needed on cu13.
if [ "$CUDA_MAJOR" = "12" ]; then
  uv pip install --force-reinstall sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/
  uv pip install --force-reinstall sgl-deep-gemm --index-url https://docs.sglang.ai/whl/cu129/ --no-deps
fi

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
