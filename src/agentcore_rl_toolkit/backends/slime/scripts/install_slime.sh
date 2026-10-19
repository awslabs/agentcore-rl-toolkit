#!/usr/bin/env bash
set -euo pipefail

# Install the slime training backend with CUDA 13
#
# Usage:
#   bash src/agentcore_rl_toolkit/backends/slime/scripts/install_slime.sh

TORCH_BACKEND=cu130               # uv --torch-backend for the PyTorch ecosystem
: "${CUDA_HOME:=/usr/local/cuda-13.0}"
export CUDA_HOME

# Install agentcore-rl-toolkit. Megatron-LM and slime are cloned into the cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
uv pip install -e "${REPO_ROOT}[gateway,rollout]"

# Install the CUDA 13 PyTorch ecosystem
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --torch-backend="$TORCH_BACKEND"
uv pip install "flash-attn==2.8.3" \
  --torch-backend="$TORCH_BACKEND" \
  --index "https://wheels.astral.sh/simple/cu130/"

# Explicitly exclude transformer-engine-cu12 to avoid "Multiple libcudart
# libraries found" errors.
echo "transformer-engine-cu12 ; sys_platform == 'never'" | \
  MAX_JOBS=128 uv pip install --no-cache --no-build-isolation \
    --overrides - \
    "transformer_engine[pytorch,core-cu13]==2.11"

NVCC_APPEND_FLAGS="--threads 4" APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_PARALLEL_BUILD=8 \
  uv pip install -v --no-build-isolation --no-cache-dir \
  "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"

# torch_memory_saver's TMS_CUDA_MAJOR sets the compiled .so suffix (_cu13) and
# must match what its runtime detector reads from torch.version.cuda, so we
# derive it from torch rather than hardcoding.
export TMS_CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')"
uv pip install -v "git+https://github.com/zhuzilin/torch_memory_saver.git@8d30c59ca12a68d9deccbc9c6599076a1218cbc5" \
  --no-cache-dir --force-reinstall --no-build-isolation

uv pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# Install sglang
uv pip install --prerelease=allow "sglang==0.5.15.post1" --torch-backend="$TORCH_BACKEND"

# We have to git clone and install from local because wheel file does not expose megatron.training that is required by slime
uv pip install pybind11

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
git checkout "a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e"
uv pip install -r "requirements.txt"
# slime's wandb init calls wandb.util.generate_id, removed in wandb 0.28.2.
uv pip install "wandb<0.28.2"
uv pip install -e . --no-deps
cd ..

uv pip install "https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl" --force-reinstall

# numpy<2 for Megatron. scipy, pyarrow, and pandas are capped below the first
# releases that require numpy>=2; pyarrow doesn't declare that requirement, and the
# pandas 3.1 prerelease arrives via sglang's --prerelease=allow.
# --no-config so this isn't silently overridden.
uv pip install --no-config "numpy<2" "scipy<1.14" "pyarrow<26" "pandas<3.1"

# Apply slime's official patches to megatron + sglang.
SLIME_PATCH_DIR="$(cd slime/docker/patch/latest && pwd)"
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

# Megatron patch: apply against the repo without leaving cwd (git -C).
git -C Megatron-LM update-index --refresh >/dev/null 2>&1 || true
git -C Megatron-LM apply --3way "$SLIME_PATCH_DIR/megatron.patch"

# sglang patches: apply into site-packages without leaving cwd (patch -d).
patch -d "$SITE_PACKAGES" -p2 -F0 -N < "$SLIME_PATCH_DIR/sglang.patch"
patch -d "$SITE_PACKAGES" -p2 -F0 -N < "$SLIME_PATCH_DIR/sglang-top_p.patch"
patch -d "$SITE_PACKAGES" -p2 -F0 -N < "$SLIME_PATCH_DIR/sglang-release_hicache.patch"
patch -d "$SITE_PACKAGES" -p2 -F0 -N < "$SLIME_PATCH_DIR/sglang-pull_weights.patch"
# megatron-sglang-aligned.patch and sglang-deterministic.patch are skipped: they are
# only for GLM-5 train/rollout log-prob alignment, which also needs DeepEP/DeepGEMM
# builds, and the latter patches sgl-kernel CUDA sources the sglang wheel doesn't ship.
