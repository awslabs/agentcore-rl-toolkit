#!/usr/bin/env bash
set -euo pipefail

# Install the slime training backend with CUDA 13
#
# Usage:
#   bash src/agentcore_rl_toolkit/backends/slime/scripts/install_slime.sh

TORCH_BACKEND=cu130               # uv --torch-backend for the PyTorch ecosystem
: "${CUDA_HOME:=/usr/local/cuda-13.0}"
export CUDA_HOME

# Pins follow slime's build_conda.sh / docker/Dockerfile at SLIME_COMMIT.
SLIME_COMMIT="540ea919e45b8b929198abce008a103be9cfc779"
SGLANG_VERSION="0.5.21"
MEGATRON_COMMIT="4b4acac9a1d28ea6829c8d4f566d75698a21249d"
PATCH_VERSION="v0.5.21"
TMS_COMMIT="bdab8c55f490a00a746ec34e301391617f5ca878"
TRANSFORMER_ENGINE_VERSION="2.18.0"
SLIME_NCCL_VERSION="2.30.7"

# sglang pins transformers==5.12.1 and tokenizers==0.22.2; transformers>=5.16.1
# requires tokenizers>=0.23.1, so both are overridden for every resolve below.
OVERRIDES_FILE="$(mktemp)"
cat > "$OVERRIDES_FILE" <<'REQ'
transformers==5.16.1
tokenizers==0.23.1
transformer-engine-cu12 ; sys_platform == 'never'
REQ
# Keep the CUDA 13 PyTorch stack fixed across later resolving installs.
CONSTRAINTS_FILE="$(mktemp)"
cat > "$CONSTRAINTS_FILE" <<'REQ'
torch==2.13.0
torchvision==0.28.0
torchaudio==2.11.0
REQ
export UV_OVERRIDE="$OVERRIDES_FILE" UV_CONSTRAINT="$CONSTRAINTS_FILE"

# Install agentcore-rl-toolkit. Megatron-LM and slime are cloned into the cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

# Install the CUDA 13 PyTorch ecosystem
uv pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --torch-backend="$TORCH_BACKEND"
uv pip install -e "${REPO_ROOT}[gateway,rollout]"
uv pip install "flash-attn==2.8.3.post1" \
  --torch-backend="$TORCH_BACKEND" \
  --index "https://wheels.astral.sh/simple/cu130/"

# The override above excludes transformer-engine-cu12 to avoid "Multiple
# libcudart libraries found" errors.
NVTE_WITH_NCCL_EP=0 MAX_JOBS=128 uv pip install --no-cache --no-build-isolation \
  "transformer_engine[pytorch,core-cu13]==${TRANSFORMER_ENGINE_VERSION}"

NVCC_APPEND_FLAGS="--threads 4" APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_PARALLEL_BUILD=8 \
  uv pip install -v --no-build-isolation --no-cache-dir \
  "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"

# torch_memory_saver's TMS_CUDA_MAJOR sets the compiled .so suffix (_cu13) and
# must match what its runtime detector reads from torch.version.cuda, so we
# derive it from torch rather than hardcoding.
export TMS_CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')"
uv pip install -v "git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT}" \
  --no-cache-dir --reinstall-package torch-memory-saver --no-build-isolation

uv pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# Install sglang
uv pip install "sglang==${SGLANG_VERSION}" --torch-backend="$TORCH_BACKEND"

# We have to git clone and install from local because wheel file does not expose megatron.training that is required by slime
uv pip install pybind11 "setuptools>=80.0.0" "packaging>=24.2"

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout "$MEGATRON_COMMIT"
uv pip install -e . --no-build-isolation --config-settings editable_mode=compat
cd ..

uv pip install --reinstall-package pyjwt PyJWT

# Install slime
git clone https://github.com/THUDM/slime.git
cd slime
git checkout "$SLIME_COMMIT"
uv pip install -r "requirements.txt" "sglang==${SGLANG_VERSION}" --torch-backend="$TORCH_BACKEND"
# slime's wandb init calls wandb.util.generate_id, removed in wandb 0.28.2.
uv pip install "wandb<0.28.2"
uv pip install -e . --no-deps
cd ..

uv pip install --no-deps --reinstall \
  "https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl"
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# kernels 0.15.x breaks the transformers.integrations.hub_kernels import.
uv pip install "numpy==2.3.5" "scipy==1.18.1" "pyarrow==25.0.1" "kernels<0.15.0"
# PyTorch's metadata pins an older NCCL.
uv pip install --no-deps --reinstall "nvidia-nccl-cu13==${SLIME_NCCL_VERSION}"

# Apply slime's official patches to megatron + sglang.
SLIME_PATCH_DIR="$(cd "slime/docker/patch/${PATCH_VERSION}" && pwd)"
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

# Megatron patch: apply against the repo without leaving cwd (git -C).
git -C Megatron-LM update-index --refresh >/dev/null 2>&1 || true
git -C Megatron-LM apply --3way "$SLIME_PATCH_DIR/megatron.patch"

# sglang patches: apply into site-packages without leaving cwd (git -C). The wheel
# ships no sglang tests, so only sglang/ paths apply (--include matches after -p2).
# git apply skips excluded paths and, inside a repository, paths outside the cwd,
# both with exit status 0. The ceiling stops git from resolving paths against an
# enclosing repository (a .venv in this checkout), and each patch must apply files.
for patch_name in sglang sglang-top_p sglang-release_hicache sglang-pull_weights; do
  if ! applied="$(GIT_CEILING_DIRECTORIES="$(dirname "$SITE_PACKAGES")" git -C "$SITE_PACKAGES" \
      apply -v -p2 --include='sglang/*' "$SLIME_PATCH_DIR/${patch_name}.patch" 2>&1)" \
    || ! grep -q '^Applied patch' <<< "$applied"; then
    echo "$applied" >&2
    echo "${patch_name}.patch did not apply" >&2
    exit 1
  fi
done
# megatron-sglang-aligned.patch and sglang-deterministic.patch are skipped: they are
# only for GLM-5 train/rollout log-prob alignment, which also needs DeepEP/DeepGEMM
# builds, and the latter patches sgl-kernel CUDA sources the sglang wheel doesn't ship.
