# shellcheck shell=bash
# Local SkyRL Tinker endpoint layout (one 8-GPU host; tested on 8x H100 80 GB).
# Adapted from agentcore-rl-toolkit/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/environment.sh
# (PR #158), which assumes an Ubuntu DLAMI with /usr/local/cuda-13.0 and /opt/dlami/nvme.
# Sourced inside the endpoint container (docker.sh), which bind-mounts these host paths at the
# same locations. Override any of them in the environment.
export SKYRL_COMMIT="${SKYRL_COMMIT:-f4ecdd09c2df6085db92e2d08d5ec366e9ba8506}"   # PR #158's tested revision
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
export MODEL_REVISION="${MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"   # snapshot the published model used
export TINKER_PORT="${TINKER_PORT:-18080}"

export SKYRL_HOME="${SKYRL_HOME:-$HOME/skyrl}"
export SKYRL_STATE_DIR="${SKYRL_STATE_DIR:-$HOME/skyrl-state}"
export TMPDIR="$SKYRL_HOME/tmp"
export SKYRL_DIR="$SKYRL_HOME/SkyRL"
# Serve the snapshot straight from the Hugging Face cache (setup.sh downloads it if missing).
export MODEL_DIR="${MODEL_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3.5-9B/snapshots/$MODEL_REVISION}"
export UV_PROJECT_ENVIRONMENT="$SKYRL_HOME/venv"
export UV_CACHE_DIR="$SKYRL_HOME/uv-cache"
export UV_PYTHON_INSTALL_DIR="$SKYRL_HOME/python"   # persistent: the venv symlinks this interpreter
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export XDG_CACHE_HOME="$SKYRL_HOME/cache"
export FLASHINFER_WORKSPACE_BASE="$SKYRL_HOME"   # flashinfer JIT cache ignores XDG_CACHE_HOME
export VLLM_CACHE_ROOT="$SKYRL_HOME/cache/vllm"
export TORCH_EXTENSIONS_DIR="$SKYRL_HOME/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$SKYRL_HOME/torchinductor"
export TRITON_CACHE_DIR="$SKYRL_HOME/triton-cache"
export CUDA_CACHE_PATH="$SKYRL_HOME/cuda-cache"
export PYTHONPYCACHEPREFIX="$SKYRL_HOME/pycache"
export RAY_TMPDIR="$TMPDIR/ray"

# CUDA 13.0 toolkit from the nvidia/cuda:13.0.2-devel-ubuntu22.04 image (same path as the DLAMI).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"   # host runs (no toolkit) pass the pip CUDA 13 toolkit
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# SWEgrep turns generate up to 16,384 tokens while ~190 rollouts share the engines; the 300 s
# default would fail long turns that are merely queued.
export SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC="${SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC:-1800}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"   # PR #158 sets 0; then TE fell back to unfused attention
                                                # for Qwen3.5-9B (56 GiB score matrix, OOM at 61k tokens)
export CUDA_DEVICE_MAX_CONNECTIONS=1
