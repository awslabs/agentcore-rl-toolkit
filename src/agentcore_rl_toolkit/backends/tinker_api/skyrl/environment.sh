# Remote EC2 layout. This does not change the launcher's local TMPDIR.
export SKYRL_HOME="$HOME/skyrl"
export SKYRL_STATE_DIR="$HOME/skyrl-state"
export TMPDIR=/opt/dlami/nvme/skyrl/tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
# Source, dependencies, weights and reusable compilation caches stay on EBS.
export SKYRL_DIR="$SKYRL_HOME/SkyRL"
export MODEL_DIR="$SKYRL_HOME/models/${MODEL_ID##*/}"
export UV_PROJECT_ENVIRONMENT="$SKYRL_HOME/venv"
export UV_CACHE_DIR="$SKYRL_HOME/uv-cache"
export HF_HOME="$SKYRL_HOME/hf-cache"
export XDG_CACHE_HOME="$SKYRL_HOME/cache"
export TORCH_EXTENSIONS_DIR="$SKYRL_HOME/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$SKYRL_HOME/torchinductor"
export TRITON_CACHE_DIR="$SKYRL_HOME/triton-cache"
export CUDA_CACHE_PATH="$SKYRL_HOME/cuda-cache"
export PYTHONPYCACHEPREFIX="$SKYRL_HOME/pycache"
export RAY_TMPDIR="$TMPDIR/ray"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# SkyRL starts its own Ray runtime, separate from SkyPilot's runtime.
unset RAY_ADDRESS
