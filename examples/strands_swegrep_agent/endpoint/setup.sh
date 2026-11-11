#!/usr/bin/env bash
# Install the SkyRL Tinker endpoint on this host (idempotent). Adapted from PR #158's setup.sh:
# no sudo / instance-store mount, model weights reused from the local HF cache.
set -euo pipefail
cd "$(dirname "$0")"
source environment.sh
mkdir -p "$TMPDIR" "$SKYRL_HOME" "$SKYRL_STATE_DIR"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
command -v uv >/dev/null || { echo "FATAL: uv not on PATH" >&2; exit 1; }

if [ ! -d "$SKYRL_DIR/.git" ]; then
  git init -q "$SKYRL_DIR"
  git -C "$SKYRL_DIR" remote add origin https://github.com/NovaSky-AI/SkyRL.git
fi
if ! git -C "$SKYRL_DIR" cat-file -e "$SKYRL_COMMIT^{commit}" 2>/dev/null; then
  git -C "$SKYRL_DIR" fetch -q --depth 1 origin "$SKYRL_COMMIT"
fi
git -C "$SKYRL_DIR" checkout -q --detach "$SKYRL_COMMIT"

cd "$SKYRL_DIR"
test -x "$CUDA_HOME/bin/nvcc" || { echo "FATAL: no nvcc under CUDA_HOME=$CUDA_HOME (run via docker.sh)" >&2; exit 1; }
uv sync --frozen --python 3.12 --extra tinker --extra megatron --extra aws
"$CUDA_HOME/bin/nvcc" --version | tail -2

if [ ! -f "$MODEL_DIR/config.json" ]; then
  uv run --frozen --extra tinker --extra megatron --extra aws \
    hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
fi
uv run --frozen --extra tinker --extra megatron --extra aws python - <<'EOF'
import torch, vllm, transformer_engine, megatron.core, tinker
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("vllm", vllm.__version__, "tinker", tinker.__version__)
EOF
echo "SkyRL endpoint environment ready: $UV_PROJECT_ENVIRONMENT (model $MODEL_DIR)"
