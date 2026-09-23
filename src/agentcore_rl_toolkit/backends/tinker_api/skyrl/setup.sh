#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source environment.sh

sudo mkdir -p /opt/dlami/nvme/skyrl
sudo chown "$(id -u):$(id -g)" /opt/dlami/nvme/skyrl
mkdir -p "$TMPDIR" "$SKYRL_HOME" "$SKYRL_STATE_DIR"
nvidia-smi
test -x "$CUDA_HOME/bin/nvcc"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh -o "$TMPDIR/install-uv.sh"
  bash "$TMPDIR/install-uv.sh"
fi
if [ ! -d "$SKYRL_DIR/.git" ]; then
  git init "$SKYRL_DIR"
  git -C "$SKYRL_DIR" remote add origin https://github.com/NovaSky-AI/SkyRL.git
fi
if ! git -C "$SKYRL_DIR" cat-file -e "$SKYRL_COMMIT^{commit}" 2>/dev/null; then
  git -C "$SKYRL_DIR" fetch --depth 1 origin "$SKYRL_COMMIT"
fi
git -C "$SKYRL_DIR" checkout --detach "$SKYRL_COMMIT"
cd "$SKYRL_DIR"
uv sync --frozen --python 3.12 --extra tinker --extra fsdp
uv run --frozen --extra tinker --extra fsdp \
  hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
