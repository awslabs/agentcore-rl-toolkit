#!/usr/bin/env zsh

set -eux

mkdir -p /agent
cd /agent

curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/agent/uv" sh
export PATH="/agent/uv:$PATH"
export UV_CACHE_DIR=/agent/uv_cache
export UV_PYTHON_INSTALL_DIR=/agent/uv_python

uv python install 3.13
uv venv -p 3.13
