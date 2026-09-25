#!/usr/bin/env zsh

set -eux

source /agent/.venv/bin/activate
uv pip install '/agent[strands]'
uv pip install '/rl_toolkit[a2a-server]'
