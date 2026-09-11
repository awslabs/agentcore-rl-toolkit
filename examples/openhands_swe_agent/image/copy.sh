#!/usr/bin/env zsh

set -eux

source /agent/.venv/bin/activate
uv pip install '/agent[server,strands]'

# With deps, not --no-deps: the toolkit's top-level __init__ imports .app eagerly, so
# even the wire module pulls in bedrock_agentcore and boto3.
uv pip install /rl_toolkit
