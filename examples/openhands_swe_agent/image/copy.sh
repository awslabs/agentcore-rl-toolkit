#!/usr/bin/env zsh

set -eux

source /agent/.venv/bin/activate
uv pip install '/agent[server,strands]'

# The wire protocol this harness speaks. Installed with its declared dependencies,
# not --no-deps: the toolkit's top-level __init__ imports .app eagerly, so even
# `import agentcore_rl_toolkit.rollout_session.wire` pulls in bedrock_agentcore and
# boto3. A --no-deps install therefore yields a package that cannot be imported at
# all, and naming the subset this container happens to reach would be a second copy
# of the toolkit's requirements to keep in step. See the Dockerfile.
uv pip install /rl_toolkit
