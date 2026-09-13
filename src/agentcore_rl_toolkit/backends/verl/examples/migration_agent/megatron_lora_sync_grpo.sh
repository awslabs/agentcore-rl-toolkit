#!/bin/bash
# GRPO on MigrationBench (Java 8->17) with rollouts as AgentCore Runtime sessions, on
# Megatron expert parallelism + LoRA. The run itself is `config/main.yaml`'s defaults list;
# this script only sets the environment verl and the agent loop read, then hands over.
#
# Prerequisites (see README.md for the full list):
#   1. Deploy examples/strands_migration_agent/rl_app.py to AgentCore Runtime and prepare a
#      data bucket of repo tarballs (that example's README).
#   2. python preprocess_migrationbench.py --s3-bucket-name <data-bucket> \
#          --output-dir "$dataset_path_prefix"                              (once)
#   3. ACR containers must be able to reach this host on the rollout gateway's port
#      (auto-assigned; open the trainer CPU nodes' ports to ACR egress).
#   4. Megatron deps, in a Python 3.12 venv:
#        uv sync --python 3.12 --extra verl --extra rollout --group verl-megatron
#
# Hydra overrides pass through: ./megatron_lora_sync_grpo.sh trainer.total_epochs=3
set -x

export HYDRA_FULL_ERROR=1

# verl imports these as a side effect of `import verl`, in every process that does. It does
# not forward the variable to Ray workers, so on a multi-node cluster export it in each
# node's `ray start` shell too.
#   ...experimental.verl  registers the rollout_session_agent_loop
#   ...verl.trainer       registers the agentcore_* trainer modes
export VERL_USE_EXTERNAL_MODULES=agentcore_rl_toolkit.backends.experimental.verl,agentcore_rl_toolkit.backends.verl.trainer

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Every one of these is read as ${oc.env:...} during config composition, so a missing one
# fails the launch rather than surfacing mid-run.
: "${AWS_REGION:?region of the agent runtime and the output bucket}"
: "${agentcore_runtime_arn:?arn:aws:bedrock-agentcore:<region>:<acct>:runtime/<name>}"
: "${rollout_output_s3:?s3://bucket/prefix for rollout dumps and agent results}"
: "${dataset_path_prefix:?directory holding the migrationbench_agent_*.parquet files}"
: "${model_path_prefix:?directory holding Qwen3-Coder-30B-A3B-Instruct}"
: "${checkpoint_path_prefix:?directory to write checkpoints under}"

python3 "$SCRIPT_DIR/main.py" "$@"
