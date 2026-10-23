# Strands AppWorld Agent

An RL-trainable [AppWorld](https://github.com/StonyBrookNLP/appworld) agent using Bedrock AgentCore RL Toolkit. The agent solves day-to-day tasks by interacting with simulated app APIs (Spotify, Venmo, Gmail, etc.) via a Python REPL.

The agent uses a **ReAct code agent** pattern: a single `execute` tool runs Python code in an AppWorld sandbox environment. The agent discovers and calls APIs on-demand through the pre-loaded `apis` object, keeping tool context minimal regardless of how many APIs exist.

## Installation

Run from the repository root:

```bash
cd examples/strands_appworld_agent

# Ensure git-lfs is installed (appworld stores bundle files in Git LFS)
git lfs install

uv venv --python 3.12
source .venv/bin/activate
UV_GIT_LFS=1 uv pip install -e .
```

Run the remaining commands from `examples/strands_appworld_agent`.
After installing, set up AppWorld data:

```bash
appworld install
appworld download data
```

## Run RL App Locally with a vLLM Server

### Start a local vLLM server

```bash
# In a separate directory/environment
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 8192 --port 4000 --enable-auto-tool-choice --tool-call-parser hermes
```

### Prepare the environment file

Prepare the environment file so Strands runs in non-interactive server mode:

```bash
cp .env.example .env
```

### Start the application server

```bash
# Start the server in one terminal
python rl_app.py

# Submit a request in another terminal
# Task id list could be found at data split files in the downloaded AppWorld
# dataset (data/datasets, dataset is downloaded by running the command
# `appworld download data`)
curl -X POST http://localhost:8080/invocations \
     -H "Content-Type: application/json" \
     -d '{
       "task_id": "3d9a636_1",
       "_config": {
         "base_url": "http://localhost:4000/v1",
         "model_id": "Qwen/Qwen3-4B-Instruct-2507",
         "sampling_params": {"max_completion_tokens": 8192}
       }
     }'
```

The request returns `{"reward": ...}`.

## Deploy

Run from `examples/strands_appworld_agent`. Set the subnet and security group IDs for the
host running your inference server:

```bash
uv tool install bedrock-agentcore-starter-toolkit

agentcore configure --entrypoint rl_app.py --name strands_appworld_agent \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --vpc --subnets subnet-xxxxxxxxxxxxxxxxx --security-groups sg-xxxxxxxxxxxxxxxxx \
  --disable-memory --non-interactive
agentcore deploy --agent strands_appworld_agent
```

For a publicly reachable inference server, omit the VPC flags.
To redeploy after editing the agent, run `agentcore deploy --agent strands_appworld_agent`.

## Evaluate

Copy the evaluation config:

```bash
cp config.example.toml config.toml
```

Set the Runtime ARN returned by the deployment and the model settings in `config.toml`:

```toml
[agentcore]
agent_arn = "arn:aws:bedrock-agentcore:REGION:ACCOUNT_ID:runtime/AGENT_ID"

[eval]
base_url = "http://INFERENCE_SERVER_IP:4000/v1"
model_id = "Qwen/Qwen3-4B-Instruct-2507"
sampling_params = {temperature = 0.0}
```

Use `--max_concurrent` to limit parallel tasks and `--tps_limit` to limit
session starts per second.

Ensure you have run `appworld download data` to download appworld dataset folder into the same path where you will run evaluation, evaluation script will load AppWorlds's built-in dataset splits (`train`, `dev`, `test_normal`, `test_challenge`) via `--dataset` from the dataset folder.

```bash
# Run full evaluation on the normal test split
python evaluate.py --dataset test_normal --exp_id my_eval --max_concurrent 50 --timeout 1800

# Quick test with a few tasks
python evaluate.py --dataset test_normal --exp_id my_eval_test --limit 5

# Only difficulty-1 tasks from the train split
python evaluate.py --dataset train --difficulty 1 --exp_id my_eval_easy
```

CLI arguments override `config.toml` values.

Results are saved as `<output_dir>/<exp_id>.jsonl`; pass `--output_dir ./results`
to choose a directory. The default is under the operating system's temporary directory.

## File Structure

```
strands_appworld_agent/
├── rl_app.py            # ReAct code agent
├── reward.py            # AppWorldReward (pass_count / num_tests)
├── evaluate.py          # Batch evaluation
├── Dockerfile           # ACR container (AppWorld data baked in)
├── config.example.toml  # Configuration template
├── .env.example         # Environment variable template
└── pyproject.toml       # Python dependencies
```
