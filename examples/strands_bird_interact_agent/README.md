# Strands BIRD-Interact Agent

An RL-trainable agent for [BIRD-Interact](https://github.com/bird-bench/BIRD-Interact) (arXiv 2510.05318; code MIT,
data CC BY-SA 4.0), Lite, in the a-Interact setting, using Bedrock AgentCore RL Toolkit. Each task is an ambiguous
business question over one of 18 PostgreSQL databases: the agent explores the schema, column meanings and a
knowledge base, asks a simulated user to resolve the ambiguities, and submits SQL that the user tests; a correct
submission brings a follow-up question. The reward is upstream's: 0.7 for the first question, plus 0.3 for the
follow-up.

The agent has upstream's nine actions as tools, each with its cost in the user's patience: `execute`, `get_schema`,
`get_all_column_meanings`, `get_column_meaning`, `get_all_external_knowledge_names`, `get_knowledge_definition`,
`get_all_knowledge_definitions` (the database environment), `ask` and `submit` (the user). The environment service
(`env/`) runs them.

## Protocol

Upstream's a-Interact (BIRD-Interact@451fe2c, `bird_interact_agent/batch_run_bird_interact/`): its system prompt,
adapted to tool calling (`prompts/system.md`: the text action format and its toy example are dropped; the task
description, costs and tips are kept), the action costs, and the budget: 6 plus 2 per labeled ambiguity plus
patience 6, as in upstream's official runs. When the budget runs out, the next action must be `submit`, and that
submission ends the task. `execute` and `submit` run on a freshly reset copy of the task's database, as upstream
resets the database before each. A query's answer is checked by comparing result sets with the gold SQL; a data
management task's by upstream's test cases. The user simulator is upstream's two-stage one (v2 prompts: the encoder
classifies the question against the labeled ambiguities, the decoder answers in the user's words) with Claude
Haiku 4.5 on Bedrock at temperature 0 (upstream: gemini-2.0-flash). 60 turns (upstream's official runs), 8K output
tokens per turn, 128K context; a reply without a tool call costs 0.2 and gets an error, as upstream answers a reply
without an action. An unavailable database or user simulator fails the rollout.

**Gold.** The authors withhold the gold SQL and test cases to prevent leakage; request them as upstream's README
describes. The environment service reads them from your path and keeps them: invocation payloads carry only the
task id. Do not commit or publish the gold file.

On 50 tasks (`--split pilot50`, 1 attempt, on AgentCore Runtime):

| Model | Reward | Phase 1 | Phase 2 |
|---|---|---|---|
| Claude Opus 5 (high) | 0.632 | 68% | 52% |
| GPT-6 Astra (high) | 0.606 | 66% | 48% |

## Installation

```bash
cd examples/strands_bird_interact_agent

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[eval,env]"
```

Run the remaining commands from `examples/strands_bird_interact_agent`.

## Start the environment service

On a host in the VPC the runtime will use, with Docker, about 5 GB of disk, and AWS credentials for Bedrock (the
user simulator):

```bash
# Upstream's PostgreSQL image (pinned), which loads the 18 Lite databases, published on 127.0.0.1
python env/db.py start /data/bird
python env/db.py check /data/bird

# Serve it on the host's private address (never a public one), with your gold file
python env/serve.py --host PRIVATE_IP --data /data/bird --gold /secure/bird_interact_gt_kg_testcases_1008.jsonl
curl http://PRIVATE_IP:18734/health        # {"tasks": 300, ...}
```

Allow TCP 18734 from the runtime's security group only. The agent's SQL runs on a per-episode copy of its database,
as upstream's superuser inside the container; the copies are dropped when the episode ends.

## Run Locally

With the service reachable at `BIRD_ENV_URL` and AWS credentials for Bedrock:

```bash
export BIRD_ENV_URL=http://PRIVATE_IP:18734
python test_local.py --split smoke2 --model_id us.anthropic.claude-opus-5 \
  --bedrock_fields '{"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}'
```

Without `--base_url`, `model_id` is a Bedrock model and `bedrock_fields` its additional request fields (e.g.
`'{"reasoning": {"effort": "high"}}'` for `us.openai.gpt-6-astra`). With `--base_url`, the model is served by an
OpenAI-compatible endpoint (vLLM, or the rollout gateway during training).

## Deploy

```bash
uv tool install bedrock-agentcore-starter-toolkit

agentcore configure --entrypoint rl_app.py --name strands_bird_interact_agent \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --vpc --subnets subnet-xxxxxxxxxxxxxxxxx --security-groups sg-xxxxxxxxxxxxxxxxx \
  --disable-memory --non-interactive
agentcore deploy --agent strands_bird_interact_agent --env BIRD_ENV_URL=http://PRIVATE_IP:18734
```

Use the subnets and security groups from which the environment service (and, for training, the inference server)
is reachable. The runtime's execution role needs Bedrock model access for Bedrock models.

## Evaluate

```bash
cp config.example.toml config.toml        # set agent_arn and the model

# All 300 tasks, avg@4
python evaluate.py --split all --repeats 4 --exp_id all_opus5 --max_concurrent 32

# Quick test
python evaluate.py --split smoke2 --exp_id smoke
```

Splits: `all` (300 tasks); `pilot50` and `smoke2`, 50 and 2 tasks drawn with seed 0. Records go to
`$TMPDIR/strands_bird_interact_agent-evaluations/<exp_id>.jsonl`; the summary line gives the mean reward and the
phase 1 and phase 2 success rates.

## File Structure

```
rl_app.py                  the Strands agent and its loop; @app.entrypoint returns {"reward": ...}
tools.py                   the nine actions, as clients of the environment service
reward.py                  upstream's phase rewards
prompts/system.md          upstream's system prompt, adapted to tool calling
models.py                  the typed invocation payload
tasks.py                   the task ids from Hugging Face (pinned revision) and the splits
evaluate.py                evaluation on AgentCore Runtime
test_local.py              local run, without AgentCore Runtime
env/serve.py               the environment service: tasks with the gold, sessions
env/bird_env.py            one episode: actions, budget, user simulator, tests (ported from upstream)
env/user_simulator_prompts.json  upstream's v2 simulator prompts
env/db.py                  the databases: upstream's PostgreSQL image, check
```
