# Strands ExCyTIn Agent

An RL-trainable agent for [ExCyTIn-Bench](https://github.com/microsoft/SecRL) (arXiv 2507.14201; code MIT, data
CDLA-Permissive-2.0, for research use), using Bedrock AgentCore RL Toolkit. Each task is a question about a security
incident in a Microsoft Sentinel environment: given an alert and its context, the agent finds a related fact (a
compromised account, a malicious IP, a process) by querying the incident's logs, over 20 tables, with SQL. The
reward is upstream's evaluation: 1 for a correct answer, else partial credit for the steps of the gold solution
the answer reaches.

The agent has one tool, `execute_sql`: read-only SQL on the incident's MySQL database, served by the environment
service (`env/`).

## Protocol

Upstream's setup (microsoft/SecRL@a1b234f7): its system prompt adapted to tool calling (`prompts/system.md`), the
question's context and question as the user message, 25 turns, 8K output tokens per turn, 128K context, tool output
cut at 100,000 characters. There is no forced answer. As upstream's SQL action, a query runs for at most 30 s, and a
result over 100,000 characters shows its first 15 rows. The questions are upstream's latest set
(`secgym/questions/opus5/`). The reward is upstream's evaluator (`reward.py`, prompts in `prompts/`), with
gpt-oss-120b on Bedrock as the judge (upstream: GPT-4o): an answer check with reflection, then, for a wrong answer,
a step check whose correct steps earn discounted credit. Training data must not touch an evaluated database, so 3
of the 8 incidents (drawn with seed 0: 39, 166, 322) are held out for evaluation. An unavailable database fails the
rollout.

On 50 tasks (`--split pilot50`, 1 attempt, on AgentCore Runtime):

| Model | Success | Reward |
|---|---|---|
| Claude Opus 5 (high) | TBD | TBD |
| GPT-6 Astra (high) | TBD | TBD |

## Installation

```bash
cd examples/strands_excytin_agent

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[env]"
```

Run the remaining commands from `examples/strands_excytin_agent`.

## Start the environment service

On a host in the VPC the runtime will use, with Docker and about 10 GB of disk:

```bash
# Download upstream's data (pinned revision) and load the 8 incidents into MySQL 9.0, published on 127.0.0.1
python env/db.py start /data/excytin
python env/db.py check /data/excytin       # per incident: tables, rows, and that its reader can only read it

# Serve it on the host's private address (never a public one)
python env/serve.py --host PRIVATE_IP --data /data/excytin
curl http://PRIVATE_IP:18733/health
```

Allow TCP 18733 from the runtime's security group only. The service connects to each incident as a MySQL user that
may only read that incident's database; the passwords are generated into `/data/excytin/secrets/`.

## Run Locally

With the service reachable at `EXCYTIN_ENV_URL` and AWS credentials for Bedrock:

```bash
export EXCYTIN_ENV_URL=http://PRIVATE_IP:18733
python test_local.py --split smoke2 --model_id us.anthropic.claude-opus-5 \
  --bedrock_fields '{"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}'
```

Without `--base_url`, `model_id` is a Bedrock model and `bedrock_fields` its additional request fields (e.g.
`'{"reasoning": {"effort": "high"}}'` for `us.openai.gpt-6-astra`). With `--base_url`, the model is served by an
OpenAI-compatible endpoint (vLLM, or the rollout gateway during training).

## Deploy

```bash
uv tool install bedrock-agentcore-starter-toolkit

agentcore configure --entrypoint rl_app.py --name strands_excytin_agent \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --vpc --subnets subnet-xxxxxxxxxxxxxxxxx --security-groups sg-xxxxxxxxxxxxxxxxx \
  --disable-memory --non-interactive
agentcore deploy --agent strands_excytin_agent --env EXCYTIN_ENV_URL=http://PRIVATE_IP:18733
```

Use the subnets and security groups from which the environment service (and, for training, the inference server)
is reachable. The runtime's execution role needs Bedrock model access for the agent and the judge.

## Evaluate

```bash
cp config.example.toml config.toml        # set agent_arn and the model

# The 244 evaluation questions, avg@4
python evaluate.py --split eval --repeats 4 --exp_id eval_opus5 --max_concurrent 32

# Quick test
python evaluate.py --split smoke2 --exp_id smoke
```

Splits: `eval` (the 244 test questions of the 3 held-out incidents); `pilot50` and `smoke2`, 50 and 2 of them drawn
with seed 0; `train` (the 261 train questions of the other 5 incidents). Records go to
`$TMPDIR/strands_excytin_agent-evaluations/<exp_id>.jsonl`; the summary line gives the success rate and the mean
reward.

## File Structure

```
rl_app.py            the Strands agent and its budget; @app.entrypoint returns {"reward": ...}
tools.py             execute_sql, as a client of the environment service
reward.py            upstream's evaluator
prompts/             upstream's system prompt (adapted to tool calling) and judge prompts
models.py            the typed invocation payload
tasks.py             the questions from upstream's repository (pinned commit) and the splits
evaluate.py          evaluation on AgentCore Runtime
test_local.py        local run, without AgentCore Runtime
env/serve.py         the environment service
env/db.py            the incident databases: download, load into MySQL, check
```
