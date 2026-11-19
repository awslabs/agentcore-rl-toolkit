# Strands CompLit Agent

An RL-trainable [CompLit](https://github.com/dayoon-ko/CompLit) agent using Bedrock AgentCore RL Toolkit. A
researcher describes, in their own words, a paper they want to find; the agent searches arXiv, reads candidate
papers, and answers with the one paper that meets every requirement, or says that none does. The reward is 1 if
the answer's arXiv id matches the gold paper, else 0.

The two tools are served by an environment service (`env/`) on a host the runtime reaches in its VPC:
- `arxiv_search` runs over a local BM25 index of the full arXiv metadata (3.2M papers), so rollouts do not depend
  on the rate-limited arXiv API.
- `read_paper` fetches [ar5iv](https://ar5iv.labs.arxiv.org/) full text through one cache and one request pace.

## Protocol

Upstream's setup with web search off: upstream's prompts and scoring (`prompts.py`, `reward.py`), 15 turns, 8K
output tokens per turn, 128K context, and a forced answer when the budget runs out. A model refusal scores 0; an
unavailable environment fails the rollout.

**Cumulative** (`--split cumulative`) is multi-turn: the Implicit queries, with the requirements arriving over 3-5
messages from upstream's simulated researcher (`simulator.py`). The simulator runs on a Bedrock model at
temperature 0, Claude Haiku 4.5 by default (`--simulator_model_id`; upstream: gpt-5.4-mini).

Accuracy on AgentCore Runtime:

| Model | Implicit (`--split pilot50`, avg@4) | Cumulative (`--split cumulative_pilot50`, 1 attempt) |
|---|---|---|
| Claude Opus 5 (high) | 63.0% | 54.0% |
| GPT-6 Astra (high) | 78.5% | 14.0% |

Claude Opus 5 refuses some epidemic-modeling tasks; in Cumulative, GPT-6 Astra mostly answers that no paper meets
every requirement.

## Installation

```bash
cd examples/strands_complit_agent

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[eval,env]"
```

Run the remaining commands from `examples/strands_complit_agent`.

## Start the environment service

On a host in the VPC the runtime will use (about 20 GB of disk and some cores for search):

```bash
# Build the search index (about 15 minutes, 8 GB): downloads the pinned arXiv metadata snapshot
python env/build_index.py --data-dir /data/arxiv-metadata --out /data/index/arxiv.sqlite

# Serve it on the host's private address (never a public one)
COMPLIT_INDEX=/data/index/arxiv.sqlite COMPLIT_CACHE_DIR=/data/cache \
  python env/serve.py --host PRIVATE_IP --port 18731
curl http://PRIVATE_IP:18731/health        # {"papers": 3224635}
```

Allow TCP 18731 from the runtime's security group only. The service reads public data (arXiv metadata, CC0, and
ar5iv pages) and writes only its page cache. Settings: `COMPLIT_AR5IV_MIN_INTERVAL` (seconds between ar5iv
requests, default 1) and `COMPLIT_SEARCH_WORKERS` (search processes, default 10).

## Run Locally

With the service reachable at `COMPLIT_ENV_URL` and AWS credentials for Bedrock:

```bash
export COMPLIT_ENV_URL=http://PRIVATE_IP:18731
python test_local.py --split smoke2 --model_id us.anthropic.claude-opus-5 \
  --bedrock_fields '{"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}'
```

Without `--base_url`, `model_id` is a Bedrock model and `bedrock_fields` its additional request fields (e.g.
`'{"reasoning": {"effort": "high"}}'` for `us.openai.gpt-6-astra`). With `--base_url`, the model is served by an
OpenAI-compatible endpoint (vLLM, or the rollout gateway during training).

## Deploy

```bash
uv tool install bedrock-agentcore-starter-toolkit

agentcore configure --entrypoint rl_app.py --name strands_complit_agent \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --vpc --subnets subnet-xxxxxxxxxxxxxxxxx --security-groups sg-xxxxxxxxxxxxxxxxx \
  --disable-memory --non-interactive
agentcore deploy --agent strands_complit_agent --env COMPLIT_ENV_URL=http://PRIVATE_IP:18731
```

Use the subnets and security groups from which the environment service (and, for training, the inference server)
is reachable. The runtime's execution role needs Bedrock model access for Bedrock models, and for the Cumulative
setting for the simulator model.

## Evaluate

```bash
cp config.example.toml config.toml        # set agent_arn and the model

# Implicit, avg@4
python evaluate.py --split implicit --repeats 4 --exp_id implicit_opus5 --max_concurrent 32

# Cumulative (multi-turn)
python evaluate.py --split cumulative --exp_id cumulative_opus5 --max_concurrent 32

# Quick test
python evaluate.py --split smoke2 --exp_id smoke
```

Splits: the settings `standard` (350), `implicit` (220), `unmet` (300) and `cumulative` (220), and subsets drawn
with seed 0: `pilot50` / `smoke2` (50 and 2 Implicit tasks), `cumulative_pilot50` / `cumulative_smoke2`. Records go
to `$TMPDIR/strands_complit_agent-evaluations/<exp_id>.jsonl`; the summary line is the mean reward.

## File Structure

```
rl_app.py            the Strands agent: tools, budget, forced answer; @app.entrypoint returns {"reward": ...}
tools.py             arxiv_search and read_paper, as clients of the environment service
reward.py            upstream's scoring
prompts.py           upstream's prompts
simulator.py         upstream's simulated researcher (Cumulative), on a Bedrock model
models.py            the typed invocation payload
tasks.py             CompLit from Hugging Face (dayoon/CompLit, pinned revision) and the splits
evaluate.py          evaluation on AgentCore Runtime
test_local.py        local run, without AgentCore Runtime
env/serve.py         the environment service
env/arxiv_tools.py   search over the index, ar5iv full text with pacing and cache
env/build_index.py   the search index from the pinned arXiv metadata snapshot
```
