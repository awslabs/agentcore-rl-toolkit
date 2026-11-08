# Strands HERB Agent (evaluation)

A [HERB](https://github.com/SalesforceAIResearch/HERB) agent using Bedrock AgentCore RL Toolkit. HERB is deep search
over a software company's internal records: Slack messages, documents, meeting transcripts and chats, GitHub PRs,
URLs, and employee and customer directories. The agent answers a question from the records, or recognizes that the
records do not answer it.

**License: evaluation only.** HERB's code and data are CC BY-NC 4.0, and the data was generated with GPT-4o and must
not be used to develop models that compete with OpenAI (HERB's README). This example contains no HERB code or data: it
downloads them from HERB's repository at a pinned commit, and it provides no training recipe.

The agent has nine tools, HERB's own: `retrieve_all` (hybrid BM25 and dense retrieval over the unstructured records)
and eight structured lookups of employees, roles, customers, PRs and URLs.

## Protocol

HERB's agent setup: its system prompt and tools, 40 turns, 8K output tokens per turn, 128K context, tool output cut
at 200,000 characters. There is no forced answer and no fallback answer: an episode that runs out of turns or context
has no answer. The reward is HERB's scoring:

| Question type | Reward |
|---|---|
| content (a paragraph) | the judge's 0-100 rating against the reference, / 100 |
| person, customer, artifact (PR, URL) | set F1 of the items the judge extracts from the answer |
| unanswerable | 1 if the answer does not answer the question, else 0; no answer scores 0 |

Differences from HERB:
- The judge and the structured tools' helper model are Claude Haiku 5.5 instead of GPT-4o.
- The dense embeddings are Cohere Embed v4 on Bedrock instead of OpenAI text-embedding-3-large.
- The unanswerable judge has one added rule: an answer concluding that nothing matching the question exists does not
  answer it.
- `role_to_id_search` formats its prompt with the key the template expects (HERB's raises on every call).

Scores on 50 questions (`--split pilot50`: 25 answerable, 25 unanswerable; 1 attempt, on AgentCore Runtime), x 100:

| Model | content | person | customer | artifact | answerable macro | unanswerable | overall |
|---|---|---|---|---|---|---|---|
| Claude Opus 5 (high) | 47.6 | 92.3 | 80.0 | 100.0 | 80.0 | 20.0 | 50.0 |
| GPT-6 Astra (high) | 85.0 | 93.7 | 80.0 | 98.1 | 89.2 | 28.0 | 58.6 |

The overall score is the mean of the answerable macro average and the unanswerable score. On unanswerable questions
both models mostly keep searching until the context runs out.

## Installation

```bash
cd examples/strands_herb_agent

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
python scripts/fetch_herb.py herb          # HERB's code and data (pinned, sha256-checked)
python scripts/build_index.py herb index   # the retrieval index (Cohere Embed v4 on Bedrock)
```

Run the remaining commands from `examples/strands_herb_agent`, with AWS credentials for Bedrock (the model under
test, Cohere Embed v4 and Claude Haiku 5.5).

## Run Locally

```bash
python test_local.py --split smoke2 --model_id us.anthropic.claude-opus-5 \
  --bedrock_fields '{"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}'
```

Without `--base_url`, `model_id` is a Bedrock model and `bedrock_fields` its additional request fields (e.g.
`'{"reasoning": {"effort": "high"}}'` for `us.openai.gpt-6-astra`). With `--base_url`, the model is served by an
OpenAI-compatible endpoint (vLLM).

## Deploy

The image downloads HERB and includes the index built above.

```bash
uv tool install bedrock-agentcore-starter-toolkit

agentcore configure --entrypoint rl_app.py --name strands_herb_agent \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --disable-memory --non-interactive
agentcore deploy
```

The runtime's execution role needs Bedrock model access.

## Evaluate

```bash
cp config.example.toml config.toml        # set agent_arn and the model

# All 1,514 questions, avg@4
python evaluate.py --split all --repeats 4 --exp_id all_opus5 --max_concurrent 32

# Quick test
python evaluate.py --split smoke2 --exp_id smoke
```

Splits: `answerable` (815 questions), `unanswerable` (699), `all`; `pilot50`, the first 25 of 100 answerable and of
50 unanswerable questions drawn with seed 0; `smoke2`, one of each. Records go to
`$TMPDIR/strands_herb_agent-evaluations/<exp_id>.jsonl`; the summary is HERB's: per-type scores, the answerable macro
average, the unanswerable score and the overall score.

## File Structure

```
rl_app.py              the Strands agent: system prompt, tools, budget; @app.entrypoint returns {"reward": ...}
tools.py               retrieve_all and HERB's structured tools as Strands tools
retrieval.py           HERB's hybrid retrieval over the index (Cohere Embed v4 on Bedrock)
upstream.py            HERB's code, loaded from the download at run time
helper_llm.py          Claude Haiku 5.5 on Bedrock in place of HERB's GPT-4o helper
reward.py              HERB's scoring
models.py              the typed invocation payload
tasks.py               HERB's questions and the splits
scripts/fetch_herb.py  HERB's code and data from its repository (pinned, sha256-checked)
scripts/build_index.py the retrieval index
evaluate.py            evaluation on AgentCore Runtime
test_local.py          local run, without AgentCore Runtime
```
