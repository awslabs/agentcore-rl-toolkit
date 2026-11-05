# Strands HSCodeComp Agent

An RL-trainable [HSCodeComp](https://github.com/AIDC-AI/Marco-Search-Agent/tree/main/Marco-DeepResearch-Family/HSCodeComp)
agent using Bedrock AgentCore RL Toolkit. Given an e-commerce product listing (title, attributes, price, category
path), the agent classifies the product into its 10-digit US HTS code. The reward is 1 if the code matches the gold
code, else 0.

The agent has five tools:
- `web_search`, through the [AgentCore Gateway Web Search Tool](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html).
- `web_extract`, an HTTP fetch with text extraction (HTML, PDF, JSON). U.S. Customs rulings come from the CROSS API.
- The three official HTS tools (`hts_tools.py`), over the HTS 2025 Revision 15 tables baked into the image.

## Protocol

The official setup: the official system prompt with the General Rules of Interpretation (`prompts/`, translated
from Chinese), 40 turns, 8K output tokens per turn, 128K context, tool output cut at 20,000 characters. There is no
forced answer: an episode that runs out of turns or context has no answer. The reward is the official rule
(`reward.py`): the first `\boxed{...}` with at least 10 characters, compared digit by digit. Leakage control:
searches exclude huggingface.co, and results under github.com/AIDC-AI are dropped. An unavailable search gateway
fails the rollout.

10-digit accuracy on 100 tasks (`--split pilot100`, 1 attempt, on AgentCore Runtime):
- Claude Opus 5 (high): TBD
- GPT-6 Astra (high): TBD

## Installation

```bash
cd examples/strands_hscodecomp_agent

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[eval]"
python scripts/download_hts.py data        # the HTS tables, for local runs
```

Run the remaining commands from `examples/strands_hscodecomp_agent`.

## Create the search gateway

Create an AgentCore Gateway with IAM inbound authorization and a Web Search Tool target, in a
[Region that offers it](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html#gateway-target-connector-web-search-tool-availability).
The gateway's service role must trust `bedrock-agentcore.amazonaws.com` and allow `bedrock-agentcore:InvokeGateway`
and `bedrock-agentcore:InvokeWebSearch`
([policy](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-add-target-api-target-config.html#gateway-add-target-api-connector-web-search-service-role)).

```bash
aws bedrock-agentcore-control create-gateway --name hscode-search --protocol-type MCP \
  --authorizer-type AWS_IAM --role-arn arn:aws:iam::ACCOUNT_ID:role/GATEWAY_SERVICE_ROLE    # prints the gatewayUrl
aws bedrock-agentcore-control create-gateway-target --gateway-identifier GATEWAY_ID --name web-search \
  --target-configuration '{"mcp": {"connector": {"source": {"connectorId": "web-search", "version": "1.2.0"},
    "configurations": [{"name": "WebSearch", "parameterValues": {"domainFilter": {"exclude": ["huggingface.co"]}}}]}}}' \
  --credential-provider-configurations '[{"credentialProviderType": "GATEWAY_IAM_ROLE"}]'
```

Callers need `bedrock-agentcore:InvokeGateway` on the gateway. The Web Search Tool's
[acceptable use](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html#gateway-target-connector-web-search-tool-acceptable-use)
applies to its results.

## Run Locally

With AWS credentials for Bedrock and the gateway:

```bash
export HSCODECOMP_GATEWAY_URL=https://GATEWAY_ID.gateway.bedrock-agentcore.REGION.amazonaws.com/mcp HTS_DATA_DIR=data
python test_local.py --split smoke2 --model_id us.anthropic.claude-opus-5 \
  --bedrock_fields '{"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}'
```

Without `--base_url`, `model_id` is a Bedrock model and `bedrock_fields` its additional request fields (e.g.
`'{"reasoning": {"effort": "high"}}'` for `us.openai.gpt-6-astra`). With `--base_url`, the model is served by an
OpenAI-compatible endpoint (vLLM, or the rollout gateway during training).

## Deploy

```bash
uv tool install bedrock-agentcore-starter-toolkit

agentcore configure --entrypoint rl_app.py --name strands_hscodecomp_agent \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --disable-memory --non-interactive
agentcore deploy --agent strands_hscodecomp_agent --env HSCODECOMP_GATEWAY_URL=https://GATEWAY_ID.gateway.bedrock-agentcore.REGION.amazonaws.com/mcp
```

The runtime's execution role needs `bedrock-agentcore:InvokeGateway` on the gateway and Bedrock model access for
Bedrock models. For training with a self-hosted inference server, add the VPC flags (`--vpc --subnets ...
--security-groups ...`) of a network that reaches it.

## Evaluate

```bash
cp config.example.toml config.toml        # set agent_arn and the model

# All 632 tasks, avg@4
python evaluate.py --split all --repeats 4 --exp_id all_opus5 --max_concurrent 32

# Quick test
python evaluate.py --split smoke2 --exp_id smoke
```

Splits: `all` (632 tasks); `pilot100`, 100 tasks drawn with seed 0; `smoke2`, the first two of them. Records go to
`$TMPDIR/strands_hscodecomp_agent-evaluations/<exp_id>.jsonl`; the summary line is the mean reward.

## File Structure

```
rl_app.py              the Strands agent: system prompt, tools, budget; @app.entrypoint returns {"reward": ...}
tools.py               web_search (the gateway), web_extract, and the HTS tools as Strands tools
hts_tools.py           the official HTS tools
reward.py              the official scoring
prompts/               the official system prompt and GRI rules, translated from Chinese
models.py              the typed invocation payload
tasks.py               HSCodeComp from Hugging Face (AIDC-AI/HSCodeComp, pinned revision) and the splits
scripts/download_hts.py  the HTS tables from the official repository (pinned, sha256-checked)
evaluate.py            evaluation on AgentCore Runtime
test_local.py          local run, without AgentCore Runtime
```
