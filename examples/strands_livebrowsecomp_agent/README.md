# Strands LiveBrowseComp Agent

An RL-trainable agent for [LiveBrowseComp](https://huggingface.co/datasets/Forival/LiveBrowseComp) (arXiv
2605.28721, 335 questions), using Bedrock AgentCore RL Toolkit. Each question combines several obscure clues that
identify one entity; the answer is short. The agent searches the live web and reads pages; the reward is 1 if the
official judge finds the answer correct, else 0.

The agent has two tools:
- `search`, through the [AgentCore Gateway Web Search Tool](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html).
- `visit`, a page fetch whose text a fixed model (gpt-oss-120b on Bedrock) summarizes toward the agent's goal; the
  summarizer is the same for every policy.

## Protocol

The paper's protocol (Appendix C): its default system prompt, 250 steps, 8K output tokens per turn, 256K context.
When the steps or the context run out, the paper's force-answer prompt gets one more turn. The reward is the
official judge: the BrowseComp grader template with gpt-oss-120b, as in the paper. Leakage control: searches exclude huggingface.co and the benchmark's site, and results and
pages of its repository and paper, or with its canary string, are dropped. A failing gateway or summarizer fails the
rollout.

Accuracy on 50 tasks (`--split pilot50`, 1 attempt, on AgentCore Runtime):
- Claude Opus 5 (high): 36%
- GPT-6 Astra (high): 48%

## Installation

```bash
cd examples/strands_livebrowsecomp_agent

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[eval]"
```

Run the remaining commands from `examples/strands_livebrowsecomp_agent`.

## Create the search gateway

Create an AgentCore Gateway with IAM inbound authorization and a Web Search Tool target, in a
[Region that offers it](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html#gateway-target-connector-web-search-tool-availability).
The gateway's service role must trust `bedrock-agentcore.amazonaws.com` and allow `bedrock-agentcore:InvokeGateway`
and `bedrock-agentcore:InvokeWebSearch`
([policy](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-add-target-api-target-config.html#gateway-add-target-api-connector-web-search-service-role)).

```bash
aws bedrock-agentcore-control create-gateway --name livebrowsecomp-search --protocol-type MCP \
  --authorizer-type AWS_IAM --role-arn arn:aws:iam::ACCOUNT_ID:role/GATEWAY_SERVICE_ROLE    # prints the gatewayUrl
aws bedrock-agentcore-control create-gateway-target --gateway-identifier GATEWAY_ID --name web-search \
  --target-configuration '{"mcp": {"connector": {"source": {"connectorId": "web-search", "version": "1.2.0"},
    "configurations": [{"name": "WebSearch", "parameterValues": {"domainFilter": {"exclude": ["huggingface.co"]}}}]}}}' \
  --credential-provider-configurations '[{"credentialProviderType": "GATEWAY_IAM_ROLE"}]'
```

Callers need `bedrock-agentcore:InvokeGateway` on the gateway. The Web Search Tool's
[acceptable use](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html#gateway-target-connector-web-search-tool-acceptable-use)
applies to its results.

Wikipedia is fetched live. Wikipedia often refuses AgentCore Runtime's addresses (HTTP 403); the agent then sees
the error, as for any page it cannot open.

## Run Locally

With AWS credentials for Bedrock (the agent, the summarizer and the judge) and the gateway:

```bash
export LIVEBROWSECOMP_GATEWAY_URL=https://GATEWAY_ID.gateway.bedrock-agentcore.REGION.amazonaws.com/mcp
python test_local.py --split smoke2 --model_id us.anthropic.claude-opus-5 \
  --bedrock_fields '{"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}'
```

Without `--base_url`, `model_id` is a Bedrock model and `bedrock_fields` its additional request fields (e.g.
`'{"reasoning": {"effort": "high"}}'` for `us.openai.gpt-6-astra`). With `--base_url`, the model is served by an
OpenAI-compatible endpoint (vLLM, or the rollout gateway during training).

## Deploy

```bash
uv tool install bedrock-agentcore-starter-toolkit

agentcore configure --entrypoint rl_app.py --name strands_livebrowsecomp_agent \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --disable-memory --non-interactive
agentcore deploy --agent strands_livebrowsecomp_agent --env LIVEBROWSECOMP_GATEWAY_URL=https://GATEWAY_ID.gateway.bedrock-agentcore.REGION.amazonaws.com/mcp
```

The runtime's execution role needs `bedrock-agentcore:InvokeGateway` on the gateway and Bedrock model access for
the agent, the summarizer and the judge. For training with a self-hosted inference server, add the VPC flags
(`--vpc --subnets ... --security-groups ...`) of a network that reaches it.

## Evaluate

```bash
cp config.example.toml config.toml        # set agent_arn and the model

# All 335 tasks, avg@4
python evaluate.py --split all --repeats 4 --exp_id all_opus5 --max_concurrent 32

# Quick test
python evaluate.py --split smoke2 --exp_id smoke
```

Splits: `all` (335 tasks); `pilot50` and `smoke2`, 50 and 2 tasks drawn with seed 0. Records go to
`$TMPDIR/strands_livebrowsecomp_agent-evaluations/<exp_id>.jsonl`; the summary line is the mean reward.

## File Structure

```
rl_app.py                  the Strands agent: tools, budget, forced answer; @app.entrypoint returns {"reward": ...}
tools.py                   search (the gateway) and visit (fetch and summarize) as Strands tools
reward.py                  the official judge, on Bedrock
prompts/                   the paper's system prompt, force-answer prompt and judge template
models.py                  the typed invocation payload
tasks.py                   LiveBrowseComp from Hugging Face (pinned revision), decrypted, and the splits
evaluate.py                evaluation on AgentCore Runtime
test_local.py              local run, without AgentCore Runtime
```
