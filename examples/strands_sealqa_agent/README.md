# Strands Seal-Hard Agent

An RL-trainable agent for Seal-Hard, the hard split of [SealQA](https://huggingface.co/datasets/vtllms/sealqa)
(254 questions), using Bedrock AgentCore RL Toolkit. The questions are fact-seeking, and web search returns
conflicting, noisy or unhelpful results for them. The agent searches the live web and reads pages; the reward is 1
if the official grader judges the answer correct, else 0.

The agent has two tools:
- `search`, through the [AgentCore Gateway Web Search Tool](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html).
- `visit`, a page fetch whose text a fixed model (gpt-oss-120b on Bedrock) summarizes toward the agent's goal; the
  summarizer is the same for every policy.

## Protocol

SealQA publishes no agent harness: the question is the whole prompt, with no system prompt, as officially. 40 turns,
8K output tokens per turn, 128K context; an episode that runs out of turns or context has no answer. The reward is
the official grader with gpt-oss-120b at temperature 0 as the judge (officially GPT-4o-mini). Leakage control: searches exclude
huggingface.co, and results and pages of the SealQA paper or with the dataset's canary string are dropped. A failing
gateway or summarizer fails the rollout.

Accuracy on 50 tasks (`--split pilot50`, 1 attempt, on AgentCore Runtime):
- Claude Opus 5 (high): 70%
- GPT-6 Astra (high): 74%

## Installation

```bash
cd examples/strands_sealqa_agent

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[eval]"
```

Run the remaining commands from `examples/strands_sealqa_agent`.

## Create the search gateway

Create an AgentCore Gateway with IAM inbound authorization and a Web Search Tool target, in a
[Region that offers it](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html#gateway-target-connector-web-search-tool-availability).
The gateway's service role must trust `bedrock-agentcore.amazonaws.com` and allow `bedrock-agentcore:InvokeGateway`
and `bedrock-agentcore:InvokeWebSearch`
([policy](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-add-target-api-target-config.html#gateway-add-target-api-connector-web-search-service-role)).

```bash
aws bedrock-agentcore-control create-gateway --name sealqa-search --protocol-type MCP \
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
export SEALQA_GATEWAY_URL=https://GATEWAY_ID.gateway.bedrock-agentcore.REGION.amazonaws.com/mcp
python test_local.py --split smoke2 --model_id us.anthropic.claude-opus-5 \
  --bedrock_fields '{"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}'
```

Without `--base_url`, `model_id` is a Bedrock model and `bedrock_fields` its additional request fields (e.g.
`'{"reasoning": {"effort": "high"}}'` for `us.openai.gpt-6-astra`). With `--base_url`, the model is served by an
OpenAI-compatible endpoint (vLLM, or the rollout gateway during training).

## Deploy

```bash
uv tool install bedrock-agentcore-starter-toolkit

agentcore configure --entrypoint rl_app.py --name strands_sealqa_agent \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --disable-memory --non-interactive
agentcore deploy --agent strands_sealqa_agent --env SEALQA_GATEWAY_URL=https://GATEWAY_ID.gateway.bedrock-agentcore.REGION.amazonaws.com/mcp
```

The runtime's execution role needs `bedrock-agentcore:InvokeGateway` on the gateway and Bedrock model access for
the agent, the summarizer and the judge. For training with a self-hosted inference server, add the VPC flags
(`--vpc --subnets ... --security-groups ...`) of a network that reaches it.

## Evaluate

```bash
cp config.example.toml config.toml        # set agent_arn and the model

# All 254 tasks, avg@4
python evaluate.py --split all --repeats 4 --exp_id all_opus5 --max_concurrent 32

# Quick test
python evaluate.py --split smoke2 --exp_id smoke
```

Splits: `all` (254 tasks); `pilot50` and `smoke2`, 50 and 2 tasks drawn with seed 0. Records go to
`$TMPDIR/strands_sealqa_agent-evaluations/<exp_id>.jsonl`; the summary line is the mean reward.

## File Structure

```
rl_app.py                  the Strands agent: tools, budget; @app.entrypoint returns {"reward": ...}
tools.py                   search (the gateway) and visit (fetch and summarize) as Strands tools
reward.py                  the official grader, with a Bedrock judge
prompts/grader_template.txt  the official grader template
models.py                  the typed invocation payload
tasks.py                   Seal-Hard from Hugging Face (vtllms/sealqa, pinned revision) and the splits
evaluate.py                evaluation on AgentCore Runtime
test_local.py              local run, without AgentCore Runtime
```
