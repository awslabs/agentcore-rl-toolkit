# Math agents for Runtime integration tests

Deploy either entrypoint to test the HTTP client and rollout session with a
Strands calculator agent:

| Entrypoint | App |
| --- | --- |
| `runtime_app.py` | `AgentCoreRuntimeApp` |
| `plain_app.py` | `BedrockAgentCoreApp` |

Both accept `{"prompt": "What is 7 + 5?", "answer": "12"}` with `_config`
containing `base_url`, `model_id`, and `api_key` for the rollout gateway. They
return `{"reward": 0.0}` or `{"reward": 1.0}` by comparing the final `####` answer
with the supplied answer.

## Deploy

Use `uv` and the Python `bedrock-agentcore-starter-toolkit` CLI. Set `AWS_REGION`,
`EXECUTION_ROLE_ARN`, `SUBNET_IDS`, and `SECURITY_GROUP_IDS`; the VPC must allow
the Runtime to reach the gateway. Run from the repository root:

```bash
agent_name=runtime_math_test
entrypoint=runtime_app.py
stage=$(mktemp -d "${TMPDIR:?}/runtime-math.XXXXXX")
cp "tests/runtime/agents/math/$entrypoint" "$stage/agent.py"
cp tests/runtime/agents/math/Dockerfile tests/runtime/agents/math/requirements.txt "$stage/"
uv build --wheel --out-dir "$stage"
cd "$stage"

agentcore configure \
  --entrypoint agent.py --name "$agent_name" \
  --requirements-file requirements.txt --deployment-type container --protocol HTTP \
  --execution-role "$EXECUTION_ROLE_ARN" --region "$AWS_REGION" \
  --vpc --subnets "$SUBNET_IDS" --security-groups "$SECURITY_GROUP_IDS" \
  --disable-memory --non-interactive
agentcore deploy
```

For the ordinary app, set `entrypoint=plain_app.py` and `agent_name=plain_math_test`.
The CLI saves deployment configuration in the staging directory.

Use the resulting Runtime ARN with the
[Tinker training recipe](../../../../src/agentcore_rl_toolkit/backends/tinker_api/README.md#run-the-math-example).
