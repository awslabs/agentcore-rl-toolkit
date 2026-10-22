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

## Container inputs

Place the selected entrypoint as `agent.py`, this directory's `Dockerfile` and
`requirements.txt`, and a toolkit wheel built from the checkout in one build
directory.

Use the deployed Runtime ARN with the
[Tinker training recipe](../../../../src/agentcore_rl_toolkit/backends/tinker_api/README.md#run-the-math-example).
