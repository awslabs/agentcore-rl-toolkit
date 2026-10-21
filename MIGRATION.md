# Migrating from v0.1.x to v0.2

The v0.2 training integrations use HTTP invocations with `_config` model settings
and a scalar `reward`. An image built with `AgentCoreRLApp` cannot be used directly
with that contract. Update the agent, rebuild its image, and redeploy it before
pointing the trainer at its Runtime ARN.

`AgentCoreRLApp`, `RolloutClient`, `RolloutFuture`, and `AgentCoreS3Session` are
deprecated and retained for existing callers. Their constructors emit
`DeprecationWarning`. `AgentCoreS3Session` still expects the legacy
`payload / llm / group_id` task; selecting it in a current trainer does not
translate the new payload automatically.

## Update the agent

| Before | After |
|---|---|
| `AgentCoreRLApp()` | `AgentCoreRuntimeApp()` |
| `@app.rollout_entrypoint` | `@app.entrypoint` |
| `payload["_rollout"]` | `payload["_config"]` |
| `{"rewards": score}` | `{"reward": float(score)}` |
| S3 result polling | HTTP `start/get` |

Keep the task payload fields and agent logic. Change the model configuration
source and forward the supplied `api_key`: it identifies the rollout whose
tokens the gateway captures.

```python
from agentcore_rl_toolkit import AgentCoreRuntimeApp
from strands import Agent
from strands.models.openai import OpenAIModel

app = AgentCoreRuntimeApp()

@app.entrypoint
def invoke_agent(payload: dict):
    config = payload["_config"]
    model = OpenAIModel(
        client_args={
            "base_url": config["base_url"],
            "api_key": config.get("api_key") or "EMPTY",
        },
        model_id=config["model_id"],
        params=config.get("sampling_params", {}),
    )
    agent = Agent(model=model, tools=tools, system_prompt=system_prompt)
    response = agent(payload["prompt"])
    return {"reward": float(reward_fn(response, payload))}
```

Here `tools`, `system_prompt`, and `reward_fn` are your existing application
components. See the complete
[Math entrypoint](examples/strands_math_agent/rl_app.py).
Use a finite scalar reward; reduce any old per-turn reward list in your agent.

For persistent results, configure an S3 Files mount and point `state_dir` at it;
see the [Runtime storage contract](designs/agentcore_runtime_app.md#filesystem-storage-and-recovery).

## Update training configuration

For verl and slime, replace the old top-level Runtime ARN and result bucket with:

```yaml
rollout_session_backend:
  backend: agentcore_rl_toolkit.rollout_session.agentcore_http_session.AgentCoreHttpSession
  agentcore_runtime_arn: arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/AGENT_ID
```

Tinker/SkyRL and SageMaker already use `AgentCoreHttpSession`; retain their
backend-specific ARN configuration and deploy the updated agent. All these
trainers inject `_config` into the task payload and capture tokens at the
gateway. Do not put `_config` in dataset rows.

## Update evaluation

Use `AgentCoreHttpSession` to stop each session on success, failure, or timeout:

```python
import asyncio
import uuid

from agentcore_rl_toolkit import AgentCoreHttpClient
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession

async def evaluate_one(runtime_arn, payload, base_url, model_id):
    sid = str(uuid.uuid4())
    task = {
        **payload,
        "_config": {"base_url": base_url, "model_id": model_id, "api_key": sid},
    }
    async with AgentCoreHttpClient(runtime_arn) as client:
        async with AgentCoreHttpSession.with_client(sid, client) as session:
            async with asyncio.timeout(900):
                dump = await session.run(task)
            if dump.exception:
                raise RuntimeError(dump.exception)
            return dump.task_output
```

For batching and JSONL output, see the
[Math](examples/strands_math_agent/evaluate.py),
[MigrationBench](examples/strands_migration_agent/evaluate.py),
[OfficeBench](examples/strands_officebench_agent/evaluate.py), and
[AppWorld](examples/strands_appworld_agent/evaluate.py) evaluators.

Remove the evaluator's `--s3_bucket` / `--s3_output_bucket` option and choose
`--output_dir` for JSONL results. Also remove AppWorld's `--mode` and
`--max_pool_connections` options.
