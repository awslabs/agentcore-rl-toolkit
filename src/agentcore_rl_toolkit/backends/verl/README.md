# verl backend (rollout gateway)

Train agents deployed on Bedrock AgentCore Runtime (ACR) with [verl](https://github.com/volcengine/verl),
using the in-repo [rollout gateway](../../rollout_gateway/) for token-level trajectory
capture.

The integration keeps verl stock: it plugs into
`python -m verl.trainer.main_ppo` as a custom v1 agent loop, captures tokens with
the in-repo gateway, and leaves verl's replay buffer, filtering, checkpointing,
validation, and rollout-correction paths unchanged.

## How it works

```
verl main_ppo (v1) ──> AgentLoopWorker ──> AgentCoreAgentLoop.run()
                                             │  1. create gateway session (sid = uuid4)
                                             │  2. RolloutClient.invoke_async(session_id=sid,
                                             │       base_url=<gateway>/v1)  ────────────> ACR agent
   RolloutGateway (thread, this process) <───┼─────  agent's OpenAI/Anthropic calls
     └─ VerlSamplingBackend ──> verl LLMServerClient (token-in/token-out, sticky by sid)
                                             │  3. await S3 result (completion signal + optional reward)
                                             │  4. finish_session -> TraceRecords -> AgentLoopOutputs
```

- The **capture session key travels in the payload**: the loop generates one sid per
  rollout, passes it as both the ACR `runtimeSessionId` and `_rollout.api_key`; the
  agent container puts `payload["_rollout"]["api_key"]` in its LLM client's api-key
  slot; the gateway reads it back from the Bearer/`X-Api-Key` slot.
- The advertised `base_url` follows the OpenAI-SDK convention and **includes `/v1`**
  (the SDK appends `/chat/completions`) — pass it to an OpenAI-compatible client
  verbatim. TODO: it is not directly usable by Anthropic-SDK agents (that SDK
  appends `/v1/messages` and does not normalize an existing `/v1`, yielding
  `/v1/v1/messages`); how to serve both SDK families cleanly is unresolved.
- One gateway per AgentLoopWorker process, serving on an auto-assigned port.
  **ACR containers must be able to reach the trainer's CPU nodes on that port.**
  Use `gateway_public_host` if the Ray node IP is not what ACR can reach.
- A session's trajectory tree can fork (sub-agents, context compaction); every leaf
  becomes its own training row (`run()` returns `list[AgentLoopOutput]`) — hence the
  hard `trainer.use_v1=true` requirement.
- Failed rollouts (timeout, ACR error, agent error) never raise. They yield a
  single-pad-token fallback row: `response_mask=[1]` (verl's rollout-correction
  helper requires ≥1 valid response token per row — an all-zero mask is rejected),
  logprob 0, reward 0. See the TODO below for its current gradient semantics.

The agent must forward the trainer-supplied key when it constructs its model client:

```python
rollout_config = payload["_rollout"]
api_key = rollout_config.get("api_key") or "EMPTY"
model = OpenAIModel(
    client_args={"api_key": api_key, "base_url": rollout_config["base_url"]},
    model_id=rollout_config["model_id"],
    params=rollout_config.get("sampling_params", {}),
)
```

The `"EMPTY"` fallback supports local evaluation and unauthenticated inference
endpoints.

## Install

verl is pinned to uni-agent's blessed submodule commit `78bba31d` via
`[tool.uv.sources]`. From a checkout of this repo:

```bash
uv sync --extra verl
```

The pinned stack uses CUDA 13 wheels and requires driver >= 580.65.06 and compute
capability >= 7.5. `flash-attn` comes from Astral's prebuilt GPU wheel index, so a
local CUDA toolkit is not required for installation.

### Megatron engine

Megatron requires Python 3.12:

```bash
uv sync --extra verl --group verl-megatron
```

- Use NVIDIA Megatron-Bridge with `megatron.use_mbridge=True` and
  `megatron.vanilla_mbridge=False`.
- LoRA recipes without NVIDIA Apex must set
  `++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False`.

## Dataset contract: the `payload` column

A training row carries the agent's exact ACR invoke payload in a single **`payload`**
column, authored against the agent's own API — the trainer forwards it verbatim and
the agent never learns any trainer/dataset conventions:

```python
# one row, e.g. for examples/strands_math_agent (rl_app.py reads prompt + answer)
{"payload": {"prompt": "Natalia sold clips to...", "answer": "72"}}
```

verl's dataloader machinery needs a chat-format `prompt` column internally;
**`PayloadDataset`** synthesizes it at load time from `payload["prompt"]`, so
dataset authors never write that ceremony column. If your payloads name the
prompt field differently (say `input`), point the synthesis at it with
`+data.payload_prompt_field=input` — a mismatched field fails loudly at dataset
load, never silently:

```yaml
data:
  custom_cls:
    path: pkg://agentcore_rl_toolkit.backends.verl.dataset
    name: PayloadDataset
```

Rows that already have an explicit chat-format `prompt` column are left untouched.

The `payload` column is the **single** dataset contract: rows without it fail
loudly at the first rollout. Payload values must contain only JSON-serializable
types. Plain row fields are never forwarded because the row namespace is shared
with verl's own plumbing fields.

Dataset fields alongside `payload` are reserved for **dispatch metadata** (routing,
not agent input). In particular, `agent` is the designated field for routing rows to
different ACR endpoints if multi-endpoint training lands — don't put dispatch
concerns inside `payload`.

## Rewards

The reward is **built into the agent** (`reward_mode="built_in"`, the only supported
mode): the agent returns `{"rewards": ...}` in its session result (scalar or list;
last element wins). The score becomes `rm_scores` directly and verl skips reward
computation.

- Failed rollouts (timeout, ACR error, non-200 `status_code`) score 0.0.
- A healthy rollout that returns no reward is a contract violation: warned, scored 0.0.
- A non-numeric `rewards` value raises because it indicates a recurring
  agent-side contract error. verl contains the exception to the affected prompt
  group, so training continues without rows from that group.

**Trainer-side rewards (`reward_mode="separate"`) are not supported yet** and are
rejected at startup because verl's v1 reward managers require dataset columns that
the payload-first contract does not provide.

## Examples

- [GSM8K with FSDP full fine-tuning](examples/math_agent/)
- [MigrationBench with Megatron and LoRA](examples/migration_agent/)
- [Public setup guide](../../../../docs/site/src/content/docs/guides/verl-backend-setup.md)

Each recipe separates verl configuration in its shell script from
`AgentCoreAgentLoop` kwargs in `agentcore_agent.yaml`. Shell-script arguments accept
Hydra overrides, but loop kwargs are loaded worker-side and are not CLI-addressable;
edit the YAML or use `${oc.env:...}` interpolation.

## Token budgets

The integration keeps four limits separate:

- `rollout.max_model_len` is the inference engine's model-context capacity. It
  must be set explicitly; stock verl validates it against the model's Hugging
  Face `max_position_embeddings`.
- `prompt_length` is verl's fixed storage width for the leading context of each
  emitted training row; it does not cap the prompts the gateway sends to the
  inference engine. A row may begin at the first model turn or at a later
  trajectory fork, so its leading context can include a long accumulated
  multi-turn prompt.
- `response_length` is verl's storage width for everything after the leading
  prompt region and the gateway's cumulative trajectory budget. It must not
  exceed `max_model_len`.
- `max_tokens_per_turn` is a required `agentcore_agent.yaml` setting. It becomes
  the gateway's default `max_new_tokens` for each model call; a smaller request
  limit and the remaining model context can clamp it further.

To let variable-length prompts use the full model window, set
`response_length = max_model_len` and enable
`actor_rollout_ref.model.use_remove_padding=true`. This gives verl a nominal
`prompt_length + response_length` padded width, but the gateway limits valid
tokens to `max_model_len`. If both lengths equal `max_model_len`, the nominal
row width is therefore `2 * max_model_len`; remove-padding avoids most model
compute on padding, but the wider fixed-width staging tensors still increase
memory and transfer overhead. Set `prompt_length` high enough for the leading
contexts expected in emitted rows (or to `max_model_len` to rule out overflow).
If a leading context does exceed `prompt_length`, the adapter preserves its
overflow at the front of the response region with loss mask and rollout logprob
zero. Training remains correct, but verl's stock length metrics count those
overflow tokens as part of the response region, so overflow should be a
fallback rather than the normal configuration.

Trace-less rollout failures raise from the agent loop and produce no synthetic
training row. Other sessions for the same prompt remain trainable.

## Troubleshooting

- **Every rollout fails, warning about "static session 'EMPTY'"**: the deployed agent
  image predates the session-key contract and sends a fixed api key — its turns
  accumulate under one shared session while each rollout's real session drains empty.
  Rebuild/redeploy the agent image with the contract above.
- **Agent-side 404s on every rollout**: an agent constructing its own URL paths
  instead of using an OpenAI/Anthropic SDK may miss the `/v1` convention (see above).
- Stop-string text trimmed by verl's rollout servers is excluded from trained ids
  (same behavior as verl's own agent loops).
- Sub-agents that reuse the same session id fork within one tree and are captured; a
  sub-agent given a *different* session id becomes a separate tree and is not joined
  to the episode (cross-session grouping is future work).
