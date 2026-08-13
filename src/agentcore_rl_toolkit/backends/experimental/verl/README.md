# Experimental verl backend (rollout gateway)

Train agents deployed on Bedrock AgentCore Runtime (ACR) with [verl](https://github.com/volcengine/verl),
using the in-repo [rollout gateway](../../../rollout_gateway/) for token-level trajectory
capture. Validated end to end: GRPO on GSM8K with a live ACR math agent reaches
**~0.93 validation reward in one epoch** (Qwen3-4B-Instruct, 8 GPUs, see
`examples/math_agent/fsdp_fft_sync_grpo.sh`).

This backend replaces the legacy `backends/verl` integration (rllm-model-gateway +
`AgentCoreTrainer(RayPPOTrainer)` subclass) with a design that keeps verl stock:

| | Legacy `backends/verl` | This backend |
|---|---|---|
| Entrypoint | custom `main.py` + `AgentCoreTrainer(RayPPOTrainer)` | stock `python -m verl.trainer.main_ppo` |
| Trainer | legacy RayPPOTrainer (deprecated upstream) | v1 trainer (`trainer.use_v1=true`, required) |
| Token capture | external `rllm-model-gateway` subprocess, per-session URLs | in-repo gateway, in-process, fixed base_url + session id in the api-key slot |
| Tokens | scraped from chat responses | gateway-rendered, token-in/token-out via verl's `LLMServerClient` |
| Multi-turn merging | trainer-side segment merge | inside the gateway (`TrajectoryManager`) |
| Rewards | fused with S3 result extraction | inline `rewards` OR verl-native `custom_reward_function` |
| verl features | custom fit loop (GRPO/REMAX only) | native: replay buffer, DAPO filtering, async modes, checkpointing, validation, reward loop, rollout correction (TIS) |

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
  slot; the gateway reads it back from the Bearer/`X-Api-Key` slot. (Because sid ==
  runtimeSessionId, older agent images that still send `context.session_id` produce
  the same value and keep working.)
- The advertised `base_url` follows the OpenAI-SDK convention and **includes `/v1`**
  (the SDK appends `/chat/completions`) — pass it to an OpenAI-compatible client
  verbatim. TODO: it is not directly usable by Anthropic-SDK agents (that SDK
  appends `/v1/messages` and does not normalize an existing `/v1`, yielding
  `/v1/v1/messages`); how to serve both SDK families cleanly is unresolved.
- One gateway per AgentLoopWorker process, serving on an auto-assigned port.
  **ACR containers must be able to reach the trainer's CPU nodes on that port** (same
  networking requirement as the legacy backend). Use `gateway_public_host` if the Ray
  node IP is not what ACR can reach.
- A session's trajectory tree can fork (sub-agents, context compaction); every leaf
  becomes its own training row (`run()` returns `list[AgentLoopOutput]`) — hence the
  hard `trainer.use_v1=true` requirement.
- Failed rollouts (timeout, ACR error, agent error) never raise. They yield an inert
  single-pad-token row: `response_mask=[1]` (verl's rollout-correction helper requires
  ≥1 valid response token per row — an all-zero mask is rejected), logprob 0, reward 0.

## Install

verl is pinned to uni-agent's blessed submodule commit `78bba31d` via
`[tool.uv.sources]`. From a checkout of this repo:

```bash
uv sync --extra verl-experimental
```

Notes from the field:
- The extra adds `transferqueue` and `orjson` explicitly: verl declares its runtime
  deps in `requirements.txt` (which its Docker images install), not in setup.py, so a
  git install of verl alone misses them.
- `flash-attn` installs as a prebuilt wheel from Astral's GPU index
  (`https://wheels.astral.sh/simple/cu130/`, wired in `[tool.uv.sources]`), so there is no
  hour-long local build and no local CUDA toolkit (`nvcc`) is needed.
- The stack is cu130 throughout, matching verl's reference stack at the pinned commit:
  needs driver >= 580.65.06 and compute capability >= 7.5 (CUDA 13 dropped Pascal and
  Volta, so V100 and older are out).

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
    path: pkg://agentcore_rl_toolkit.backends.experimental.verl.dataset
    name: PayloadDataset
```

Rows that already have an explicit chat-format `prompt` column are left untouched.

The `payload` column is the **single** dataset contract: rows without it fail
loudly at the first rollout. Payload values must be plain JSON types (what the
stock parquet read path produces); a custom dataset class that puts
non-serializable values (e.g. numpy arrays) in the payload fails loudly when the
invoke body is serialized. Plain row fields are deliberately never forwarded —
the row namespace is shared with verl's own plumbing fields, and verl-shaped
column values are not agent-shaped (a verl-compatible `prompt` column is always a
chat-format message list, while agents typically expect a plain string).
Converting an existing verl dataset is a few lines:

```python
import pandas as pd

df = pd.read_parquet("existing_verl_dataset.parquet")
df["payload"] = df.apply(
    lambda r: {"prompt": r["prompt"][0]["content"],   # chat list -> the agent's string
               "answer": r["reward_model"]["ground_truth"]},
    axis=1,
)
df[["prompt", "payload"]].to_parquet("agentcore_dataset.parquet", index=False)
```

Sibling fields of `payload` are reserved for **dispatch metadata** (routing, not agent
input). In particular, `agent` is the designated field for routing rows to different
ACR endpoints if multi-endpoint training lands — don't put dispatch concerns inside
`payload`.

## Rewards

The reward is **built into the agent** (`reward_mode="built_in"`, the only supported
mode): the agent returns `{"rewards": ...}` in its session result (scalar or list;
last element wins). The score becomes `rm_scores` directly and verl skips reward
computation.

- Failed rollouts (timeout, ACR error, non-200 `status_code`) score 0.0.
- A healthy rollout that returns no reward is a contract violation: warned, scored 0.0.
- A non-numeric `rewards` value **raises**. That indicates broken agent-side reward
  code, which would be broken on every rollout, and silently scoring 0.0 would
  flatten every GRPO group's advantages — a run that trains on nothing while looking
  healthy. verl contains the raise per prompt group (logged with a traceback, group
  tagged `failure`, replay buffer still samples), so training continues; the affected
  rollouts contribute no rows.

**Trainer-side rewards (`reward_mode="separate"`) are not supported yet** and are
rejected at startup. Handing scoring to verl's reward loop requires `data_source` and
`reward_model.ground_truth` dataset columns — every v1 reward manager indexes them
unguarded, *before* merging the session result into `extra_info` — so a payload-only
row raises `KeyError` before the reward function ever runs. Synthesizing a shape for
those columns without a concrete reward function to validate against would be a
guess, so the mode stays closed until there is one.

## Run the GSM8K example

```bash
cd examples/math_agent
export AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/...   # deployed strands_math_agent rl_app
export ACR_S3_BUCKET=your-bucket
python preprocess_gsm8k.py --output-dir gsm8k
./fsdp_fft_sync_grpo.sh                                       # console logging
./fsdp_fft_sync_grpo.sh trainer.logger='["console","wandb"]'  # with wandb
```

The example owns two config files, split by which config tree they feed:
`fsdp_fft_sync_grpo.sh` sets verl-tree knobs (overridable from the CLI via its
trailing `"$@"`), and `agentcore_agent.yaml` — the per-run agent-loop config —
carries the loop's kwargs (ARN, bucket, `max_tokens_per_turn`,
`max_rollout_time`, ...).
Loop kwargs are NOT CLI-addressable (verl loads the YAML worker-side, outside
hydra's override grammar): edit the YAML, or use `${oc.env:...}` interpolation
for values that should vary per launch. Except for required
`max_tokens_per_turn`, unset kwargs fall back to `AgentCoreAgentLoop.__init__`
defaults.

## Token budgets

The integration keeps three limits separate:

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

The script name encodes the configuration axes: FSDP engine, full fine-tune, sync
trainer mode, GRPO. Its defaults are the validated stable configuration; three
settings act as the trust region and matter for stability (an early run with lr=2e-5
and no KL/TIS collapsed to reward 0 around step 60 — see the script header):
`lr=5e-6`, `use_kl_loss=true`, and truncated importance sampling
(`algorithm.rollout_correction.rollout_is=token`), which corrects the vLLM-sampler vs
FSDP-trainer probability gap using the gateway-captured rollout logprobs.

Checkpoints land under `checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}` (verl's own
layout), and the script sets `trainer.resume_mode=disable` so no run auto-resumes.
If you re-enable resume, the per-experiment dir keeps it scoped: verl's `auto` mode
only ever searches `trainer.default_local_dir`.

**Wall-clock note:** validation dominates. Each val prompt is a live ACR agent
round-trip (the GSM8K test split is 1319 prompts ≈ 12 min/eval), versus ~100 s per
training step. Budget `trainer.test_freq` and `data.val_batch_size` accordingly.

## Agent-side contract

The agent app sets its api key from the trainer-supplied `_rollout` config (see
`examples/strands_math_agent/rl_app.py`):

```python
@app.rollout_entrypoint
def invoke_agent(payload: dict, context):
    api_key = payload["_rollout"].get("api_key") or "EMPTY"   # trajectory-capture session key
    model = OpenAIModel(client_args={"api_key": api_key, "base_url": payload["_rollout"]["base_url"]}, ...)
```

`"EMPTY"` keeps local runs and the rllm/legacy path (per-session URLs, api key ignored)
working unchanged.

## Troubleshooting

- **Every rollout degenerates, warning about "static session 'EMPTY'"**: the deployed
  agent image predates the session-key contract and sends a fixed api key — its turns
  accumulate under one shared session while each rollout's real session drains empty.
  Rebuild/redeploy the agent image with the contract above.
- **Agent-side 404s on every rollout**: an agent constructing its own URL paths
  instead of using an OpenAI/Anthropic SDK may miss the `/v1` convention (see above).
- **Run finishes instantly at "Training Progress: 100%"**: verl auto-resumed from a
  previous run's checkpoint dir. The script prevents this with
  `trainer.resume_mode=disable`; if you re-enable resume and override `CKPTS_DIR`,
  keep it per-experiment so a new experiment can't pick up another run's state.
- Stop-string text trimmed by verl's rollout servers is excluded from trained ids
  (same behavior as verl's own agent loops).
- Sub-agents that reuse the same session id fork within one tree and are captured; a
  sub-agent given a *different* session id becomes a separate tree and is not joined
  to the episode (cross-session grouping is future work).
- The gateway assumes verl's Ray-based `LLMServerClient` can be awaited from the
  gateway thread's event loop (uni-agent does the same); if a Ray upgrade breaks this,
  marshal via `run_coroutine_threadsafe` onto the worker loop.
