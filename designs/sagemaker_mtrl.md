# SageMaker MTRL: one entrypoint for ART backends and SageMaker RFT

| Field | Value |
| --- | --- |
| Status | Proposed |
| Implementation | In progress |
| Date | 2026-10-29 |

## Summary

`@sagemaker_mtrl` lets one `AgentCoreRuntimeApp` entrypoint, written for ART's
`_config` contract, serve both ART's training backends (slime, verl, Tinker)
and SageMaker RFT's multi-turn RL (MTRL) rollouts. All SageMaker-specific work
happens inside ART: request translation, inference authentication and
trajectory headers, sampling parameters, and completion and reward reporting.
An agent adds one decorator and no SageMaker code:

```python
from agentcore_rl_toolkit import AgentCoreRuntimeApp, sagemaker_mtrl

app = AgentCoreRuntimeApp()

@app.entrypoint
@sagemaker_mtrl
def invoke_agent(payload, context):
    config = payload["_config"]  # base_url, model_id, api_key, whichever trainer called
    result = run_agent(payload, config)
    return {"reward": compute_reward(result)}
```

## Problem

The two callers use incompatible contracts.

| | ART backends | SageMaker RFT |
| --- | --- | --- |
| Request | `{**task, "_config": {base_url, model_id, api_key}}` | `{"prompt": <row JSON string>, "metadata": {jobArn, rolloutId, endpoint, region}, "inferenceParams": {...}}` |
| Inference | `_config.base_url`; `api_key` is the gateway session ID | `<endpoint>/v1` with a SageMaker bearer token and `X-Amzn-SageMaker-Job-Arn` / `X-Amzn-SageMaker-Trajectory-Id` on every call |
| Sampling | Owned by the trainer | `inferenceParams` (`temperature`, `topP`, `maxTokens`) |
| Outcome | The trainer reads `reward` from the result | The agent calls `/complete-rollout` and `/update-reward` |

The SageMaker SDK's `sagemaker_rft_handler` reports the outcome, but stacking
it on an ART entrypoint leaves the rest to each agent, and has three defects in
this composition:

- It accepts only `payload`. `functools.wraps` exposes the handler's signature
  through `__wrapped__`, so bedrock-agentcore passes `context` to a handler that
  declares it, and the SDK's wrapper raises `TypeError`.
- It does not translate the request, so an agent written for `_config` cannot
  run a SageMaker request.
- A result without `reward` marks the trajectory ready with no reward instead
  of failed.

## Contract

The wrapper selects a path per request:

| Request | Path |
| --- | --- |
| Carries `_config` | The handler receives the payload unchanged and nothing is reported to SageMaker. |
| No `_config`, a string `prompt` and an object `metadata` | Translated and reported, as below. |
| Neither | The handler receives the payload unchanged. |

`_config` is checked first, so a dataset row's own `metadata` cannot select the
SageMaker path for an ART backend request.

SageMaker's request carries `prompt`, `metadata`, and optionally
`inferenceParams` (`temperature`, `topP`, `maxTokens`). ART reads two `metadata`
fields itself: `endpoint`, which is required, and `region`, which defaults to
`AWS_REGION` through `generate_token`. The SDK reads the job and trajectory in
whichever spellings it accepts. A request without an endpoint, or one whose job
or trajectory the SDK cannot name, is refused.

On the SageMaker path the handler receives
`{**row, "_config": {"base_url", "model_id", "api_key"}}`, where `row` is
`json.loads(payload["prompt"])`. SageMaker's `metadata` and `inferenceParams`
are withheld, so they cannot collide with fields of the row. A row that is not
an object or contains `_config` is refused.

The handler keeps its signature, including `context`, and may be synchronous or
`async`. Generator handlers are refused at decoration time because the outcome
is reported from a returned dict.

| Handler outcome | Reported to SageMaker | Returned to the app |
| --- | --- | --- |
| Dict with `status: "error"` | `failed`, reward 0 | The result |
| Dict with a non-empty `turn_rewards` list of finite numbers | `ready`, then `turn_rewards` | The result |
| Dict without `turn_rewards`, with a finite scalar `reward` | `ready`, then `[reward]` | The result |
| Dict with invalid `turn_rewards`, or neither field valid (missing, `NaN`, `bool`, non-numeric) | `failed`, reward 0 | The result with `status: "error"` and an `error` message |
| Non-dict result, or an exception | `failed`, reward 0 | The exception, persisted as `error` |

A feedback call that fails after a successful handler propagates, so the
invocation record shows that SageMaker was not told. One that fails while
reporting a failure is logged, and the record keeps the handler's error.

`turn_rewards` exists so that one result serves every caller. The SageMaker
path also gives the handler `_config`, so the handler cannot tell which trainer
called it. ART's backends train on the scalar `reward` and reject a list in it,
and SageMaker accepts per-turn rewards. A handler that computes per-turn rewards
returns both: `{"reward": total, "turn_rewards": [...]}`. ART's backends ignore
`turn_rewards`, and the invocation record keeps the handler's result as
returned.

Options:

- `row=`: `payload -> row` for datasets that encode `prompt` differently.
- `model_id=`: the `_config.model_id` for SageMaker rollouts (default
  `"default"`).
- `trusted_endpoint=`: the regular expression `metadata.endpoint` must match
  (see [Endpoint trust](#endpoint-trust)).

## Design decisions

### Loopback inference proxy

SageMaker inference needs a bearer token and two trajectory headers on every
call. `_config` carries neither headers nor a refreshable credential, and the
agent's model client is built from `_config` alone. On the SageMaker path,
`_config.base_url` addresses a proxy that ART starts on first use and that
forwards `/v1/*` to `<endpoint>/v1/*`, adding the token and headers. Streaming
responses pass through unbuffered.

The proxy keeps the agent free of SageMaker code, and the token never enters
the payload, the result, or the invocation record. `api_key` is a per-rollout
key that selects the rollout, as the rollout gateway's session ID does, and
stops routing when the handler returns.

Extending `_config` with a `headers` field would avoid the proxy but requires
every agent to forward the headers to its client, and leaves a fixed token in
the payload for the whole episode. The SDK's framework adapters (`wrap_model`)
inject the headers, but only for the frameworks they cover.

### Reporting through the feedback client

ART reports the outcome by calling `RolloutFeedbackClient` itself:
`complete_rollout("ready")` and `update_reward(...)` on success, and
`report_error(...)` on failure. It also sets and clears the SDK's rollout
context around the handler, so agents that read it, for example through the
SDK's framework adapters, keep working.

Steering `sagemaker_rft_handler` instead would mean shaping results to match its
private result handling, such as its `status` values and its treatment of a
list `reward`. Those conventions would then leak into ART's result contract, and
an SDK change to them would silently change what SageMaker is told. With the
feedback client, ART owns the success and failure rules, and the `status:
"error"` signal is ART's documented contract.

### Token refresh

The token is a SigV4 presign, so it stops verifying when the credentials that
signed it expire, regardless of its own 12-hour lifetime. The proxy
regenerates it every 15 minutes, so episodes may outlast a role session.

### Sampling parameters

The proxy sets `inferenceParams` on completions requests, overriding the
agent's values, so rollouts sample with the settings SageMaker trains with.
`maxTokens` sets `max_completion_tokens` when the request uses that field and
`max_tokens` otherwise.

### Endpoint trust

One endpoint, SageMaker's RFT Runtime Service, serves both inference
(`<endpoint>/v1/*`) and feedback (`/complete-rollout`, `/update-reward`). The
wrapper uses `metadata.endpoint` only. The SDK's feedback client prefers that
value over its environment fallbacks, so both reach the same host. Without it,
the wrapper refuses the request rather than reproduce the SDK's fallbacks.

The endpoint comes from caller-supplied `metadata`, and both inference and
feedback carry a token signed with the Runtime's credentials. The wrapper
refuses it, before anything is reported, unless it fully matches
`https://job-runtime[.<stage>].sagemaker.<region>.api.aws`, or another pattern
passed as `trusted_endpoint=`. A refused request reaches neither the endpoint
nor the feedback APIs, and the trajectory remains pending.

### Handler wrapper, not an app option

`AgentCoreRuntimeApp` treats payloads and results as opaque, and its protocol
envelope is the only field it removes. Translation is therefore a choice made by
the decorated handler. The wrapper composes with the app's foreground and
background execution, protocol envelopes, and invocation records unchanged.

### Dependencies

The SageMaker SDK and the proxy's aiohttp dependency are in the
`sagemaker_mtrl` extra and imported only when a SageMaker request arrives, so
images serving only ART backends do not need them.

### SDK coupling

All SageMaker knowledge is in `_MtrlAdapter` and `_MtrlRollout`; the loopback
forwarder and the handler wrapper are protocol-independent. ART implements the
request shape (`prompt`, `metadata.endpoint`, `metadata.region`,
`inferenceParams`), the `/v1` inference path, and its own reporting rules. The
SDK does the rest through public functions:

| SDK function | Used for |
| --- | --- |
| `make_inference_headers` | The trajectory headers, from `metadata` in any spelling the SDK accepts |
| `generate_token` | The inference token, with its region fallback |
| `RolloutFeedbackClient` | Feedback, from the same `metadata` |
| `set_rollout_context`, `clear_rollout_context` | The SDK's context, for agents that read it |

The only SDK behavior ART reproduces is the mapping from `inferenceParams` to
OpenAI request fields, which the SDK keeps inside its framework adapters.
Contract tests pin it to the SDK's Strands adapter and check that the feedback
client prefers `metadata.endpoint`. The `sagemaker_mtrl` extra bounds
`sagemaker-train` to the tested major version, and the tests must pass before
the bound is raised. A public SDK function for the `inferenceParams` mapping
would remove the reproduced logic.

## Security

- The proxy binds `127.0.0.1` on an ephemeral port and is reachable only inside
  the Runtime session's microVM. The per-rollout key routes requests; the
  loopback bind is the access boundary.
- Requests with an unknown or expired key receive `401` and are not forwarded.
- The SageMaker token is held by the proxy and is never returned to the
  handler.
- Endpoint trust is enforced as described above.

## Limitations

- `turn_rewards` is reported as given. Whether its length must match the
  rollout's turns is SageMaker's contract, and the wrapper does not check it.
- A failed trajectory reports reward 0. The SDK's partial reward on failure is
  not used.
- A background request without a protocol envelope gets a new invocation ID,
  so a retried SageMaker request may run again. The feedback client skips
  calls for trajectories SageMaker reports as already processed.
- The proxy serves one process; each Runtime session runs one app process.

## Validation

`tests/runtime/test_sagemaker_rft.py` runs the real SDK feedback client
against a loopback stand-in for the SageMaker endpoint, with feedback calls
recorded.
It covers synchronous and `async` handlers with `context` in background
execution, streaming and non-streaming completions, the token, headers, and
sampling parameters received upstream, key revocation after the rollout, each
failed-reward case, `turn_rewards` reporting and its failure cases, snake_case
and camelCase metadata, `status: "error"`, feedback failures, unusable rows and
metadata, refused, trusted, and configured endpoints, and pass-through of ART
backend payloads. Contract tests check the `inferenceParams` mapping against
the SDK's Strands adapter and the feedback client's use of `metadata.endpoint`. A live test against a SageMaker job is pending.

## Changes

| File | Change |
| --- | --- |
| `src/agentcore_rl_toolkit/runtime/sagemaker_mtrl.py` | New: the wrapper, the SageMaker adapter, and the loopback forwarder |
| `src/agentcore_rl_toolkit/__init__.py`, `runtime/__init__.py` | Export `sagemaker_mtrl` |
| `pyproject.toml`, `uv.lock` | New `sagemaker_mtrl` extra, with `sagemaker-train>=1.27.0,<2` |
| `tests/runtime/test_sagemaker_rft.py` | Wrapper tests |
| `tests/rollout_gateway/test_sagemaker_sdk_backend.py` | Its SDK stub extends an installed SDK instead of replacing it, so the wrapper tests run in the full suite |
| `designs/agentcore_runtime_app.md`, `designs/README.md`, `AGENTS.md` | Link to this design |
