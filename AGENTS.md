# AGENTS.md

This document provides context, patterns, and guidelines for AI coding assistants working in this repository. For human contributors, see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Quick Reference

### Key Commands

```bash
# Install dependencies (root package)
uv sync

# Run tests
uv run pytest tests/

# Build and push Docker image to ECR (current approach, may change)
./scripts/build_docker_image_and_push_to_ecr.sh \
  --dockerfile=examples/strands_math_agent/.bedrock_agentcore/strands_math_agent_rl/Dockerfile \
  --tag=latest \
  --context=examples/strands_math_agent

# Run example locally
cd examples/strands_math_agent && uv sync && uv run python rl_app.py
```

### Key Files

| File | Purpose |
|------|---------|
| `src/agentcore_rl_toolkit/app.py` | `AgentCoreRLApp` base class, `@rollout_entrypoint` decorator |
| `src/agentcore_rl_toolkit/runtime/` | `AgentCoreRuntimeApp`: optional HTTP `start/get` invocation lifecycle with filesystem records |
| `src/agentcore_rl_toolkit/client.py` | `RolloutClient` and `RolloutFuture` for training integration and batch evaluation |
| `src/agentcore_rl_toolkit/reward_function.py` | `RewardFunction` base class |
| `src/agentcore_rl_toolkit/rollout_gateway/` | In-repo token-level trajectory capture layer: `RolloutGateway`, `Renderer`, `SamplingBackend`, `TraceRecord` (see [Rollout Gateway](#rollout-gateway)) |
| `src/agentcore_rl_toolkit/backends/experimental/verl/` | Experimental verl extras that have not graduated into `backends/verl/`: `RolloutSessionAgentLoop` (sandbox-session rollouts, used by the SWE agent example), its task runner, and trainer mixins |
| `src/agentcore_rl_toolkit/backends/experimental/sagemaker/` | Experimental SageMaker Training Sessions backend: single-process GRPO loop (`train_grpo.py`) over the rollout gateway, no local GPUs |
| `src/agentcore_rl_toolkit/backends/verl/` | verl backend: `AgentCoreAgentLoop` plugged into verl's standard main_ppo entrypoint via the rollout gateway |
| `src/agentcore_rl_toolkit/backends/tinker_api/` | CPU training loop over a Tinker-compatible endpoint, with HF rendering and synchronous group-relative updates |
| `src/agentcore_rl_toolkit/backends/slime/` | slime backend: `generate` + `normalize_episode_rewards` hooks for slime's `--custom-generate-function-path` / `--custom-reward-post-process-path` (see [slime backend](#slime-backend-backendsslime)) |
| `src/agentcore_rl_toolkit/sandbox/` | Sandbox SDK: `SandboxClient`, `Sandbox`, `ExecResult` — run programs in arbitrary images on ACR (see [Sandbox SDK](#sandbox-sdk)) |
| `sandboxd/` | Go daemon (`agentcore-sandboxd`) for session health and recoverable command execution |
| `examples/strands_math_agent/` | GSM8K math agent example |
| `examples/strands_migration_agent/` | Java migration agent example |
| `examples/strands_officebench_agent/` | OfficeBench office automation agent example |
| `examples/strands_appworld_agent/` | AppWorld API interaction agent example |

---

## Product Overview

### What is ACR

This repo provides an SDK that helps developers train their agents with **Bedrock AgentCore Runtime (ACR)**.

ACR can be viewed as Lambda functions with session continuity:
- **Session routing**: Requests with the same session ID route to the same container for multi-turn interactions
- **Session isolation**: Different session IDs use separate runtime sessions (microVMs) for strong isolation
- **Auto-scaling**: New runtime sessions spin up instantly when needed
- **Sandboxed execution**: Each session runs in a secure microVM environment

These properties make ACR ideal for deploying LLM agents, and especially suited for online RL training which requires running many parallel agent rollouts securely and efficiently.

### Why This SDK

For online RL training techniques like GRPO, developers need to:
1. Gather rollouts and corresponding rewards
2. Invoke the model being trained (hosted on a training cluster) instead of using a model API

**Goal**: Help developers adapt their production agent with minimal friction for RL training with ACR, so most of the production codebase can be directly reused while enjoying ACR's security and efficiency benefits.

### Background: BedrockAgentCoreApp

ACR containers must serve `/invocations` (agent logic) and `/ping` (health) on port 8080 —
see the [HTTP protocol contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html#container-requirements-http).
AWS's [`BedrockAgentCoreApp`](https://github.com/aws/bedrock-agentcore-sdk-python) wraps that
contract (`@app.entrypoint`, `@app.websocket`, `@app.ping`, `@app.async_task`); see
`examples/*/basic_app.py` for the production shape this repo adapts from, and the
[runtime overview](https://aws.github.io/bedrock-agentcore-starter-toolkit/user-guide/runtime/overview.html)
for the full API.

### What agentcore-rl-toolkit Provides

When performing rollout in ACR during RL, we need to collect the rollout and reward and return them to the training engine. A naive approach of waiting synchronously requires maintaining many TCP connections, which is brittle and hard to manage.

#### Design Pattern 1: Fire-and-forget with background async processing

With `@app.rollout_entrypoint` decorator replacing `@app.entrypoint`:
- Agent processing moves to the background immediately
- Server returns an in-progress message right away
- Health status from `/ping` is automatically managed (busy while working, idle when done)
- ACR can manage session lifecycle to avoid early termination or wasteful idle sessions

#### Design Pattern 2: S3-based result delivery with HEAD polling

Since the client won't get results directly from HTTP:
- `@app.rollout_entrypoint` requires returning rollout and reward from the entrypoint
- Rollout data is saved to S3 with a predictable key returned in the immediate HTTP response
- Client polls S3 using efficient HEAD requests to detect when each result is available
- No additional messaging infrastructure required — S3 is the single source of truth

On the client side, `RolloutClient` and `RolloutFuture` are the complement to these server-side patterns — they handle submitting requests to ACR and polling S3 for results, so both sides work together to manage long-running async agent tasks end-to-end. See the [Evaluation](#evaluation) section for details.

#### Core Classes

**AgentCoreRuntimeApp** (`src/agentcore_rl_toolkit/runtime/`)
- Exported from `agentcore_rl_toolkit`; retains `@app.entrypoint` and ordinary HTTP behavior.
- `background=True` runs requests without `_agentcore_runtime` in the background
  with generated invocation IDs; ordinary retries may execute again. Explicit
  envelopes retain their protocol behavior. See the
  [SageMaker RFT example](designs/agentcore_runtime_app.md#sagemaker-rft-example)
  and [integration tests](tests/runtime/test_sagemaker_rft.py).
- Requests with `_agentcore_runtime` use versioned `start/get`, foreground/background execution, and session-scoped invocation IDs.
- Records live under configurable `state_dir`, defaulting to `.agentcore_runtime` under the OS temporary directory (`TMPDIR`).
- One app process serves one Runtime session; live tasks are keyed by invocation ID. Filesystem records retain session-specific paths; use a managed mount for stop/resume persistence.
- Protocol handlers return JSON values; terminal errors use an `error` string. Terminal errors, `interrupted`, and `not_found` are HTTP 200 states, while storage failures are operation errors.
- `AgentCoreHttpClient` drives this app; the existing `RolloutClient` still uses S3. See [the app design](designs/agentcore_runtime_app.md) for HTTP mapping and implementation decisions, and [RIP](designs/runtime_invocation_protocol.md) for the shared lifecycle contract.
- `AgentCoreHttpClient.invoke()` returns a foreground result or a background `InvocationHandle`.
  `AgentCoreHttpSession` owns per-rollout cleanup;
  the Tinker backend uses it with flat task payloads and singular `reward`.
- Tests: `uv run pytest tests/runtime/`.
- Live tests: deploy `tests/runtime/live_agent.py` with the checkout's package,
  the HTTP contract, and `idleRuntimeSessionTimeout=60`, then run
  `RUNTIME_TEST_ARN=arn:... uv run pytest tests/runtime/test_live.py`.
  Without the ARN they skip. Each test stops its own sessions; the Runtime stays
  deployed for reuse. The agent needs no model, S3, or managed session storage.

**AgentCoreRLApp** (`src/agentcore_rl_toolkit/app.py`)
- Inherits `BedrockAgentCoreApp` - drop-in replacement
- Provides `@app.rollout_entrypoint` decorator
- Expects `_rollout` dict in payload with `RolloutConfig` fields (`exp_id`, `input_id`, `s3_bucket`) plus optional pass-through config (`base_url`, `model_id`, `sampling_params`)
- Framework-agnostic: works with any agent framework, not just Strands

#### Utilities

**RewardFunction** (`src/agentcore_rl_toolkit/reward_function.py`)
- Base class for reward implementations
- Can be any function that outputs a scalar

### Rollout Gateway

`src/agentcore_rl_toolkit/rollout_gateway/` is an in-repo, backend-agnostic layer that
captures **token-level, loss-maskable trajectories** from agent rollouts for RL training.
It is used by the toolkit's `backends/{slime,verl,tinker_api}` integrations.

**Why it exists.** RL training needs per-token ids, logprobs, and a loss mask for every
model turn — not just the final text. The gateway captures these transparently: an agent
points its OpenAI/Anthropic client at the gateway (just change `base_url`), and the
gateway records the trajectory as a side effect of serving the request.

**How it works.** The gateway *owns tokenization*. Rather than scraping token ids out of
a chat response (which is engine-specific and brittle), it renders canonical messages to
`token_ids` itself, sends those to a token-in/token-out inference backend, and gets back
`token_ids` + logprobs. Owning both directions makes loss-masking well-defined and
eliminates cross-backend retokenization drift. This is also what lets it support
sample-only backends like Tinker, which cannot render themselves.

**Core components:**

| Component | File | Role |
|---|---|---|
| `TraceRecord` | `trace.py` | Torch-free output: `token_ids`, `loss_mask`, `logprobs`, `reward`, `rollout_id`. Each training backend converts this to its native sample type in its own process. |
| `TrajectoryManager` | `trajectory.py` | Per-session message **tree**. Handles multi-turn concatenation and parallel tool-call branches; re-tokenization drift splits records while preserving generated tokens (CLEAN / FORK). Tokenizer-free and torch-free. |
| `Renderer` | `render.py` | Tokenization seam. `HfTemplateRenderer` (HF `apply_chat_template`), shared by all sampling backends. |
| `SamplingBackend` | `sampling_backends/` | The one per-engine seam: `token_ids -> token_ids + logprobs` as a `TurnRecord`. Impls: `VllmHttpBackend`, `SglangHttpBackend`, `TinkerSdkBackend`, `SageMakerSdkBackend`. Placement rule: engine seams for independently reachable inference services (HTTP endpoints, hosted SDKs like Tinker and SageMaker) live here; seams over trainer-internal handles (e.g. `VerlSamplingBackend` over verl's Ray-based `LLMServerClient`) live with that trainer's integration under `backends/`. |
| Adapters | `adapters/` | Wire-protocol translation: `OpenAIAdapter` (`/v1/chat/completions`), `AnthropicAdapter` (`/v1/messages`). An agent drives the gateway in its *native* protocol unmodified (just point `base_url` at it); both normalize to one canonical message form and share one `TrajectoryManager`. |
| `RolloutGateway` | `gateway.py` | Assembles tokenizer + renderer + backend + adapters onto one aiohttp app sharing one `TrajectoryManager`. Session identity rides in the api-key / Bearer slot; `base_url` is a fixed gateway address (no per-session URLs). |
| `ThreadedGatewayServer` | `server.py` | Serves an assembled gateway on a background thread with its own event loop — the deployment shape for synchronous trainers (slime, verl). Async trainers can mount `gateway.app` into their own loop instead. |

**Session model.** A session id (in the Bearer slot) keys one trajectory tree.
`gateway.create_session(sid)` → agent turns are captured → `gateway.finish_session(sid)`
drains the tree into `list[TraceRecord]`.

**Multi-API & sub-agents.**
- *Multi-API* (works today): OpenAI- and Anthropic-protocol turns for the same session id
  fold into the same trajectory tree, so one training run can capture agents that speak
  different wire protocols.
- *Dynamic sub-agents as in-session forks* (works today): when a harness spawns a sub-agent
  (e.g. Claude Code's Task tool) whose LLM calls reuse the **same** session id, its distinct
  system prompt doesn't match the parent's branch, so `TrajectoryManager` **forks** a new leaf
  automatically (`_find_mount_point` matches by role + message equality). `get_trajectory`
  walks *all* leaves, so the parent trajectory and each sub-agent trajectory are all captured
  and correlated under one tree — the common self-spawning-sub-agent case needs no extra
  wiring. (Verifying this end-to-end against a real Claude Code harness is a to-do; the design
  is in place, but only synthetic/single-agent flows have been exercised so far.)
- *Sub-agents that run under a **distinct** session id* (grouping pending): if the harness
  gives a sub-agent its own session id (its own Bearer key), it becomes a separate tree. Tying
  those separate trees into one episode requires stamping a shared `rollout_id` across their
  `TraceRecord`s — that stamping lives in the (not-yet-landed) dispatch layer, so this
  cross-session case is not wired in this package yet.

**Dependencies.** The gateway is trainer-side and lives behind extras — the base install
(agent-side `AgentCoreRLApp` / `RolloutClient`) stays lean:
- `pip install agentcore-rl-toolkit[gateway]` → `aiohttp` + `transformers` + `jmespath`.
- Tool/reasoning parsing: when the tokenizer's chat template is recognized (sha256 hash
  lookup in `rollout_gateway/response_schemas.py`), `HfTemplateRenderer` derenders the
  whole output in one pass via `tokenizer.parse_response(text, schema=...)` — reasoning,
  text, and tool calls in the model family's actual format (Qwen2.5/3/3.5/3.6,
  GLM4-MoE, GPT-OSS, Nemotron-3; schemas vendored from huggingface/trl, see NOTICE). A
  parse failure degrades to raw text with `ill_formed=True`, never an exception.
  Unrecognized templates fall back to a `</think>` split for tool-free parsing, but
  tool-bearing requests are **rejected** (there is no implicit tool parser — the
  dependency-free `<tool_call><function=...>` XML regex understands one format and
  would silently miss every other; opt into it explicitly with
  `tool_parser=parse_tool_uses`); the gateway itself never imports an inference
  engine. Injecting a `reasoning_parser` / `tool_parser` callable into
  `HfTemplateRenderer` disables schema detection and takes full control. The slime
  backend (`backends/slime/`) uses the default schema-based parsing (`HfTemplateRenderer` with no custom parsers); injecting the same
  SGLang-native parsers via the `reasoning_parser` / `tool_parser` kwargs is a natural
  extension if tool-bearing or reasoning-bearing models are served.
- For Tinker API training (`backends/tinker_api/`), install `[gateway,tinker_api]`
  for the official service or `[gateway,tinker_skyrl]` for the tested SkyRL version.
  `TinkerSdkBackend` uses the Tinker SDK for token sampling and the gateway uses
  `HfTemplateRenderer`. SkyRL and its GPU dependencies live in a separate service
  environment. The synchronous recipe and instructions for connecting to an existing endpoint are in
  `src/agentcore_rl_toolkit/backends/tinker_api/README.md`.
- For the SageMaker backend (`SageMakerSdkBackend`), install the Training Sessions SDK
  (`uv sync --group sagemaker-integration`, or `pip install sagemaker-train`). Unlike the other
  sampling backends it imports `sagemaker.train.training_session` at module scope, so the
  module is only importable with the SDK present; nothing else in the gateway imports it.

The core (`TraceRecord`, `TrajectoryManager`, `Renderer` protocol, `SamplingBackend`
protocol) imports torch-free and aiohttp-free; `RolloutGateway` is exposed lazily so
importing the package never requires aiohttp. Tests live in `tests/rollout_gateway/`.

**Status.** The capture layer above is implemented and tested. Training-backend consumers:
the **verl backend** (`backends/verl/`, see below), the **Tinker API backend**
(`backends/tinker_api/`), the **slime backend**
(`backends/slime/`, see below), and the **experimental SageMaker backend**
(`backends/experimental/sagemaker/`, see below). Other
backends' dispatch/reward-join glue is not yet on the main branch — a prototype
dispatcher is parked on the `wip/online-rl-dispatch` branch.

### External Integrations

[rLLM natively integrates with AgentCore Runtime](https://docs.rllm-project.com/agent-runtimes/agentcore).
rLLM maintains its installation, configuration, training, and troubleshooting
instructions upstream. This repository provides agent adaptation and deployment
guides; its training integrations and rollout gateway are maintained separately.

### verl backend (`backends/verl/`)

The verl backend connects AgentCore rollouts to verl through `AgentCoreAgentLoop`
and the rollout gateway. See `backends/verl/README.md` for its architecture, setup,
contracts, limitations, and troubleshooting.

### Experimental SageMaker backend (`backends/experimental/sagemaker/`)

GRPO against **Amazon SageMaker Training Sessions**: SageMaker hosts the policy weights,
the sampler, and the optimizer behind an SDK, so **no local GPU cluster is needed** — the
RL loop is a single-process asyncio program driven by one YAML config. Contrast with
slime/verl, where the trainer owns the GPUs and the gateway is embedded in it. Requires
the `sagemaker-train` wheel (see above); user-facing setup lives in
`SETUP.md` and `docs/site/src/content/docs/guides/sagemaker-backend-setup.md`.

Key pieces:
- `rollout_gateway/sampling_backends/sagemaker_sdk.py` — `SageMakerSdkBackend`: gateway
  `SamplingBackend` over the SageMaker `SamplingClient`. Renderer-free like
  `TinkerSdkBackend` (the SDK cannot render; the gateway owns tokenization). Three SDK
  properties shape it: **(a)** weights are pinned at sampling-client creation, so
  `save_weights_and_get_sampling_client()` returns a *new* client each update and reusing
  the old one silently samples stale weights — `set_sampling_client()` rebinds under a
  `threading.Lock` while the gateway's background thread may be mid-`generate()`
  (in-flight calls finish against the old weights, subsequent calls use the new client);
  **(b)** `generate()` awaits `APIFuture.result_async()`, which holds a thread only per
  status poll (blocking `result()` in `to_thread` held one of the stock `min(32, cpu+4)`
  threads per in-flight sample); the polls run on the loop's default executor, sized to
  `poll_threads` (default 256);
  **(c)** the service rejects `maxTokens < 512`, so the backend declares
  `min_new_tokens = 512` and the gateway adapter returns `context_length_exceeded` when
  less room than that is left (other backends default to 1, i.e. no room).
  Both are covered in `tests/rollout_gateway/test_sagemaker_sdk_backend.py`, which stubs
  the `sagemaker` module so the tests need no SDK wheel and no AWS calls.
- `train_grpo.py` — the loop, run as
  `python -m agentcore_rl_toolkit.backends.experimental.sagemaker.train_grpo --config config.yaml`.
  Per step: pop `batch_size` payloads, run `responses_per_prompt` ACR rollouts per payload
  concurrently, center/normalize rewards within each group, convert every trajectory-tree
  leaf to a datum, `forward_backward(loss_fn=...)` + `optim_step(AdamParams(...))`, then
  rebind the sampler to the updated weights. LoRA only
  (`create_lora_training_client`); `loss` selects the surrogate
  (`ppo` | `importance_sampling` | `cispo`). The training session bills until stopped, so
  `tc.stop()` is the first statement in the `finally` and a `SIGTERM` handler cancels the
  task rather than letting the signal kill the interpreter before teardown.
- `rollout.py` — `run_one_rollout()`: drives an `AgentCoreRuntimeApp` agent through
  `AgentCoreHttpClient` / `AgentCoreHttpSession` (`AgentCoreRLApp` agents are not
  supported). One UUID per rollout serves as gateway session id, ACR
  `runtimeSessionId`, **and** `api_key` (sent in the payload's `_config` with
  `base_url` / `model_id`). The agent returns a scalar `{"reward": ...}`; reward is
  0.0 (logged, never raised) on timeout, ACR error, agent exception, or a
  missing/non-numeric reward.
- `datum.py` — `trace_record_to_datum()`: `TraceRecord` + advantage → a SageMaker datum
  (shifted `input`/`target_tokens`, with `weights` / `logprobs` / `advantages` left-padded
  over the prompt span). Returns `None` for an empty or all-zero loss mask; a step with no
  valid datums is skipped rather than submitted. The loss mask is applied to `advantages`
  as well as `weights`: masked positions inside the response span (tool results, user
  turns) carry a stored logprob of `0.0`, so a service-side
  surrogate that consulted only `advantages` would score them against a fabricated old
  policy. Whether all four `lossFn` values honor `weights` is not observable from the
  client SDK (it ships `lossFnName` + datums and does no loss math), so the datum is made
  self-consistent instead of relying on that.
- `config.py` / `config.yaml.example` — typed dataclass + template; `load_config` silently
  drops unknown YAML keys. Local `config.yaml` is gitignored (holds ARNs).
- `prepare_datasets/prepare_gsm8k.py` — GSM8K → Parquet. Same payload-first dataset
  contract as the verl backend (one `payload` column forwarded verbatim to the
  agent's entrypoint), but read directly by `rollout.load_dataset()` (Parquet or JSONL)
  with no synthesized `prompt` column, since there is no verl dataloader to satisfy.

Current limits: GRPO only (`SageMakerSdkBackend`, `run_one_rollout()`,
`trace_record_to_datum()`, and the gateway assembly are reusable for another algorithm),
LoRA only, rewards agent-side only, and one gateway in one process — it scales by ACR
concurrency, not by local hardware.

"No GPU cluster" is not "runs anywhere": `train_grpo.py` binds the gateway to
`local_ip()` and hands the agent `http://<private-ip>:<gateway_port>/v1`, so every
sampling call is an *inbound* connection from the ACR runtime to the loop host. The
host must be an EC2 instance in a VPC, the runtime must be deployed into that VPC
(`--vpc --subnets --security-groups`), and the security group must allow inbound TCP on
`gateway_port` (default `0` = ephemeral, so pin it to write a narrow rule). A laptop
cannot work; failures surface as rollouts scored `0.0`, not as a crash, because
`run_one_rollout()` logs and returns a zero reward. Documented under "Networking" in
`docs/site/src/content/docs/guides/sagemaker-backend-setup.md`.

**Vendored response schemas.** The response-schema registry is adapted from
[trl](https://github.com/huggingface/trl) (Apache-2.0; see `NOTICE`).

| This repo | upstream source | Baseline commit |
|---|---|---|
| `rollout_gateway/response_schemas.py` | `trl/chat_template_utils.py` (schema dicts) + `trl/chat_templates/*.jinja` (hash table) | `7073af94` |

For `response_schemas.py`, re-sync means updating the schema dicts and recomputing the
sha256 hashes of the covered chat templates — preserving the marked local-additions
block at the end of `_TEMPLATE_HASHES` (templates with no TRL equivalent, e.g.
Qwen3-Coder → `qwen3_5`). Bump the baseline commit here when you re-sync.

### slime backend (`backends/slime/`)

Plugs into [slime](https://github.com/THUDM/slime) (Megatron-LM + SGLang GRPO trainer)
via two hook entry points, requiring no changes to slime itself:

- `--custom-generate-function-path agentcore_rl_toolkit.backends.slime.integration.rollout.generate`
- `--custom-reward-post-process-path agentcore_rl_toolkit.backends.slime.integration.rewards.normalize_episode_rewards`
- `--custom-config-path /path/to/config.yaml` (see `examples/math_agent/config.yaml.example`)

**Key pieces:**

- `integration/rollout.py` — `AgentCoreRLConfig` (dataclass read from the YAML),
  `AgentCoreRLService` (singleton: one `RolloutGateway` + `ThreadedGatewayServer` +
  `AgentCoreHttpClient` per process; slime runs every `generate` call on one persistent
  event loop, so one client's connection pool serves them all), and
  `generate(args, sample, sampling_params)` — the per-sample hook. Drives an
  `AgentCoreRuntimeApp` agent (e.g. `examples/strands_math_agent/runtime_app.py`).
  Creates a gateway session keyed by a fresh `uuid4` (= ACR `runtimeSessionId` =
  `api_key` in the payload's `_config`, alongside `base_url` / `model_id`), runs the
  invocation in the background through `AgentCoreHttpSession`, which stops the Runtime
  session on exit, drains `gateway.finish_session` into `TraceRecord`s, and maps each
  record to a slime `Sample`. `tps_limit` caps new sessions per second and
  `request_tps_limit` caps every invoke/get/stop request. A timeout, invocation error, or
  agent exception scores captured turns 0; a rollout with no captured turns returns an
  `ABORTED` sample with `remove_sample=True` rather than raising — slime drops those
  rows from the batch.
- `integration/rewards.py` — `normalize_episode_rewards(args, samples)`: GRPO group
  normalization that stays correct when one rollout forks into multiple trajectory
  leaves. Slime's built-in normalizer groups positionally (reshape to `(-1, n_samples_per_prompt)`),
  which is wrong for forked rows. This groups by `group_index`, dedups by `rollout_id`,
  and normalizes within each group's unique rollouts. Active only when
  `advantage_estimator` is in `(grpo, gspo, cispo, reinforce_plus_plus_baseline)` and
  `rewards_normalization=True`; returns identity otherwise.
- `integration/rollout.py:_agent_reward` — rewards are agent-owned (scalar `{"reward": ...}`
  in the returned dict). `None` is returned when the agent omits it, signalling slime to
  fall back to its own `rm_hub`; a non-numeric reward or non-dict result raises (broken
  agent-side code would silently zero every group).

**Dataset contract.** Each slime dataset row must carry the ACR invocation payload in its
`metadata.payload` field (a dict without a `_config` key). Slime's `--input-key prompt` supplies the conversation
for the positional advantage estimator; the actual agent input lives in `metadata.payload`.

**Installation.**

```bash
bash src/agentcore_rl_toolkit/backends/slime/scripts/install_slime.sh
```

Requires CUDA 13 (`CUDA_HOME=/usr/local/cuda-13.0`). The script installs: cu13 PyTorch +
flash-attn, transformer-engine-cu13, apex, torch_memory_saver, nvidia-modelopt, sglang,
and clones + installs slime + Megatron-LM from source.

**Running (GSM8K example).**

```bash
cd src/agentcore_rl_toolkit/backends/slime/examples/math_agent
cp config.yaml.example config.yaml  # fill in agent_runtime_arn
SLIME_DIR=... MODEL_DIR=... TRAIN_DATA_PATH=... MODEL_TYPE=... bash train.sh
```

See `examples/math_agent/SETUP.md` for the full walkthrough.

**CUDA 13 quirks** (handled inside `train.sh`):
- `/usr/local/cuda` → `cuda-12.*` confuses TE's cuDNN-frontend into loading two
  `libcudart` versions ("Multiple libcudart found"). `train.sh` redirects `CUDA_HOME` to
  `cuda-13.0`, strips cu12 paths from `LD_LIBRARY_PATH`, and drops an empty
  `libcudart.so.12` decoy so the cu12 resolver finds nothing real.
- `CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13` makes standalone cudnn-frontend ≥1.26
  skip the libcudart probe entirely.
- Megatron train actors get a separate Ray `runtime_env` that drops `CUDA_HOME`/`CUDNN_*`/
  `LD_LIBRARY_PATH`; `--train-env-vars` re-pins them for those actors.

### Sandbox SDK

The Sandbox SDK executes argv lists in arbitrary images on AgentCore Runtime.
The Python client calls an independent Go daemon inside each sandbox.

**Code entry points:**

- `src/agentcore_rl_toolkit/sandbox/`: `client.py` owns session and execution handles;
  `types.py` defines result types.
- `sandboxd/main.go`: HTTP/session dispatch; `process.go`: execution; `store.go`: records.

**Key constraints:**

- Execution ownership is independent of HTTP connections. Retries reuse invocation
  IDs, `get` never executes work, and terminal results are persisted before completion.
- Completion waits for process exit and output EOF. Execution timeout kills only the
  direct process; descendants may survive until session termination.
- Execution timeout raises `ExecTimeoutError` with the persisted result. Local
  `ExecHandle.result(timeout=...)` timeout raises `TimeoutError` and leaves work running.

See the [SDK design](designs/sandbox_sdk.md) for API semantics and decisions,
the [daemon README](sandboxd/README.md) for wire format and configuration, and
the [quickstart](examples/sandbox_quickstart/README.md) for deployment and usage.

```bash
sandboxd/build.sh --stage examples/sandbox_quickstart
uv run pytest tests/sandbox/
(cd sandboxd && go test -race ./...)
```

Live ACR tests require `SANDBOX_RUNTIME_ARN`. Go CI runs in
`.github/workflows/sandboxd.yml`.

### Migration Guide (basic_app → rl_app)

See `examples/strands_math_agent` for a complete example adapting from `basic_app.py` to `rl_app.py`.

#### Step 1: Switch to AgentCoreRLApp & Add Reward Function

- `AgentCoreRLApp` is a thin wrapper around `BedrockAgentCoreApp` — framework-agnostic
- Users implement the reward function for their use case

```diff
- from bedrock_agentcore.runtime import BedrockAgentCoreApp
+ from agentcore_rl_toolkit import AgentCoreRLApp
+ from reward import GSM8KReward

- app = BedrockAgentCoreApp()
+ app = AgentCoreRLApp()
+ reward_fn = GSM8KReward()
```

#### Step 2: Create Model & Agent Inside Entrypoint

- Model config (`base_url`, `model_id`) comes from the `_rollout` payload, not environment variables
- Optional `sampling_params` (e.g., `max_completion_tokens`, `temperature`) can also be passed via `_rollout` for training-engine-controlled generation settings
- Use standard `OpenAIModel` — no custom model wrappers needed. For evaluation, `base_url` can point directly to any OpenAI-compatible endpoint (vLLM, SGLang, LiteLLM, etc.), or you can use `BedrockModel` directly
- `api_key` is set from `payload["_rollout"].get("api_key")` — the training engine passes the trajectory-capture session key in the `_rollout` config (the verl backend keys token capture off the api-key slot). Fall back to `"EMPTY"` (the standard vLLM convention for unauthenticated servers) for local runs, evaluation endpoints, and gateways with per-session URLs, which ignore the api key
- Model and agent are created per-invocation inside the entrypoint
- This gives flexibility for the training engine to pass runtime configuration (inference address, sampling parameters, system prompt, etc.) to accommodate different learning scenarios
- This is safe because RL rollouts are single-invocation — the agent doesn't need persistent conversation history across requests, so there's no need to keep model/agent as global state

```diff
- model = BedrockModel(model_id="us.anthropic.claude-sonnet-5-5")
- agent = Agent(model=model, tools=[calculator], system_prompt="...")

- @app.entrypoint
- def invoke_agent(payload):
-     response = agent(user_input)
+ @app.rollout_entrypoint
+ def invoke_agent(payload: dict, context):
+     base_url = payload["_rollout"]["base_url"]
+     model_id = payload["_rollout"]["model_id"]
+     params = payload["_rollout"].get("sampling_params", {})
+     api_key = payload["_rollout"].get("api_key") or "EMPTY"  # session key for trajectory-capture gateways
+     model = OpenAIModel(client_args={"api_key": api_key, "base_url": base_url}, model_id=model_id, params=params)
+     agent = Agent(model=model, tools=[calculator], system_prompt="...")
+     response = agent(user_input)
```

#### Step 3: Compute Rewards & Return Result

The `@rollout_entrypoint` decorator automatically:
- Executes the function in the background (works with both sync and async functions)
- Saves the returned dict to S3 with a predictable key
- Handles errors and saves error results for client awareness

The return value must be a JSON-serializable dict when S3 save is configured. Any dict structure is accepted — there are no required keys. For training, return rewards; for evaluation, return whatever artifacts you need (metrics, conversation history, etc.). You can also persist raw data and run evaluation logic client-side.

**Reserved keys**: The SDK injects metadata into the saved S3 JSON. Avoid using these keys in your return dict:
- `status_code`, `stop_reason` — added only if not already present in your dict
- `input_id`, `s3_bucket`, `result_key`, `payload` — always overwritten with SDK values

```diff
-   return response.message["content"][0]["text"]
+   rewards = reward_fn(response_text=response.message["content"][0]["text"], ground_truth=answer)
+   return {"rewards": rewards}
```

Other valid return patterns:
```python
# Evaluation-only (no rollout_data needed)
return {"rewards": rewards, "metrics": {"latency_ms": elapsed}}

# Custom artifacts
return {"summary": "...", "artifacts": {...}}
```

Each example in `/examples` contains `basic_app.py` and `rl_app.py` to demonstrate this adaptation.

### Deployment to ACR

This package relies on [bedrock-agentcore-starter-toolkit](https://github.com/aws/bedrock-agentcore-starter-toolkit) for deployment:
- CLI tool to generate Dockerfiles, build images, push to ECR, and launch on ACR
- We prioritize container (ECR image) deployment for operational simplicity

**Current workflow:**
1. Dockerfiles are generated in `examples/{agent_name}/.bedrock_agentcore/{app_name}/Dockerfile`
2. Use `scripts/build_docker_image_and_push_to_ecr.sh` to build and push:
   ```bash
   ./scripts/build_docker_image_and_push_to_ecr.sh \
     --dockerfile=examples/strands_math_agent/.bedrock_agentcore/strands_math_agent_rl/Dockerfile \
     --tag=latest \
     --context=examples/strands_math_agent
   ```
3. Training engine takes ECR URI as config for deployment
4. Model config (`base_url`, `model_id`, and optionally `sampling_params`) is passed via the `_rollout` payload at invocation time

### Evaluation

Users can evaluate agents before and after training using the same `rl_app.py`.

**RolloutClient** (`src/agentcore_rl_toolkit/client.py`) provides both sync and async invocation patterns:

**Sync API** (blocking — suitable for scripts and simple loops):
- **`invoke()`**: Returns a `RolloutFuture` for fine-grained control — ideal for training loops (e.g., GRPO) where you submit individual rollouts and group results by `input_id`
- **`run_batch()`**: Higher-level API for batch evaluation — manages concurrency, timeouts, and polling automatically

**Async API** (non-blocking — suitable for `asyncio` event loops in RL training frameworks):
- **`invoke_async()`**: Like `invoke()` but doesn't block the event loop. Cold starts on one request don't block submission of others.
- **`run_batch_async()`**: Like `run_batch()` but returns an async iterator with concurrent submission.
- **`RolloutFuture`** supports `await future`, `future.result_async(timeout=...)`, and `future.done_async()`.

Concretely, `invoke()` / `invoke_async()` sends the request to ACR and returns a `RolloutFuture` immediately — meaning ACR has received the request and a background agent session is processing it. Calling `future.result(timeout=...)` or `await future.result_async(timeout=...)` blocks/waits until the result appears in S3, polling with exponential backoff. It returns the result (rewards, metrics, etc.) once the agent finishes and writes to S3.

Both sync and async patterns share the same infrastructure:
- **Rate limiting**: Handles ACR TPS limits (25)
- **Concurrency control**: Manages ACR session limits (1000/account) and model API rate limits
- **S3 HEAD polling**: Polls S3 for completed results using efficient HEAD requests
- **Automatic session cancellation**: Sessions are automatically cancelled after result fetch, timeout, or error — callers don't need to manage ACR session lifecycle

**Async usage example:**

```python
import asyncio
from agentcore_rl_toolkit import RolloutClient

client = RolloutClient(agent_runtime_arn="arn:...", s3_bucket="my-bucket", exp_id="exp-1")

async def run():
    # Fire all requests concurrently (cold starts don't block each other)
    tasks = [asyncio.create_task(client.invoke_async(p)) for p in payloads]
    futures = await asyncio.gather(*tasks)

    # Wait for all results concurrently
    results = await asyncio.gather(*[f.result_async(timeout=300) for f in futures])
    # Or without timeout: results = await asyncio.gather(*futures)

    # Or use run_batch_async for managed concurrency:
    async for item in client.run_batch_async(payloads, max_concurrent_sessions=100):
        if item.success:
            process(item.result)
```

**Note:** For evaluation, pass the appropriate `base_url`, `model_id`, and optionally `sampling_params` in the `_rollout` payload to point to the desired inference server (training cluster or hosted cloud model).

---

## Environment Variables

| Variable | Description | When Required |
|----------|-------------|---------------|
| `AWS_REGION` | AWS region | Always |
| `AWS_ACCOUNT` | AWS account ID | Deployment |
| `ECR_REPO_NAME` | ECR repository name | Deployment |

**Note:** `BASE_URL` and `MODEL_ID` are no longer set via environment variables. They are passed in the `_rollout` payload field along with optional `sampling_params`, allowing the training engine to configure them per-invocation.

See `.env.example` for template. The build script sources `.env` for deployment values.

---

## Common Tasks

### Adding a New Example Agent

1. Create folder in `examples/{agent_name}/`
2. Add `basic_app.py` (production version using `BedrockAgentCoreApp`)
3. Add `rl_app.py` (RL-adapted version using `AgentCoreRLApp` + `OpenAIModel`)
4. Add `reward.py` with `RewardFunction` implementation
5. Add `pyproject.toml` with example-specific dependencies
6. Run `uv sync` in the example folder

### Adding Support for a New Framework

Agents use their framework's native OpenAI-compatible model class (e.g., `OpenAIModel` for Strands), so framework-specific model wrappers are not needed. Point the model's `base_url` at the training inference server and token capture is handled at the infrastructure layer — no per-framework code lives in this package.

### Running Tests

```bash
uv run pytest tests/
```

The rollout gateway tests (`tests/rollout_gateway/`) need the gateway's runtime deps
(`aiohttp`, `transformers`), which are included in the `dev` extra:

```bash
uv sync --extra dev
uv run pytest tests/rollout_gateway/
```

The verl backend tests (`tests/backends/verl/`) run against
the installed verl distribution and skip when verl is absent (conftest-level
`importorskip`). Run them from an env with the `verl` extra synced; in CI
they run in `.github/workflows/verl-integration.yml`, which syncs the
`verl-ci` dependency group (same pinned verl, but CPU torch and no vllm/flash-attn —
the LLM server client is the faked seam, so no inference engine is needed).

---

## Development Tips

### Per-Example Environments

Each example has its own `pyproject.toml` and uv environment:
```bash
cd examples/strands_math_agent
uv sync  # Creates .venv in this folder
source .venv/bin/activate
```

To use the latest local source of `agentcore-rl-toolkit` (e.g., for testing unreleased changes):
```bash
uv pip install -e ../../ --force-reinstall --no-deps
```

### Code Conventions

- Commit messages must be [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`) — a commitizen pre-commit hook rejects anything else
- GitHub pull request descriptions must not hard-wrap prose at a fixed column width. Write each Markdown paragraph as one physical line, regardless of length, and let GitHub handle visual wrapping.
- In pull request descriptions, insert newlines only for Markdown structure: paragraph boundaries, headings, list items, blockquotes, tables, code blocks, and similar constructs.
- This pull request formatting rule does not apply to git commit messages or repository documentation; follow their existing wrapping conventions.
- When implementing or changing behavior covered by a document in `designs/`, update the relevant design content in the same change. Do not change its `Implementation` status unless explicitly requested.
- The `Implementation` field permits exactly three values with this capitalization: `Proposal`, `In progress`, or `Done`. Do not use synonyms, parenthetical explanations, component lists, or other qualifiers.
- Preserve the existing `Implementation` state unless the user explicitly requests a state transition. Implementing, testing, refactoring, or adding a consumer does not authorize changing this field.
- Return a JSON-serializable dict from `@rollout_entrypoint` (any structure accepted — no required keys)
- Create model and agent inside the entrypoint function (not at module level) so config comes from the `_rollout` payload
- Use standard `OpenAIModel` for OpenAI-compatible inference endpoints (token capture during training is handled at the infrastructure layer)
- Implement reward functions as classes inheriting `RewardFunction`

### Documentation

- State concrete behavior directly. Do not add assurances about unrelated
  side effects or explain nonexistent features. Include negative guarantees
  only when readers need them to use the API correctly.
- Avoid repeating explanations across design docs, READMEs, and docstrings.
  Keep detailed contracts in one primary location and link to them elsewhere.
- Design docs explain decisions and relevant tradeoffs, not conversation
  history or implementation progress. Include rejected alternatives only
  when they help explain a design decision.
- Scope implementation choices to the implementation they describe.
  Do not present defaults or example wiring as interface requirements.

### Symlink Note

`CLAUDE.md` is a symlink to `AGENTS.md` to support both instruction formats for AI coding assistants.

---

## External References

- **ACR Documentation**: https://docs.aws.amazon.com/bedrock-agentcore/
- **ACR Runtime Guide**: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html
- **bedrock-agentcore-sdk-python** (provides `BedrockAgentCoreApp`): https://github.com/aws/bedrock-agentcore-sdk-python
- **bedrock-agentcore-starter-toolkit** (CLI tools, Dockerfile generation): https://github.com/aws/bedrock-agentcore-starter-toolkit
- **Runtime SDK Overview**: https://aws.github.io/bedrock-agentcore-starter-toolkit/user-guide/runtime/overview.html
- **HTTP Protocol Contract**: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html#container-requirements-http
