# AGENTS.md

This document provides context, patterns, and guidelines for AI coding assistants working in this repository. For human contributors, see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Table of Contents

- [Quick Reference](#quick-reference)
- [Project Structure](#project-structure)
- [Product Overview](#product-overview)
  - [What is ACR](#what-is-acr)
  - [Why This SDK](#why-this-sdk)
  - [Background: BedrockAgentCoreApp](#background-bedrockagentcoreapp)
  - [What agentcore-rl-toolkit Provides](#what-agentcore-rl-toolkit-provides)
  - [Rollout Gateway](#rollout-gateway)
  - [Sandbox SDK](#sandbox-sdk)
  - [Migration Guide (basic_app → rl_app)](#migration-guide-basic_app--rl_app)
  - [Deployment to ACR](#deployment-to-acr)
  - [Evaluation](#evaluation)
- [Environment Variables](#environment-variables)
- [Common Tasks](#common-tasks)
- [Development Tips](#development-tips)
- [External References](#external-references)

---

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
| `src/agentcore_rl_toolkit/client.py` | `RolloutClient` and `RolloutFuture` for training integration and batch evaluation |
| `src/agentcore_rl_toolkit/reward_function.py` | `RewardFunction` base class |
| `src/agentcore_rl_toolkit/rollout_gateway/` | In-repo token-level trajectory capture layer: `RolloutGateway`, `Renderer`, `SamplingBackend`, `TraceRecord` (see [Rollout Gateway](#rollout-gateway)) |
| `src/agentcore_rl_toolkit/sandbox/` | Sandbox SDK: `SandboxClient`, `Sandbox`, `ExecResult` — run shell commands in arbitrary images on ACR (see [Sandbox SDK](#sandbox-sdk)) |
| `sandboxd/` | Go health shim (`agentcore-sandboxd`) that makes arbitrary Docker images satisfy the ACR container contract |
| `examples/strands_math_agent/` | GSM8K math agent example |
| `examples/strands_migration_agent/` | Java migration agent example |
| `examples/strands_officebench_agent/` | OfficeBench office automation agent example |
| `examples/strands_appworld_agent/` | AppWorld API interaction agent example |

---

## Project Structure

```
agentcore-rl-toolkit/
├── src/agentcore_rl_toolkit/
│   ├── __init__.py                 # Public exports
│   ├── app.py                      # AgentCoreRLApp base class
│   ├── client.py                   # RolloutClient for batch evaluation
│   ├── reward_function.py          # RewardFunction base class
│   ├── sandbox/                    # Sandbox SDK (sync client for command execution)
│   │   ├── client.py               # SandboxClient (start/attach) + Sandbox (exec/terminate)
│   │   └── types.py                # ExecResult, SandboxProtocolError
│   └── rollout_gateway/            # Token-level trajectory capture layer (trainer-side)
│       ├── trace.py                # TraceRecord — torch-free output boundary
│       ├── trajectory.py           # TrajectoryManager — per-session message tree
│       ├── render.py               # Renderer protocol; HfTemplateRenderer, TinkerRenderer
│       ├── parsing.py              # tool/reasoning output parsing (sglang optional)
│       ├── gateway.py              # RolloutGateway — assembles the serving unit
│       ├── adapters/               # OpenAI + Anthropic wire protocol adapters
│       └── sampling_backends/      # SamplingBackend impls (vLLM/SGLang HTTP, Tinker SDK)
├── examples/
│   ├── strands_math_agent/         # GSM8K example
│   │   ├── .bedrock_agentcore/     # Dockerfiles for deployment
│   │   ├── basic_app.py            # Production agent
│   │   ├── rl_app.py               # RL-adapted agent
│   │   ├── reward.py               # GSM8KReward implementation
│   │   └── pyproject.toml          # Example-specific dependencies
│   ├── strands_migration_agent/    # Java migration example
│   │   ├── rl_app.py               # RL-adapted migration agent
│   │   ├── evaluate.py             # Batch evaluation script (sync)
│   │   ├── evaluate_async.py       # Batch evaluation script (async)
│   │   ├── reward.py               # MigrationReward implementation
│   │   └── pyproject.toml          # Example-specific dependencies
│   ├── strands_officebench_agent/  # OfficeBench example
│   │   ├── dev_app.py              # RL-adapted office automation agent
│   │   ├── evaluate.py             # Batch evaluation script
│   │   ├── reward.py               # OfficeBenchReward implementation
│   │   ├── tools.py                # Office automation tools
│   │   └── pyproject.toml          # Example-specific dependencies
│   ├── strands_appworld_agent/    # AppWorld example
│   │   ├── rl_app.py               # RL-adapted AppWorld code agent
│   │   ├── evaluate.py             # Async batch evaluation script
│   │   ├── deploy.py               # Deploy container to AgentCore
│   │   ├── reward.py               # AppWorldReward implementation
│   │   ├── Dockerfile              # ACR container (AppWorld data baked in)
│   │   └── pyproject.toml          # Example-specific dependencies
│   └── sandbox_quickstart/         # Sandbox SDK example (no agent server)
│       ├── Dockerfile              # debian-slim + prebuilt sandboxd binary
│       ├── build_and_push.sh       # binary + image + ECR push in one command
│       ├── deploy.py               # create the sandbox runtime (temp scaffolding)
│       ├── run_sandbox.py          # start -> exec -> terminate demo
│       └── README.md               # binary build + deploy + run walkthrough
├── sandboxd/                       # Go health shim (self-contained module, stdlib only)
│   ├── main.go                     # /ping + /invocations busy-state server
│   ├── main_test.go
│   └── build.sh                    # cross-compile (arm64 default; docker fallback)
├── tests/
│   ├── test_rollout_entrypoint.py
│   ├── test_client.py
│   ├── test_async_client.py
│   └── sandbox/                    # Sandbox SDK tests (mocked boto3)
├── scripts/
│   └── build_docker_image_and_push_to_ecr.sh
├── pyproject.toml
└── uv.lock
```

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

When deploying an agent on ACR, developers follow the [HTTP protocol contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html#container-requirements-http):
- `/invocations` endpoint: Receives requests and processes them through agent logic
- `/ping` endpoint: Health checks for AWS monitoring

AWS provides `BedrockAgentCoreApp` from the [bedrock-agentcore-sdk-python](https://github.com/aws/bedrock-agentcore-sdk-python):

**BedrockAgentCoreApp features:**
- HTTP service wrapper with `/invocations`, `/ping`, `/ws` endpoints
- Built-in logging, error handling, and session management

**Key Decorators:**
- `@app.entrypoint` - Define your agent's main logic
- `@app.websocket` - WebSocket handler for bi-directional streaming
- `@app.ping` - Custom health checks
- `@app.async_task` - Background processing

**Example:**

```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator

app = BedrockAgentCoreApp()

load_dotenv()

model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")

agent = Agent(
    model=model,
    tools=[calculator],
    system_prompt=(
        "Your task is to solve the math problem. "
        + "Use calculator when applicable. "
        + 'Let\'s think step by step and output the final answer after "####".'
    ),
)


@app.entrypoint
def invoke_agent(payload):
    """
    Invoke the agent with a payload
    """
    user_input = payload.get("prompt")

    print("User input:", user_input)

    response = agent(user_input)

    return response.message["content"][0]["text"]


if __name__ == "__main__":
    app.run()
```

More details can be found at https://aws.github.io/bedrock-agentcore-starter-toolkit/user-guide/runtime/overview.html.

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
It is the successor to the `rllm-model-gateway` dependency used by the current
`backends/{slime,verl}` integrations.

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
| `TrajectoryManager` | `trajectory.py` | Per-session message **tree**. Handles multi-turn concatenation, parallel tool-call branches, and heals re-tokenization drift between turns (CLEAN / REALIGN / FORK). Tokenizer-free and torch-free. |
| `Renderer` | `render.py` | Tokenization seam. `HfTemplateRenderer` (default, HF `apply_chat_template`) or `TinkerRenderer` (needs `tinker-cookbook`, installed manually). |
| `SamplingBackend` | `sampling_backends/` | The one per-engine seam: `token_ids -> token_ids + logprobs` as a `TurnRecord`. Impls: `VllmHttpBackend`, `SglangHttpBackend`, `TinkerSdkBackend`. |
| Adapters | `adapters/` | Wire-protocol translation: `OpenAIAdapter` (`/v1/chat/completions`), `AnthropicAdapter` (`/v1/messages`). An agent drives the gateway in its *native* protocol unmodified (just point `base_url` at it); both normalize to one canonical message form and share one `TrajectoryManager`. |
| `RolloutGateway` | `gateway.py` | Assembles tokenizer + renderer + backend + adapters onto one aiohttp app sharing one `TrajectoryManager`. Session identity rides in the api-key / Bearer slot; `base_url` is a fixed gateway address (no per-session URLs). |

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
- `pip install agentcore-rl-toolkit[gateway]` → `aiohttp` + `transformers`.
- Tool/reasoning parsing uses a dependency-free regex (common `<tool_call>` format) and
  `</think>` split; no engine parser dependency. A caller needing engine-grade parsing
  for an exotic format installs `sglang`/`vllm` and passes a parser name explicitly.
- For the Tinker backend (`TinkerSdkBackend` + `TinkerRenderer`), install `tinker` and
  `tinker-cookbook` manually — they require Python ≥3.11, so they are not declared as an
  extra (this package supports ≥3.10). Both pull torch.

The core (`TraceRecord`, `TrajectoryManager`, `Renderer` protocol, `SamplingBackend`
protocol) imports torch-free and aiohttp-free; `RolloutGateway` is exposed lazily so
importing the package never requires aiohttp. Tests live in `tests/rollout_gateway/`.

**Status.** The capture layer above is implemented and tested. Wiring it into a training
backend (the per-backend rollout function + `TraceRecord → native sample` conversion, and
the ACR dispatch/reward-join glue) is not yet on the main branch — a prototype dispatcher
is parked on the `wip/online-rl-dispatch` branch until a backend consumes it end-to-end.

**Vendored from slime (upstream baselines).** Several files are adapted from
[slime](https://github.com/THUDM/slime) (Apache-2.0; see `NOTICE`). To check what changed
upstream before re-syncing, diff the source file against the baseline commit below:

| This repo | slime source | Baseline commit |
|---|---|---|
| `rollout_gateway/trajectory.py` | `slime/agent/trajectory.py` | `90c212b5` |
| `rollout_gateway/adapters/{common,openai,anthropic}.py` | `slime/agent/adapters/` | `90c212b5` |
| `rollout_gateway/parsing.py` | `slime/agent/parsing.py` | `90c212b5` |

Re-sync workflow: `git -C <slime> diff 90c212b5..HEAD -- slime/agent/<file>` shows upstream
changes since the lift. Our copies are intentionally modified (torch-free; `Sample` →
`TraceRecord`; injected backend/renderer seams; sglang parser hook removed), so treat the diff
as a review aid, not an automatic merge. Bump the baseline commit here when you re-sync.

### Sandbox SDK

`src/agentcore_rl_toolkit/sandbox/` runs shell commands in **arbitrary Docker images**
(e.g. SWE-bench-style coding environments) deployed as ACR runtimes — the substrate for
coding-agent evaluation and RL rollouts.

**How it works.** Most coding-environment images are not HTTP agent servers, but ACR
requires containers to expose `/ping` and `/invocations` on port 8080. The bridge is
`agentcore-sandboxd` (Go, stdlib-only, source in `sandboxd/`): a health shim added to the
image that manages the Healthy/HealthyBusy ping state. Command execution does NOT go
through the shim — the client uses ACR's native `InvokeAgentRuntimeCommand` API, which
runs shell commands in the same session/container and streams back stdout/stderr/exit code.

```python
from agentcore_rl_toolkit.sandbox import SandboxClient

client = SandboxClient(runtime_arn="arn:aws:bedrock-agentcore:...:runtime/...")
with client.start() as sb:                      # session starts, ping -> HealthyBusy
    result = sb.exec("cd /app && pytest -q", timeout=900)
    result.exit_code, result.stdout, result.stderr, result.timed_out
# __exit__ -> terminate(): ping -> Healthy, then StopRuntimeSession
sb = client.attach(session_id)                  # reconnect to a live session
```

**Key semantics:**

- **Nonzero exit and timeout are data, not exceptions** (`ExecResult.exit_code`,
  `ExecResult.timed_out` with partial output). Exceptions are reserved for infrastructure
  failures: `ClientError`/`EventStreamError` propagate; `SandboxProtocolError` means the
  deployed container isn't behaving like sandboxd (wrong image) or the stream was invalid.
- **Commands are stateless** — each `exec()` runs in a fresh process. `cwd=`/`env=` params
  are composed into the command string per call (`cd ... && export ... && <command>`).
- **The command API does not invoke a shell itself** (it word-splits argv-style), so the
  client wraps every command in `<shell> -c '...'` on the wire — default `/bin/sh` for
  arbitrary-image portability, overridable via `SandboxClient(shell=...)`/`exec(shell=...)`.
  Users just write shell strings; pipes, `;`, and `$VAR` work.
- **Verbs**: `start`/`attach`/`terminate` are the per-session data plane. `create` is
  reserved for future control-plane provisioning (`CreateAgentRuntime` from an ECR image).
- `terminate()` is idempotent and best-effort: it flips the ping to Healthy first so the
  idle reaper collects the session even if `StopRuntimeSession` fails.
- The base image must contain a shell (`InvokeAgentRuntimeCommand` executes shell
  commands); scratch/distroless images won't work.
- Sync-only today. Planned next phases: TTL leak protection + async client (aiobotocore),
  then detached exec (`spawn`/`ExecHandle`), interactive shells, file transfer.

**Building the shim binary** (static, cross-compiled; works on x86 hosts, falls back to a
golang container if Go isn't installed):

```bash
sandboxd/build.sh                 # -> sandboxd/dist/agentcore-sandboxd-linux-arm64
sandboxd/build.sh --stage examples/sandbox_quickstart   # also copy into a build context
```

See `examples/sandbox_quickstart/` for the full walkthrough (wrap image → push to ECR →
create runtime → run). Python tests: `tests/sandbox/` (boto3 fully mocked). Go tests:
`cd sandboxd && go test -race ./...` (CI: `.github/workflows/sandboxd.yml`).

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
- `api_key` is set to `"EMPTY"` — the standard vLLM convention for servers that don't require authentication
- Model and agent are created per-invocation inside the entrypoint
- This gives flexibility for the training engine to pass runtime configuration (inference address, sampling parameters, system prompt, etc.) to accommodate different learning scenarios
- This is safe because RL rollouts are single-invocation — the agent doesn't need persistent conversation history across requests, so there's no need to keep model/agent as global state

```diff
- model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
- agent = Agent(model=model, tools=[calculator], system_prompt="...")

- @app.entrypoint
- def invoke_agent(payload):
-     response = agent(user_input)
+ @app.rollout_entrypoint
+ def invoke_agent(payload: dict):
+     base_url = payload["_rollout"]["base_url"]
+     model_id = payload["_rollout"]["model_id"]
+     params = payload["_rollout"].get("sampling_params", {})
+     model = OpenAIModel(client_args={"api_key": "EMPTY", "base_url": base_url}, model_id=model_id, params=params)
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

### Building and Pushing Docker Images

```bash
# Ensure .env is configured with AWS_REGION, AWS_ACCOUNT, ECR_REPO_NAME
./scripts/build_docker_image_and_push_to_ecr.sh \
  --dockerfile=examples/strands_math_agent/.bedrock_agentcore/strands_math_agent_rl/Dockerfile \
  --tag=my-tag \
  --context=examples/strands_math_agent
```

### Running an Example Locally

```bash
cd examples/strands_math_agent
uv sync
uv run python rl_app.py
```

---

## Development Tips

### Package Management

- This package uses **uv** for dependency management
- All dependencies are installed in `.venv` at each level
- You can inspect source code of dependencies in `.venv/lib/python*/site-packages/`

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

### Finding Source Code

When source locations are unclear:
```python
import module_name
print(module_name.__file__)  # Shows the file path
```

### Pre-commit Hooks

This repo uses pre-commit hooks that run automatically on `git commit`:
- **ruff**: Linting and auto-formatting (will fix issues automatically)
- **commitizen**: Enforces [Conventional Commits](https://www.conventionalcommits.org/) format (e.g., `feat:`, `fix:`, `docs:`)
- Standard checks: trailing whitespace, YAML/TOML validation

To install hooks locally:
```bash
uv run pre-commit install
```

### Code Conventions

- Return a JSON-serializable dict from `@rollout_entrypoint` (any structure accepted — no required keys)
- Create model and agent inside the entrypoint function (not at module level) so config comes from the `_rollout` payload
- Use standard `OpenAIModel` for OpenAI-compatible inference endpoints (token capture during training is handled at the infrastructure layer)
- Implement reward functions as classes inheriting `RewardFunction`

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
- **rLLM SDK (reference)**: https://rllm-project.readthedocs.io/en/latest/core-concepts/sdk/#1-define-your-agent-function
- **rllm-model-gateway** (token capture proxy for RL training): https://github.com/rllm-org/rllm/tree/main/rllm-model-gateway | [PyPI](https://pypi.org/project/rllm-model-gateway/)
- **AgentCore math training example** (rllm + Tinker backend): https://github.com/rllm-org/rllm/blob/main/examples/agentcore_math/train_agentcore_math_tinker.sh
