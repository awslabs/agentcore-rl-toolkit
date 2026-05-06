---
title: "Core API rename: rollout-server / rollout-launcher / rollout-config"
description: "ART's public surface uses generic names (app.py, client.py,"
---
# Core API rename: rollout-server / rollout-launcher / rollout-config

## Summary

ART's public surface uses generic names (`app.py`, `client.py`,
`RolloutClient`, `payload["_rollout"]`) that don't convey the
two-sided architecture. This roadmap renames module files, collector-side
classes, and the rollout payload schema in a single breaking change, so
the `ls` view, the `import` view, and the rollout-body view all use
the same vocabulary.

**Breaking change, one PR, one version bump (0.2.0).** No shims, no
deprecation warnings — users update imports / payload keys or pin
`<0.2`. At this stage of the toolkit's adoption, the cost of carrying
deprecation scaffolding outweighs the migration burden.

`AgentCoreRLApp`, `RolloutFuture`, and `RewardFunction` stay
unchanged.

### Changes at a glance

**Module files:**

| Today | Renamed |
|---|---|
| `app.py` | `rollout_server.py` |
| `client.py` | `rollout_launcher.py` |
| `reward_function.py` | `reward.py` |

**Public classes:**

| Today | Renamed |
|---|---|
| `RolloutClient` | `RolloutLauncher` |
| `BatchResult` | `RolloutBatch` |
| `AsyncBatchResult` | `AsyncRolloutBatch` |
| `BatchItem` | `RolloutResult` |

**Rollout payload:**

| Today | Renamed |
|---|---|
| `payload["_rollout"]` | `payload["rollout_config"]` |
| _(user hardcodes `api_key="EMPTY"`)_ | `rollout_config["api_key"]` (backend-injected) |

---

## Why

**Generic filenames.** `app.py` / `client.py` could live in any
Python project; nothing in the directory listing indicates ART's
two-sided architecture.

**Generic class names.** `RolloutClient` reads as an HTTP client;
`BatchItem`/`BatchResult` read as generic dataclasses and bury the
`run_batch` relationship. The `Item → .result → Result` hop is
confusing.

**Misleading payload conventions.** `_rollout` is the one required
public field in every rollout entrypoint, but the leading underscore
signals "private, don't touch." And the name doesn't describe the
contents — it's a settings bundle (`base_url`, `model_id`,
`sampling_params`, `session_id`, `input_id`, `exp_id`, `s3_bucket`),
not a rollout.

**`api_key="EMPTY"` hardcoded literal.** The OpenAI SDK requires a
non-empty `api_key`; the vLLM/SGLang ecosystem convention is to pass
`"EMPTY"` when the target doesn't check auth. Not ART's invention,
but surfacing in every user's code is avoidable. Backend injection
(`"EMPTY"` for local; real key for Tinker) keeps the literal out of
user files.

**One rename, not three.** Splitting across PRs leaves users looking
at renamed classes inside `app.py`, or old payload keys in renamed
files. Coherence requires a single pass.

## Post-rename mental model

> ART has two sides of one wire.
>
> The **rollout side** runs on ACR: an `AgentCoreRLApp` with a
> `@rollout_entrypoint` handler (in `rollout_server.py`). It reads
> per-rollout settings from `payload["rollout_config"]` (`base_url`,
> `model_id`, `api_key`, `sampling_params`), runs the agent,
> computes a reward, returns a dict. Results save to S3.
>
> The **launcher side** runs in your trainer or evaluator: a
> `RolloutLauncher` submits rollouts and collects results (in
> `rollout_launcher.py`). `launcher.invoke(...)` returns a
> `RolloutFuture`; `launcher.run_batch(...)` returns a `RolloutBatch`
> whose entries are `RolloutResult`s.
>
> You subclass `RewardFunction` (in `reward.py`) for scoring, called
> inside the rollout.

`ls src/agentcore_rl_toolkit/` after the rename:

```
__init__.py
rollout_server.py       # AgentCoreRLApp
rollout_launcher.py     # RolloutLauncher, RolloutFuture, RolloutBatch, ...
reward.py               # RewardFunction
backends/
frameworks/
```

Class-level asymmetry (`App` vs `Launcher`) is kept deliberately:
the App subclasses an AWS-managed runtime; the Launcher is a plain
Python object. File-level symmetry (`rollout_server.py` vs
`rollout_launcher.py`) shows the pairing on `ls`.

## Migration

The exact renames are already listed in the [Changes at a glance](#changes-at-a-glance) tables. Two points worth stating explicitly:

- **Missing `api_key` in `rollout_config` raises**, not defaults. Trainers set it themselves — slime ships `"EMPTY"` (gateway in front of unauth'd SGLang); rllm ships a real key when routing through Tinker.
- `AgentCoreRLApp`-level import paths stay at the package root: `from agentcore_rl_toolkit import AgentCoreRLApp` keeps working; only `from agentcore_rl_toolkit.app import …` breaks.

Users who can't migrate pin `agentcore-rl-toolkit<0.2`. At this adoption stage we don't ship a dedicated migration guide — the Changes-at-a-glance tables plus release notes are enough.

## Backend trainer integration impact

The rename touches three backend integrations differently depending on
where their code lives. Summary table first, details below.

| Backend | Code location | Touched by this PR? | Required downstream work |
|---|---|---|---|
| **slime** | in-tree (`src/agentcore_rl_toolkit/backends/slime/`) | ✅ yes | none — bundled in the 0.2.0 PR |
| **rllm** | out-of-tree ([rllm-org/rllm](https://github.com/rllm-org/rllm), `rllm/experimental/engine/remote_runtime/agentcore_runtime.py`) | ❌ no | ~5-line patch in rllm repo; land there when they bump their ART pin to `>=0.2` |
| **verl** | not integrated yet | ❌ no | pending investigation — patch to be integrated into the art codebase as part of the verl integration work |

### slime (in-tree)

Two touch sites, both handled by the rename PR: `RolloutClient` →  `RolloutLauncher` in `backends/slime/integration/rollout.py` (import + constructor). Slime never touches `payload["_rollout"]` directly — it passes kwargs through `invoke_async(...)`, so the payload-key flip is contained in the renamed `rollout_launcher.py` that slime consumes. Slime-backend users bump their `agentcore-rl-toolkit` pin and keep going.

### rllm (out-of-tree)

Lives in [rllm-org/rllm](https://github.com/rllm-org/rllm) at `rllm/experimental/engine/remote_runtime/agentcore_runtime.py`. File a tracking issue on that repo when 0.2.0 ships; changes needed there:

1. `RolloutClient` → `RolloutLauncher` (import + constructor).
2. Audit for any direct `payload["_rollout"]` writes (shouldn't exist — the launcher builds it).
3. For Tinker/real-API-key paths, pass via `invoke(..., api_key=<key>)` so it lands in `rollout_config["api_key"]`.

Until they update, rllm pinned to `agentcore-rl-toolkit<0.2` continues to work.

### verl (pending investigation)

Not integrated in-tree or out-of-tree today. When integration work starts, the adapter gets written against 0.2+ directly and lands under `src/agentcore_rl_toolkit/backends/verl/`. Open a tracking issue for scope visibility; this PR neither blocks nor pre-empts it.

## Out of scope

- Renaming `AgentCoreRLApp`, `RolloutFuture`, `RewardFunction`.
- Changing method names on the launcher.
- Renames for backend module paths (e.g. `agentcore_rl_toolkit.backends.slime.integration.rollout.generate_rollout`) — those are used as strings in user `train.sh` scripts and in out-of-tree backends' code.
- Eliminating `"EMPTY"` itself. The SDK still requires non-empty
  `api_key`; we only stop surfacing the literal in user code.
- Promoting `rollout_config` to a `TypedDict` or dataclass.

## Open questions

1. `RolloutResult.result` — keep `.result` or rename to `.data` to
   avoid `rollout_result.result`? Recommend: keep `.result`; revisit
   if users complain.
2. `api_key` naming — bare or scoped (`inference_api_key`)? Recommend:
   bare; nesting under `rollout_config` makes scope clear and matches
   the OpenAI-SDK argument name.
3. Structured payload — promote to `TypedDict`? Recommend: stay dict
   for 0.2; consider follow-up if users ask for autocomplete.

---

## Task checklist (single PR, 0.2.0)

**Core renames**
- [ ] `git mv src/agentcore_rl_toolkit/app.py → rollout_server.py`
      (keeps `AgentCoreRLApp`).
- [ ] `git mv src/agentcore_rl_toolkit/client.py → rollout_launcher.py`
      and rename `RolloutClient` → `RolloutLauncher`,
      `BatchResult` → `RolloutBatch`,
      `AsyncBatchResult` → `AsyncRolloutBatch`,
      `BatchItem` → `RolloutResult`.
- [ ] `git mv src/agentcore_rl_toolkit/reward_function.py → reward.py`
      (keeps `RewardFunction`).
- [ ] Update `agentcore_rl_toolkit/__init__.py` — only new names in
      `__all__`.

**Payload**
- [ ] `AgentCoreRLApp.rollout_entrypoint`: read `rollout_config` only;
      raise `KeyError` if missing `api_key`.
- [ ] `RolloutLauncher`: inject `rollout_config` (with `api_key`) on
      send; drop the `_rollout` code path.

**Consumers (same PR)**
- [ ] Update examples: `strands_math_agent`, `strands_appworld_agent`,
      `strands_migration_agent`, `strands_officebench_agent`.
  - [ ] Swap imports and class names.
  - [ ] Swap `payload["_rollout"]` → `payload["rollout_config"]`.
  - [ ] Replace `api_key="EMPTY"` literal with `cfg["api_key"]`.
- [ ] Update the in-tree slime backend
      (`backends/slime/integration/rollout.py`): 2-line import/constructor
      swap.
- [ ] File a tracking issue on rllm-org/rllm (see [rllm section](#rllm-out-of-tree)).

**Docs**
- [ ] Update `docs/site/scripts/gen_api.py` `MODULES` allowlist.
- [ ] Regenerate API reference (`pnpm gen:api`); commit diff.
- [ ] Update the Overview guide with the new mental-model paragraph.
- [ ] Update the prepare-agent-for-RL guide's canonical code block.

**Release**
- [ ] Bump version to `0.2.0` in `pyproject.toml`.
- [ ] Release notes for 0.2.0 (breaking change) — call out the renames
      + `rollout_config` payload key.

**Tests**
- [ ] New imports resolve (`from agentcore_rl_toolkit import
      RolloutLauncher`).
- [ ] Old imports fail loudly (`ImportError`, not silent alias).
- [ ] `payload["rollout_config"]` round-trips launcher → server.
- [ ] Missing `api_key` raises a clear `KeyError`, not defaults to
      `"EMPTY"` silently.

## Milestones

1. **M1** — spike in a throwaway branch; size the diff.
2. **M2** — PR lands on `main`; 0.2.0 released.

M1 ≈ half day. M2 ≈ 1 day.
