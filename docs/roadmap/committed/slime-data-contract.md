---
title: "Slime backend: use-case-independent rollout payload"
description: "Make the JSONL row's `metadata` field the agent payload verbatim, so migration/appworld/officebench agents can train via slime without any slime integration changes."
---
# Slime backend: use-case-independent rollout payload

## Summary

Today `_sample_to_payload` in the slime integration hardcodes the
math-agent shape: it emits `payload["prompt"]` and `payload["answer"]`
(from `sample.prompt` / `sample.label`) plus a nested
`payload["metadata"]` for anything else. That fits GSM8K but not the
three other in-tree examples (appworld needs `task_id`; migration
needs `repo_uri`/`metadata_uri`/...; officebench needs
`task_uri`/`testbed_uri`). Each has a different top-level payload
shape, and there's no way to express that through the current
pipeline.

This plan changes one thing: **the JSONL row's `metadata` dict becomes
the agent payload verbatim**. No top-level key injection, no nesting.
Each agent declares whatever payload shape it wants by choosing what
keys to put in `metadata`.

No new abstractions. No adapter class. No new CLI flag. Just one
simplification in `_sample_to_payload` and matching JSONL shape
examples in the docs.

### Changes at a glance

| Today | Proposed |
|---|---|
| `_sample_to_payload` builds payload from `sample.prompt`, `sample.label`, `sample.metadata`, plus a fall-through copy of Sample fields | `_sample_to_payload` returns `dict(sample.metadata)` — that's it |
| Payload structure: `{"prompt": ..., "answer": ..., "metadata": {...}, "_rollout": {...}}` | Payload structure: `{**metadata, "_rollout": {...}}` — agent-defined keys at the top level |
| Agents read `payload["prompt"]`, `payload["answer"]`, `payload["metadata"]["..."]` (nested) | Agents read `payload["<whatever>"]` directly |
| JSONL shape: `{prompt, label}` hardcoded for math | JSONL shape: `{prompt, metadata: {<agent-specific>}}` uniform across agents |

## Why

**The four in-tree agents want four different payload shapes.** Each
already parses its own `InvocationRequest` Pydantic model, or reads
specific keys like `task_id`. The current slime integration only
delivers `prompt` + `answer` + a bag-of-metadata at the top level,
which doesn't match:

- **math** reads `payload["prompt"]`, `payload["answer"]` — fits current shape.
- **appworld** reads `payload["task_id"]` — needs `task_id` at the top level.
- **migration** does `InvocationRequest(**payload)` — needs `prompt`, `repo_uri`, `metadata_uri`, `require_maximal_migration`, `use_dependency_search_tool`, `apply_static_update` all at the top.
- **officebench** does `InvocationRequest(**payload)` — needs `task_uri`, `testbed_uri` at the top.

Three of four agents can't run today without bespoke data-prep
changes or framework edits.

**The clean fix is to let users own the payload shape.** slime's
Dataset already threads arbitrary dicts through to `Sample.metadata`.
The rollout function just needs to hand that dict to the agent
verbatim. The data author chooses the schema; the framework doesn't
have an opinion.

**Why not keep metadata nested under `payload["metadata"]`?** That's
what the current code does, and it forces every agent to add a
`.get("metadata", {})` hop that serves no purpose. Worse, migration
and officebench use `InvocationRequest(**payload)` — they'd need
`InvocationRequest(**payload["metadata"])`, which breaks the Pydantic
model's self-documentation. Flattening to top-level matches how
agents already want to read the payload.

**Why not drop `prompt` as a top-level JSONL key?** slime uses
`Sample.prompt` for tokenization, length filtering, and SGLang seed
tokens. It has to be a real string at the JSONL top level for slime's
Dataset. If the agent's actual input also happens to be a prompt
string (math, migration), the data author duplicates it inside
`metadata`. That's a small cost for a major simplification.

### What happens when `prompt` appears in both?

slime reads the two fields on different sides of the data layer, so
they never collide:

| Field | Read by | Used for |
|---|---|---|
| top-level `prompt` | slime's Dataset via `prompt_key` | slime's tokenizer (length filter), chat template application, SGLang seed tokens |
| `metadata.prompt` | agent via `payload["prompt"]` | whatever the agent does with it |

**Consequence:** you can legitimately set them to different values —
e.g. a short sentinel top-level `prompt` to minimize slime's length
filter, and a long templated instruction in `metadata.prompt`. Or
identical strings (what the data-prep recipe does for simplicity).
Nothing forces them to match.

**Practical footgun:** if a user duplicates the prompt and later edits
only one, the two silently diverge — the agent sees one string and
slime's length filter sees another. Editorially fragile. The data-prep
recipe in SETUP.md will show a pattern that writes both from the same
source variable to avoid this.

## Post-change JSONL shapes and agent code

The slime-visible field (`prompt`) and the agent-visible payload
(`metadata` contents) are deliberately separate. slime gets a prompt
for its tokenizer; the agent gets whatever it wants.

### math

```jsonl
{"prompt": "Natalia sold clips to 48 friends...",
 "metadata": {"prompt": "Natalia sold clips to 48 friends...", "answer": "72"}}
```

Agent (`rl_app.py`):

```python
user_input = payload["prompt"]
answer = payload["answer"]
```

### appworld

```jsonl
{"prompt": "AppWorld task seed",
 "metadata": {"task_id": "82e20b2_1"}}
```

Agent:

```python
task_id = payload["task_id"]
```

(The top-level `"prompt"` field is a throwaway string slime tokenizes
for length filtering. AppWorld's agent doesn't read it.)

### migration

```jsonl
{"prompt": "Migrate the Java project at {repo_path}...",
 "metadata": {
   "prompt": "Migrate the Java project at {repo_path}...",
   "repo_uri": "s3://migration-bench/repos/abc.tar.gz",
   "metadata_uri": "s3://migration-bench/meta/abc.json",
   "require_maximal_migration": false,
   "use_dependency_search_tool": false,
   "apply_static_update": false
 }}
```

Agent:

```python
request = InvocationRequest(**payload)   # unchanged from today
```

### officebench

```jsonl
{"prompt": "OfficeBench task seed",
 "metadata": {
   "task_uri": "s3://bucket/officebench/1-1/config.json",
   "testbed_uri": "s3://bucket/officebench/1-1/testbed.tar.gz"
 }}
```

Agent:

```python
request = InvocationRequest(**payload)   # unchanged from today
```

## Design

### The one code change

`src/agentcore_rl_toolkit/backends/slime/integration/rollout.py::_sample_to_payload`
collapses to:

```python
def _sample_to_payload(sample) -> dict:
    """The agent payload is the JSONL row's `metadata` dict, verbatim.

    slime's Dataset reads the JSONL row's `metadata` field into
    `Sample.metadata`; we pass it to the agent unchanged. The JSONL's
    top-level `prompt` field is for slime (tokenization, length
    filtering); the agent's payload shape is entirely defined by
    whatever the data author put in `metadata`.
    """
    if hasattr(sample, "metadata") and isinstance(sample.metadata, dict):
        return dict(sample.metadata)   # shallow copy to isolate from Sample state
    return {}
```

Downstream (in `_process_one_episode`), the RolloutLauncher still
injects `_rollout` / `rollout_config` into the payload as today. The
final wire-format is:

```python
{**metadata, "_rollout": {base_url, model_id, sampling_params, ...}}
```

### What goes away

- The `sample.prompt → payload["prompt"]` injection.
- The `sample.label → payload["answer"]` injection (math-specific).
- The fall-through `to_dict()` loop that copies unfiltered Sample
  fields into the payload — which was carrying silent leakage risk
  (e.g. future slime Sample fields landing in agent payloads).
- The nested `payload["metadata"]` wrapping. `metadata` *is* the
  payload now.

### JSONL prep script

SETUP.md's 3.2 prep snippet updates from:

```python
with open(out, "w") as f:
    for i, row in enumerate(ds):
        if i >= 64: break
        answer = row['answer'].split('####')[-1].strip()
        f.write(json.dumps({'prompt': row['question'], 'label': answer}) + '\n')
```

to:

```python
with open(out, "w") as f:
    for i, row in enumerate(ds):
        if i >= 64: break
        answer = row['answer'].split('####')[-1].strip()
        f.write(json.dumps({
            'prompt': row['question'],                              # slime
            'metadata': {'prompt': row['question'], 'answer': answer},  # agent payload
        }) + '\n')
```

### train.sh flag

slime's `--label-key` is no longer needed (nothing consumes
`Sample.label`). Remove from `train.sh`. `--input-key prompt` stays —
slime still tokenizes the top-level prompt.

## Downstream impact

### math agent

**Breaks** unless JSONL is regenerated. The existing
`data/gsm8k_tiny.jsonl` uses `{prompt, label}`; that has no
`metadata` → `Sample.metadata` is empty → payload is empty. Fix: the
prep script update above.

Agent code change: `payload.get("answer")` → `payload["answer"]`
(or no change if `.get` is retained; it still works).

### appworld agent

Already reads `payload["task_id"]`. Needs its JSONL authored with
`{prompt, metadata: {task_id}}`. Zero code change in `rl_app.py`.

### migration agent

Already does `InvocationRequest(**payload)`. Works unchanged once
JSONL is authored as `{prompt, metadata: {<InvocationRequest fields>}}`.

### officebench agent

Same as migration — works unchanged with the new JSONL shape.

### SlimeRunner

No change needed — it just ships `--input-key prompt` and
`--metadata-key metadata` (the latter is already slime's default).

### rllm / verl

No change.

## Effect on slime scheduling and Megatron training

A natural worry: if we move the agent's "real" prompt into metadata,
does slime get confused about what to tokenize, how to batch, or
which samples are too long? Investigation says the effect is
**narrow and bounded**.

### Slime-side scheduling: mostly unaffected

`Sample.prompt` is read by slime in three places; only one affects
scheduling:

| Reader | Affects scheduling? | Notes |
|---|---|---|
| `filter_long_prompt` at Dataset load | **Yes** — filters rows whose tokenized `prompt` exceeds `rollout_max_prompt_len` | Happens once at dataset init, before rollout begins. Short top-level `prompt` → no rows filtered. |
| `slime.rollout.sglang_rollout` (slime's built-in rollout fn) | N/A | We replaced this with our `generate_rollout`; code path is dead on our side. |
| Logging (first-rollout / finish-rollout / rm_hub) | No | Pretty-printing only; no scheduling or training effect. |

Rollout ordering, batching, SGLang engine routing, and concurrency are
all driven by dataset **index**, not prompt content. `data_source.py::
get_samples` does `self.dataset.samples[offset : offset + N]` and then
deep-copies each row `n_samples_per_prompt` times. Prompt text has
zero influence on this path.

**Consequence for the data-contract change:**

- If the top-level `prompt` is short (e.g. a sentinel for appworld/
  officebench), length filtering is effectively disabled — every row
  passes. Safe.
- If the top-level `prompt` is the real instruction (math, migration),
  length filtering works as today — rows whose tokenized instruction
  exceeds the context budget are dropped before rollout.
- If the *agent's* true prompt is long and lives only in `metadata`,
  slime can't see its length. SGLang may then truncate at inference
  time. Not broken, but less clean than pre-filtering. Authors who
  want pre-filtering should put the real prompt at the top level too
  (the `prompt` appears twice pattern).

### Per-turn Samples passed to Megatron: unaffected

Megatron trains on per-turn `Sample` objects built by `trace_to_sample`
in `integration/traces.py`, not on the dataset rows. Those Samples
have their own freshly-constructed fields:

| Sample field | Source | Used by Megatron? |
|---|---|---|
| `tokens` | `TraceRecord.prompt_token_ids + completion_token_ids` (live from gateway) | **Yes** — the tokens trained on |
| `response_length` | `len(completion_token_ids)` | Yes |
| `loss_mask` | `[1] * response_length` | Yes |
| `rollout_log_probs` | `TraceRecord.logprobs` | Yes (off-policy correction) |
| `reward` | `extract_reward(result)` — from agent's S3 output | Yes |
| `group_index`, `index`, `session_id` | bookkeeping | Yes (grouping, logging) |
| `metadata` | `{"task_index", "gateway_session_id", "turn_index"}` + `"task_metadata"` (copied from dataset row) | Only two keys read by Megatron: `raw_reward`, `round_number` (neither set by our path) |
| `prompt`, `label` | Copied from dataset row for logging/traceability only | **No** — Megatron never reads these |

**Megatron's training tensors are synthesized entirely from
gateway-captured traces.** The dataset row's contents — `prompt`,
`metadata`, whatever — do not flow into `tokens`, `loss_mask`, or
gradients. They only affect:

1. What the agent receives as its payload (the primary behavior change).
2. What gets preserved in `Sample.metadata["task_metadata"]` for
   traceability / logging.

### One non-functional nuance: per-turn Sample.prompt / label become less useful for logging

Today, `_process_one_episode` copies the dataset-row's `prompt` and
`label` onto each per-turn Sample (lines 323-325 of `rollout.py`) for
traceability:

```python
for s in samples:
    s.prompt = sample.prompt
    s.label = sample.label
    if sample.metadata:
        s.metadata["task_metadata"] = sample.metadata
```

After the data-contract change:
- `sample.prompt` will be whatever the data author put at the JSONL
  top level (potentially a short sentinel).
- `sample.label` will typically be `None` (the `label_key` CLI arg is
  dropped).

So slime's first-rollout / finish-rollout log lines (which display
`sample.prompt` and `sample.label`) will show the sentinel instead of
the real prompt, and no label. The **real** agent input is preserved
in `sample.metadata["task_metadata"]` — logs that want to show it can
read from there.

This is cosmetic, not functional. No gradients, rewards, or
scheduling decisions depend on the affected fields.

## Backwards compatibility

**Not backward compatible for math's existing JSONL shape.** The
current `{prompt, label}` rows silently become empty-metadata rows
under the new rule, and the agent receives `{"_rollout": {...}}` with
no `prompt`/`answer`. Fix is trivial (regenerate the JSONL with the
new prep script), but it's a breaking change that ships alongside the
code change.

Options:

1. **Break cleanly** — ship the code change + prep-script update
   together as a minor version bump; users regenerate JSONLs.
2. **Parallel-support period** — keep the old `prompt`/`label` injection
   as a fallback when `sample.metadata` is empty. Adds ~6 lines of
   back-compat code; we'd remove it when cleaning up the other 0.2
   changes.

Recommend (2) for one release, then remove in the next minor.

## Out of scope

- **Changing how agents receive `_rollout` / `rollout_config`.** That's
  the RolloutLauncher's contract, unrelated to data shape. Stays as-is.
- **Supporting `prompt` as a list of chat messages.** slime has
  `apply_chat_template=True` for that; out of scope for this plan.
- **Auto-deriving the slime `prompt` from metadata.** Requires a
  slime-side hook. Not worth the lift; data authors duplicate the
  string.
- **Validating agent payload shape against a schema.** Each agent
  already does its own validation (Pydantic, direct key access).
  No new framework layer.
- **Renaming the JSONL top-level `metadata` key.** slime's default is
  `"metadata"`; we'd have to plumb a rename through slime's CLI to
  change it. Not worth the churn.

## Open questions

1. **Hard break vs. parallel-support period?** Recommend parallel
   support for one release; log a deprecation warning when
   `sample.metadata` is empty and `sample.prompt`/`sample.label` are
   non-empty (the old math shape).
2. **Do we remove `_label_key` from `train.sh`?** Yes, with the
   JSONL change. Keeping it costs nothing but adds noise.
3. **Should SETUP.md show all four JSONL shapes or just math?**
   Recommend all four — it anchors the agent-agnostic narrative.

## Task checklist (single PR, 0.2.0)

**Core**

- [ ] `integration/rollout.py::_sample_to_payload`: collapse to the
      3-line `dict(sample.metadata)` version. Delete the
      prompt/label/to_dict fall-through logic.
- [ ] Optional: add a one-release-only fallback that logs a
      `DeprecationWarning` and injects `prompt`/`answer` when
      `sample.metadata` is empty.

**Examples**

- [ ] Regenerate `examples/strands_math_agent/data/gsm8k_tiny.jsonl`
      (and equivalent scripts) with the `{prompt, metadata: {...}}`
      shape.
- [ ] Update math `rl_app.py` to read `payload["prompt"]`,
      `payload["answer"]` (no change, but verify).
- [ ] Update appworld `rl_app.py` — already reads
      `payload["task_id"]`; verify with a sample JSONL.
- [ ] migration `rl_app.py` — `InvocationRequest(**payload)`
      unchanged; verify.
- [ ] officebench `rl_app.py` — same.
- [ ] Add JSONL sample files to all four `examples/` dirs showing
      the expected shape.

**Docs**

- [ ] SETUP.md 3.2: update the data-prep snippet to the new shape,
      show one non-math example (e.g. migration) as a side-by-side
      so the agent-agnostic nature is obvious.
- [ ] Regenerate API reference.

**Tests**

- [ ] Unit: `_sample_to_payload(Sample(metadata={"task_id": "x"}))`
      → `{"task_id": "x"}`.
- [ ] Unit: `_sample_to_payload(Sample(metadata={}))` → `{}`.
- [ ] Unit: `_sample_to_payload(Sample())` (no metadata attr) → `{}`.

**Release**

- [ ] Ship alongside the core-api-rename as 0.2.0 (both are
      breaking; consolidate the user-migration story).

## Milestones

1. **M1** — update `_sample_to_payload`, regenerate math JSONL, run
    3B smoke. Half day.
2. **M2** — update 3 other example JSONLs + SETUP.md. Half day.
3. **M3** — PR review. Calendar.

---

## Appendix A — Identity and `group_index` in slime

This section is background investigation, not part of the proposed
change. It explains how slime assigns identities to Samples today,
so readers can reason about whether metadata-derived `input_id`s
(content hashes, row offsets, etc.) are needed. **Conclusion: not
needed for the data-contract change itself, but useful context if
cross-run identity ever becomes a concrete need.**

### The three IDs in play

```
result_key = f"{exp_id}/{input_id}/{session_id}.json"
             ^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^^^^
             run       which row  which rollout attempt
                       in dataset of that row
```

- `exp_id` — user-chosen run name. Propagated via `SlimeRunner(exp_id=...)`.
- `session_id` — per-attempt UUID from `uuid.uuid4()` in `integration/rollout.py`.
- `input_id` — **currently set equal to `session_id`** in our code
  (`integration/rollout.py:294`), so the "which row" level is
  effectively unused; each attempt looks unique.

### How slime assigns per-Sample identity

`slime.rollout.data_source.PromptDataset` maintains two persistent
counters:

```python
self.sample_group_index = 0   # monotonic across the whole run
self.sample_index = 0         # monotonic across the whole run
```

`get_samples(N)` — called once per rollout step to produce N prompts'
worth of training samples — does:

```python
prompt_samples = self.dataset.samples[offset : offset + N]

samples = []
for prompt_sample in prompt_samples:
    group = []
    for _ in range(args.n_samples_per_prompt):    # fan out K copies per row
        sample = copy.deepcopy(prompt_sample)
        sample.group_index = self.sample_group_index    # shared by all K copies
        sample.index = self.sample_index                # unique per copy
        self.sample_index += 1
        group.append(sample)
    self.sample_group_index += 1                        # next row gets next group_index
    samples.append(group)
```

Both counters are **checkpointed** in `save()` / loaded in
`load_state_dict()`, so resuming a job keeps the numbering monotonic.

### Concrete numbering example

`rollout_batch_size=32`, `n_samples_per_prompt=8`, JSONL has 64 rows:

| Rollout step | JSONL rows consumed | `group_index` range | `sample.index` range |
|---|---|---|---|
| 0 | rows 0..31 | 0..31 | 0..255 (32 × 8) |
| 1 | rows 32..63 | 32..63 | 256..511 |
| 2 (epoch wraps) | rows 0..31 (epoch 1) | 64..95 | 512..767 |
| 3 | rows 32..63 | 96..127 | 768..1023 |

Key property: after the epoch wrap, the **same dataset row gets a
different `group_index`** than its first visit. `group_index` is an
identity of "this *visit* to the row," not "identity of the row
itself."

### What `group_index` is used for

`slime.ray.rollout.py:1248`:

```python
all_sample_groups = group_by(all_samples, lambda s: s.group_index)
```

This is the **GRPO group boundary**. All Samples sharing a
`group_index` are the K rollouts of the same prompt-visit, and GRPO
mean-centers their rewards against each other. So `group_index`
serves *within-rollout-step* identity — it's exactly what GRPO needs
and nothing more.

### Why this doesn't drive the data-contract change

The data-contract proposal only changes **what goes into
`payload["metadata"]`** and what the agent reads. It doesn't touch
how slime assigns `group_index`, `sample.index`, or what slime
records in `metadata["task_metadata"]`. All three counters continue
to work identically.

### Where content-derived IDs would matter (future work, not this plan)

If a future need arises for:

- **Cross-run row identity** ("did exp-124 process the same row-42
  content that exp-123 did?")
- **Cross-epoch row identity** (`group_index` visits differ)
- **Cross-dataset correlation** ("which samples in eval-v2 test the
  same inputs as eval-v1?")

…then a content hash of `sample.metadata` would give a stable ID:

```python
import hashlib, json
input_id = hashlib.sha256(
    json.dumps(sample.metadata, sort_keys=True).encode()
).hexdigest()[:16]
```

The groundwork is laid by the data-contract change — metadata becomes
the agent payload and the authoritative per-row state — so a
content-hash `input_id` becomes trivially implementable later. But it
is explicitly deferred: the current plan preserves
`input_id == session_id` behavior.

### Non-content row identity: `group_index` as a quick win

If someone wants "find all attempts of the same row within a run"
without cross-run stability, swapping:

```python
# in integration/rollout.py::_process_one_episode
input_id=session_id
# →
input_id=f"g{sample.group_index}"
```

…gives S3 keys like `exp-123/g42/abcd.json` where all `n_samples_per_prompt`
attempts of group 42 sort together. Also deferred; just noting the
5-line version exists.
