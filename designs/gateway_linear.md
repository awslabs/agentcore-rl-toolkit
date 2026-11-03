# Linear-history mode for the rollout gateway (generate-time prefix healing)

| Field | Value |
| --- | --- |
| Status | Accepted |
| Implementation | Shipped |
| Date | 2026-08-31 |
| Pull request | [#119](https://github.com/awslabs/agentcore-rl-toolkit/pull/119), [#143](https://github.com/awslabs/agentcore-rl-toolkit/pull/143) |

## Summary

`history_mode="linear"` is an opt-in gateway mode for agents where cosmetic
re-tokenization drift of a previously-generated assistant turn is
out-of-distribution-insignificant. In that mode the gateway heals drift at generation time:
before each `backend.generate` it substitutes the exact token ids it already served for the
earlier assistant turns, so the model always generates on the canonical, drift-free context.
Each recorded prompt then extends the turn it continues exactly, so every branch of the
`TrajectoryManager` tree stays CLEAN. An append-only session is one training sample, with
every generated turn trained on its true multi-turn context; retries, edits, parallel
branches, and sub-agent forks that reuse the session id add one sample per branch.

Default behavior (`history_mode="tree"`) is unchanged.

The design is implemented in `src/agentcore_rl_toolkit/rollout_gateway/linear.py`
(`LinearHealer`), wired through `RolloutGateway` and `BaseAdapter`, with anchors read from
`TrajectoryManager.heal_anchor`.

## Motivation

In tree mode, the gateway renders and encodes the whole conversation the agent replays.
A prior **assistant** turn's served tokens came from generation; rendering and encoding
its parsed message dict (`text` + `tool_calls`) can produce different IDs due to
whitespace, tool-call JSON spacing, or reasoning re-keying. `TrajectoryManager`
(`rollout_gateway/trajectory.py`) detects that drift and, to stay safe for harnesses that
may branch or edit history, resolves it by FORK (split the rollout into a second sample).

For a multi-turn tool-calling agent whose history is append-only, that safety is pure
overhead. On a real multi-turn SWE-agent run (OpenHands driving Qwen3-Coder-30B, forking on
token drift) the large majority of sessions forked into several records each, even though
every rollout was append-only. Each fork re-emits the shared prefix as `loss_mask=0` context
and splits the trajectory, so it is a large, avoidable training-efficiency loss.

## Design

**Template rendering** (`render_text` in pseudocode) converts messages to text via
`apply_chat_template(..., tokenize=False)`. **Encoding**, also called tokenization,
converts text to token IDs (`encode`). The APIs `Renderer.render()` and
`HfTemplateRenderer.render_delta()` combine these steps and return token IDs.

Because the gateway controls the tokens handed to the backend, it can splice the exact ids
it already served for prior turns over IDs recomputed from the client's messages **before**
generation:

```
anchor = manager.heal_anchor(sid, messages)    # deepest generated turn the request matches
fed    = anchor.served_ids + close + delta_tail(messages[len(anchor.messages):])
out    = backend.generate(fed)
manager.record_turn(sid, TurnRecord(prompt_ids=fed, output_ids=out), ...)
```

`anchor.served_ids` is the anchor turn's `prompt_ids + output_ids`, `close` is the
template's between-message glue, and `delta_tail` is the genuinely-new tokens (new
observation messages + generation prompt). The new turn is recorded below the anchor, and
its prompt extends the anchor's served ids exactly, so the manager appends it to that
branch without forking.

### Anchoring on the tree

`TrajectoryManager.heal_anchor` walks the session tree with the same message matching
`record_turn` mounts by — role plus dict equality, which is order-independent — then moves
up to the nearest generated assistant node. Its root-path messages and tools schema come
back with its served ids; `record_turn` stores the tools schema on each generated node for
this purpose. Because each turn's prompt ids were themselves healed, every branch carries
the served ids of all generated turns on its path. The manager's existing tree
linearization emits one record per branch and trains a shared turn on one branch only.

The healer keeps no per-session history. The tree that `record_turn` advances is the only
state, so a failed or disconnected generation, which is never recorded, cannot advance it.

For an append-only harness the anchor is always the session's latest turn. Other harnesses
anchor deeper in the tree:

| Request | Anchor |
| --- | --- |
| Next turn of an append-only history | latest turn |
| Retry of turn N | turn N-1 |
| Edit of a message after turn K | turn K |
| Return to an older branch | that branch's last matching turn |
| First turn, or a new root (e.g. a sub-agent with its own first message) | none: full render |
| Tools schema differs from the anchor's | full render |

Matching at dict equality (rather than by a raw token-prefix check on the client's render)
keeps the healer in agreement with `TrajectoryManager`. A harness may replay a prior tool
call with its arguments re-keyed or its JSON re-spaced: the resulting message is still `==`
to what the gateway stored, but it renders to different tokens under an order-sensitive chat
template. A token-prefix check would treat that cosmetic re-serialization as non-linear.

### Computing `delta_tail`

`delta_tail` is rendered from the gateway's **own** stored messages, so no hand-rolled
delimiter logic is needed and the client's re-serialization of prior turns never enters the
fed tokens:

```
new        = messages[len(anchor.messages):]
text_prev  = render_text(anchor.messages,       tools=anchor.tools, add_generation_prompt=False)
text_ext   = render_text(anchor.messages + new, tools=anchor.tools, add_generation_prompt=True)
assert text_ext.startswith(text_prev)
delta_tail = encode(text_ext[len(text_prev):])  # new messages + generation prompt
```

Incremental healing renders four texts: the two histories above and two closer probes
described below. Only the closer and new tail are encoded into token IDs.

The tail is sliced from text because the same append boundary may not exist in token
space. For example, the Qwen3-Coder tokenizer encodes `\n` as `[198]` but `\n\n` as
`[271]`, so a newline on each side becomes one token when encoded together. Encoding
the new newline separately lets the healer append it without changing the served ids.

If the text prefix or closer check fails, `render_delta()` returns `None`.
`LinearHealer` then uses `Renderer.render()` to render and encode both histories
(`r_prev`, `r_ext`), check their token prefix, and probe the closer in token space.
Renderers without `render_delta()` also use this path. If full-history healing fails, the
request is rendered in full and counted as `unhealable`.

### Isolating the assistant closer

The anchor's served ids end at the model's last generated content; to rejoin them to
`delta_tail` we reinsert the template's between-message glue (Qwen: `<|im_end|>\n`). That
glue must be **message-type independent** — the same after a text turn and a tool-call turn.
`render_delta()` probes it by rendering the prior turns with the last
assistant replaced by two distinct *text* bodies and taking the common suffix of the two
strings. It checks that the suffix also terminates the real `text_prev`, then encodes
only that short suffix. The full-history fallback uses `LinearHealer._assistant_close`
to perform the same probe and suffix check on token IDs.
Probing with text bodies avoids a tool-call turn's `<tool_call>…</tool_call>` wrapper, which
is itself content-independent and would otherwise be mistaken for the closer.

Two details, both handled: the served output already contains part of the closer (with
`no_stop_trim=True` the served ids end with the stop token `<|im_end|>`), so `_trim_overlap`
drops the prefix of the glue the served ids already end with — the reinserted glue is
just `\n`, never a doubled `<|im_end|>`. And the tool-call wrapper `</tool_call>` is
generated by the model, so it lives in the served output and is preserved verbatim,
correctly not treated as closer.

## Integration

- `rollout_gateway/linear.py` — `LinearHealer`, tokenizer-free and torch-free; it only
  calls the injected `Renderer` and reads anchors from the injected `TrajectoryManager`.
  `heal` (before generation) returns the ids to feed the backend. `make_healer` builds the
  healer for a `history_mode` and rejects unknown modes.
- `rollout_gateway/trajectory.py` — `heal_anchor` and `HealAnchor`; `record_turn` stores
  the tools schema on generated nodes.
- `rollout_gateway/gateway.py` — `RolloutGateway.__init__` takes `history_mode` and
  `linear_on_nonlinear`. In linear mode it builds **one** `LinearHealer` over the shared
  `TrajectoryManager` and injects it into every co-mounted adapter, so a session heals the
  same whichever wire protocol its turns arrive on.
- `rollout_gateway/adapters/common.py` — `BaseAdapter` calls `heal` before
  `backend.generate` in `_run_turn` and feeds the healed ids to both `generate` and
  `TurnRecord`. Session teardown drops per-sid healer counters.

### Config surface

- `history_mode`: `"tree"` (default) | `"linear"`. Gateway-global; sessions of different
  modes do not coexist within a run.
- `linear_on_nonlinear`: `"reset"` (default) | `"error"` | `"passthrough"` — behavior when
  a request does not extend the session's latest turn (see below).

## When the history is not append-only

A request is *non-linear* when it does not extend the session's latest generated turn: a
retry, an edit or drop of a prior message, a parallel branch, a sub-agent with its own
first message, a compaction, or a tools change. `linear_on_nonlinear` selects the behavior:

- `"reset"` (default) — anchor on the deepest generated turn the request still matches, as
  in the table above, or render in full when none matches or its tools schema differs.
- `"error"` — raise and fail the rollout, for runs that require an append-only history.
- `"passthrough"` — stop healing this session and render the rest of its requests in full,
  so the standard tree/FORK path handles them.

Healing from the deepest match rather than from the latest turn only matters for these
requests. Restarting from a full render instead would generate the non-linear turn on the
re-tokenized prior turns and split it from the shared prefix; and a healer that tracked
only the latest turn would also lose the anchor when the agent returns to an older branch.

## Observability

`LinearHealer` tracks `healed_turns` and `healed_prefix_tokens` (turns healed and
served prefix tokens spliced); `nonlinear` (requests that did not extend the session's
latest turn); `unhealable` (anchored requests the template could not splice, rendered in
full); and `close_unresolved` (the subset of `unhealable` where the closer probe couldn't
isolate the glue). A high `nonlinear` rate on a harness expected to be append-only is worth
investigating, and `linear_on_nonlinear="error"` turns it into a failure.

These are kept two ways. `LinearHealer.counters` is the run-cumulative total across every
session the (shared) healer has seen. Per session, `finish_session` pops that session's own
counters (`LinearHealer.pop_stats`, a fixed schema of `LinearHealer.COUNTER_KEYS` with zeros
for counters that never fired) and attaches them to every record's metadata under
`linear_healer`,
riding out on the existing `extra_metadata` → `TraceRecord.metadata` seam — so a downstream
consumer sees each rollout's healing stats without touching the healer.

## Validation

**Unit tests** (`tests/rollout_gateway/test_linear_healer.py`) drive the real `LinearHealer`
+ `TrajectoryManager` through the same heal → generate → record pipeline
`BaseAdapter._run_turn` uses, with a template-shaped fake renderer that reproduces
assistant-only drift: renderer-driven drift forks without healing and collapses to one
sample with it (loss mask covering every generated turn, served logprobs preserved);
multi-turn drift stays one sample; retries, edits, and returns to an older branch anchor
on the right turn and give one CLEAN sample per branch; new roots, tools changes, and
unspliceable templates fall back to the full render; the closer probe isolates the glue for
both text and tool-call turns and does not double the closer when the served output already
ends with the stop token; and the three `linear_on_nonlinear` paths behave as specified. The
template-level assumptions (generation-prompt tokens equal the assistant open; token-prefix
consistency turn to turn; the closer probe returns `<|im_end|>\n` for both text and
tool-call turns and is a genuine suffix of a tool-call `r_prev`; served ids end with the
stop token) were confirmed against the live Qwen3-Coder-30B tokenizer/template.

`tests/rollout_gateway/test_render.py` covers four incremental-path properties with a local
HF/Rust tokenizer: correct delta extraction without encoding old history, multi-turn
trajectory preservation, fallback on incompatible text prefixes or closers, and preservation
of the served token prefix across a BPE merge boundary.

## Assumptions

- Cosmetic re-tokenization drift is treated as OOD-insignificant, so training on the
  canonical/healed context is acceptable.
- A prior message the harness replays must be `==` to what the gateway stored for the
  request to anchor past it. Cosmetic re-serialization (re-keyed tool-call arguments,
  re-spaced JSON) keeps dict equality; a semantic rewrite of a prior message is non-linear
  and anchors on the turn before it.
- Incremental healing requires a stable text prefix and a matching closer. The
  type-independent-closer assumption is validated for Qwen3-Coder-30B; re-check it when
  targeting a template family whose between-message glue could differ after tool-call vs text
  turns (the `close_unresolved` fallback keeps such a case correct, just unhealed).
