# Linear-history mode for the rollout gateway (generate-time prefix healing)

| Field | Value |
| --- | --- |
| Status | Accepted |
| Implementation | Shipped |
| Date | 2026-08-31 |
| Pull request | [#119](https://github.com/awslabs/agentcore-rl-toolkit/pull/119), [#143](https://github.com/awslabs/agentcore-rl-toolkit/pull/143) |

## Summary

`history_mode="linear"` is an opt-in gateway mode for agents whose conversation is
strictly **linear and append-only** and where cosmetic re-tokenization drift of a
previously-generated assistant turn is out-of-distribution-insignificant. In that mode the
gateway heals drift at generation time: before each `backend.generate` it substitutes the
exact token ids it already served for the earlier assistant turns, so the model always
generates on the canonical, drift-free context. The recorded trajectory is then append-only
by construction, so `TrajectoryManager` stays permanently CLEAN — one training sample per
session, every generated turn trained on its true multi-turn context, no dropped turns and
no cosmetic forks.

Default behavior (`history_mode="tree"`) is unchanged.

The accepted v1 design is implemented in
`src/agentcore_rl_toolkit/rollout_gateway/linear.py`
(`LinearHealer`), wired through `RolloutGateway` and `BaseAdapter`. Unit-tested in
`tests/rollout_gateway/test_linear_healer.py`; core template assumptions validated against
the real Qwen3-Coder-30B tokenizer. Remaining follow-ups at the end.

## Motivation

In tree mode, the gateway renders and encodes the whole conversation the agent replays.
A prior **assistant** turn's served tokens came from generation; rendering and encoding
its parsed message dict (`text` + `tool_calls`) can produce different IDs due to
whitespace, tool-call JSON spacing, or reasoning re-keying. `TrajectoryManager`
(`rollout_gateway/trajectory.py`) detects that drift and, to stay safe for harnesses that
may branch or edit history, resolves it by FORK (split the rollout into a second sample).

For a strictly linear, multi-turn tool-calling agent that safety is pure overhead. On a
real multi-turn SWE-agent run (OpenHands driving Qwen3-Coder-30B, forking on token drift)
the large majority of sessions forked into several records each, even though every rollout
was linear. Each fork re-emits the shared prefix as `loss_mask=0` context and splits the
trajectory, so it is a large, avoidable training-efficiency loss.

## Design

**Template rendering** (`render_text` in pseudocode) converts messages to text via
`apply_chat_template(..., tokenize=False)`. **Encoding**, also called tokenization,
converts text to token IDs (`encode`). The APIs `Renderer.render()` and
`HfTemplateRenderer.render_delta()` combine these steps and return token IDs.

Because the gateway controls the tokens handed to the backend, it can splice the exact ids
it already served for prior turns over IDs recomputed from the client's messages **before**
generation, so the model always generates on the canonical, drift-free context:

```
fed_{N+1} = served_prefix + close_N + delta_tail
O_{N+1}   = backend.generate(fed_{N+1})
served_prefix = fed_{N+1} + O_{N+1}    # extend the canonical sequence
```

where `served_prefix` is the canonical ids through the last generated turn, `close_N` is the
template's between-message glue, and `delta_tail` is the genuinely-new tokens (new
observation messages + generation prompt). `O_{N+1}` is then sampled on the canonical
context, `TrajectoryManager` sees an append-only sequence, appends the tail, and never forks
— one session, one `TraceRecord`, every generated turn trained on its true multi-turn
context.

### Computing `delta_tail`

Linearity is decided in **message space**, at the same altitude `TrajectoryManager` matches
replayed history: dict equality (which is order-independent), not raw token identity. The
stored `prev_messages` (whose last element is the gateway's own parsed assistant message)
must be a prefix of the incoming `messages` under `==`, and the tools schema must be
unchanged. The genuinely-new messages are then `new = messages[len(prev_messages):]`, and
`delta_tail` is rendered from the gateway's **own** stored messages, so no hand-rolled
delimiter logic is needed and the client's re-serialization of prior turns never enters the
fed tokens:

```
assert messages[:len(prev_messages)] == prev_messages and tools == prev_tools  # linearity check
new        = messages[len(prev_messages):]
text_prev  = render_text(prev_messages,       tools=prev_tools, add_generation_prompt=False)
text_ext   = render_text(prev_messages + new, tools=prev_tools, add_generation_prompt=True)
assert text_ext.startswith(text_prev)
delta_tail = encode(text_ext[len(text_prev):])  # new messages + generation prompt
```

Incremental healing renders four texts: the two histories above and two closer probes
described below. Only the closer and new tail are encoded into token IDs.

The tail is sliced from text because the same append boundary may not exist in token
space. For example, the Qwen3-Coder tokenizer encodes `\n` as `[198]` but `\n\n` as
`[271]`, so a newline on each side becomes one token when encoded together. Encoding
the new newline separately lets the healer append it without changing `served_prefix`.

If the text prefix or closer check fails, `render_delta()` returns `None`.
`LinearHealer` then uses `Renderer.render()` to render and encode both histories
(`r_prev`, `r_ext`), check their token prefix, and probe the closer in token space.
Renderers without `render_delta()` also use this path. If full-history healing fails,
`linear_on_nonlinear` determines whether to reset, raise, or disable healing.

Deciding linearity at the message level (rather than by a raw token-prefix check on the
client's render) is what keeps the healer in agreement with `TrajectoryManager`. A harness
may replay a prior tool call with its arguments re-keyed or its JSON re-spaced: the resulting
message is still `==` to what the gateway stored — so the manager, matching by dict equality,
keeps it linear — but it renders to different tokens under an order-sensitive chat template.
A token-prefix check would wrongly flag that cosmetic re-serialization as non-linear and
reset, needlessly. Matching at dict equality keeps cosmetic drift healed and reserves the
non-linear path for a genuine edit/branch (see *When the assumption breaks*).

### Isolating the assistant closer

The canonical `served_prefix` ends at the model's last generated content; to rejoin it to
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
drops the prefix of the glue the served prefix already ends with — the reinserted glue is
just `\n`, never a doubled `<|im_end|>`. And the tool-call wrapper `</tool_call>` is
generated by the model, so it lives in the served output and is preserved verbatim,
correctly not treated as closer.

## Integration

- `rollout_gateway/linear.py` — `LinearHealer`, the per-sid healer. Tokenizer-free and
  torch-free; it only calls the injected `Renderer`. `heal` (read-only, before generation)
  returns the ids to feed the backend; `commit` advances per-sid state after a turn is
  actually recorded. `heal`/`commit` are split because `commit` needs the served `output_ids`
  (which only exist after generation) and must fire only for turns that are truly recorded —
  a failed or retried generation must not advance the canonical prefix — keeping healer state
  in lockstep with `TrajectoryManager`.
- `rollout_gateway/gateway.py` — `RolloutGateway.__init__` gains `history_mode` and
  `linear_on_nonlinear`. In linear mode it builds **one** shared `LinearHealer` and injects it
  into every co-mounted adapter (like the shared `TrajectoryManager`), so a session's
  canonical state is coherent regardless of which wire protocol its turns arrive on.
- `rollout_gateway/adapters/common.py` — `BaseAdapter` calls `heal` before
  `backend.generate` in `_run_turn`; the healer handles rendering and encoding internally.
  The adapter feeds the healed ids to both `generate` and `TurnRecord`, and calls `commit`
  before `record_turn`. Session teardown drops per-sid healer state.
  The existing `TrajectoryManager` is unchanged.
- `rollout_gateway/__init__.py` — exports `LinearHealer`.

### Config surface

- `history_mode`: `"tree"` (default, current behavior) | `"linear"`. Gateway-global; linear and tree
  sessions do not coexist within a run.
- `linear_on_nonlinear`: `"reset"` (default) | `"error"` | `"passthrough"` — behavior when a
  turn breaks the append-only assumption.

## When the assumption breaks

The message-level linearity check fails when the incoming history is not an append-only
extension of what the gateway stored: a dropped/edited/branched prior message (a genuine
dict change, not merely a re-serialization), context compaction, a changed tools schema, or
an LLM-client retry that re-issues the same prompt and produces a second generation from the
same prefix. The same fallback path also fires when the closer probe cannot isolate the glue
(a message-type-dependent template — `close_unresolved` counter). Behavior is controlled by
`linear_on_nonlinear`:

- `"reset"` (default) — drop per-sid state and re-anchor to the fully rendered and encoded
  prompt, treating the jump as the start of a fresh linear segment. The turns before and after
  each stay drift-free; only the single jump turn conditions on that prompt. Correct for benign,
  agent-controlled jumps; increments a counter so a run that resets constantly is visible.
- `"error"` — raise and fail the rollout, for runs that want a hard guarantee the assumption
  holds.
- `"passthrough"` — stop healing this session and route the rest through the standard
  tree/FORK path (default behavior) for the remainder of the session.

`"reset"` re-anchoring is only sound because we train on the canonical/healed context by
design: the reset turn's slightly drifted context is accepted for the same reason the whole
mode is.

## Observability

`LinearHealer` tracks, per turn: `healed_turns` and `healed_prefix_tokens` (turns healed and
canonical prefix tokens spliced); `nonlinear` (message-level linearity failures — a genuine
history edit/branch or tools change, not cosmetic re-serialization); `close_unresolved`
(turns where the closer probe couldn't isolate the glue and fell back). A high `nonlinear`
rate means the "linear" assumption is wrong for that harness and tree mode is the better fit
— what happened to those jumps is the (known) `linear_on_nonlinear` mode, so there is no
separate reset counter.

These are kept two ways. `LinearHealer.counters` is the run-cumulative total across every
session the (shared) healer has seen. Per session, `finish_session` pops that session's own
counters (`LinearHealer.pop_stats`, a fixed schema of `LinearHealer.COUNTER_KEYS` with zeros
for counters that never fired) and attaches them to every record's metadata under
`linear_healer`, riding out on the existing `extra_metadata` → `TraceRecord.metadata` seam —
so a downstream consumer sees each rollout's healing stats without touching the healer. The
per-session counters survive `reset`/`passthrough` jumps within a session (only per-sid
healing *state* is cleared on a jump), so the final record still reports the full `nonlinear`
total for that session.

## Validation

**Unit tests** (`tests/rollout_gateway/test_linear_healer.py`) drive the real `LinearHealer`
+ `TrajectoryManager` through the same heal → generate → commit pipeline
`BaseAdapter._run_turn` uses, with a template-shaped fake renderer that reproduces
assistant-only drift: renderer-driven drift forks without healing and collapses to one
sample with it (loss mask covering every generated turn, served logprobs preserved);
multi-turn drift stays one sample; the closer probe isolates the glue for both text and
tool-call turns and does not double the closer when the served output already ends with the
stop token; and the three `linear_on_nonlinear` paths behave as specified. The template-level
assumptions (generation-prompt tokens equal the assistant open; token-prefix consistency turn to
turn; the closer probe returns `<|im_end|>\n` for both text and tool-call turns and is a
genuine suffix of a tool-call `r_prev`; served ids end with the stop token) were confirmed
against the live Qwen3-Coder-30B tokenizer/template.

`tests/rollout_gateway/test_render.py` covers four incremental-path properties with a local
HF/Rust tokenizer: correct delta extraction without encoding old history, multi-turn
trajectory preservation, fallback on incompatible text prefixes or closers, and preservation
of the served token prefix across a BPE merge boundary.

## Assumptions and target use case

- The agent harness intends a strictly **linear, append-only** history: it replays prior
  turns without semantically editing them. It may re-serialize a prior assistant turn
  cosmetically — tool-call `arguments` re-keyed, JSON re-spaced — because such a replay is
  still `==` (dict equality) to what the gateway stored, so the message-level linearity check
  keeps it linear; the only turn-to-turn divergence it need not tolerate is a message-level
  rewrite that actually changes a prior dict. The target is a multi-turn, tool-calling SWE
  agent (validated with OpenHands driving Qwen3-Coder-30B); the harness never branches or
  edits prior turns.
- Cosmetic re-tokenization drift is treated as OOD-insignificant, so training on the
  canonical/healed context is acceptable.
- Incremental healing requires a stable text prefix and a matching closer. The
  type-independent-closer assumption is validated for Qwen3-Coder-30B; re-check it when
  targeting a template family whose between-message glue could differ after tool-call vs text
  turns (the `close_unresolved` fallback keeps such a case correct, just unhealed).

## Non-goals

- Message-level rewrites where a prior message's **dict** changes *semantically* (not just
  its key order or JSON spacing): treated as non-linear and re-anchored/failed per
  `linear_on_nonlinear`. Cosmetic re-serialization that keeps the dict `==` is handled — the
  linearity check matches at dict equality — but an actual content change is out of scope for
  healing.
- Sub-agent / parallel-branch capture: that is what tree mode is for; linear mode re-anchors
  or falls back rather than trying to represent it.
