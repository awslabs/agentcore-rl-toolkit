# Linear-history mode for the rollout gateway (generate-time prefix healing)

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

**Status:** implemented (v1) in `src/agentcore_rl_toolkit/rollout_gateway/linear.py`
(`LinearHealer`), wired through `RolloutGateway` and `BaseAdapter`. Unit-tested in
`tests/rollout_gateway/test_linear_healer.py`; core template assumptions validated against
the real Qwen3-Coder-30B tokenizer. Remaining follow-ups at the end.

## Motivation

The gateway owns tokenization, so each turn it re-renders the whole conversation the agent
replays. Chat templates are deterministic, so system/user/tool messages re-render to
identical ids — but a prior **assistant** message rarely does: its served tokens came from
generation, while on the next turn it is re-rendered from a parsed message dict
(`text` + `tool_calls`) back through the template, yielding near-identical ids that differ
in whitespace, tool-call JSON spacing, or reasoning re-keying. `TrajectoryManager`
(`rollout_gateway/trajectory.py`) detects that drift and, to stay safe for harnesses that
may branch or edit history, resolves it by FORK (split the rollout into a second sample) or
REALIGN (drop a turn's signal).

For a strictly linear, multi-turn tool-calling agent that safety is pure overhead. On a
real multi-turn SWE-agent run (OpenHands driving Qwen3-Coder-30B, `fork_threshold_tokens=0`)
**96.6% of sessions forked**, mean **6.3** records/session (max 16), even though every
rollout was linear. Each fork re-emits the shared prefix as `loss_mask=0` context and splits
the trajectory, so it is a large, avoidable training-efficiency loss.

## Design

Because the gateway controls the tokens handed to the backend, it can splice the exact ids
it already served for prior turns over the client's re-rendered version **before**
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

`delta_tail` is computed with a two-render suffix diff, so no hand-rolled delimiter logic is
needed:

```
r_prev = render(prev_messages, add_generation_prompt=False)   # drifted re-render of prior convo
r_full = render(messages,      add_generation_prompt=True)    # full incoming render
assert r_full[:len(r_prev)] == r_prev                          # append-only / linearity check
delta_tail = r_full[len(r_prev):]                              # new messages + generation prompt
```

This relies only on chat templates being prefix-consistent with
`add_generation_prompt=False` (rendering `[m0..mk]` is a prefix of `[m0..mk, mk+1]`), which
holds for turn-delimited templates such as Qwen and Llama-3. The assertion is itself the
linearity check: if it fails, the history was edited or branched (see *When the assumption
breaks*).

### Isolating the assistant closer

The canonical `served_prefix` ends at the model's last generated content; to rejoin it to
`delta_tail` we reinsert the template's between-message glue (Qwen: `<|im_end|>\n`). That
glue is **message-type independent** — the same after a text turn and a tool-call turn — so
`LinearHealer._assistant_close` recovers it by rendering the prior turns with the last
assistant replaced by two distinct *text* bodies and taking the common suffix of the two
renders (the bodies differ in every trailing token, only the glue survives). It then asserts
that glue is actually a suffix of the real `r_prev`; if a template's glue turns out to be
type-dependent the assert fails and the turn falls back rather than emit a wrong boundary.
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
  `fork_threshold_tokens` is ignored in linear mode (forking is disabled by construction) and
  a warning is logged if both are set.
- `rollout_gateway/adapters/common.py` — `BaseAdapter` heals between render and
  `backend.generate` in `_run_turn`, feeds the healed ids to both `generate` and the
  `TurnRecord`, and calls `commit` before `record_turn`. Session teardown drops per-sid healer
  state. The existing `TrajectoryManager` is unchanged.
- `rollout_gateway/__init__.py` — exports `LinearHealer`.

### Config surface

- `history_mode`: `"tree"` (default, current behavior) | `"linear"`. Gateway-global, plumbed
  like `fork_threshold_tokens`; linear and tree sessions do not coexist within a run.
- `linear_on_nonlinear`: `"reset"` (default) | `"error"` | `"passthrough"` — behavior when a
  turn breaks the append-only assumption.

## When the assumption breaks

The append-only check fails when the history is not linear from the healer's view: a
dropped/edited/branched prior message, context compaction, or an LLM-client retry that
re-issues the same prompt and produces a second generation from the same prefix. The same
fallback path also fires when the closer probe cannot isolate the glue (a message-type-
dependent template — `close_unresolved` counter). Behavior is controlled by
`linear_on_nonlinear`:

- `"reset"` (default) — drop per-sid state and re-anchor to the current render, treating the
  jump as the start of a fresh linear segment. The turns before and after each stay
  drift-free; only the single jump turn conditions on the client's render. Correct for benign,
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
canonical prefix tokens spliced); `nonlinear` (append-only-check failures); `close_unresolved`
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
assumptions (generation-prompt tokens equal the assistant open; prefix-consistency turn to
turn; the closer probe returns `<|im_end|>\n` for both text and tool-call turns and is a
genuine suffix of a tool-call `r_prev`; served ids end with the stop token) were confirmed
against the live Qwen3-Coder-30B tokenizer/template.

**Real-trace reconstruction.** We reconstructed, at the token level, the per-turn prompt
streams from real recorded multi-turn SWE-agent sessions (Qwen3-Coder-30B) and replayed them
through the real `TrajectoryManager` two ways: the drifted prompts as they actually happened,
and the canonical served prefix + drift-free new-observation tail that healing feeds. The raw
replay reproduced the recorded fork counts exactly (a fidelity check). Over 200 sessions:

| Metric | Baseline (tree/fork) | Linear (healed) | Delta |
|---|---|---|---|
| Records/session (mean) | 6.42 | 1.00 | −84.3% |
| Total tokens (context + response) | 28.72M | 9.06M | −68.5% (3.17× fewer) |
| Trained (`loss_mask=1`) tokens | 2.32M | 2.32M | unchanged |
| Useful-token fraction | 8.1% | 25.7% | — |

Training latency scales ~linearly with the sequence length forwarded, not the record count,
so the **68.5% token reduction (3.17×)** is the latency-relevant figure — and it is smaller
than the 84.3% record reduction because early forks carry short prefixes while the redundancy
is dominated by late, large-context forks. The trained-token count is identical, so the entire
delta is redundant re-emitted context.

Exactly one of the 200 sessions did not collapse to a single record: it was an LLM-client
retry artifact (two records sharing an identical prompt, responses agreeing for thousands of
tokens before diverging one token; only one attempt was seen by the harness), which
`on_nonlinear="reset"` re-anchored into two linear segments — the safe outcome, and one a
capture-time retry dedup could also collapse.

## Assumptions and target use case

- The agent harness intends a strictly **linear, append-only** history: it replays prior
  assistant **message dicts** verbatim, so the only turn-to-turn divergence is token-level
  drift (not a message-level rewrite). The target is a multi-turn, tool-calling SWE agent
  (validated with OpenHands driving Qwen3-Coder-30B); the harness never branches or edits
  prior turns.
- Cosmetic re-tokenization drift is treated as OOD-insignificant, so training on the
  canonical/healed context is acceptable.
- Chat templates are turn-delimited and prefix-consistent (Qwen, Llama-3). The
  type-independent-closer assumption is validated for Qwen3-Coder-30B; re-check it when
  targeting a template family whose between-message glue could differ after tool-call vs text
  turns (the `close_unresolved` fallback keeps such a case correct, just unhealed).

## Non-goals

- Message-level rewrites where the assistant **dict** itself changes (not just its
  tokenization): linear mode assumes the client replays prior messages verbatim, so only
  token-level drift occurs.
- Sub-agent / parallel-branch capture: that is what tree mode is for; linear mode re-anchors
  or falls back rather than trying to represent it.

## Follow-ups

- Wire `history_mode` / `linear_on_nonlinear` through the verl backend
  (`backends/verl/gateway_host.py`, `backends/verl/agent_loop.py`) alongside
  `fork_threshold_tokens`, and expose them in the configs that build the gateway.
- A sanity assertion that a healed session finishes as exactly one record with no resets
  (the per-session counters are now on record metadata; this would flag a session that
  silently degraded to forking).
- A parity harness over recorded sessions (linear-mode records are a superset-signal of
  tree-mode records — never fewer trained tokens, same or fewer samples).
