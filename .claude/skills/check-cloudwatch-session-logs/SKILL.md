---
name: check-cloudwatch-session-logs
description: Query and analyze AWS CloudWatch logs/traces for Bedrock AgentCore runtime sessions. Use when the user asks about a specific AgentCore session — why it failed, why it hung, what tools it called, how long each step took, or wants to pull the full trace for a given session_id.
---

Use this when the user asks about a specific AgentCore session — why it failed, why it hung, what tools it called, how long each step took, or wants to pull the full trace for a given `session_id`.

## Prerequisites

- `aws` CLI configured with credentials for the account hosting the AgentCore runtime (check `aws sts get-caller-identity`).
- The runtime ARN. Form: `arn:aws:bedrock-agentcore:{region}:{account}:runtime/{name}-{id}` (example: `arn:aws:bedrock-agentcore:us-west-2:000000000000:runtime/my_agent-AAAAAAAAAA`). Ask the user for the ARN if they haven't provided one.
- A session ID (36-char UUID). Ask the user if they haven't provided one — they typically have it from their client-side invocation.

## Two data sources

AgentCore splits telemetry across two CloudWatch log groups:

1. **Runtime logs** — app-level INFO/ERROR from the container. Coarse but has session start/end and top-level errors.
   - Log group: `/aws/bedrock-agentcore/runtimes/{runtime_name}-{runtime_id}-{qualifier}` (see "Resolving the log group name from the ARN" below for picking the qualifier)
   - Each event is a JSON line with `sessionId`, `level`, `message`, `logger`, `requestId`.

2. **Spans (`aws/spans`)** — OpenTelemetry traces. Fine-grained: every tool call, every model call, every S3 op, every downstream API call.
   - Log group: `aws/spans` (shared across all runtimes in the region)
   - Filter by `attributes.session.id = "<sid>"` to scope to one session.
   - Each event has `name`, `kind`, `durationNano`, `attributes.gen_ai.operation.name`, `attributes.tool.name`, plus any exceptions in `events[]`.

Use source (1) for "did the session start / end / error at the app level?". Use source (2) for "what did the agent actually do?".

## Resolving the log group name from the ARN

Given `arn:aws:bedrock-agentcore:{region}:{account}:runtime/{name}-{id}` (e.g., `arn:aws:bedrock-agentcore:us-west-2:000000000000:runtime/my_agent-AAAAAAAAAA`):

- `alias = {name}-{id}` — the part after `runtime/` (e.g., `my_agent-AAAAAAAAAA`)
- Region = 4th `:`-separated field of the ARN (e.g., `us-west-2`)
- Runtime log group = `/aws/bedrock-agentcore/runtimes/{alias}-{qualifier}`

The `{qualifier}` is the **endpoint name**. `DEFAULT` is what `invoke_agent_runtime(...)` routes to when called without a `qualifier=` argument (see [InvokeAgentRuntime API](https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_InvokeAgentRuntime.html)), but custom endpoints created via `CreateAgentRuntimeEndpoint` use a different name.

Before assuming `DEFAULT`, list which qualifiers actually have logs:

```bash
aws logs describe-log-groups --region $REGION \
  --log-group-name-prefix /aws/bedrock-agentcore/runtimes/${ALIAS}- \
  --query 'logGroups[].logGroupName' --output text
```

- **One result ending in `-DEFAULT`** → use it.
- **Multiple results** → ask the user which endpoint they invoked (the suffix after the last `-` is the qualifier).
- **No results** → the runtime hasn't been invoked yet, or it was invoked under a qualifier whose log group was never created. Ask the user to confirm the ARN and qualifier, since log groups are only created lazily on first traffic to that endpoint.

## Recipes

> **Note:** `REGION` and `ALIAS` come from the runtime ARN — `REGION` is the 4th `:`-separated field, `ALIAS` is the part after `runtime/`. Set them once before running the recipes below.

### 1. List recent sessions (to find a session_id)

```bash
REGION=us-west-2
ALIAS=my_agent-AAAAAAAAAA
QUALIFIER=DEFAULT  # the endpoint name; verify via describe-log-groups above
LG=/aws/bedrock-agentcore/runtimes/${ALIAS}-${QUALIFIER}
WINDOW=1800  # seconds

QID=$(aws logs start-query --region $REGION --log-group-name "$LG" \
  --start-time $(($(date +%s) - $WINDOW)) --end-time $(date +%s) \
  --query-string 'stats count(*) as events, earliest(@timestamp) as start, latest(@timestamp) as last by sessionId | sort last desc | limit 20' \
  --query 'queryId' --output text)

# Poll until Complete — busy regions can take >10s
while [[ "$(aws logs get-query-results --region $REGION --query-id $QID --query 'status' --output text)" != "Complete" ]]; do
  sleep 2
done
aws logs get-query-results --region $REGION --query-id $QID --output json
```

### 2. Runtime-log events for one session (app-level story)

```bash
SID=00000000-0000-0000-0000-000000000000  # replace with the real session UUID
# REGION, ALIAS, QUALIFIER, LG as in recipe 1

aws logs filter-log-events --region $REGION --log-group-name "$LG" \
  --filter-pattern "{ \$.sessionId = \"$SID\" }" \
  --start-time $(($(date +%s) - 3600))000 \
  --query 'events[].message' --output text
```

`filter-log-events` takes **epoch-milliseconds** (note trailing `000`). `start-query` takes epoch-seconds.

`--output text` joins all events with tabs onto one line — pipe through `tr '\t' '\n'` to read line-by-line.

### 3. Full OTel span timeline for a session

`aws/spans` is a regional log group that holds traces from every runtime in the account, so it's huge. Narrow the time window to the session's actual lifetime — recipe 2's first event timestamp is a good lower bound. (Sized this way, scans drop to roughly half versus a 24h sweep.) `aws/spans` doesn't index session.id efficiently, so window size dominates query cost.

```bash
SID=00000000-0000-0000-0000-000000000000  # replace with the real session UUID
# REGION, LG as in recipe 1

# Use recipe 2's earliest event as the start; pad 60s before for safety.
START_MS=$(aws logs filter-log-events --region $REGION --log-group-name "$LG" \
  --filter-pattern "{ \$.sessionId = \"$SID\" }" \
  --start-time $(($(date +%s) - 86400))000 \
  --output json | jq '[.events[].timestamp] | min')
SPANS_START=$(( START_MS / 1000 - 60 ))

QID=$(aws logs start-query --region $REGION --log-group-name aws/spans \
  --start-time $SPANS_START --end-time $(date +%s) \
  --query-string "filter attributes.session.id = \"$SID\" | fields @timestamp, name, kind, attributes.gen_ai.operation.name, attributes.tool.name, durationNano, status.code | sort @timestamp asc | limit 500" \
  --query 'queryId' --output text)

while [[ "$(aws logs get-query-results --region $REGION --query-id $QID --query 'status' --output text)" != "Complete" ]]; do
  sleep 2
done
aws logs get-query-results --region $REGION --query-id $QID --output json
```

Typical output shape — alternating model calls and tool calls:

```
[t+0]    POST /invocations              (entry)
[t+0.1]  S3.GetObject                   (repo fetch)
[t+30]   chat                           op=chat           (model call)
[t+30.5] execute_tool shell             op=execute_tool   (tool exec)
[t+31]   execute_event_loop_cycle       (strands loop)
...
```

### 4. Pull exceptions across all spans

```bash
QID=$(aws logs start-query --region us-west-2 --log-group-name aws/spans \
  --start-time $(($(date +%s) - 3600)) --end-time $(date +%s) \
  --query-string 'filter status.code = "ERROR" | fields @timestamp, name, attributes.session.id, status.message | sort @timestamp desc | limit 50' \
  --query 'queryId' --output text)

while [[ "$(aws logs get-query-results --region us-west-2 --query-id $QID --query 'status' --output text)" != "Complete" ]]; do
  sleep 2
done
aws logs get-query-results --region us-west-2 --query-id $QID --output json
```

## Analysis patterns

When inspecting, answer these in order:

1. **Did it start?** First event should be `POST /invocations`. If missing, the runtime never received the request (check agent ARN, network, IAM).
2. **Did the agent call the model?** Look for `chat` spans with `gen_ai.operation.name=chat`. If none, the model never responded — likely chat-template/tool-parser mismatch (returns empty `end_turn`).
3. **Did tools execute?** `execute_tool` spans. Successive tool calls with growing `durationNano` typically mean the agent is building/testing (Maven, npm, tests). Single tool call with no follow-up = agent gave up or errored.
4. **Did it finish cleanly?** Last runtime log event should be `Async task completed: invoke_agent`. If not, the session is either still running or was killed by `session_timeout`.
5. **Any exceptions?** Check `events[].name=="exception"` inside spans. `status.code=ERROR` surfaces these in the top-level fields.

## Constructing the CloudWatch console URL

If the user wants to click through to the UI:

```python
from urllib.parse import quote

def cloudwatch_session_url(arn: str, session_id: str, qualifier: str = "DEFAULT") -> str:
    region = arn.split(":")[3]
    alias = arn.split("/")[-1]                    # e.g., my_agent-AAAAAAAAAA
    name = alias.rsplit("-", 1)[0]                # e.g., my_agent
    resource_id = quote(f"{arn}/runtime-endpoint/{qualifier}:{qualifier}", safe="")
    return (
        f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
        f"#gen-ai-observability/agent-core"
        f"/agent-alias/{alias}/endpoint/{qualifier}/agent/{name}/session/{session_id}"
        f"?resourceId={resource_id}&serviceName={name}.{qualifier}"
    )
```

## When a query returns zero events

If recipe 2 or 3 returns nothing for a session you're sure ran, work through these in order before assuming the session never executed:

1. **Wrong qualifier?** Re-run `aws logs describe-log-groups --log-group-name-prefix /aws/bedrock-agentcore/runtimes/${ALIAS}-` and confirm the suffix matches the qualifier the client invoked with. `DEFAULT` is only correct when no `qualifier=` was passed.
2. **Wrong region?** The ARN's 4th `:`-field is the runtime region; the AWS CLI default region (or `AWS_REGION`) must match it.
3. **Stale time window?** `filter-log-events --start-time` and `start-query --start-time` both default to "recent." Older sessions need a larger window — the runtime ARN's `creationTime` from `describe-log-groups` is a hard lower bound.
4. **Typo'd or wrongly-cased session_id?** AgentCore session IDs are case-sensitive lowercased UUIDs. Copy from the source rather than retyping.
5. **Session never reached the runtime?** If recipe 2 returns no events even with a wide window, the runtime never received the request — check the client-side `invoke_agent_runtime` response, network/IAM, and the agent ARN itself. Recipe 1 ("list recent sessions") confirms whether *any* sessions are landing.

## Gotchas

- **`filter-log-events` does NOT accept `@timestamp` in its filter pattern.** Use `start-time`/`end-time` args (epoch-ms) instead.
- **`attributes.session.id` (with dots)** — CloudWatch Logs Insights requires backtick-quoting only for keys with dashes, hyphens, or reserved words. Plain dots work unquoted.
- **Runtime logs vs spans differ in completeness.** A session that returned fast with no model call will have ~3 runtime-log events but zero spans (no instrumented work happened). Conversely, a session that crashed mid-stream may have 100+ spans but only a single "task started" runtime log — useful for debugging in-flight failures.
- **Session ID must be lowercased UUID.** AgentCore treats them case-sensitively.
- **Spans don't always arrive in order.** Use `sort @timestamp asc` explicitly.
