#!/usr/bin/env python3
"""Harbor harness — agent_type "claude_code": Claude Code CO-LOCATED in the box.

Claude Code runs *inside* the task sandbox: this app bootstraps it into the
session and runs ``claude -p`` there, so its native Bash/Read/Write tools
operate directly on the task's own filesystem — the same box the verifier will
grade. (This harness process only orchestrates; the agent lives in the box.)

    launcher ──invoke──▶ THIS app ──sb.exec──▶ task runtime [node + claude -p]

Flow per rollout:
  1. open a session on the task runtime
  2. bootstrap: fetch a pinned Node build (the task images ship no node) and
     ``npm i -g @anthropic-ai/claude-code`` (~5-10s)
  3. run ``claude -p <instruction>`` over Bedrock — creds come from the task
     runtime's OWN IAM role (verified reachable in-box), nothing is injected.
     stream-json output doubles as the per-turn transcript AND keepalive bytes
     on the exec stream, so long runs never trip the SandboxClient's 900s idle read.
  4. grade with the task's shipped verifier (injected only now, agent frozen)

The prompt is EXACTLY the task instruction (Claude Code's own system prompt; no
scaffolding added, no "remote sandbox" note — co-located, the box IS local).

Async contract (same for both harness images): @rollout_entrypoint — the
invoke ACKS immediately; the record (or a status_code=500 error doc) is saved
to s3://<s3_bucket>/<exp_id>/<input_id>/<session_id>.json.

Payload: task_runtime_arn, instruction (required); tests_tar_b64, model,
    max_steps (-> --max-turns), agent_timeout_s (per-task agent wall-clock
    budget; on expiry the agent is stopped and the verifier still grades its
    partial work — the leaderboard's AgentTimeout semantics), _rollout.
"""
from __future__ import annotations

import base64
import json
import time

from harbor_sandbox import REGION, HarborSandboxClient

from agentcore_rl_toolkit import AgentCoreRLApp
from agentcore_rl_toolkit.sandbox import SandboxClient

# REGION is imported from harbor_sandbox.config (.env-driven) so the in-box
# Bedrock calls follow the deploy region, not the Dockerfile-baked AWS_REGION.
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
# agent wall-clock budget when the payload omits one (was the old hardcoded cap).
DEFAULT_AGENT_TIMEOUT_S = 3300
# verifier (test.sh) budget when the payload omits [verifier].timeout_sec. Was a
# hardcoded 180s that truncated heavy verifiers mid-build -> reward=None.
DEFAULT_VERIFIER_TIMEOUT_S = 900
# SandboxClient.exec enforces a service-side timeout ceiling of 3600s, so a single
# `claude -p` exec cannot run longer than this. Tasks whose task.toml budget
# exceeds it (TB-2: 7200s, 12000s) are CAPPED here — the run is stopped at 3600s
# and still graded (partial work). We record requested-vs-applied so the cap is
# auditable rather than silent. (>3600s budgets need a background-run+poll loop.)
EXEC_TIMEOUT_MAX_S = 3600

# Client-side botocore read_timeout for the sandbox connection. This is a PER-READ
# idle timeout (max silence between streamed stream-json events), NOT a total cap.
# The SandboxClient default is 900s (sized for cold-start start()); that is fatal
# for the long agent exec: a task may legitimately run one silent shell op (compile,
# train, MCMC) for its ENTIRE budget with zero stream-json, so the idle timer must
# outlast the longest possible command. The longest command is the 3600s service
# cap itself, so read_timeout must be STRICTLY greater than it — equal is a dead
# heat (read_timeout==budget is exactly what killed the 900s-budget tasks and would
# re-kill train-fasttext at 3600). +300s covers the terminal result's round-trip
# after the service kills the command. Over-provisioning is free: read_timeout only
# bites during silence, and the 3600s service cap still bounds a genuinely-hung one.
SANDBOX_READ_TIMEOUT_S = EXEC_TIMEOUT_MAX_S + 300  # 3900

# Pinned Node build fetched into the box at session start (the task images ship
# no node). NA is picked in-box from uname -m (arm64 on the microVM substrate).
NODE_VER = "v20.18.1"

# The harness assumes ONLY a Linux container with a shell, not whatever the TASK
# image happens to ship, so the fetcher falls back curl -> wget -> python ->
# `apt-get install curl`; `echo DL=<tool>` records which path was taken. Kept
# single-quote-free (python fetchers use \" not ') so the sandbox exec wrapper
# stays a simple `/bin/sh -c '...'`.
_BOOTSTRAP = r"""
set -e
case "$(uname -m)" in aarch64|arm64) NA=arm64;; *) NA=x64;; esac
URL="https://nodejs.org/dist/NODEVER/node-NODEVER-linux-$NA.tar.gz"
have(){ command -v "$1" >/dev/null 2>&1; }
if have node && have npm; then
  echo "DL=preinstalled"
else
  # no fetcher in the image? install curl via the package manager on the box.
  if ! { have curl || have wget || have python3 || have python; }; then
    if have apt-get; then
      echo "DL=apt-installing-curl"
      apt-get update >/dev/null 2>&1 || true
      DEBIAN_FRONTEND=noninteractive apt-get install -y curl >/dev/null 2>&1 || true
    fi
  fi
  if have curl; then echo "DL=curl"; curl -fsSL "$URL" -o /tmp/node.tgz;
  elif have wget; then echo "DL=wget"; wget -qO /tmp/node.tgz "$URL";
  elif have python3; then echo "DL=python3";
    python3 -c "import urllib.request as u; u.urlretrieve(\"$URL\",\"/tmp/node.tgz\")";
  elif have python; then echo "DL=python"; python -c "import urllib; urllib.urlretrieve(\"$URL\",\"/tmp/node.tgz\")";
  else echo "no fetcher and apt-get install curl unavailable/failed" >&2; exit 3;
  fi
  mkdir -p /opt/node
  # --no-same-owner: even with caps, avoid chowning to the uid in the tarball
  tar xzf /tmp/node.tgz -C /opt/node --strip-components=1 --no-same-owner >/dev/null
fi
export PATH=/opt/node/bin:$PATH
# stdout silenced, stderr KEPT: on failure the npm error must reach the
# bootstrap-failed exception detail instead of vanishing into /dev/null.
npm i -g @anthropic-ai/claude-code >/dev/null
echo BOOTSTRAP_OK $(node --version) $(claude --version)
""".replace("NODEVER", NODE_VER)

app = AgentCoreRLApp()


def _resolve_sandbox(payload):
    """Start a task's sandbox. Two addressing modes:
      * legacy: payload["task_runtime_arn"] — a pre-created runtime; open a session.
      * harbor: payload["benchmark"] + payload["task_id"] — ensure the runtime via
        HarborSandboxClient.create (idempotent). With payload["lease"]=true the
        harness deletes it after the rollout (the lease pattern).

    Returns (sandbox_client, release_fn) — release_fn is a no-op unless leased.
    """
    arn = payload.get("task_runtime_arn")
    if arn:
        return SandboxClient(runtime_arn=arn, read_timeout=SANDBOX_READ_TIMEOUT_S), (lambda: None)
    bench, task = payload.get("benchmark"), payload.get("task_id")
    if not (bench and task):
        raise ValueError("payload must include either 'task_runtime_arn' or " "'benchmark' + 'task_id'")
    # read_timeout forwards through **client_kwargs to the SandboxClient ctor.
    client = HarborSandboxClient.create(bench, task, read_timeout=SANDBOX_READ_TIMEOUT_S)
    if payload.get("lease"):
        return client, client.release  # instance path: deletes its OWN runtime
    return client, (lambda: None)


def _claude_env(model: str) -> dict:
    return {
        "PATH": "/opt/node/bin:/usr/local/bin:/usr/bin:/bin",
        "CLAUDE_CODE_USE_BEDROCK": "1",  # creds = the TASK runtime's own role
        "ANTHROPIC_MODEL": model,
        "AWS_REGION": REGION,
        "IS_SANDBOX": "1",  # allow --dangerously-skip-permissions as root in a sandbox
        "HOME": "/root",
    }


def _parse_stream(stdout: str) -> dict:
    """Pull the agent's Bash commands and the final result out of Claude Code's
    stream-json transcript."""
    commands = []
    result, num_turns, cost = None, None, None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        t = obj.get("type")
        if t == "assistant":
            for blk in obj.get("message", {}).get("content", []):
                if blk.get("type") == "tool_use" and blk.get("name") == "Bash":
                    cmd = blk.get("input", {}).get("command")
                    if cmd:
                        commands.append(cmd)
        elif t == "result":
            result = obj.get("result")
            num_turns = obj.get("num_turns")
            cost = obj.get("total_cost_usd")
    return {"commands": commands, "result": result, "num_turns": num_turns, "cost_usd": cost}


def _stage_b64(sb, b64: str, dest: str, chunk: int = 50000) -> None:
    """Decode a base64 blob to <dest> INSIDE the box, written in slices.

    A single ``printf '%s' '<b64>' | base64 -d`` puts the whole blob in one
    command, but SandboxClient.exec caps body.command at 65536 bytes, so big
    test tarballs (build-pov-ray, sam-cell-seg, video-processing, ...) blow the
    cap and die with a ValidationException. Append the base64 in <=50000-char
    slices (the base64 alphabet has no single-quotes, so '...'-wrapping is safe
    and never re-expanded) and decode once at the end."""
    tmp = dest + ".b64"
    sb.exec(f"rm -f {tmp}", timeout=30)
    for i in range(0, len(b64), chunk):
        sb.exec(f"printf '%s' '{b64[i:i + chunk]}' >> {tmp}", timeout=60)
    sb.exec(f"base64 -d {tmp} > {dest} && rm -f {tmp}", timeout=60)


def grade_with_verifier(sb, tests_tar_b64: str, verifier_timeout_s: float = DEFAULT_VERIFIER_TIMEOUT_S) -> dict:
    """Run the task's own shipped verifier in the sandbox (uniform corpus
    contract: tests/test.sh -> /logs/verifier/reward.txt).

    The test.sh budget is the task's OWN `[verifier].timeout_sec` (threaded via
    the payload), clamped to exec's 3600s ceiling. A hardcoded cap here (was
    180s) silently truncates heavy verifiers mid-build — they never write
    reward.txt, so the row grades reward=None (indeterminate, NOT a real fail).
    Faithful to the leaderboard, which gives each verifier its declared budget."""
    if not tests_tar_b64:
        return {"reward": None, "verifier_tail": "(no tests supplied)"}
    verify_s = min(int(verifier_timeout_s), EXEC_TIMEOUT_MAX_S)
    sb.exec("rm -rf /tests && mkdir -p /tests /logs/verifier", timeout=30)
    _stage_b64(sb, tests_tar_b64, "/tmp/_tests.tgz")
    unpack = sb.exec("tar xzf /tmp/_tests.tgz -C /tests 2>&1", timeout=120)
    run = sb.exec("bash /tests/test.sh 2>&1", timeout=verify_s)
    reward_txt = sb.exec("cat /logs/verifier/reward.txt 2>/dev/null", timeout=30)
    reward = None
    try:
        reward = int(reward_txt.stdout.strip())
    except Exception:
        pass
    return {
        "reward": reward,
        "verifier_tail": (run.stdout or "")[-1500:],
        "unpack_err": (unpack.stdout or "")[:300] if unpack.exit_code != 0 else "",
        "verifier_budget_s": verify_s,
        "verifier_timed_out": bool(getattr(run, "timed_out", False)),
    }


def run_rollout(
    sb_client,
    instruction,
    model,
    max_steps,
    tests_tar_b64,
    agent_timeout_s,
    verifier_timeout_s=DEFAULT_VERIFIER_TIMEOUT_S,
) -> dict:
    client = sb_client
    timing = {}

    t0 = time.time()
    with client.start() as sb:
        timing["start_s"] = round(time.time() - t0, 1)

        # --- bootstrap Claude Code into the box ---
        t0 = time.time()
        boot = sb.exec(_BOOTSTRAP, timeout=300)
        timing["bootstrap_s"] = round(time.time() - t0, 1)
        if "BOOTSTRAP_OK" not in (boot.stdout or ""):
            raise RuntimeError(f"claude-code bootstrap failed: {(boot.stderr or boot.stdout or '')[-300:]}")

        # --- stage the instruction as the raw prompt (base64 avoids all quoting) ---
        # chunked so a very long instruction can't blow exec's 64KB command cap.
        b64 = base64.b64encode(instruction.encode()).decode()
        _stage_b64(sb, b64, "/tmp/prompt.txt")

        # --- run Claude Code; prompt = the instruction verbatim, nothing added ---
        # cwd = the first task-like dir that exists, so relative paths resolve.
        cwd = (
            "d=$(for x in /home/user /app /workspace /root; do "
            '[ -d "$x" ] && echo "$x" && break; done); cd "${d:-/}" && '
        )
        run_cmd = (
            cwd + "cat /tmp/prompt.txt | claude -p "
            "--output-format stream-json --verbose "
            f"--max-turns {max_steps} --dangerously-skip-permissions"
        )
        t0 = time.time()
        # per-task agent budget: on expiry sb.exec stops the agent (timed_out)
        # and we STILL grade its partial work below — faithful AgentTimeout.
        # int(): body.timeout is validated as an integer (float 900.0 is rejected).
        # min(..., EXEC_TIMEOUT_MAX_S): exec's own 3600s ceiling; larger budgets
        # are capped (and recorded below) since one exec cannot outlive it.
        budget_s = int(agent_timeout_s)
        exec_timeout_s = min(budget_s, EXEC_TIMEOUT_MAX_S)
        run = sb.exec(run_cmd, timeout=exec_timeout_s, env=_claude_env(model))
        timing["agent_s"] = round(time.time() - t0, 1)
        parsed = _parse_stream(run.stdout or "")

        # --- grade with the shipped verifier (agent is frozen) ---
        t0 = time.time()
        result = grade_with_verifier(sb, tests_tar_b64, verifier_timeout_s)
        timing["grade_s"] = round(time.time() - t0, 1)

    result.update(
        {
            "agent_type": "claude_code",
            "model": model,
            "stop_reason": "agent_timed_out" if run.timed_out else "end_turn",
            "agent_budget_s": budget_s,
            "agent_budget_applied_s": exec_timeout_s,
            "budget_capped_by_exec_max": exec_timeout_s < budget_s,
            "steps": len(parsed["commands"]),
            "num_turns": parsed["num_turns"],
            "agent_result": (parsed["result"] or "")[:400],
            "cost_usd": parsed["cost_usd"],
            "timing": timing,
            "commands": [c[:500] for c in parsed["commands"]],
        }
    )
    return result


@app.rollout_entrypoint
def invoke(payload: dict) -> dict:
    """One rollout per invocation, run as a background task by the SDK.

    Exceptions are NOT caught here on purpose: the SDK's error path saves
    {"status_code": 500, "stop_reason": str(e), "traceback": ...} to the same
    S3 key, so the launcher learns about failures the same way as results.
    """
    instruction = payload.get("instruction")
    if not instruction:
        raise ValueError("payload must include 'instruction'")
    sb_client, release = _resolve_sandbox(payload)
    try:
        return run_rollout(
            sb_client=sb_client,
            instruction=instruction,
            model=payload.get("model", DEFAULT_MODEL),
            max_steps=int(payload.get("max_steps", 25)),
            tests_tar_b64=payload.get("tests_tar_b64", ""),
            agent_timeout_s=float(payload.get("agent_timeout_s", DEFAULT_AGENT_TIMEOUT_S)),
            verifier_timeout_s=float(payload.get("verifier_timeout_s", DEFAULT_VERIFIER_TIMEOUT_S)),
        )
    finally:
        # no-op unless payload["lease"]: frees the runtime slot. A delete failure
        # (AccessDenied, or ResourceNotFound when the runtime never became ready)
        # must NOT run as an unguarded finally — that would REPLACE the rollout's
        # real return value / real exception with the cleanup error (this is how
        # circuit-fibsqrt's true ResourceNotFound got masked as DeleteAgentRuntime
        # AccessDenied). Swallow it so the genuine result/cause is what's recorded.
        try:
            release()
        except Exception as e:  # noqa: BLE001 - cleanup best-effort
            print(f"lease release failed (ignored, slot may leak): {e!r}")


if __name__ == "__main__":
    app.run()
