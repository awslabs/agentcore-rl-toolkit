#!/usr/bin/env python3
"""bench — one-command evaluation of a Harbor benchmark on AgentCore Runtime.

    python bench.py --benchmark tmax/TMax-15K-Harbor --task-root ./tasks \
                    --agent claude-code --model us.anthropic.claude-sonnet-4-6

One rollout per task: the payload carries Harbor coordinates (benchmark +
task_id) — the HARNESS creates/ensures the task sandbox itself via
HarborSandboxClient (idempotent), runs the agent, grades with the task's own
shipped verifier, saves the record to S3, and (lease mode, default) removes the
runtime afterwards. No pre-created task runtimes: quota use is transient
(~ n-concurrent), and this launcher only invokes + polls.

Restrict the task set with --tasks / --exclude (comma list or @file) and --limit.
Results: a jsonl record per task plus a solve-rate summary.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import sys
import tarfile
import time
from pathlib import Path

import boto3
import tomllib
from harbor_sandbox import REGION, S3_BUCKET, ensure_dataset

from agentcore_rl_toolkit import RolloutClient

ROOT = Path(__file__).resolve().parent

# --agent choices. A harness deploys under `harness_<agent>_v1` (see
# harness/deploy_harness.py); we resolve that name to an ARN at startup.
AGENTS = ("strands", "claude-code")

# transient infra failures worth ONE re-run.
_TRANSIENT_ERR = re.compile(
    r"read timed out|timed out|ConnectionPool|Failed to launch|"
    r"ResourceNotFound|No endpoint|no agent found|Throttl|TooManyRequests|"
    r"runtimeClientError|ServiceException|InternalServer",
    re.I,
)

_SDK_KEYS = ("status_code", "input_id", "s3_bucket", "result_key", "payload")


def harness_arn(agent: str) -> str:
    """Resolve --agent to its deployed harness runtime ARN. Deploy the harness
    first (see harness/deploy_harness.py)."""
    wanted = f"harness_{agent.replace('-', '_')}_v1"
    ctrl = boto3.client("bedrock-agentcore-control", region_name=REGION)
    kw: dict = {"maxResults": 100}
    while True:
        resp = ctrl.list_agent_runtimes(**kw)
        for r in resp.get("agentRuntimes", []):
            if r.get("agentRuntimeName") == wanted:
                return r["agentRuntimeArn"]
        kw["nextToken"] = resp.get("nextToken")
        if not kw["nextToken"]:
            break
    raise SystemExit(
        f"no harness runtime named {wanted!r} for agent {agent!r} " f"— deploy it first (harness/deploy_harness.py)"
    )


def tests_tar_b64(tests_dir: Path) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(tests_dir.rglob("*")):
            tf.add(p, arcname=str(p.relative_to(tests_dir)))
    return base64.b64encode(buf.getvalue()).decode()


# Fallback budgets for tasks whose task.toml omits a per-task timeout. Threading
# the task's own [agent]/[verifier].timeout_sec into the harness reproduces the
# leaderboard's per-task AgentTimeout semantics (stop at the budget, still grade
# partial work); these defaults only apply when a task declares neither.
DEFAULT_AGENT_TIMEOUT_S = 3300.0
DEFAULT_VERIFIER_TIMEOUT_S = 900.0
# head-room the OUTER client wedge gets over the largest per-task budget, so the
# faithful per-task budget always fires first and the wedge only catches a hang.
WEDGE_MARGIN_S = 900.0


def _task_section_timeout_s(task_dir: Path, section: str, default: float) -> float:
    """`[<section>].timeout_sec` from the task's `task.toml`, else `default`."""
    toml = task_dir / "task.toml"
    if toml.exists():
        try:
            v = (tomllib.loads(toml.read_text()).get(section) or {}).get("timeout_sec")
            if v:
                return float(v)
        except Exception:
            pass
    return default


def build_payload(task_dir: Path, args) -> dict:
    return {
        "benchmark": args.benchmark,
        "task_id": task_dir.name,
        "lease": not args.keep_runtimes,
        "instruction": (task_dir / "instruction.md").read_text(),
        "tests_tar_b64": tests_tar_b64(task_dir / "tests"),
        "model": args.model,
        "max_steps": args.max_steps,
        "agent_timeout_s": _task_section_timeout_s(task_dir, "agent", args.agent_timeout_default),
        "verifier_timeout_s": _task_section_timeout_s(task_dir, "verifier", DEFAULT_VERIFIER_TIMEOUT_S),
    }


def normalize(item) -> dict:
    if not item.success:
        return {"error": str(item.error)[:300]}
    doc = item.result
    if doc.get("status_code") == 500:
        return {"error": str(doc.get("stop_reason", ""))[:300], "traceback": str(doc.get("traceback", ""))[:1500]}
    return {k: v for k, v in doc.items() if k not in _SDK_KEYS}


def run_jobs(client, jobs, n_concurrent, timeout) -> list:
    """jobs = [(task_id, payload)]; returns records in job order, prints live."""
    records = {}
    for item in client.run_batch([p for _, p in jobs], max_concurrent_sessions=n_concurrent, timeout=timeout):
        tid = jobs[item.index][0]
        rec = normalize(item)
        rec.update({"task_id": tid, "elapsed_s": round(item.elapsed or 0.0, 1)})
        records[item.index] = rec
        n = len(records)
        if "error" in rec:
            print(f"  [{n}/{len(jobs)}] {tid} ERROR ({rec['elapsed_s']}s): {rec['error'][:90]}")
        elif n % 25 == 0 or n <= 3:
            print(
                f"  [{n}/{len(jobs)}] {tid} reward={rec.get('reward')} "
                f"steps={rec.get('steps')} ({rec['elapsed_s']}s)"
            )
    return [records[i] for i in sorted(records)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, help="Harbor benchmark identifier ('org/name')")
    p.add_argument(
        "--task-root",
        required=True,
        dest="task_root",
        help="Harbor dataset dir whose immediate subdirectories ARE the "
        "tasks (each with instruction.md, environment/, tests/, "
        "task.toml). If empty, the tasks are pulled from the Harbor "
        "registry (needs the 'harbor' package; see README).",
    )
    p.add_argument("--agent", default="claude-code", choices=sorted(AGENTS))
    p.add_argument("--model", default="us.anthropic.claude-sonnet-4-6")
    p.add_argument("--n-concurrent", type=int, default=48, dest="n_concurrent")
    p.add_argument("--max-steps", type=int, default=100, dest="max_steps")
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="OUTER client wedge deadline per rollout. Default: auto = "
        "max per-task agent budget + margin, so the faithful "
        "per-task budget (task.toml [agent].timeout_sec, enforced "
        "in the harness) always fires first.",
    )

    p.add_argument(
        "--agent-timeout-default",
        type=float,
        default=DEFAULT_AGENT_TIMEOUT_S,
        dest="agent_timeout_default",
        help="agent wall-clock budget for tasks whose task.toml omits " "[agent].timeout_sec (e.g. the tmax corpus)",
    )
    p.add_argument(
        "--keep-runtimes",
        action="store_true",
        dest="keep_runtimes",
        help="leave each task's runtime alive after its rollout " "(default: delete it, so quota use stays transient)",
    )
    p.add_argument("--tasks", default="", help="comma list or @file to restrict")
    p.add_argument("--exclude", default="", help="comma list or @file to skip")
    p.add_argument("--limit", type=int, default=0, help="first N tasks (0 = all)")
    p.add_argument("--exp-id", default=None, dest="exp_id")
    p.add_argument("--out", default=str(ROOT / "bench_results.jsonl"))
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("agentcore_rl_toolkit.client").setLevel(logging.INFO)

    # ---- task scope ---------------------------------------------------------
    def idset(spec):
        if not spec:
            return set()
        return set(Path(spec[1:]).read_text().split() if spec.startswith("@") else spec.split(","))

    task_root = Path(args.task_root)
    ensure_dataset(args.benchmark, task_root)  # pull tasks if not already on disk
    only, excl = idset(args.tasks), idset(args.exclude)
    tasks = [
        d
        for d in sorted(task_root.iterdir())
        if d.is_dir() and (d / "instruction.md").exists() and d.name not in excl and (not only or d.name in only)
    ]
    if args.limit:
        tasks = tasks[: args.limit]

    print(
        f"bench: {len(tasks)} tasks | benchmark={args.benchmark} agent={args.agent} "
        f"model={args.model} lease={not args.keep_runtimes} "
        f"n-concurrent={args.n_concurrent}"
    )

    t_payload = time.time()
    jobs = [(d.name, build_payload(d, args)) for d in tasks]
    print(f"payloads built in {time.time()-t_payload:.0f}s")

    # per-task agent budgets (faithful, enforced harness-side); size the OUTER
    # client wedge above the largest so the per-task budget always fires first.
    budgets = sorted(p["agent_timeout_s"] for _, p in jobs) or [DEFAULT_AGENT_TIMEOUT_S]
    print(
        f"agent budgets (task.toml [agent].timeout_sec): min {budgets[0]:.0f}s  "
        f"p50 {budgets[len(budgets)//2]:.0f}s  max {budgets[-1]:.0f}s"
    )
    if args.timeout is None:
        args.timeout = budgets[-1] + WEDGE_MARGIN_S
        print(
            f"outer wedge auto-set to {args.timeout:.0f}s "
            f"(max budget {budgets[-1]:.0f}s + {WEDGE_MARGIN_S:.0f}s margin)"
        )
    elif args.timeout <= budgets[-1]:
        print(
            f"WARNING: --timeout {args.timeout:.0f}s <= max per-task budget "
            f"{budgets[-1]:.0f}s — the client wedge may fire before the faithful "
            f"per-task budget, abandoning rollouts as errors instead of grading them"
        )

    exp_id = args.exp_id or f"harbor-bench/{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    client = RolloutClient(
        agent_runtime_arn=harness_arn(args.agent),
        s3_bucket=S3_BUCKET,
        exp_id=exp_id,
        max_pool_connections=max(10, args.n_concurrent),
    )

    # ---- run (wall measured) ------------------------------------------------
    t0 = time.time()
    records = run_jobs(client, jobs, args.n_concurrent, args.timeout)

    retry_idx = [i for i, r in enumerate(records) if "error" in r and _TRANSIENT_ERR.search(r["error"])]
    if retry_idx:
        print(f"\nretrying {len(retry_idx)} transient failure(s) ...")
        for i, rec in zip(
            retry_idx, run_jobs(client, [jobs[i] for i in retry_idx], args.n_concurrent, args.timeout), strict=True
        ):
            records[i] = rec
    wall = time.time() - t0

    # ---- results + summary --------------------------------------------------
    Path(args.out).write_text("\n".join(json.dumps(r) for r in records) + "\n")
    ok = [r for r in records if "error" not in r]
    solved = [r for r in ok if r.get("reward") == 1]
    el = sorted(r["elapsed_s"] for r in ok) or [0]

    print("\n================ bench summary ================")
    print(
        f"wall: {wall/3600:.2f} h ({wall:.0f}s) for {len(records)} tasks "
        f"@ n-concurrent {args.n_concurrent}  ({len(records)/max(wall/3600,1e-9):.0f} tasks/h)"
    )
    print(f"graded: {len(ok)}/{len(records)}  errors: {len(records)-len(ok)}")
    print(f"solve: {len(solved)}/{len(ok)} ({100*len(solved)/max(1,len(ok)):.1f}%)")
    print(f"rollout p50 {el[len(el)//2]:.0f}s  p90 {el[int(0.9*(len(el)-1))]:.0f}s  max {el[-1]:.0f}s")
    print(f"records -> {args.out} | S3 -> s3://{S3_BUCKET}/{exp_id}/")


if __name__ == "__main__":
    main()
