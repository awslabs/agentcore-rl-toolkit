#!/usr/bin/env python3
"""Build a Harbor task or dataset into ECR under the ratified naming scheme.

Benchmark-agnostic: every image URI comes from naming.resolve, so the SAME
builder serves tmax, terminal-bench-2, and any future org/name dataset. The
build is deliberately minimal — the task's ORIGINAL environment/ image, then a
COPY-only wrapper that adds just the sandboxd binary (the AgentCore runtime
contract: port 8080, no logic). Nothing else is injected, so the numbers
reflect the most original image. COPY-only means the wrap stage needs no qemu;
the env stage still runs the task's own Dockerfile (RUN lines there need qemu
when cross-building arm64-on-x86).

Idempotent by ECR tag (existing tags are listed up front and skipped), so a run
is safely resumable. Drive a remote NATIVE builder with DOCKER_HOST=ssh://host
(docker runs there; ECR auth stays client-side, so the builder needs no IAM).

    python -m harbor_sandbox.build \
        --task-root /tmp/tb2/terminal-bench-2 \
        --benchmark terminal-bench/terminal-bench-2 --arch arm64
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import REGION
from .naming import resolve

WRAPPER = Path(__file__).resolve().parent / "wrapper"
# Single source of truth for the health-shim binary is the repo-root sandboxd/
# (source + build.sh + prebuilt dist/); we stage from it rather than vendor a copy.
SANDBOXD = Path(__file__).resolve().parents[3] / "sandboxd"


def ensure_sandboxd(arch: str) -> None:
    """Stage the sandboxd health-shim binary into the wrapper build context.

    Reuses the repo's prebuilt ``sandboxd/dist/`` binary if present, else builds
    it via ``sandboxd/build.sh`` (local Go toolchain or a golang container). The
    binary is git-ignored here — never vendored."""
    binary = f"agentcore-sandboxd-linux-{arch}"
    dst = WRAPPER / binary
    if dst.exists():
        return
    src = SANDBOXD / "dist" / binary
    if src.exists():
        shutil.copy2(src, dst)
    else:
        subprocess.run([str(SANDBOXD / "build.sh"), "--arch", arch, "--stage", str(WRAPPER)], check=True)


def sh(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def ecr_login(benchmark: str, arch: str) -> tuple[str, set[str]]:
    """Ensure the repo exists + docker login; return (repo, existing tags)."""
    n = resolve(benchmark, "_probe_", arch=arch)
    repo, registry = n.ecr_repo, n.image_uri.split("/")[0]
    subprocess.run(
        ["aws", "ecr", "describe-repositories", "--region", REGION, "--repository-names", repo], capture_output=True
    ).returncode == 0 or subprocess.run(
        ["aws", "ecr", "create-repository", "--region", REGION, "--repository-name", repo], capture_output=True
    )
    pw = subprocess.run(
        ["aws", "ecr", "get-login-password", "--region", REGION], capture_output=True, text=True, check=True
    ).stdout.strip()
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", registry],
        input=pw,
        text=True,
        check=True,
        capture_output=True,
    )
    tags, token = set(), None
    while True:
        cmd = [
            "aws",
            "ecr",
            "list-images",
            "--region",
            REGION,
            "--repository-name",
            repo,
            "--max-results",
            "1000",
            "--query",
            "{t:imageIds[].imageTag,n:nextToken}",
            "--output",
            "json",
        ]
        if token:
            cmd += ["--next-token", token]
        out = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
        tags.update(x for x in (out.get("t") or []) if x)
        token = out.get("n")
        if not token:
            return repo, tags


def build_task(task_dir: Path, benchmark: str, *, arch: str = "arm64", push: bool = True) -> tuple[str, str]:
    """Build one task image (original env stage + COPY-only sandboxd wrap) and push."""
    tid = task_dir.name
    env_ctx = task_dir / "environment"
    if not (env_ctx / "Dockerfile").exists():
        return tid, "skip: no environment/Dockerfile"
    uri = resolve(benchmark, tid, arch=arch).image_uri
    plat = ["--platform", f"linux/{arch}", "--provenance=false"]
    env_tag = f"harbor-env-{arch}:{tid}"

    rc, out = sh(["docker", "build", *plat, "-t", env_tag, str(env_ctx)])
    if rc:
        return tid, f"env build failed: {out[-250:]}"

    wrap = [
        "docker",
        "build",
        *plat,
        "--build-arg",
        f"BASE={env_tag}",
        "--build-arg",
        f"SANDBOXD=agentcore-sandboxd-linux-{arch}",
        "-t",
        uri,
        str(WRAPPER),
    ]
    rc, out = sh(wrap)
    if rc:
        return tid, f"wrap failed: {out[-250:]}"

    if push:
        rc, out = sh(["docker", "push", uri])
        if rc:
            return tid, f"push failed: {out[-250:]}"
    sh(["docker", "rmi", "-f", env_tag, uri])  # drop tags, keep shared layer cache
    return tid, "ok"


def _idset(spec: str) -> set[str]:
    if not spec:
        return set()
    return set(Path(spec[1:]).read_text().split() if spec.startswith("@") else spec.split(","))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task-root", required=True, help="Harbor dataset dir (task dirs inside)")
    p.add_argument("--benchmark", required=True, help="Harbor org/name identifier")
    p.add_argument("--arch", default="arm64", choices=["arm64"])
    p.add_argument("--tasks", default="", help="restrict to these task ids: comma list or @file")
    p.add_argument("--exclude", default="", help="skip these task ids: comma list or @file")
    p.add_argument("--limit", type=int, default=0, help="first N tasks (0 = all)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--no-skip-existing", action="store_true", help="rebuild even if the ECR tag exists")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    ensure_sandboxd(args.arch)  # stage the wrap-stage binary from repo-root sandboxd/

    root = Path(args.task_root)
    only, excl = _idset(args.tasks), _idset(args.exclude)
    tasks = sorted(
        d
        for d in root.iterdir()
        if d.is_dir()
        and (d / "environment" / "Dockerfile").exists()
        and d.name not in excl
        and (not only or d.name in only)
    )
    if args.limit:
        tasks = tasks[: args.limit]

    repo, existing = ecr_login(args.benchmark, args.arch)
    if not args.no_skip_existing:
        skip = {resolve(args.benchmark, d.name, arch=args.arch).image_tag for d in tasks} & existing
        tasks = [d for d in tasks if resolve(args.benchmark, d.name, arch=args.arch).image_tag not in existing]
    else:
        skip = set()
    print(
        f"benchmark={args.benchmark} repo={repo} arch={args.arch} "
        f"todo={len(tasks)} skip-existing={len(skip)} concurrency={args.concurrency}",
        flush=True,
    )

    t0, n, fails = time.time(), 0, []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(build_task, d, args.benchmark, arch=args.arch) for d in tasks]
        for fut in cf.as_completed(futs):
            tid, status = fut.result()
            n += 1
            if status != "ok":
                fails.append((tid, status))
            if n % 10 == 0 or status != "ok":
                rate = n / max(1e-9, time.time() - t0) * 3600
                print(
                    f"  [{n}/{len(tasks)}] {tid:32} {status[:60]}  "
                    f"({rate:.0f}/h, eta {(len(tasks)-n)/max(1e-9,rate):.1f}h)",
                    flush=True,
                )

    print(f"\ndone: {len(tasks)-len(fails)} ok, {len(fails)} failed in {(time.time()-t0)/60:.1f}m")
    for tid, s in fails[:20]:
        print(f"  FAIL {tid}: {s}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
