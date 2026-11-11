"""Upload a snapshot of every task's repository at its base commit to S3, for the agent (the runtime has
no internet access). Run after cdk/deploy.sh; rerun if GitHub throttles, finished snapshots are skipped.

  uv run python scripts/prepare_corpus.py [TASKS.jsonl ...] [--workers 8]

Each snapshot is `git archive` of a blobless clone, keeping only text and code files up to 1 MB, uploaded
to s3://$SWEGREP_CORPUS_BUCKET/repos/. The default task files are the training data and both benchmarks.
"""

import argparse
import io
import json
import os
import subprocess
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from rollout.repo_store import snapshot_key

ROOT = Path(__file__).resolve().parents[1]
TASK_FILES = [
    "data/train.jsonl",
    "data/val.jsonl",
    "data/benchmarks/swe-explore.jsonl",
    "data/benchmarks/contextbench-lite.jsonl",
]
CLONES = Path(os.environ.get("SWEGREP_CLONE_DIR", Path.home() / ".cache/swegrep_clones"))
MAX_FILE_BYTES = 1_000_000
MAX_SNAPSHOT_BYTES = 400_000_000

# Every gold file of the training data and benchmarks has one of these suffixes or names.
KEEP_SUFFIXES = {
    ".py", ".pyx", ".pyi", ".txt", ".rst", ".md", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".json", ".c",
    ".h", ".cc", ".cpp", ".hpp", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb", ".sh", ".html",
    ".css", ".xml", ".sql", ".tex", ".in", ".template", ".pony", ".svelte", ".vue", ".mjs", ".cjs", ".mts",
    ".cts", ".scss", ".less", ".pcss", ".styl", ".tpl", ".hbs", ".erb", ".asciidoc", ".adoc", ".mod", ".sum",
    ".cue", ".pot", ".snap", ".kt", ".kts", ".scala", ".swift", ".cs", ".php", ".m", ".mm", ".proto",
    ".gradle", ".properties", ".bzl", ".cmake", ".zig", ".ex", ".exs", ".clj", ".hs", ".lua", ".pl", ".pm",
    ".r", ".jl", ".plist", ".preview", ".vcproj", ".vcxproj", ".txtar", ".j2", ".feature", ".options", ".ps1",
    ".test", ".mdx", ".jq", ".bats", ".lock",
}  # fmt: skip
KEEP_NAMES = {
    "Makefile", "makefile", "GNUmakefile", "Dockerfile", "Containerfile", "Jenkinsfile", "Rakefile",
    "Gemfile", "Procfile", "Vagrantfile", "Brewfile", "BUILD", "WORKSPACE", "CHANGELOG", "README", "LICENSE",
    "NOTICE", "CODEOWNERS", "managed-schema",
}  # fmt: skip
# "javadoc" is generated HTML that overflowed MAX_SNAPSHOT_BYTES; other doc directories hold gold files.
SKIP_DIRS = {
    ".git", "node_modules", ".tox", "__pycache__", ".mypy_cache", ".pytest_cache", "javadoc",
}  # fmt: skip
_repo_locks: dict[str, threading.Lock] = {}  # one clone or fetch per repository at a time
_guard = threading.Lock()


def repo_lock(repo: str) -> threading.Lock:
    with _guard:
        return _repo_locks.setdefault(repo, threading.Lock())


def git_archive(repo: str, commit: str) -> bytes:
    """The repository's tree at commit, as a tar (git archive honours export-ignore)."""
    clone = CLONES / repo.replace("/", "__")
    git = ["git", "-C", str(clone)]
    with repo_lock(repo):
        if not (clone / ".git").exists():  # blobless: full clones of every repository fill a 500 GB disk
            url = f"https://github.com/{repo}.git"
            subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", url, str(clone)], check=True)
    archive = subprocess.run([*git, "archive", "--format=tar", commit], capture_output=True)
    if archive.returncode != 0:  # the commit may only be on a pull-request ref: fetch it by hash
        with repo_lock(repo):
            if subprocess.run([*git, "fetch", "--quiet", "origin", commit], capture_output=True).returncode != 0:
                subprocess.run([*git, "fetch", "--quiet", "--all"], check=True)
        archive = subprocess.run([*git, "archive", "--format=tar", commit], capture_output=True, check=True)
    return archive.stdout


def snapshot(repo: str, commit: str) -> tuple[bytes, int]:
    """The filtered snapshot as tar.gz, and the number of files in it."""
    out = io.BytesIO()
    kept = total = dropped = 0
    with (
        tarfile.open(fileobj=io.BytesIO(git_archive(repo, commit))) as src,
        tarfile.open(fileobj=out, mode="w:gz", compresslevel=6) as dst,
    ):
        for member in src:
            path = Path(member.name)
            if (
                not member.isreg()
                or not 0 < member.size <= MAX_FILE_BYTES
                or SKIP_DIRS.intersection(path.parts)
                or (path.suffix.lower() not in KEEP_SUFFIXES and path.name not in KEEP_NAMES)
            ):
                continue
            if total + member.size > MAX_SNAPSHOT_BYTES:
                dropped += 1
                continue
            dst.addfile(member, src.extractfile(member))
            total += member.size
            kept += 1
    if dropped:
        print(f"WARNING {repo}@{commit[:8]}: over {MAX_SNAPSHOT_BYTES:,} bytes, dropped {dropped} files", flush=True)
    return out.getvalue(), kept


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*", type=Path, default=[ROOT / f for f in TASK_FILES])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    bucket = os.environ["SWEGREP_CORPUS_BUCKET"]
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"), config=Config(max_pool_connections=64))
    rows = [json.loads(line) for path in args.tasks for line in path.read_text().splitlines()]
    pairs = list(dict.fromkeys((row["repo"], row["base_commit"]) for row in rows))
    print(f"{len(pairs)} snapshots -> s3://{bucket}/repos/")

    def upload(repo: str, commit: str) -> str:
        key = snapshot_key(repo, commit)
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return "skipped"
        except ClientError as e:
            if e.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
        data, files = snapshot(repo, commit)
        s3.put_object(Bucket=bucket, Key=key, Body=data)
        print(f"{repo}@{commit[:8]}: {files} files", flush=True)
        return "uploaded"

    def try_upload(pair: tuple[str, str]) -> str:
        try:
            return upload(*pair)
        except Exception as e:  # noqa: BLE001 - reported, and retried on the next run
            print(f"FAILED {pair[0]}@{pair[1][:8]}: {e}", flush=True)
            return "failed"

    CLONES.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        outcomes = list(pool.map(try_upload, pairs))
    print({outcome: outcomes.count(outcome) for outcome in set(outcomes)})
    if "failed" in outcomes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
