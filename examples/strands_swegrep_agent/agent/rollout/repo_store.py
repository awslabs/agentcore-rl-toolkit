"""Repository checkouts: ``ensure_checkout(repo, commit)`` downloads the snapshot that
scripts/prepare_corpus.py uploaded to s3://$SWEGREP_CORPUS_BUCKET/repos/ and extracts it once
(the runtime has no internet access)."""

import os
import tarfile
import threading
from pathlib import Path

import boto3
from botocore.config import Config

# A warm runtime container keeps its checkouts, so later tasks on the same repo skip the download.
CACHE_ROOT = Path(os.environ.get("SWEGREP_CACHE_DIR", "/tmp/swegrep_repos"))

_s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"), config=Config(max_pool_connections=64))
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def checkout_dir(repo: str, commit: str) -> Path:
    return CACHE_ROOT / f"{repo.replace('/', '__')}__{commit}"


def snapshot_key(repo: str, commit: str) -> str:
    return f"repos/{checkout_dir(repo, commit).name}.tar.gz"


def ensure_checkout(repo: str, commit: str) -> Path:
    dest = checkout_dir(repo, commit)
    done = dest / ".swegrep_extracted"
    with _locks_guard:
        lock = _locks.setdefault(dest.name, threading.Lock())
    with lock:  # one download per snapshot; different snapshots in parallel
        if not done.exists():
            dest.mkdir(parents=True, exist_ok=True)
            archive = dest.with_name(dest.name + ".tar.gz")
            _s3.download_file(os.environ["SWEGREP_CORPUS_BUCKET"], snapshot_key(repo, commit), str(archive))
            with tarfile.open(archive) as tar:
                tar.extractall(dest, filter="data")  # refuses paths and links that leave dest
            archive.unlink()
            done.touch()
    return dest
