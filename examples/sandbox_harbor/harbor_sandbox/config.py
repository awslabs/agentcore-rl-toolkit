"""Deployment constants for the Harbor-on-AgentCore example — one source of truth.

Account-specific identifiers (account id, role, bucket) are NOT committed: the
values below are placeholders, overridden at import time from a git-ignored
``.env`` beside this file (copy ``.env.example`` and fill it in). That
``.env`` is staged into the harness image by build_push.sh, so the same values
reach the in-container code, and build_push.sh sources it too — one file feeds
both shell and Python. Everything here is deployment-level, NOT per-benchmark:
benchmark-specific names are DERIVED in ``naming.resolve()``.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_env(path: Path) -> None:
    """Minimal .env reader: ``KEY=value`` lines, ``#`` comment lines, optional
    quotes. Real environment variables win (``setdefault``), so an exported var
    or deploy-time injection overrides the file."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env(Path(__file__).with_name(".env"))

# -- overridable scalars (placeholders until .env / the environment supplies them) --
REGION = os.environ.get("REGION", "us-west-2")
ACCOUNT = os.environ.get("ACCOUNT", "000000000000")
S3_BUCKET = os.environ.get("S3_BUCKET", "your-rollout-bucket")  # rollout records + run outputs
ROLE_NAME = os.environ.get("ROLE_NAME", "your-runtime-role")  # runtime execution role (in ACCOUNT)
HARNESS_REPO = os.environ.get("HARNESS_REPO", "your-harness-repo")  # ECR repo for the harness images
IDLE_SESSION_TIMEOUT_S = 900  # runtime lifecycle: idle -> stop
MAX_LIFETIME_S = 28800  # runtime lifecycle: hard cap (8h)

# -- derived (built from the resolved values above) --
ECR_REGISTRY = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/{ROLE_NAME}"

# Network config for the serverless arm64 microVM runtime, passed to
# create_agent_runtime.
NETWORK_CONFIG = {"networkConfiguration": {"networkMode": "PUBLIC"}}
