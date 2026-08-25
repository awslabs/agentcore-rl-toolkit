"""resolve() maps a (benchmark, task_id) to the URIs that locate its resources —
the ECR image and the AgentCore runtime — so any script can check whether a
resource exists and address it by benchmark + task id.

AgentCore runtime names must be < 40 chars, so the runtime name is a fixed-length
hash of the image URI rather than the raw ``<benchmark>-<task_id>``; that rule
lives here so callers never synthesize names by hand.

    >>> n = resolve("tmax/TMax-15K-Harbor", "task_000606_03976796")
    >>> n.image_uri     # ...amazonaws.com/harbor_bench/tmax/tmax-15k-harbor:task_000606_03976796-arm64
    >>> n.runtime_name  # sb_tmax15kh_39102efe2637

Convention: repo = harbor_bench/<org>/<name> lowercased (ECR forbids uppercase);
tag = <task_id>-<arch>; runtime = sb_<benchcode>_<sha256(image_uri)[:12]>.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .config import ECR_REGISTRY
from .errors import ValidationError


@dataclass(frozen=True)
class SandboxNames:
    benchmark: str
    task_id: str
    arch: str
    ecr_repo: str
    image_tag: str
    image_uri: str
    runtime_name: str


def resolve(benchmark: str, task_id: str, *, arch: str = "arm64", suffix: str | None = None) -> SandboxNames:
    """Map (benchmark, task_id) to its ECR image + AgentCore runtime names.

    ``benchmark`` ('org/name') is taken verbatim — split only to build the repo
    path, never matched against a list. Whether it actually EXISTS is Harbor
    Hub's call (ensure_dataset pulls it and fails if unknown), so there is one
    source of truth; here we only check it parses as 'org/name'.

    ``suffix`` appends a per-caller discriminator to the runtime name
    (``sb_<code>_<hash12>_<suffix>``): concurrent rollouts of the SAME task
    (e.g. a GRPO group) each get a private runtime instead of colliding on the
    deterministic name. The image URIs are unchanged.
    """
    org, _, name = benchmark.partition("/")
    if not org or not name or "/" in name:
        raise ValidationError(f"{benchmark!r} is not a Harbor id ('org/name')")
    repo = f"harbor_bench/{org.lower()}/{name.lower()}"
    uri = f"{ECR_REGISTRY}/{repo}:{task_id}-{arch}"
    code = re.sub(r"[^a-z0-9]", "", name.lower())[:8] or "bench"
    runtime = f"sb_{code}_{hashlib.sha256(uri.encode()).hexdigest()[:12]}"
    if suffix:
        runtime = f"{runtime}_{re.sub(r'[^a-zA-Z0-9_]', '', suffix)[:8]}"
    return SandboxNames(benchmark, task_id, arch, repo, f"{task_id}-{arch}", uri, runtime)
