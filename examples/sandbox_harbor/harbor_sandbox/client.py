"""HarborSandboxClient — SandboxClient plus Harbor-aware runtime lifecycle.

Subclass of the toolkit's ``SandboxClient`` (session API: start/exec/stop) that
adds create/release for a task's AgentCore runtime:

    sb = HarborSandboxClient.create("tmax/TMax-15K-Harbor", "task_000606_03976796")
    with sb.start() as s:
        s.exec("uname -m")
    sb.release()   # delete THIS client's runtime (the lease pattern)

Self-contained: no local corpus or config files. A task's existence and
built-ness are answered by ONE remote source of truth — the conventional ECR
tag (see ``naming.resolve``). ``create`` is idempotent (create-if-missing /
reuse-if-present) and does NOT build images: a missing tag raises
``ImageNotFoundError``. Runtime naming:
``sb_<benchcode>_<sha256(image_uri)[:12]>``.
"""
from __future__ import annotations

import logging
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from agentcore_rl_toolkit.sandbox import SandboxClient

from .config import IDLE_SESSION_TIMEOUT_S, MAX_LIFETIME_S, NETWORK_CONFIG, REGION, ROLE_ARN
from .errors import ImageNotFoundError
from .naming import resolve

logger = logging.getLogger(__name__)


def _control(region: str = REGION):
    return boto3.client(
        "bedrock-agentcore-control", region_name=region, config=Config(retries={"max_attempts": 10, "mode": "adaptive"})
    )


def _find_runtime_id(ctrl, name: str) -> str | None:
    kw: dict = {"maxResults": 100}
    while True:
        resp = ctrl.list_agent_runtimes(**kw)
        for r in resp.get("agentRuntimes", []):
            if r.get("agentRuntimeName") == name:
                return r["agentRuntimeId"]
        kw["nextToken"] = resp.get("nextToken")
        if not kw["nextToken"]:
            return None


class HarborSandboxClient(SandboxClient):
    """SandboxClient that also owns runtime lifecycle for Harbor benchmarks.
    Instantiable anywhere with AWS credentials — dev host, harness container, CI
    — since all validation runs against ECR + the AgentCore control plane."""

    @classmethod
    def create(
        cls, benchmark: str, task_id: str, arch: str = "arm64", wait_ready_s: int = 180, **client_kwargs
    ) -> "HarborSandboxClient":
        """Ensure the task's runtime exists and is READY; return a client bound
        to it (identical whether it created the runtime or reused a live one).

        Raises ImageNotFoundError if the conventional ECR tag is absent.
        """
        names = resolve(benchmark, task_id, arch=arch)

        # the ECR tag is the single source of truth for "this task exists AND is
        # built" (task existence + built-ness in one check).
        ecr = boto3.client("ecr", region_name=REGION)
        try:
            ecr.describe_images(repositoryName=names.ecr_repo, imageIds=[{"imageTag": names.image_tag}])
        except (ecr.exceptions.ImageNotFoundException, ecr.exceptions.RepositoryNotFoundException):
            raise ImageNotFoundError(
                f"{names.image_uri} not in ECR — unknown task or not built " f"for {arch}"
            ) from None

        body = dict(
            agentRuntimeArtifact={"containerConfiguration": {"containerUri": names.image_uri}},
            roleArn=ROLE_ARN,
            protocolConfiguration={"serverProtocol": "HTTP"},
            lifecycleConfiguration={"idleRuntimeSessionTimeout": IDLE_SESSION_TIMEOUT_S, "maxLifetime": MAX_LIFETIME_S},
            **NETWORK_CONFIG,
        )
        name = names.runtime_name
        ctrl = _control()
        deadline = time.time() + wait_ready_s

        # create-or-reuse: a lost create race, or a create right after release()
        # (same name still DELETING), both resolve to one live runtime.
        while True:
            try:
                arn = ctrl.create_agent_runtime(agentRuntimeName=name, **body)["agentRuntimeArn"]
                logger.info(f"created runtime {name}")
                break
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") != "ConflictException":
                    raise
            rid = _find_runtime_id(ctrl, name)
            info = ctrl.get_agent_runtime(agentRuntimeId=rid) if rid else None
            if info and info["status"] != "DELETING":
                arn = info["agentRuntimeArn"]
                break
            if time.time() > deadline:
                raise RuntimeError(f"runtime {name}: create conflict unresolved " f"after {wait_ready_s}s")
            time.sleep(3)

        # wait for control-plane READY (seconds on both substrates)
        rid = arn.split("/")[-1]
        while ctrl.get_agent_runtime(agentRuntimeId=rid)["status"] != "READY":
            if time.time() > deadline:
                raise RuntimeError(f"runtime {name} not READY after {wait_ready_s}s")
            time.sleep(3)
        return cls(runtime_arn=arn, **client_kwargs)

    def release(self) -> bool:
        """Delete THIS client's runtime (the lease pattern: the object knows its
        own ARN, so no coordinates are re-derived). Idempotent — returns False if
        the runtime was already gone or still mid-deletion."""
        ctrl = _control(self._parse_region_from_arn(self.runtime_arn))
        try:
            ctrl.delete_agent_runtime(agentRuntimeId=self.runtime_arn.split("/")[-1])
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "ResourceNotFoundException" or (code == "ConflictException" and "DELETING" in str(e)):
                return False
            raise
