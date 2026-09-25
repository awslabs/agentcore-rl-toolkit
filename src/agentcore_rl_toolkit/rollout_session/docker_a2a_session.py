"""A rollout session that drives the agent over A2A on a local ``docker run`` container (plain HTTP, no ACR)."""

from __future__ import annotations

import asyncio
import json
import logging

import backoff
import httpx
from a2a.client import Client

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_role_credentials
from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict, measure_span_persistent
from agentcore_rl_toolkit.rollout_session.a2a_client import A2ARolloutSession, build_a2a_client

logger = logging.getLogger(__name__)


class DockerA2ASession(A2ARolloutSession):
    """A2A rollout session backed by an A2A server in a local Docker container."""

    def __init__(
        self,
        session_id: str,
        session_state: PersistentDict,
        *,
        agent_image_uri: str,
        iam_role_arn: str,
        log_group: str,
        log_region: str,
    ):
        super().__init__(session_id, session_state)
        self.agent_image_uri = agent_image_uri
        self.iam_role_arn = iam_role_arn
        self.log_group = log_group
        self.log_region = log_region

        # Set once `docker run` succeeds so shutdown skips a container that never started.
        self._running = False
        self.endpoint: str | None = None
        self._httpx: httpx.AsyncClient | None = None

    async def _client(self) -> Client:
        assert self.endpoint is not None, "client requested before the container started"
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        return build_a2a_client(self._httpx, f"{self.endpoint}/")

    async def setup(self, task: dict) -> None:
        await self.session_state.update(
            {
                "agent_image_uri": self.agent_image_uri,
                "iam_role_arn": self.iam_role_arn,
            }
        )
        async with measure_span_persistent("container_start", self.session_state):
            self.endpoint = await self._start_container()
            logger.info("Launched agent at %s for %s", self.endpoint, self.session_id)
        await self.session_state.set("endpoint", self.endpoint)
        await _wait_ping_ok(await self._client_httpx(), f"{self.endpoint}/ping")

        async with measure_span_persistent("task_setup", self.session_state):
            await super().setup(task)
            logger.info("Prepared environment at %s for %s", self.endpoint, self.session_id)

    async def _client_httpx(self) -> httpx.AsyncClient:
        await self._client()  # opens the client
        assert self._httpx is not None
        return self._httpx

    async def shutdown(self) -> None:
        if self._running:
            self._running = False
            self.endpoint = None
            await _docker("stop", "-t", "0", self.session_id, check=False)
        if self._httpx is not None:
            await self._httpx.aclose()
            self._httpx = None

    async def _start_container(self) -> str:
        """``docker run`` the agent image, then wait for its endpoint. Named after the session so shutdown finds it."""
        await _docker(
            "run",
            "--name",
            self.session_id,
            "--rm",
            "--detach",
            "--cpus",
            "2",
            "--network",
            "bridge",
            # Daemon-side cloudwatch logs outlive the `--rm` container.
            "--log-driver",
            "awslogs",
            "--log-opt",
            f"awslogs-region={self.log_region}",
            "--log-opt",
            f"awslogs-group={self.log_group}",
            "--log-opt",
            "awslogs-create-group=true",
            "--log-opt",
            f"awslogs-stream={self.session_id}",
            *[c for k, v in get_role_credentials(self.iam_role_arn).items() for c in ["-e", f"{k}={v}"]],
            self.agent_image_uri,
        )
        self._running = True
        ip = await wait_ip_address(self.session_id)
        return f"http://{ip}:9000"


async def _docker(*args: str, check: bool = True) -> str:
    """Run a docker CLI command over an async subprocess, returning stdout."""
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if check and proc.returncode != 0:
        raise RuntimeError(f"command docker {args} returned non-zero exit status {proc.returncode}: {stderr.decode()}")
    return stdout.decode()


@backoff.on_exception(backoff.constant, Exception, max_time=60)
async def wait_ip_address(container_name: str) -> str:
    out = await _docker("inspect", "-f", "{{json .NetworkSettings.Networks}}", container_name)
    networks = json.loads(out)
    ip = networks[next(iter(networks.keys()))]["IPAddress"]
    assert ip is not None and ip != ""
    return ip


@backoff.on_exception(backoff.constant, Exception, interval=3)
async def _wait_ping_ok(client: httpx.AsyncClient, url: str) -> None:
    """Wait for the container's A2A ``/ping`` to answer 200; it refuses connections at first."""
    response = await client.get(url, timeout=httpx.Timeout(5.0))
    assert response.status_code == 200
