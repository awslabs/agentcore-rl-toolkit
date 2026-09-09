"""A rollout session that runs the agent as a local Docker container."""

import asyncio
import json
import logging

import aiohttp
import backoff
from pydantic import BaseModel

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_role_credentials
from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict, measure_span_persistent
from agentcore_rl_toolkit.rollout_session.lifecycle import RolloutSession
from agentcore_rl_toolkit.rollout_session.wire import (
    InvocationRequest,
    InvocationResponse,
    RolloutDumpRequest,
    RolloutDumpResponse,
    RolloutSetupRequest,
    RolloutStartRequest,
    RolloutStatusRequest,
    RolloutStatusResponse,
)

logger = logging.getLogger(__file__)


class DockerSession(RolloutSession):
    """A rollout session backed by an HTTP server in a local Docker container."""

    def __init__(
        self,
        session_id,
        session_state: PersistentDict,
        *,
        agent_image_uri: str,
        iam_role_arn: str,
        log_group: str,
        log_region: str,
    ):
        self.session_id = session_id
        self.session_state = session_state
        self.agent_image_uri = agent_image_uri
        self.iam_role_arn = iam_role_arn
        self.log_group = log_group
        self.log_region = log_region

        # Set only once `docker run` succeeds, so shutdown skips a container that
        # never started.
        self._running = False
        self.endpoint: str | None = None

    async def __aenter__(self) -> "DockerSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    async def setup(self, task: dict) -> None:
        await self.session_state.update(
            {
                "agent_image_uri": self.agent_image_uri,
                "iam_role_arn": self.iam_role_arn,
            }
        )

        async with measure_span_persistent("container_start", self.session_state):
            self.endpoint = await self._start_container()
            logger.info(f"Launched agent at {self.endpoint} for {self.session_id}")
        await self.session_state.set("endpoint", self.endpoint)

        async with measure_span_persistent("task_setup", self.session_state):
            await start_and_wait_setup(self.endpoint, task)
            logger.info(f"Prepared environment at {self.endpoint} for {self.session_id}")

    async def run(self, task: dict) -> RolloutDumpResponse:
        assert self.endpoint is not None, "run called before setup"
        rollout = await start_and_wait_rollout(self.endpoint, self.session_id, task)
        logger.info(f"Received rollout from endpoint at {self.endpoint} for {self.session_id}")
        return rollout

    async def shutdown(self) -> None:
        if self._running:
            self._running = False
            self.endpoint = None
            await _docker("stop", "-t", "0", self.session_id, check=False)

    async def _start_container(self) -> str:
        """``docker run`` the agent image, then wait for the container's endpoint.

        The container is named after the session so the IP lookup and
        :meth:`shutdown` can find it.
        """
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
            # Daemon-side cloudwatch logs, one stream per container, so output
            # outlives the `--rm` container.
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
            "/agent/entrypoint.sh",
        )
        self._running = True
        ip = await wait_ip_address(self.session_id)
        return f"http://{ip}:8080"


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


# The wire protocol over HTTP: each call POSTs an InvocationRequest to /invocations.
# A container refuses connections for a while after it starts, hence the waiting.


async def start_and_wait_setup(endpoint: str, task: dict):
    async with aiohttp.ClientSession() as session:
        await wait_status_ok(session, f"{endpoint}/ping")
        await http_call_check(session, "POST", f"{endpoint}/invocations", RolloutSetupRequest(task_input=task), "setup")
        response = await wait_setup_done(session, endpoint, "setup")
        assert response.exception is None, f"Setup failed with exception: {response.exception}"


async def start_and_wait_rollout(endpoint: str, rollout_id: str, task: dict) -> RolloutDumpResponse:
    async with aiohttp.ClientSession() as session:
        await wait_status_ok(session, f"{endpoint}/ping")
        rollout_start = RolloutStartRequest(
            rollout_id=rollout_id,
            task_input=task,
        )
        await http_call_check(session, "POST", f"{endpoint}/invocations", rollout_start, rollout_id)

        @backoff.on_predicate(backoff.constant, lambda x: x is None, interval=10)
        async def get_terminal_rollout() -> RolloutDumpResponse | None:
            response: RolloutStatusResponse = await http_call_check(
                session, "POST", f"{endpoint}/invocations", RolloutStatusRequest(), rollout_id
            )  # type: ignore
            if response.done:
                return await http_call_check(
                    session, "POST", f"{endpoint}/invocations", RolloutDumpRequest(), rollout_id
                )  # type: ignore

        rollout = await get_terminal_rollout()
        assert rollout is not None, f"Failed to get rollout for {rollout_id}"
        return rollout


@backoff.on_predicate(backoff.constant, lambda x: not x.done, interval=10)
async def wait_setup_done(session, endpoint, rollout_id) -> RolloutStatusResponse:
    return await http_call_check(session, "POST", f"{endpoint}/invocations", RolloutStatusRequest(), rollout_id)  # type: ignore


async def http_call_check(
    session: aiohttp.ClientSession, method: str, endpoint: str, payload: BaseModel, rollout_id: str
):
    json_body = InvocationRequest(payload=payload).model_dump(mode="json")  # type: ignore
    async with session.request(method, endpoint, json=json_body) as http_response:
        text = await http_response.text()
        assert http_response.status == 200, (
            f"Received {http_response.status} status code from {endpoint} for {rollout_id} "
            f"with request {json_body} and response: {text}"
        )
        return InvocationResponse.model_validate_json(text).payload


@backoff.on_exception(backoff.constant, Exception, interval=3)
async def wait_status_ok(session: aiohttp.ClientSession, endpoint: str):
    async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=5)) as response:
        assert response.status == 200
