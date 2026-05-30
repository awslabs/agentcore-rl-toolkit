"""AgentCore LLM server manager.

Subclasses verl's LLMServerManager to:
1. Use AgentCoreVLLMReplica for ghost-request protection.
2. Launch and manage the rllm-model-gateway subprocess.
3. Register vLLM server addresses with the gateway.
"""

import logging
import socket
import subprocess
import sys
import time

from omegaconf import DictConfig
from rllm_model_gateway import GatewayClient
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.workers.rollout.llm_server import LLMServerManager

from agentcore_rl_toolkit.backends.verl.vllm_replica import AgentCoreVLLMReplica

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_HEALTH_POLL_INTERVAL = 0.5
_HEALTH_POLL_TIMEOUT = 30.0


def _get_routable_ip() -> str:
    """Return the machine's routable IPv4 address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip: str = s.getsockname()[0]
            if not ip.startswith("127.") and ip != "::1":
                return ip
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        for info in infos:
            ip = str(info[4][0])
            if not ip.startswith("127.") and ip != "::1":
                return ip
    except Exception:
        pass

    return "127.0.0.1"


class AgentCoreLLMServerManager(LLMServerManager):
    """LLM server manager with rllm-model-gateway for AgentCore rollout.

    Overrides the replica class to use AgentCoreVLLMReplica and adds gateway
    lifecycle management (start, register workers, stop).
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
    ):
        # Set replica class before super().__init__ which reads it
        self.rollout_replica_class = AgentCoreVLLMReplica
        super().__init__(config=config, worker_group=worker_group, rollout_resource_pool=rollout_resource_pool)

        # Gateway state
        self._gateway_process = None
        self._gateway_client = None
        self._gateway_url = None

        agentcore_config = config.actor_rollout_ref.rollout.agentcore
        self._gateway_port = getattr(agentcore_config, "gateway_port", 9090)
        self._gateway_host = _get_routable_ip()
        self._gateway_store = getattr(agentcore_config, "gateway_store", "memory")

    def _start_gateway(self):
        """Launch model gateway as a subprocess and poll until healthy."""
        cmd = [
            sys.executable,
            "-m",
            "rllm_model_gateway",
            "--host",
            "0.0.0.0",
            "--port",
            str(self._gateway_port),
            "--store",
            self._gateway_store,
            "--log-level",
            "warning",
        ]

        logger.info("Starting model gateway subprocess: %s", " ".join(cmd))
        self._gateway_process = subprocess.Popen(cmd)

        gateway_url = f"http://{self._gateway_host}:{self._gateway_port}"
        self._gateway_url = gateway_url
        self._gateway_client = GatewayClient(gateway_url)

        deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT
        while time.monotonic() < deadline:
            try:
                self._gateway_client.health()
                logger.info("Model gateway healthy at %s", gateway_url)
                return
            except Exception as e:
                if self._gateway_process.poll() is not None:
                    raise RuntimeError(
                        f"Gateway process exited unexpectedly (rc={self._gateway_process.returncode})"
                    ) from e
                time.sleep(_HEALTH_POLL_INTERVAL)

        self._gateway_process.terminate()
        raise TimeoutError(f"Model gateway did not become healthy within {_HEALTH_POLL_TIMEOUT}s")

    def _register_gateway_workers(self):
        """Register all vLLM server addresses with the gateway."""
        for server_address in self.server_addresses:
            if server_address.startswith("http"):
                url = server_address
            else:
                url = f"http://{server_address}"
            worker_id = self._gateway_client.add_worker(url=url)
            logger.info("Registered gateway worker %s -> %s", worker_id, url)

    def _stop_gateway(self):
        """Terminate the gateway subprocess."""
        if self._gateway_client is not None:
            try:
                self._gateway_client.close()
            except Exception:
                pass
            self._gateway_client = None

        if self._gateway_process is not None:
            self._gateway_process.terminate()
            try:
                self._gateway_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._gateway_process.kill()
            self._gateway_process = None

    @property
    def gateway_url(self) -> str:
        """Public URL for the model gateway."""
        if self._gateway_url is None:
            raise RuntimeError("Gateway not started. Call start_gateway() first.")
        return self._gateway_url

    def start_gateway(self):
        """Start the gateway and register vLLM workers. Call after create()."""
        self._start_gateway()
        self._register_gateway_workers()

    def __del__(self):
        self._stop_gateway()
