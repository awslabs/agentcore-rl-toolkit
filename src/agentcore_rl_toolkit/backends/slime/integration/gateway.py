"""Manages rllm-model-gateway lifecycle for slime + ACR integration."""

import logging
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _import_gateway_client():
    try:
        from rllm_model_gateway import GatewayClient

        return GatewayClient
    except ImportError as err:
        raise ImportError(
            "rllm-model-gateway is required for slime integration. "
            "Install with: pip install agentcore-rl-toolkit[slime]"
        ) from err


@dataclass
class GatewayConfig:
    port: int = 9090
    host: str | None = None
    db_path: str | None = None
    add_logprobs: bool = True
    add_return_token_ids: bool = True
    strip_vllm_fields: bool = False
    # Cumulative token mode: gateway rewrites turn N>1 to /v1/completions with a
    # pre-tokenized, prefix-extending prompt (drift-free multi-turn). Requires
    # `model` (tokenizer source) and a `renderer_family` the model supports.
    cumulative_token_mode: bool = False
    renderer_family: str = "auto"
    # Served model checkpoint (path or HF id). The gateway loads its tokenizer to
    # render each turn's prompt to token ids for SGLang /generate and to decode
    # completion ids for parsing; in cumulative mode it also resolves the renderer.
    # Always required for the slime backend (use_sglang is always on); sourced from
    # slime's --hf-checkpoint at the call site.
    model: str | None = None
    # SGLang tool-call / reasoning parser names for parsing /generate output text
    # (same parsers SGLang's /v1/chat/completions uses). Required for tool-using
    # agents in use_sglang mode; without the tool parser, tool calls come back as
    # plain assistant text.
    sglang_tool_call_parser: str | None = None
    sglang_reasoning_parser: str | None = None
    # Log level for the gateway subprocess (passed as --log-level). Also controls
    # uvicorn access logs and httpx request logs, which are very chatty at INFO
    # (one line per /v1/chat/completions and /generate call). Default WARNING so
    # the training stdout is dominated by our own batch/metric logs.
    log_level: str = "warning"


class SlimeGatewayManager:
    """Manages rllm-model-gateway for slime training with ACR agents.

    Starts a gateway process, registers SGLang engine(s) as workers,
    and provides session management for per-episode trace capture.
    """

    def __init__(self, config: GatewayConfig | None = None):
        self._config = config or GatewayConfig()
        self._process: subprocess.Popen | None = None
        self._client = None

    @property
    def client(self):
        if self._client is None:
            raise RuntimeError("Gateway not started. Call start() first.")
        return self._client

    def start(self, sglang_router_url: str) -> None:
        """Start gateway process, register SGLang router as worker.

        Args:
            sglang_router_url: SGLang router URL, e.g. "http://10.0.0.1:30000/v1"
        """
        GatewayClient = _import_gateway_client()
        cfg = self._config
        cmd = [
            "python",
            "-m",
            "rllm_model_gateway",
            "--port",
            str(cfg.port),
        ]
        if cfg.host:
            cmd.extend(["--host", cfg.host])
        if cfg.db_path:
            cmd.extend(["--db-path", cfg.db_path])
        if cfg.log_level:
            cmd.extend(["--log-level", cfg.log_level])

        if not cfg.model:
            raise ValueError(
                "GatewayConfig.model is required for the slime backend, "
                "the gateway needs the served checkpoint to load its tokenizer. "
                "Pass it through --hf-checkpoint of slime's training script."
            )
        cmd.append("--use-sglang")
        cmd.extend(["--model", cfg.model])
        # SGLang output parsers (tool calls / reasoning). The same parser names
        # slime passes to its SGLang server, so the gateway parses /generate output
        # identically. Without the tool parser, tool calls come back as plain text.
        if cfg.sglang_tool_call_parser:
            cmd.extend(["--sglang-tool-call-parser", cfg.sglang_tool_call_parser])
        if cfg.sglang_reasoning_parser:
            cmd.extend(["--sglang-reasoning-parser", cfg.sglang_reasoning_parser])
        # Cumulative mode adds the cross-turn bridge; it needs --renderer-family
        # for a local checkpoint path (renderers can auto-infer it only for HF ids).
        if cfg.cumulative_token_mode:
            cmd.append("--cumulative-token-mode")
            if cfg.renderer_family and cfg.renderer_family != "auto":
                cmd.extend(["--renderer-family", cfg.renderer_family])
        # Note: add_logprobs, add_return_token_ids, strip_vllm_fields are
        # gateway config options set via GatewayClient after startup, not CLI args.

        self._process = subprocess.Popen(cmd)
        # Tokenizer (always) + renderer (cumulative) load can take longer than a
        # plain proxy start, so allow a generous health window.
        self._wait_for_health(timeout=120)

        base = f"http://{cfg.host or 'localhost'}:{cfg.port}"
        self._client = GatewayClient(base)
        self._client.add_worker(url=sglang_router_url)
        logger.info("Gateway started at %s, worker registered: %s", base, sglang_router_url)

    def _wait_for_health(self, timeout: float) -> None:
        """Poll gateway /health until it responds."""
        import httpx

        base = f"http://{self._config.host or 'localhost'}:{self._config.port}"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(f"{base}/health", timeout=2)
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"Gateway did not become healthy within {timeout}s")

    def create_session(self, session_id: str, sampling_params: dict | None = None) -> str:
        """Create a gateway session, return session URL for the agent's base_url."""
        # sampling_params support depends on rllm-model-gateway version
        try:
            self.client.create_session(session_id=session_id, sampling_params=sampling_params)
        except TypeError:
            # Fallback for gateway versions that don't support sampling_params
            self.client.create_session(session_id=session_id)
        return self.client.get_session_url(session_id)

    def get_traces(self, session_id: str) -> list:
        """Retrieve all TraceRecords for a completed session."""
        return self.client.get_session_traces(session_id)

    def delete_session(self, session_id: str) -> None:
        """Clean up session data after trace retrieval."""
        try:
            self.client.delete_session(session_id)
        except Exception:
            logger.warning("Failed to delete session %s", session_id, exc_info=True)

    def add_worker(self, url: str) -> None:
        """Register an additional SGLang engine URL."""
        self.client.add_worker(url=url)

    def shutdown(self) -> None:
        """Terminate the gateway process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            logger.info("Gateway process terminated")
