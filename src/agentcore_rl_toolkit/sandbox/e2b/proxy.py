"""A loopback E2B API backed by AgentCore Runtime sessions.

The unmodified ``e2b`` SDK talks to this proxy as if it were the E2B service.
Control-plane calls (create, kill, timeout, ...) become ``InvokeAgentRuntime`` and
``StopRuntimeSession`` calls. Data-plane calls carry ``E2b-Sandbox-Id`` and
``E2b-Sandbox-Port`` headers; the proxy signs each with SigV4 and streams it to
that port in the sandbox's session through the session endpoint, where envd
serves E2B's process and filesystem APIs.

``templateID`` is the AgentCore runtime ARN. The runtime image runs
``agentcore-sandboxd`` on its serving port and envd on ``envd_port``, which must
be in the runtime's ``allowedPorts``; see ``examples/sandbox_e2b``.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import hmac
import http.client
import json
import logging
import re
import secrets
import select
import socket
import ssl
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, parse_qsl, quote, urlsplit

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

E2B_ENVD_PORT = 49983  # the envd port the SDK addresses
DEFAULT_ENVD_PORT = 10983
DEFAULT_ENVD_VERSION = "0.10.2"
SERVICE = "bedrock-agentcore"

_RUNTIME_ARN = re.compile(r"arn:aws[a-z-]*:bedrock-agentcore:([a-z0-9-]+):\d{12}:runtime/[A-Za-z0-9_-]+")
_SANDBOX_PATH = re.compile(r"(?:/v2)?/sandboxes/([^/]+)(?:/([a-z]+))?")
_SESSION_ENDPOINT_HEADER = "x-amzn-bedrock-agentcore-runtime-session-endpoint-url"
_PLATFORM_ERROR_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Error"
_SESSION_ENDPOINT_DOMAINS = (".on.aws", ".amazonaws.com", ".amazonaws.com.cn", ".api.aws")
_HOP_BY_HOP = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ]
)
_REQUEST_DROPPED = _HOP_BY_HOP | {
    "host",
    "content-length",
    "authorization",
    "x-api-key",
    "e2b-sandbox-id",
    "e2b-sandbox-port",
    "x-amz-date",
    "x-amz-security-token",
    "x-amz-content-sha256",
}
_STREAM_CHUNK_BYTES = 64 * 1024


class _ProxyError(Exception):
    """An E2B-style API error: ``{"code": status, "message": ...}``."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class _Sandbox:
    sandbox_id: str
    runtime_arn: str
    region: str
    host: str  # session endpoint hostname, without a port prefix
    access_token: str
    metadata: dict[str, str]
    started_at: dt.datetime
    end_at: dt.datetime
    timer: threading.Timer | None = field(default=None, repr=False)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(t: dt.datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


def _canonical_query(url: str) -> str:
    """``url`` with its query encoded as SigV4 canonicalizes it."""
    parts = urlsplit(url)
    if not parts.query:
        return url
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    query = "&".join(f"{quote(k, safe='-_.~')}={quote(v, safe='-_.~')}" for k, v in pairs)
    return parts._replace(query=query).geturl()


def _client_error(exc: Exception) -> _ProxyError:
    if isinstance(exc, ClientError):
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 500
        error = exc.response.get("Error", {})
        return _ProxyError(status, f"AgentCore {error.get('Code', 'error')}: {error.get('Message', exc)}")
    return _ProxyError(502, f"AgentCore request failed: {exc!r}")


def _dropped(sock: socket.socket) -> bool:
    """Whether an idle kept-alive upstream connection was closed (readable means EOF)."""
    return bool(select.select([sock], [], [], 0)[0])


class AgentCoreE2BProxy:
    """Serves the E2B API on ``url``, one AgentCore Runtime session per sandbox.

    Pass ``api_key=proxy.api_key, api_url=proxy.url, sandbox_url=proxy.url`` to
    ``Sandbox.create(template=<runtime ARN>)``, or ``proxy.environ`` to processes
    that use the SDK. Requests are signed with
    ``session``'s credentials (default ``boto3.Session()``) in the runtime's Region,
    so the caller needs ``bedrock-agentcore:InvokeAgentRuntime``,
    ``InvokeAgentRuntimeProxy`` and ``StopRuntimeSession`` on the runtime.

    Sandbox state and timeouts live in this process. ``close()`` (or leaving a
    ``with`` block) stops the sessions of sandboxes that are still running.
    """

    def __init__(
        self,
        *,
        session: boto3.Session | None = None,
        qualifier: str = "DEFAULT",
        envd_port: int = DEFAULT_ENVD_PORT,
        envd_version: str = DEFAULT_ENVD_VERSION,
        default_user: str = "user",
        default_workdir: str | None = "/home/user",
        host: str = "127.0.0.1",
        port: int = 0,
        read_timeout: float = 900,
        connect_timeout: float = 15,
    ):
        self.session = session or boto3.Session()
        self._credentials = self.session.get_credentials()
        if self._credentials is None:
            raise ValueError("the boto3 session has no AWS credentials")
        self.qualifier = qualifier
        self.envd_port = envd_port
        self.envd_version = envd_version
        self.default_user = default_user
        self.default_workdir = default_workdir
        self.read_timeout = read_timeout
        self.connect_timeout = connect_timeout
        self.api_key = "e2b_" + secrets.token_hex(20)
        self._tls = ssl.create_default_context()
        self._lock = threading.Lock()
        self._sandboxes: dict[str, _Sandbox] = {}
        self._clients: dict[str, Any] = {}
        self._closed = False
        self._server = _Server((host, port), _handler_class(self))
        bound_host, bound_port = self._server.server_address[:2]
        self.url = f"http://{bound_host}:{bound_port}"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.1},  # how long close() may wait for the serving thread
            name="agentcore-e2b-proxy",
            daemon=True,
        )
        self._thread.start()

    @property
    def environ(self) -> dict[str, str]:
        """Environment variables that point the E2B SDK at this proxy."""
        return {"E2B_API_URL": self.url, "E2B_SANDBOX_URL": self.url, "E2B_API_KEY": self.api_key}

    def close(self) -> None:
        """Stops serving, then stops the sessions of sandboxes that are still running."""
        with self._lock:
            self._closed = True
            sandbox_ids = list(self._sandboxes)
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
        for sandbox_id in sandbox_ids:
            self._kill(sandbox_id)

    def __enter__(self) -> AgentCoreE2BProxy:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- AgentCore calls ---------------------------------------------------------------

    def _client(self, region: str) -> Any:
        with self._lock:
            client = self._clients.get(region)
            if client is None:
                config = Config(
                    retries={"mode": "adaptive", "max_attempts": 5}, user_agent_extra="agentcore-rl-toolkit"
                )
                client = self.session.client("bedrock-agentcore", region_name=region, config=config)
                self._clients[region] = client
            return client

    def _stop_session(self, sandbox: _Sandbox) -> None:
        try:
            self._client(sandbox.region).stop_runtime_session(
                agentRuntimeArn=sandbox.runtime_arn, runtimeSessionId=sandbox.sandbox_id, qualifier=self.qualifier
            )
        except (ClientError, BotoCoreError) as exc:
            logger.info("stopping sandbox %s: %s", sandbox.sandbox_id, exc)

    def _signed_headers(self, sandbox: _Sandbox, method: str, url: str, headers: dict[str, str]) -> dict[str, str]:
        # Without X-Amz-Content-Sha256 the session endpoint expects the empty-payload
        # hash, so the body is attached after signing and can be any size.
        request = AWSRequest(method=method, url=_canonical_query(url), headers=headers)
        SigV4Auth(self._credentials.get_frozen_credentials(), SERVICE, sandbox.region).add_auth(request)
        return dict(request.headers.items())

    def _open_upstream(self, host: str) -> http.client.HTTPConnection:
        connection = http.client.HTTPSConnection(host, 443, timeout=self.connect_timeout, context=self._tls)
        connection.connect()
        connection.sock.settimeout(self.read_timeout)
        return connection

    def _envd_request(self, sandbox: _Sandbox, method: str, path: str, body: bytes) -> tuple[int, bytes]:
        host = f"{self.envd_port}--{sandbox.host}"
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        connection = self._open_upstream(host)
        try:
            connection.request(
                method, path, body=body, headers=self._signed_headers(sandbox, method, f"https://{host}{path}", headers)
            )
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    # --- Control plane -----------------------------------------------------------------

    def _create(self, body: dict[str, Any]) -> tuple[int, Any]:
        runtime_arn = body.get("templateID")
        match = _RUNTIME_ARN.fullmatch(runtime_arn) if isinstance(runtime_arn, str) else None
        if match is None:
            raise _ProxyError(400, f"templateID must be an AgentCore runtime ARN, got {runtime_arn!r}")
        unsupported = [
            name
            for name, value in (
                ("autoPause", body.get("autoPause")),
                ("allow_internet_access=false", body.get("allow_internet_access") is False),
                ("network", body.get("network")),
                ("mcp", body.get("mcp")),
                ("volumeMounts", body.get("volumeMounts")),
            )
            if value
        ]
        if unsupported:
            raise _ProxyError(400, f"not supported on AgentCore Runtime: {', '.join(unsupported)}")
        region = match.group(1)
        # Parsed before the session starts, so a bad request leaves no session behind.
        timeout = dt.timedelta(seconds=int(body.get("timeout", 300)))
        metadata = dict(body.get("metadata") or {})
        env_vars = dict(body.get("envVars") or {})
        sandbox_id = uuid.uuid4().hex + uuid.uuid4().hex[:8]  # runtimeSessionId needs >= 33 characters
        try:
            response = self._client(region).invoke_agent_runtime(
                agentRuntimeArn=runtime_arn,
                runtimeSessionId=sandbox_id,
                qualifier=self.qualifier,
                contentType="application/json",
                payload=json.dumps({"action": "start"}).encode(),
            )
            response["response"].read()
        except (ClientError, BotoCoreError) as exc:
            raise _client_error(exc) from exc
        headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        endpoint_url = response.get("sessionEndpointUrl") or headers.get(_SESSION_ENDPOINT_HEADER)
        now = _now()
        sandbox = _Sandbox(
            sandbox_id=sandbox_id,
            runtime_arn=runtime_arn,
            region=region,
            host=(urlsplit(endpoint_url).hostname or "") if endpoint_url else "",
            access_token=secrets.token_urlsafe(32),
            metadata=metadata,
            started_at=now,
            end_at=now + timeout,
        )
        if not sandbox.host.endswith(_SESSION_ENDPOINT_DOMAINS):
            self._stop_session(sandbox)
            raise _ProxyError(
                500,
                f"no AWS session endpoint URL in the response ({endpoint_url!r}); "
                f"the runtime's allowedPorts must include {self.envd_port}",
            )
        init = {
            "accessToken": sandbox.access_token,
            "envVars": env_vars,
            "defaultUser": self.default_user,
            "timestamp": _iso(now),
        }
        if self.default_workdir:
            init["defaultWorkdir"] = self.default_workdir
        try:
            status, data = self._envd_request(sandbox, "POST", "/init", json.dumps(init).encode())
        except OSError as exc:
            status, data = 0, repr(exc).encode()
        if status != 204:
            self._stop_session(sandbox)
            raise _ProxyError(502, f"envd /init on port {self.envd_port} failed: {status} {data[:200]!r}")
        with self._lock:
            closed = self._closed
            if not closed:
                self._sandboxes[sandbox_id] = sandbox
                self._reset_timer(sandbox)
        if closed:
            self._stop_session(sandbox)
            raise _ProxyError(503, "the proxy is closed")
        return 201, self._sandbox_json(sandbox)

    def _reset_timer(self, sandbox: _Sandbox) -> None:
        """Replaces the sandbox's timeout timer; the caller holds ``_lock``."""
        if sandbox.timer is not None:
            sandbox.timer.cancel()
        delay = max(0.0, (sandbox.end_at - _now()).total_seconds())
        sandbox.timer = threading.Timer(delay, self._kill, args=(sandbox.sandbox_id,))
        sandbox.timer.daemon = True
        sandbox.timer.start()

    def _kill(self, sandbox_id: str) -> bool:
        with self._lock:
            sandbox = self._sandboxes.pop(sandbox_id, None)
        if sandbox is None:
            return False
        if sandbox.timer is not None:
            sandbox.timer.cancel()
        self._stop_session(sandbox)
        return True

    def _sandbox_json(self, sandbox: _Sandbox) -> dict[str, Any]:
        return {
            "templateID": sandbox.runtime_arn,
            "sandboxID": sandbox.sandbox_id,
            "clientID": "agentcore",
            "envdVersion": self.envd_version,
            "envdAccessToken": sandbox.access_token,
        }

    def _detail_json(self, sandbox: _Sandbox) -> dict[str, Any]:
        # AgentCore does not report per-session resources; the SDK requires the fields.
        return {
            **self._sandbox_json(sandbox),
            "startedAt": _iso(sandbox.started_at),
            "endAt": _iso(sandbox.end_at),
            "cpuCount": 0,
            "memoryMB": 0,
            "diskSizeMB": 0,
            "state": "running",
            "metadata": sandbox.metadata,
        }

    def _control(self, method: str, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        path = urlsplit(path).path
        if path == "/v2/sandboxes":
            if method == "POST":
                return self._create(body)
            if method == "GET":
                with self._lock:
                    return 200, [self._detail_json(s) for s in self._sandboxes.values()]
        match = _SANDBOX_PATH.fullmatch(path)
        if match is None:
            raise _ProxyError(404, f"{method} {path} is not supported on AgentCore Runtime")
        sandbox_id, action = match.groups()
        if action is None and method == "DELETE":
            if not self._kill(sandbox_id):
                raise _ProxyError(404, f"sandbox {sandbox_id} not found")
            return 204, None
        with self._lock:
            sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is None:
            raise _ProxyError(404, f"sandbox {sandbox_id} not found")
        if action is None and method == "GET":
            return 200, self._detail_json(sandbox)
        if method == "POST" and action in ("timeout", "connect"):
            timeout = dt.timedelta(seconds=int(body["timeout"])) if "timeout" in body else None
            with self._lock:
                if self._sandboxes.get(sandbox_id) is not sandbox:
                    raise _ProxyError(404, f"sandbox {sandbox_id} not found")
                if timeout is not None:
                    sandbox.end_at = _now() + timeout
                self._reset_timer(sandbox)
            return (200, self._sandbox_json(sandbox)) if action == "connect" else (204, None)
        raise _ProxyError(400, f"{method} {path} is not supported on AgentCore Runtime")


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        logger.debug("connection from %s failed", client_address, exc_info=True)


def _handler_class(proxy: AgentCoreE2BProxy) -> type[BaseHTTPRequestHandler]:
    class Handler(_Handler):
        pass

    Handler.proxy = proxy
    return Handler


class _Handler(BaseHTTPRequestHandler):
    proxy: AgentCoreE2BProxy
    protocol_version = "HTTP/1.1"
    upstream: http.client.HTTPConnection | None = None
    upstream_host: str | None = None

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("%s " + format, self.address_string(), *args)

    def finish(self) -> None:
        self._close_upstream()
        super().finish()

    def handle_any(self) -> None:
        try:
            body = self._read_body()
        except ValueError as exc:
            self._send_json(400, {"code": 400, "message": str(exc)}, close=True)
            return
        sandbox_id = self.headers.get("E2b-Sandbox-Id")
        if sandbox_id is None:
            self._handle_control(body)
        else:
            self._handle_data(sandbox_id, body)

    do_GET = do_POST = do_PUT = do_DELETE = handle_any

    # --- Control plane -----------------------------------------------------------------

    def _handle_control(self, body: bytes) -> None:
        key = self.headers.get("X-API-Key", "")
        if not hmac.compare_digest(key.encode(), self.proxy.api_key.encode()):
            self._send_json(401, {"code": 401, "message": "invalid API key"})
            return
        try:
            payload = json.loads(body) if body else {}
            status, result = self.proxy._control(self.command, self.path, payload)
        except _ProxyError as exc:
            status, result = exc.status, {"code": exc.status, "message": str(exc)}
        except (ValueError, TypeError) as exc:
            status, result = 400, {"code": 400, "message": f"invalid request: {exc}"}
        except Exception as exc:
            logger.exception("E2B control-plane request failed")
            status, result = 500, {"code": 500, "message": repr(exc)}
        self._send_json(status, result)

    # --- Data plane --------------------------------------------------------------------

    def _handle_data(self, sandbox_id: str, body: bytes) -> None:
        with self.proxy._lock:
            sandbox = self.proxy._sandboxes.get(sandbox_id)
        if sandbox is None:
            self._send_json(502, {"code": 502, "message": f"sandbox {sandbox_id} is not running"})
            return
        try:
            port = int(self.headers.get("E2b-Sandbox-Port", E2B_ENVD_PORT))
        except ValueError:
            self._send_json(400, {"code": 400, "message": "invalid E2b-Sandbox-Port"})
            return
        if port == E2B_ENVD_PORT:
            port = self.proxy.envd_port
            if not self._envd_authorized(sandbox):
                self._send_json(401, {"code": 401, "message": "missing or invalid X-Access-Token"})
                return
            user = self._requested_user()
            if user is not None and user != self.proxy.default_user:
                # The SDK names the user in Authorization, which SigV4 replaces, so envd
                # would run the request as default_user instead.
                message = (
                    f"user={user!r} is not supported on AgentCore Runtime; requests run as the proxy's "
                    f"default_user {self.proxy.default_user!r}"
                )
                self._send_json(400, {"code": "invalid_argument", "message": message})
                return
        host = f"{port}--{sandbox.host}"
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _REQUEST_DROPPED}
        headers["Content-Length"] = str(len(body))
        signed = self.proxy._signed_headers(sandbox, self.command, f"https://{host}{self.path}", headers)
        try:
            upstream = self._upstream(host)
            upstream.request(self.command, self.path, body=body, headers=signed)
            response = upstream.getresponse()
        except TimeoutError as exc:
            self._close_upstream()
            self._send_json(504, {"code": 504, "message": f"session endpoint timed out: {exc}"}, close=True)
            return
        except (OSError, http.client.HTTPException) as exc:
            self._close_upstream()
            self._send_json(502, {"code": 502, "message": f"session endpoint request failed: {exc!r}"}, close=True)
            return
        platform_error = response.getheader(_PLATFORM_ERROR_HEADER)
        if platform_error:
            response.read()
            # A gone session reads as a stopped sandbox, as E2B reports it.
            status = 502 if response.status == 404 else response.status
            self._send_json(status, {"code": status, "message": f"AgentCore session endpoint: {platform_error}"})
            return
        if not self._relay(response):
            self._close_upstream()

    def _envd_authorized(self, sandbox: _Sandbox) -> bool:
        token = self.headers.get("X-Access-Token")
        if token is not None:
            return hmac.compare_digest(token.encode(), sandbox.access_token.encode())
        # Signed file URLs carry a signature that envd verifies against the token.
        return "signature" in parse_qs(urlsplit(self.path).query)

    def _requested_user(self) -> str | None:
        """The user the SDK names in ``Authorization: Basic``, or None without one."""
        scheme, _, credentials = self.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "basic":
            return None
        try:
            return base64.b64decode(credentials, validate=True).decode().partition(":")[0]
        except (binascii.Error, UnicodeDecodeError):
            return ""

    def _upstream(self, host: str) -> http.client.HTTPConnection:
        stale = self.upstream is not None and (
            self.upstream_host != host or self.upstream.sock is None or _dropped(self.upstream.sock)
        )
        if stale:
            self._close_upstream()
        if self.upstream is None:
            self.upstream = self.proxy._open_upstream(host)
            self.upstream_host = host
        return self.upstream

    def _close_upstream(self) -> None:
        if self.upstream is not None:
            self.upstream.close()
            self.upstream = None
            self.upstream_host = None

    def _relay(self, response: http.client.HTTPResponse) -> bool:
        """Streams the upstream response as it arrives; whether it was read to the end."""
        has_body = response.status not in (204, 304) and response.status >= 200
        length = response.getheader("Content-Length")
        chunked = has_body and length is None
        try:
            self.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() not in _HOP_BY_HOP and name.lower() != "content-length":
                    self.send_header(name, value)
            if length is not None:
                self.send_header("Content-Length", length)
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            if not has_body:
                response.read()
                return True
            while True:
                try:
                    chunk = response.read1(_STREAM_CHUNK_BYTES)
                except (OSError, http.client.HTTPException) as exc:
                    # Headers are already sent; dropping the connection tells the client the body is cut short.
                    logger.debug("session endpoint response failed mid-body: %r", exc)
                    self.close_connection = True
                    return False
                if not chunk:
                    if chunked:
                        self.wfile.write(b"0\r\n\r\n")
                    return True
                self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk) if chunked else chunk)
                self.wfile.flush()
        except OSError:  # the client disconnected
            self.close_connection = True
            return False

    # --- Helpers -----------------------------------------------------------------------

    def _read_body(self) -> bytes:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            chunks = []
            while True:
                size_line = self.rfile.readline(1024).split(b";")[0].strip()
                try:
                    size = int(size_line, 16)
                except ValueError:
                    raise ValueError(f"invalid chunk size {size_line!r}") from None
                if size == 0:
                    while self.rfile.readline(1024).strip():  # trailers
                        pass
                    return b"".join(chunks)
                chunks.append(self.rfile.read(size))
                self.rfile.readline(1024)
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        if not length.isdigit():
            raise ValueError(f"invalid Content-Length {length!r}")
        return self.rfile.read(int(length))

    def _send_json(self, status: int, obj: Any, *, close: bool = False) -> None:
        data = b"" if obj is None else json.dumps(obj).encode()
        self.send_response(status)
        if obj is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(data)
