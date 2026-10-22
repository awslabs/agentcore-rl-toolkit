"""SigV4 (``execute-api``) signing for an endpoint behind API Gateway, as a local proxy.

Any HTTP client, in any library or language, talks plain HTTP to the proxy, which
forwards each request to the endpoint signed with the caller's AWS credentials, body
included, and streams the response back as it arrives. Clients only need a base URL
and an API key::

    with ExecuteApiProxy("https://abc123.execute-api.us-west-2.amazonaws.com/prod/v1") as proxy:
        client = openai.OpenAI(base_url=proxy.base_url, api_key=proxy.api_key)
        model = OpenAIModel(client_args={"base_url": proxy.base_url, "api_key": proxy.api_key}, model_id=...)

Any other HTTP operation works the same way. With ``requests``, through the proxy,
which does the signing::

    with ExecuteApiProxy(base_url, region="us-east-2") as proxy:
        response = requests.post(
            proxy.base_url + "/chat/completions",
            json={"model": model_id, "messages": messages},
            headers={"Authorization": f"Bearer {proxy.api_key}"},
        )

Or without the proxy, signing each request with ``signed_headers``. Send exactly the
signed URL and body bytes, sign every retry again, and don't follow redirects::

    url = base_url + "/chat/completions"
    body = json.dumps({"model": model_id, "messages": messages}).encode()
    headers = {"Content-Type": "application/json", **signed_headers("POST", url, body, region="us-east-2")}
    response = requests.post(url, data=body, headers=headers, allow_redirects=False)

The signing Region defaults to the boto3 session's; pass ``region=`` for an endpoint
on a custom domain, whose hostname doesn't name its Region.

The proxy listens on loopback and only forwards requests carrying its random
``api_key`` as a bearer token, so other local processes can't use it to sign their
own requests. The token is removed before forwarding. Redirects are returned to the
client, not followed, since a redirected request needs a new signature; a ``Location``
on the endpoint's origin is rewritten to the proxy, so a client that follows it is
signed again.

Each connection is served by its own thread, and upstream connections are pooled.
Needs only ``boto3``; upstream requests use ``urllib3``, a dependency of ``botocore``.
A forked child process doesn't inherit the serving thread, so start the proxy in the
process that uses it.
"""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import boto3
import urllib3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

SERVICE = "execute-api"
# The endpoint allows 15 minutes per streamed request.
DEFAULT_TIMEOUT_SECONDS = 15 * 60
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
_SIGV4_HEADERS = ("Authorization", "X-Amz-Date", "X-Amz-Security-Token")


# --- Signing ---------------------------------------------------------------------------


def _credentials_and_region(session: boto3.Session, region: str | None) -> tuple[Credentials, str]:
    """The session's credentials and the signing Region; fails if either is missing."""
    region = region or session.region_name
    if not region:
        raise ValueError("no region: pass region= or configure one for the boto3 session")
    # Loaded once, then cached by the session. Refreshing credentials (assumed roles, instance
    # profiles, SSO) renew themselves, one thread at a time, under their own lock.
    credentials = session.get_credentials()
    if credentials is None:
        raise ValueError("the boto3 session has no AWS credentials")
    return credentials, region


def signed_headers(
    method: str, url: str, body: bytes = b"", *, session: boto3.Session | None = None, region: str | None = None
) -> dict[str, str]:
    """The SigV4 headers (``Authorization``, ``X-Amz-*``) for an ``execute-api`` request.

    Only ``host`` and the ``X-Amz-*`` headers are signed, plus the body's hash, so other
    headers can be sent as they are. The Region defaults to the session's. Without
    ``session``, each call creates a ``boto3.Session()`` and resolves its credentials
    again; pass one session to reuse them across requests.
    """
    credentials, region = _credentials_and_region(session or boto3.Session(), region)
    signed = AWSRequest(method=method, url=url, data=body)
    SigV4Auth(credentials.get_frozen_credentials(), SERVICE, region).add_auth(signed)
    return {name: signed.headers[name] for name in _SIGV4_HEADERS if name in signed.headers}


# --- Proxy -----------------------------------------------------------------------------


logger = logging.getLogger(__name__)

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
# Replaced by the proxy: the token, the upstream Host and the body's framing.
_REQUEST_REPLACED = frozenset(["authorization", "host", "content-length", "x-amz-date", "x-amz-security-token"])
_STREAM_CHUNK_BYTES = 64 * 1024
_SHUTDOWN_POLL_SECONDS = 0.1  # how long close() may wait for the serving thread


class ExecuteApiProxy:
    """A loopback HTTP proxy that signs every request for ``upstream_url``'s origin.

    It starts serving when constructed; ``close()`` (or leaving a ``with`` block) stops
    it. ``base_url`` is ``upstream_url`` with the proxy's origin, so clients use it as
    their base URL, with ``api_key`` as the bearer token. Credentials come from
    ``session`` (default ``boto3.Session()``), and the Region defaults to the session's;
    either missing fails here. ``timeout`` bounds each wait for upstream bytes, not the
    whole response.

    The proxy signs on its serving threads with ``session``'s credentials, which are
    safe to share between threads. The session itself isn't, so while the proxy runs,
    use a passed ``session`` only on the thread that created the proxy.
    """

    def __init__(
        self,
        upstream_url: str,
        *,
        session: boto3.Session | None = None,
        region: str | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ):
        upstream = urlsplit(upstream_url)
        if upstream.scheme not in ("http", "https") or not upstream.hostname:
            raise ValueError(f"upstream_url must be an http(s) URL, not {upstream_url!r}")
        self.upstream_origin = f"{upstream.scheme}://{upstream.netloc}"
        self.session = session or boto3.Session()
        # Loads the credentials now, so a missing one fails here and serving threads only read the session's cache.
        _, self.region = _credentials_and_region(self.session, region)
        self.api_key = secrets.token_urlsafe(32)
        self.pool = urllib3.PoolManager(
            retries=False, timeout=urllib3.Timeout(connect=connect_timeout, read=timeout), maxsize=64, block=False
        )
        self._server = ThreadingHTTPServer((host, port), _handler_class(self))
        self._server.daemon_threads = True
        bound_host, bound_port = self._server.server_address[:2]
        self.origin = f"http://{bound_host}:{bound_port}"
        self.base_url = self.origin + upstream.path.rstrip("/")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": _SHUTDOWN_POLL_SECONDS},
            name="execute-api-proxy",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stops accepting connections; requests in flight on their own threads may finish."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
        self.pool.clear()

    def __enter__(self) -> ExecuteApiProxy:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _handler_class(proxy: ExecuteApiProxy) -> type[BaseHTTPRequestHandler]:
    class Handler(_ProxyHandler):
        pass

    Handler.proxy = proxy
    return Handler


class _ProxyHandler(BaseHTTPRequestHandler):
    proxy: ExecuteApiProxy
    protocol_version = "HTTP/1.1"  # keep-alive, so clients pool their connections to the proxy

    def do_proxy(self) -> None:
        if not self._authorized():
            self._send_error(401, "missing or wrong proxy api key")
            return
        try:
            body = self._read_body()
        except ValueError as exc:
            self._send_error(400, str(exc))
            return
        url = self.proxy.upstream_origin + self.path
        headers = self._upstream_headers()
        headers.update(signed_headers(self.command, url, body, session=self.proxy.session, region=self.proxy.region))
        try:
            response = self.proxy.pool.urlopen(
                self.command,
                url,
                body=body,
                headers=headers,
                redirect=False,
                preload_content=False,
                decode_content=False,
            )
        except urllib3.exceptions.NewConnectionError as exc:  # before TimeoutError, which it subclasses
            self._send_error(502, f"upstream connection failed: {exc}")
            return
        except urllib3.exceptions.TimeoutError as exc:
            self._send_error(504, f"upstream timed out: {exc}")
            return
        except urllib3.exceptions.HTTPError as exc:
            self._send_error(502, f"upstream request failed: {exc}")
            return
        complete = False
        try:
            complete = self._relay(response)
        finally:
            if not complete:
                response.close()  # a partly read connection can't go back to the pool
            response.release_conn()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = do_proxy

    def _authorized(self) -> bool:
        expected = f"Bearer {self.proxy.api_key}".encode()
        return hmac.compare_digest(self.headers.get("Authorization", "").encode(), expected)

    def _read_body(self) -> bytes:
        """The whole request body, since its hash is part of the signature."""
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            return b"".join(self._read_chunks())
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        if not length.isdigit():
            raise ValueError(f"invalid Content-Length {length!r}")
        return self.rfile.read(int(length))

    def _read_chunks(self) -> Iterator[bytes]:
        while True:
            size_line = self.rfile.readline(1024).split(b";")[0].strip()
            try:
                size = int(size_line, 16)
            except ValueError:
                raise ValueError(f"invalid chunk size {size_line!r}") from None
            if size == 0:
                while self.rfile.readline(1024).strip():  # trailers
                    pass
                return
            yield self.rfile.read(size)
            self.rfile.readline(1024)  # the chunk's CRLF

    def _upstream_headers(self) -> dict[str, str]:
        dropped = _HOP_BY_HOP | _REQUEST_REPLACED
        headers: dict[str, str] = {}
        for name, value in self.headers.items():
            if name.lower() not in dropped:
                headers[name] = f"{headers[name]}, {value}" if name in headers else value
        return headers

    def _relay(self, response: urllib3.BaseHTTPResponse) -> bool:
        """Sends the upstream response as it arrives, body still encoded; whether it was read to the end."""
        has_body = self.command != "HEAD" and response.status not in (204, 304) and response.status >= 200
        length = response.headers.get("Content-Length")
        chunked = has_body and length is None
        self.send_response(response.status)
        for name, value in response.headers.items():
            if name.lower() == "location" and value.startswith(self.proxy.upstream_origin + "/"):
                value = self.proxy.origin + value[len(self.proxy.upstream_origin) :]
            if name.lower() not in _HOP_BY_HOP and name.lower() != "content-length":
                self.send_header(name, value)
        if length is not None:
            self.send_header("Content-Length", length)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if not has_body:
            return True
        try:
            for chunk in _upstream_chunks(response):
                if chunk:
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk) if chunked else chunk)
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
        except urllib3.exceptions.HTTPError as exc:
            # The status line is already sent; dropping the connection tells the client the body is cut short.
            logger.warning("upstream response failed mid-body: %s", exc)
            self.close_connection = True
            return False
        except OSError:  # the client disconnected
            self.close_connection = True
            return False
        return True

    def _send_error(self, status: int, message: str) -> None:
        """An OpenAI-style error; the connection closes, since the request body may be unread."""
        body = json.dumps({"error": {"message": message, "type": "execute_api_proxy_error"}}).encode()
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("%s " + format, self.address_string(), *args)


def _upstream_chunks(response: urllib3.BaseHTTPResponse) -> Iterator[bytes]:
    """The raw body in pieces as they arrive, so streamed (SSE) responses aren't held back."""
    if response.chunked and response.supports_chunked_reads():
        yield from response.read_chunked(decode_content=False)
    elif hasattr(response, "read1"):  # urllib3>=2.3: returns what's available instead of waiting for a full chunk
        while chunk := response.read1(_STREAM_CHUNK_BYTES, decode_content=False):
            yield chunk
    else:
        yield from response.stream(_STREAM_CHUNK_BYTES, decode_content=False)
