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
own requests. The token is removed before forwarding. The request target (path and
query) is forwarded byte for byte, so API Gateway checks the signature against exactly
what the client sent. Redirects are returned to the client, not followed, since a
redirected request needs a new signature; a ``Location`` on the endpoint's origin is
rewritten to the proxy, so a client that follows it is signed again.

Each client connection is served by its own thread, with its own upstream connection,
kept alive and closed along with it. Needs only ``boto3``; upstream requests use the
standard library's ``http.client``.
A forked child process doesn't inherit the serving thread, so start the proxy in the
process that uses it.
"""

from __future__ import annotations

import hmac
import http.client
import json
import logging
import secrets
import select
import socket
import ssl
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

import boto3
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


def _with_canonical_query(url: str) -> str:
    """``url`` with its query encoded as SigV4 canonicalizes it, which botocore assumes.

    API Gateway decodes the query it receives and re-encodes it before checking the
    signature (a ``+`` reads as a space), while botocore signs a query string as written,
    so the query is put in that form for signing. The path is left alone: botocore
    canonicalizes a raw path the same way API Gateway does.
    """
    parts = urlsplit(url)
    if not parts.query:
        return url
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    query = "&".join(f"{quote(name, safe='-_.~')}={quote(value, safe='-_.~')}" for name, value in pairs)
    return urlunsplit(parts._replace(query=query))


def signed_headers(
    method: str, url: str, body: bytes = b"", *, session: boto3.Session | None = None, region: str | None = None
) -> dict[str, str]:
    """The SigV4 headers (``Authorization``, ``X-Amz-*``) for an ``execute-api`` request.

    Only ``host`` and the ``X-Amz-*`` headers are signed, plus the body's hash, so other
    headers can be sent as they are. ``url``'s query may be in any encoding, but its path
    must be exactly the path sent: clients such as ``requests`` uppercase percent-encodings
    and percent-encode non-ASCII characters in a path before sending it, so write a path
    that way. The Region defaults to the session's. Without ``session``, each call
    creates a ``boto3.Session()`` and resolves its credentials again; pass one session to
    reuse them across requests.
    """
    credentials, region = _credentials_and_region(session or boto3.Session(), region)
    signed = AWSRequest(method=method, url=_with_canonical_query(url), data=body)
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
_STREAM_CHUNK_BYTES = 64 * 1024  # the most read1() returns at once; it doesn't wait to fill it
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
        self._upstream_address = (upstream.scheme, upstream.hostname, upstream.port)
        self._tls = ssl.create_default_context() if upstream.scheme == "https" else None  # verifies certificates
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.session = session or boto3.Session()
        # Loads the credentials now, so a missing one fails here and serving threads only read the session's cache.
        _, self.region = _credentials_and_region(self.session, region)
        self.api_key = secrets.token_urlsafe(32)
        self._lock = threading.Lock()  # guards _closed and the handlers' busy flags
        self._closed = False
        self._connections: set[_ProxyHandler] = set()
        self._server = _Server((host, port), _handler_class(self))
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
        """Stops serving without waiting for requests in flight.

        New connections are refused and idle kept-alive connections are closed. A request
        already being handled finishes on its own thread, and its connection then closes;
        any later request that still arrives gets a 503 and isn't forwarded.
        """
        with self._lock:
            self._closed = True
            idle = [handler.connection for handler in self._connections if not handler.busy]
        for connection in idle:
            try:
                connection.shutdown(socket.SHUT_RDWR)  # wakes its handler, which then sees EOF
            except OSError:  # already closed by the client
                pass
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _connect(self) -> http.client.HTTPConnection:
        """A new upstream connection, read timeout set once connected."""
        scheme, host, port = self._upstream_address
        if scheme == "https":
            connection = http.client.HTTPSConnection(host, port, timeout=self.connect_timeout, context=self._tls)
        else:
            connection = http.client.HTTPConnection(host, port, timeout=self.connect_timeout)
        connection.connect()
        connection.sock.settimeout(self.timeout)
        return connection

    def __enter__(self) -> ExecuteApiProxy:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        # A connection that close() shuts down mid-request fails here; the default prints a traceback.
        logger.debug("connection from %s failed", client_address, exc_info=True)


def _handler_class(proxy: ExecuteApiProxy) -> type[BaseHTTPRequestHandler]:
    class Handler(_ProxyHandler):
        pass

    Handler.proxy = proxy
    return Handler


class _ProxyHandler(BaseHTTPRequestHandler):
    proxy: ExecuteApiProxy
    protocol_version = "HTTP/1.1"  # keep-alive, so clients pool their connections to the proxy
    busy = False  # handling a request, as opposed to waiting for the connection's next one
    upstream: http.client.HTTPConnection | None = None  # kept alive between this connection's requests

    def setup(self) -> None:
        super().setup()
        with self.proxy._lock:
            self.proxy._connections.add(self)

    def finish(self) -> None:
        with self.proxy._lock:
            self.proxy._connections.discard(self)
        self._close_upstream()
        super().finish()

    def do_proxy(self) -> None:
        with self.proxy._lock:
            closed = self.proxy._closed
            self.busy = not closed
        if closed:
            self._send_error(503, "the proxy is closed")
            return
        try:
            self._forward()
        finally:
            with self.proxy._lock:
                self.busy = False
                if self.proxy._closed:
                    self.close_connection = True

    def _forward(self) -> None:
        if not self._authorized():
            self._send_error(401, "missing or wrong proxy api key")
            return
        try:
            body = self._read_body()
        except ValueError as exc:
            self._send_error(400, str(exc))
            return
        headers = self._upstream_headers()
        url = self.proxy.upstream_origin + self.path
        headers.update(signed_headers(self.command, url, body, session=self.proxy.session, region=self.proxy.region))
        try:
            upstream = self._upstream_connection()
            # The request target goes out byte for byte; http.client adds Host and the body's Content-Length.
            upstream.request(self.command, self.path, body=body or None, headers=headers)
            response = upstream.getresponse()
        except TimeoutError as exc:
            self._close_upstream()
            self._send_error(504, f"upstream timed out: {exc}")
            return
        except (OSError, http.client.HTTPException) as exc:
            self._close_upstream()
            self._send_error(502, f"upstream request failed: {exc!r}")
            return
        if not self._relay(response):
            self._close_upstream()  # a partly read connection can't send another request

    def _upstream_connection(self) -> http.client.HTTPConnection:
        """The kept-alive upstream connection, or a new one if upstream closed or dropped it."""
        if self.upstream is not None and (self.upstream.sock is None or _dropped(self.upstream.sock)):
            self._close_upstream()
        if self.upstream is None:
            self.upstream = self.proxy._connect()
        return self.upstream

    def _close_upstream(self) -> None:
        if self.upstream is not None:
            self.upstream.close()
            self.upstream = None

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

    def _relay(self, response: http.client.HTTPResponse) -> bool:
        """Sends the upstream response as it arrives, body still encoded; whether it was read to the end."""
        has_body = self.command != "HEAD" and response.status not in (204, 304) and response.status >= 200
        length = response.getheader("Content-Length")
        chunked = has_body and length is None
        self.send_response(response.status)
        for name, value in response.getheaders():
            if name.lower() == "location" and value.startswith(self.proxy.upstream_origin + "/"):
                value = self.proxy.origin + value[len(self.proxy.upstream_origin) :]
            if name.lower() not in _HOP_BY_HOP and name.lower() != "content-length":
                self.send_header(name, value)
        if length is not None:
            self.send_header("Content-Length", length)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        if self.proxy._closed:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if not has_body:
            response.read()  # http.client sends a connection's next request only after this one is read
            return True
        while True:
            try:
                # read1 returns what has arrived, so streamed (SSE) responses aren't held back;
                # http.client takes care of upstream's chunked framing.
                chunk = response.read1(_STREAM_CHUNK_BYTES)
            except (OSError, http.client.HTTPException) as exc:
                # The status line is already sent; dropping the connection tells the client the body is cut short.
                logger.warning("upstream response failed mid-body: %r", exc)
                self.close_connection = True
                return False
            try:
                if not chunk:
                    if chunked:
                        self.wfile.write(b"0\r\n\r\n")
                    return True
                self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk) if chunked else chunk)
            except OSError:  # the client disconnected
                self.close_connection = True
                return False

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


def _dropped(sock: socket.socket) -> bool:
    """Whether an idle kept-alive connection was closed by upstream (readable means EOF or garbage)."""
    return bool(select.select([sock], [], [], 0)[0])
