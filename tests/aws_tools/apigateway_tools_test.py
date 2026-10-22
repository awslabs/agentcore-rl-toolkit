#!/usr/bin/env python
"""Unit tests for SigV4 signing of an endpoint behind API Gateway: the signer, checked
against an independent SigV4 verifier, and the signing proxy: what reaches a fake
upstream, streaming, the token guard, redirects, failures, and OpenAI / Strands clients
configured with only a base URL and API key. Local servers, static credentials, no AWS.
"""

import asyncio
import gzip
import hashlib
import hmac
import http.client
import importlib.util
import io
import json
import re
import socket
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.parse import parse_qsl, quote, urlsplit

import boto3
import httpx
from botocore.credentials import Credentials

from agentcore_rl_toolkit.aws_tools import apigateway_tools
from agentcore_rl_toolkit.aws_tools.apigateway_tools import ExecuteApiProxy

MODULE = "agentcore_rl_toolkit.aws_tools.apigateway_tools"
REGION = "us-west-2"
SECRET = "secret"
HAS_OPENAI = importlib.util.find_spec("openai") is not None
HAS_STRANDS = importlib.util.find_spec("strands") is not None
HAS_REQUESTS = importlib.util.find_spec("requests") is not None
COMPLETION = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
}
MESSAGES = [{"role": "user", "content": "x"}]


def session(token: str | None = None) -> boto3.Session:
    return boto3.Session(
        aws_access_key_id="AKIDEXAMPLE", aws_secret_access_key=SECRET, aws_session_token=token, region_name=REGION
    )


_AUTHORIZATION = re.compile(
    r"AWS4-HMAC-SHA256 Credential=(?P<key>[^/]+)/(?P<date>\d{8})/(?P<region>[^/]+)/(?P<service>[^/]+)/aws4_request, "
    r"SignedHeaders=(?P<signed>[^,]+), Signature=(?P<signature>[0-9a-f]{64})"
)


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def wait_for(condition, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for a condition")
        time.sleep(0.01)


def normalize_path(path: str) -> str:
    """Dot segments removed (RFC 3986 section 5.2.4) and consecutive slashes collapsed.

    Enough for the paths the tests send.
    """
    segments: list[str] = []
    for segment in path.split("/"):
        if segment == "..":
            if segments:
                segments.pop()
        elif segment not in ("", "."):
            segments.append(segment)
    trailing = "/" if path.endswith("/") and segments else ""
    return "/" + "/".join(segments) + trailing


def expected_signature(method: str, url: str, headers, body: bytes, signed_headers: str, scope: str) -> str:
    """SigV4 as API Gateway computes it from the request it receives (written from the spec, not botocore).

    The path is canonicalized as received, with dot segments removed and consecutive
    slashes collapsed; the query is decoded (``+`` as a space) and re-encoded. Both match
    what a live endpoint accepted.
    """
    parts = urlsplit(url)
    # Non-S3 services encode the (already URL-encoded) path a second time.
    canonical_uri = quote(normalize_path(parts.path or "/"), safe="/~")
    query = sorted((quote(k, safe="-_.~"), quote(v, safe="-_.~")) for k, v in parse_qsl(parts.query, True))
    names = signed_headers.split(";")
    canonical_headers = "".join(f"{name}:{' '.join(headers[name].split())}\n" for name in names)
    canonical_request = "\n".join(
        [
            method,
            canonical_uri,
            "&".join(f"{k}={v}" for k, v in query),
            canonical_headers,
            signed_headers,
            hashlib.sha256(body).hexdigest(),
        ]
    )
    string_to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", headers["x-amz-date"], scope, hashlib.sha256(canonical_request.encode()).hexdigest()]
    )
    date, region, service, _ = scope.split("/")
    key = _hmac(_hmac(_hmac(_hmac(f"AWS4{SECRET}".encode(), date), region), service), "aws4_request")
    return hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()


def assert_valid_signature(
    test: unittest.TestCase, method: str, url: str, headers, body: bytes, region: str = REGION
) -> None:
    """Asserts ``headers`` carry a SigV4 ``execute-api`` signature that verifies for this request."""
    headers = {name.lower(): value for name, value in headers.items()}
    match = _AUTHORIZATION.fullmatch(headers.get("authorization", ""))
    test.assertIsNotNone(match, headers.get("authorization"))
    test.assertEqual((match["key"], match["region"], match["service"]), ("AKIDEXAMPLE", region, "execute-api"))
    test.assertIn("host", match["signed"].split(";"))
    scope = f"{match['date']}/{region}/execute-api/aws4_request"
    test.assertEqual(match["signature"], expected_signature(method, url, headers, body, match["signed"], scope))


# --- Signer ----------------------------------------------------------------------------


class SignedHeadersTest(unittest.TestCase):
    URL = "https://abc123.execute-api.us-west-2.amazonaws.com/prod/v1/chat/completions"

    def sign(self, url=URL, body=b'{"a": 1}', token=None, **kwargs):
        """The headers a request carries: its Host plus the SigV4 headers."""
        headers = apigateway_tools.signed_headers("POST", url, body, session=session(token), **kwargs)
        return {"Host": urlsplit(url).netloc, **headers}

    def test_signature_verifies(self):
        assert_valid_signature(self, "POST", self.URL, self.sign(), b'{"a": 1}')

    def test_signature_covers_the_query_and_a_non_default_port(self):
        url = "https://h.example.com:8443/prod/v1/models?b=2&a=x%20y"
        assert_valid_signature(self, "POST", url, self.sign(url), b'{"a": 1}')

    def test_signature_covers_the_body(self):
        with self.assertRaises(AssertionError):
            assert_valid_signature(self, "POST", self.URL, self.sign(), b'{"a": 2}')

    def test_only_host_and_amz_headers_are_signed(self):
        headers = self.sign()
        self.assertEqual(set(headers), {"Host", "Authorization", "X-Amz-Date"})
        self.assertEqual(_AUTHORIZATION.match(headers["Authorization"])["signed"], "host;x-amz-date")

    def test_session_token_is_sent_and_signed(self):
        headers = self.sign(token="tok")
        self.assertEqual(headers["X-Amz-Security-Token"], "tok")
        self.assertIn("x-amz-security-token", _AUTHORIZATION.match(headers["Authorization"])["signed"])
        assert_valid_signature(self, "POST", self.URL, headers, b'{"a": 1}')

    def test_signs_an_empty_body_by_default(self):
        headers = {
            "Host": urlsplit(self.URL).netloc,
            **apigateway_tools.signed_headers("GET", self.URL, session=session()),
        }
        assert_valid_signature(self, "GET", self.URL, headers, b"")

    def test_signs_a_query_in_any_encoding(self):
        for query in ("q=x%2fy", "q=x/y", "e=%7e", "p=1+2", "s=a b", "flag", "b=2&a=1&a=0", "x=(a)!*'", "k=caf%c3%a9"):
            url = f"{self.URL}?{query}"
            with self.subTest(query):
                assert_valid_signature(self, "POST", url, self.sign(url), b'{"a": 1}')

    def test_signs_the_path_as_given(self):
        for path in ("/prod/v1/files/a%2fb", "/prod/v1/a/../files/./x", "/prod/v1/caf%C3%A9", "/prod/v1//x"):
            url = "https://abc123.execute-api.us-west-2.amazonaws.com" + path
            with self.subTest(path):
                assert_valid_signature(self, "POST", url, self.sign(url), b'{"a": 1}')

    def test_uses_the_sessions_region_unless_given(self):
        assert_valid_signature(self, "POST", self.URL, self.sign(), b'{"a": 1}', region=REGION)
        assert_valid_signature(self, "POST", self.URL, self.sign(region="eu-west-1"), b'{"a": 1}', region="eu-west-1")

    def test_fails_without_a_region(self):
        with self.assertRaisesRegex(ValueError, "no region"):
            apigateway_tools.signed_headers("GET", self.URL, session=mock.Mock(region_name=None))

    def test_fails_without_credentials(self):
        no_credentials = mock.Mock(region_name=REGION, get_credentials=mock.Mock(return_value=None))
        with self.assertRaisesRegex(ValueError, "no AWS credentials"):
            apigateway_tools.signed_headers("GET", self.URL, session=no_credentials)


# --- Proxy -----------------------------------------------------------------------------


def sse_events() -> list[bytes]:
    chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    events = [
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    return [f"data: {json.dumps(e)}\n\n".encode() for e in events] + [b"data: [DONE]\n\n"]


class Upstream:
    """A fake endpoint that keeps every request; the path picks the response.

    ``/stream`` sends its first event, then waits for ``release`` before the rest, so a
    client can only see the first event early if the proxy doesn't buffer. ``/slow``
    waits for ``release`` before answering at all.
    """

    def __init__(self):
        self.requests = []
        self.connections = []  # the sockets of the connections upstream accepted
        self.finished = 0  # connections that have ended
        self.release = threading.Event()
        self.stalled = False  # a client waited for /stream's rest before seeing its first event
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                upstream.connections.append(self.connection)

            def finish(self):
                super().finish()
                upstream.finished += 1

            def handle_one(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                upstream.requests.append((self.command, self.path, dict(self.headers), body))
                route = self.path.split("?")[0].rsplit("/", 1)[-1]
                if route == "slow":
                    upstream.release.wait(10)
                streaming = route in ("stream", "stream-unframed")
                if streaming or (route == "completions" and body and json.loads(body).get("stream")):
                    chunked = route != "stream-unframed"  # unframed: the body ends when the connection closes
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header(
                        "Transfer-Encoding" if chunked else "Connection", "chunked" if chunked else "close"
                    )
                    self.end_headers()
                    for i, event in enumerate(sse_events()):
                        self.wfile.write(b"%X\r\n%s\r\n" % (len(event), event) if chunked else event)
                        self.wfile.flush()
                        if i == 0 and streaming and not upstream.release.wait(5):
                            upstream.stalled = True
                    if chunked:
                        self.wfile.write(b"0\r\n\r\n")
                    else:
                        self.close_connection = True
                    return
                status, headers, reply = 200, {"Content-Type": "application/json"}, json.dumps(COMPLETION).encode()
                if route == "files":
                    reply = json.dumps(
                        {"id": "f", "object": "file", "bytes": 2, "created_at": 0, "filename": "a.jsonl",
                         "purpose": "batch", "status": "processed"}
                    ).encode()  # fmt: skip
                elif route == "gzip":
                    headers["Content-Encoding"], reply = "gzip", gzip.compress(reply)
                elif route == "redirect":
                    status, headers, reply = 307, {"Location": upstream.origin + "/prod/v1/models"}, b""
                elif route == "fail":
                    status, reply = 500, b'{"error": {"message": "boom"}}'
                elif route == "close":
                    headers["Connection"], self.close_connection = "close", True
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(reply)

            do_GET = do_POST = do_HEAD = do_DELETE = handle_one

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.base_url = self.origin + "/prod/v1"
        threading.Thread(target=self.server.serve_forever, args=(0.05,), daemon=True).start()

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()

    def assert_signed(self, test: unittest.TestCase, count: int | None = None):
        test.assertTrue(self.requests)
        if count is not None:
            test.assertEqual(len(self.requests), count)
        for method, path, headers, body in self.requests:
            assert_valid_signature(test, method, self.origin + path, headers, body)
            test.assertEqual(headers["Host"], self.origin.removeprefix("http://"))


class ProxyTestCase(unittest.TestCase):
    def setUp(self):
        self.upstream = Upstream()
        self.addCleanup(self.upstream.close)
        self.proxy = ExecuteApiProxy(self.upstream.base_url, session=session())
        self.addCleanup(self.proxy.close)
        self.auth = {"Authorization": f"Bearer {self.proxy.api_key}"}
        self.http = httpx.Client(base_url=self.proxy.base_url, headers=self.auth)
        self.addCleanup(self.http.close)


class ForwardingTest(ProxyTestCase):
    def test_base_url_keeps_the_upstream_path(self):
        self.assertRegex(self.proxy.base_url, r"^http://127\.0\.0\.1:\d+/prod/v1$")

    def test_signs_and_forwards_requests(self):
        response = self.http.post("/chat/completions?x=1", json={"model": "m"}, headers={"X-Custom": "v"})
        self.assertEqual(response.json(), COMPLETION)
        self.upstream.assert_signed(self, 1)
        _, path, headers, body = self.upstream.requests[0]
        self.assertEqual(
            (path, json.loads(body), headers["X-Custom"]), ("/prod/v1/chat/completions?x=1", {"model": "m"}, "v")
        )

    def test_never_forwards_its_token(self):
        self.http.post("/chat/completions", json={})
        _, _, headers, _ = self.upstream.requests[0]
        self.assertNotIn(self.proxy.api_key, json.dumps(headers))
        self.assertTrue(headers["Authorization"].startswith("AWS4-HMAC-SHA256 "))

    def test_signs_session_tokens(self):
        with ExecuteApiProxy(self.upstream.base_url, session=session("tok")) as proxy:
            httpx.post(
                proxy.base_url + "/chat/completions", json={}, headers={"Authorization": f"Bearer {proxy.api_key}"}
            )
        self.assertEqual(self.upstream.requests[0][2]["X-Amz-Security-Token"], "tok")
        self.upstream.assert_signed(self, 1)

    def test_signs_the_url_it_forwards(self):
        """Request targets sent byte for byte, so only the proxy decides what reaches upstream."""
        targets = [
            "/prod/v1/files/a%2fb",  # lowercase percent-encoding in the path
            "/prod/v1/models?q=x%2fy",  # ... and in the query
            "/prod/v1/models?q=x/y&e=%7e&s=a%20b&t=~",
            "/prod/v1/models?p=1+2",  # a form-encoded space
            "/prod/v1/models?b=2&a=1&flag",
            "/prod/v1/models?x=(a)!*'",
            "/prod/v1/a/../models/./x",  # dot segments
            "/prod/v1//models",
            "/prod/v1/caf%c3%a9?x=%c3%a9",
        ]
        forwarded_targets = {target: target for target in targets}  # byte for byte
        for target, forwarded in forwarded_targets.items():
            with self.subTest(target):
                self.upstream.requests.clear()
                connection = http.client.HTTPConnection(urlsplit(self.proxy.origin).netloc)
                self.addCleanup(connection.close)
                connection.request("GET", target, headers=self.auth)
                self.assertEqual(connection.getresponse().status, 200)
                self.assertEqual(self.upstream.requests[0][1], forwarded)
                self.upstream.assert_signed(self, 1)

    def test_uses_rotated_credentials_on_the_next_request(self):
        credentials = Credentials("AKIDOLD", "old")
        rotating = mock.Mock(region_name=REGION, get_credentials=mock.Mock(return_value=credentials))
        with ExecuteApiProxy(self.upstream.base_url, session=rotating) as proxy:
            credentials.access_key, credentials.secret_key = "AKIDEXAMPLE", SECRET
            httpx.post(
                proxy.base_url + "/chat/completions", json={}, headers={"Authorization": f"Bearer {proxy.api_key}"}
            )
        self.upstream.assert_signed(self, 1)

    def test_replaces_client_sigv4_headers(self):
        self.http.post(
            "/chat/completions", json={}, headers={"X-Amz-Date": "19700101T000000Z", "X-Amz-Security-Token": "stale"}
        )
        _, _, headers, _ = self.upstream.requests[0]
        self.assertNotIn("X-Amz-Security-Token", headers)
        self.upstream.assert_signed(self, 1)

    def test_signs_bodyless_methods(self):
        for method in ("GET", "HEAD", "DELETE"):
            with self.subTest(method):
                self.assertEqual(self.http.request(method, "/models").status_code, 200)
        self.upstream.assert_signed(self, 3)

    def test_signs_chunked_request_bodies(self):
        self.http.post("/chat/completions", content=iter([b'{"a":', b" 1}"]))
        self.assertEqual(self.upstream.requests[0][3], b'{"a": 1}')
        self.upstream.assert_signed(self, 1)

    def test_streams_responses_as_they_arrive(self):
        for route in ("stream", "stream-unframed"):
            self.upstream.release.clear()  # before the request starts, so upstream holds back the rest
            with self.subTest(route), self.http.stream("POST", "/" + route, json={}) as response:
                lines = response.iter_lines()
                self.assertTrue(next(lines).startswith("data: "))  # arrives while upstream holds the rest back
                self.upstream.release.set()
                rest = [line for line in lines if line]
                self.assertEqual(rest[-1], "data: [DONE]")
        self.assertFalse(self.upstream.stalled)

    def test_passes_encoded_bodies_through(self):
        response = self.http.get("/gzip")
        self.assertEqual(response.headers["Content-Encoding"], "gzip")
        self.assertEqual(response.json(), COMPLETION)

    def test_passes_upstream_errors_through(self):
        response = self.http.get("/fail")
        self.assertEqual((response.status_code, response.json()), (500, {"error": {"message": "boom"}}))

    def test_handles_concurrent_requests(self):
        def send(i):
            return httpx.post(self.proxy.base_url + "/chat/completions", json={"i": i}, headers=self.auth).status_code

        with ThreadPoolExecutor(8) as pool:
            self.assertEqual(list(pool.map(send, range(16))), [200] * 16)
        self.upstream.assert_signed(self, 16)

    def test_reuses_client_connections(self):
        with httpx.Client(base_url=self.proxy.base_url, headers=self.auth) as client:
            ports = set()
            for _ in range(3):
                with client.stream("POST", "/chat/completions", json={}) as response:
                    ports.add(response.extensions["network_stream"].get_extra_info("client_addr")[1])
                    response.read()
        self.assertEqual(len(ports), 1)

    def test_keeps_one_upstream_connection_per_client_connection(self):
        for method, path in (("POST", "/chat/completions"), ("HEAD", "/models"), ("GET", "/gzip"), ("POST", "/stream")):
            self.upstream.release.set()
            self.assertEqual(self.http.request(method, path, json={} if method == "POST" else None).status_code, 200)
        self.assertEqual(len(self.upstream.connections), 1)
        with httpx.Client(base_url=self.proxy.base_url, headers=self.auth) as other:
            other.get("/models")
        self.assertEqual(len(self.upstream.connections), 2)
        self.upstream.assert_signed(self, 5)

    def test_reconnects_after_upstream_closes_the_connection(self):
        self.assertEqual(self.http.get("/close").status_code, 200)
        self.assertEqual(self.http.get("/models").status_code, 200)
        self.assertEqual(len(self.upstream.connections), 2)

    def test_reconnects_after_upstream_drops_an_idle_connection(self):
        self.http.get("/models")
        self.upstream.connections[0].shutdown(socket.SHUT_RDWR)
        wait_for(lambda: self.upstream.finished == 1)
        self.assertEqual(self.http.get("/models").status_code, 200)
        self.assertEqual(len(self.upstream.connections), 2)

    def test_closes_its_upstream_connection_with_the_client_connection(self):
        with httpx.Client(base_url=self.proxy.base_url, headers=self.auth) as client:
            client.get("/models")
        wait_for(lambda: self.upstream.finished == 1)


class GuardTest(ProxyTestCase):
    def test_rejects_requests_without_its_token(self):
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": self.proxy.api_key}):
            with self.subTest(headers):
                response = httpx.post(self.proxy.base_url + "/chat/completions", json={}, headers=headers)
                self.assertEqual(response.status_code, 401)
                self.assertIn("api key", response.json()["error"]["message"])
        self.assertEqual(self.upstream.requests, [])

    def test_returns_redirects_pointing_back_through_the_proxy(self):
        response = self.http.get("/redirect")
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["Location"], self.proxy.origin + "/prod/v1/models")
        self.assertEqual(len(self.upstream.requests), 1)

    def test_followed_redirects_are_signed_again(self):
        response = self.http.get("/redirect", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([path for _, path, _, _ in self.upstream.requests], ["/prod/v1/redirect", "/prod/v1/models"])
        self.upstream.assert_signed(self, 2)

    def test_unreachable_upstream_is_a_502(self):
        self.upstream.close()
        response = self.http.post("/chat/completions", json={})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["type"], "execute_api_proxy_error")

    def test_slow_upstream_is_a_504(self):
        with ExecuteApiProxy(self.upstream.base_url, session=session(), timeout=0.2) as proxy:
            response = httpx.get(proxy.base_url + "/slow", headers={"Authorization": f"Bearer {proxy.api_key}"})
        self.assertEqual(response.status_code, 504)

    def connection(self) -> http.client.HTTPConnection:
        connection = http.client.HTTPConnection(urlsplit(self.proxy.origin).netloc, timeout=5)
        self.addCleanup(connection.close)
        return connection

    def test_kept_alive_connections_send_nothing_after_close(self):
        connection = self.connection()
        connection.request("GET", "/prod/v1/models", headers=self.auth)
        self.assertEqual(connection.getresponse().read() and 200, 200)
        self.proxy.close()
        try:
            connection.request("GET", "/prod/v1/models", headers=self.auth)
            status = connection.getresponse().status
        except (ConnectionError, http.client.HTTPException):
            status = None
        self.assertIn(status, (None, 503))
        self.assertEqual(len(self.upstream.requests), 1)

    def test_closes_idle_connections(self):
        connection = self.connection()
        connection.request("GET", "/prod/v1/models", headers=self.auth)
        connection.getresponse().read()
        self.proxy.close()
        self.assertEqual(connection.sock.recv(1), b"")  # the proxy closed its end

    def test_in_flight_requests_finish_after_close(self):
        connection = self.connection()
        connection.request("GET", "/prod/v1/slow", headers=self.auth)
        wait_for(lambda: self.upstream.requests)
        closer = threading.Thread(target=self.proxy.close)
        closer.start()
        closer.join(5)
        self.assertFalse(closer.is_alive(), "close() waits for in-flight requests")
        self.upstream.release.set()
        response = connection.getresponse()
        self.assertEqual((response.status, response.read()), (200, json.dumps(COMPLETION).encode()))
        self.assertEqual(response.getheader("Connection"), "close")

    def test_stops_serving_when_closed(self):
        self.proxy.close()
        with self.assertRaises(httpx.ConnectError):
            httpx.get(self.proxy.base_url + "/models", headers=self.auth)

    def test_rejects_non_http_upstreams(self):
        for url in ("ftp://example.com/v1", "example.com/v1", "https:///v1"):
            with self.subTest(url), self.assertRaises(ValueError):
                ExecuteApiProxy(url, session=session())

    def test_keeps_the_session_and_loads_its_credentials_when_built(self):
        given = mock.Mock(wraps=session(), region_name=REGION)
        with ExecuteApiProxy(self.upstream.base_url, session=given) as proxy:
            self.assertIs(proxy.session, given)
            given.get_credentials.assert_called_once_with()

    def test_fails_without_credentials(self):
        no_credentials = mock.Mock(region_name=REGION, get_credentials=mock.Mock(return_value=None))
        with self.assertRaisesRegex(ValueError, "no AWS credentials"):
            ExecuteApiProxy(self.upstream.base_url, session=no_credentials)


@unittest.skipUnless(HAS_OPENAI, "openai isn't installed")
class OpenAITest(ProxyTestCase):
    def test_sync_client(self):
        import openai

        client = openai.OpenAI(base_url=self.proxy.base_url, api_key=self.proxy.api_key)
        self.assertEqual(client.chat.completions.create(model="m", messages=MESSAGES).choices[0].message.content, "hi")
        self.upstream.assert_signed(self, 1)

    def test_async_client_and_streaming(self):
        import openai

        async def run():
            async with openai.AsyncOpenAI(base_url=self.proxy.base_url, api_key=self.proxy.api_key) as client:
                stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
                return [chunk.choices[0].delta.content async for chunk in stream]

        self.assertEqual(asyncio.run(run())[0], "hi")
        self.upstream.assert_signed(self, 1)

    def test_retries_are_signed_again(self):
        import openai

        client = openai.OpenAI(base_url=self.proxy.base_url, api_key=self.proxy.api_key, max_retries=1)
        with self.assertRaises(openai.InternalServerError):
            client.get("/fail", cast_to=object)
        self.upstream.assert_signed(self, 2)

    def test_multipart_uploads(self):
        import openai

        client = openai.OpenAI(base_url=self.proxy.base_url, api_key=self.proxy.api_key)
        client.files.create(file=("a.jsonl", io.BytesIO(b"{}")), purpose="batch")
        self.assertIn(b'name="purpose"', self.upstream.requests[0][3])
        self.upstream.assert_signed(self, 1)


@unittest.skipUnless(HAS_STRANDS, "strands-agents isn't installed")
class StrandsTest(ProxyTestCase):
    def setUp(self):
        super().setUp()
        from strands import Agent
        from strands.models.openai import OpenAIModel

        client_args = {"base_url": self.proxy.base_url, "api_key": self.proxy.api_key}
        self.model = OpenAIModel(client_args=client_args, model_id="m")
        self.agent = Agent(model=self.model, callback_handler=None)

    def test_sync_calls_each_run_on_a_new_event_loop(self):
        for _ in range(3):
            self.assertIn("hi", str(self.agent("x")))
        self.upstream.assert_signed(self, 3)

    def test_concurrent_async_calls(self):
        from strands import Agent

        async def run():
            await asyncio.gather(*(Agent(model=self.model, callback_handler=None).invoke_async("x") for _ in range(4)))

        asyncio.run(run())
        self.upstream.assert_signed(self, 4)


@unittest.skipUnless(HAS_REQUESTS, "requests isn't installed")
class DirectSigningTest(unittest.TestCase):
    """The module docstring's requests example, signing without the proxy."""

    def setUp(self):
        self.upstream = Upstream()
        self.addCleanup(self.upstream.close)

    def test_signed_canonical_urls_verify(self):
        import requests

        for path in ("/chat/completions", "/files/a%2Fb?q=x%2fy&s=a b&p=1+2&flag"):
            with self.subTest(path):
                self.upstream.requests.clear()
                url = self.upstream.base_url + path
                body = json.dumps({"model": "m", "messages": MESSAGES}).encode()
                headers = {
                    "Content-Type": "application/json",
                    **apigateway_tools.signed_headers("POST", url, body, session=session()),
                }
                response = requests.post(url, data=body, headers=headers, allow_redirects=False)
                self.assertEqual(response.status_code, 200)
                self.upstream.assert_signed(self, 1)


class ModuleTest(unittest.TestCase):
    def test_import_loads_no_http_client_library(self):
        libraries = ("httpx", "httpx2", "openai", "aiohttp")
        code = f"import sys, {MODULE}; print(sorted(m for m in {libraries} if m in sys.modules))"
        output = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
        self.assertEqual(output.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
