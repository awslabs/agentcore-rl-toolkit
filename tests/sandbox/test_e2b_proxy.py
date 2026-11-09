"""Tests for AgentCoreE2BProxy with a fake AgentCore client and a local session endpoint."""

import base64
import http.client
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest
from botocore.credentials import Credentials
from botocore.exceptions import ClientError

from agentcore_rl_toolkit.sandbox.e2b import AgentCoreE2BProxy

ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/e2b_test-AbCdE12345"
ENDPOINT_HOST = "abc123.runtime.bedrock-agentcore.us-east-1.on.aws"


class FakeSessionEndpoint:
    """Records requests and answers them like envd behind a session endpoint."""

    def __init__(self):
        self.requests = []
        self.init_status = 204
        self.release_stream = threading.Event()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def handle_one(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                endpoint.requests.append(
                    {"method": self.command, "path": self.path, "headers": dict(self.headers.items()), "body": body}
                )
                if self.path == "/init":
                    self.send_response(endpoint.init_status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path == "/process.Process/Start":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/connect+json")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for part, wait in ((b"first", False), (b"second", True)):
                        if wait:
                            endpoint.release_stream.wait(5)
                        self.wfile.write(b"%X\r\n%s\r\n" % (len(part), part))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                elif self.path == "/gone":
                    self.send_response(404)
                    self.send_header("X-Amzn-Bedrock-AgentCore-Runtime-Error", "SessionNotFound")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    data = b"hello"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

            do_GET = do_POST = handle_one

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.release_stream.set()
        self.server.shutdown()
        self.server.server_close()


class FakeSession:
    def __init__(self, client):
        self._client = client

    def get_credentials(self):
        return Credentials("AKIDEXAMPLE", "secret")

    def client(self, name, region_name=None, config=None):
        assert name == "bedrock-agentcore"
        self.region_name = region_name
        return self._client


class LocalProxy(AgentCoreE2BProxy):
    """Sends session endpoint traffic to the fake endpoint and records the target host."""

    def __init__(self, endpoint, **kwargs):
        self.endpoint = endpoint
        self.upstream_hosts = []
        super().__init__(**kwargs)

    def _open_upstream(self, host):
        self.upstream_hosts.append(host)
        return http.client.HTTPConnection("127.0.0.1", self.endpoint.server.server_address[1], timeout=5)


def invoke_response(url=f"https://{ENDPOINT_HOST}"):
    return {
        "response": io.BytesIO(b'{"status":"ok","state":"busy"}'),
        "sessionEndpointUrl": url,
        "ResponseMetadata": {"HTTPHeaders": {}},
    }


@pytest.fixture
def endpoint():
    endpoint = FakeSessionEndpoint()
    yield endpoint
    endpoint.close()


@pytest.fixture
def acr():
    client = MagicMock()
    client.invoke_agent_runtime.return_value = invoke_response()
    return client


@pytest.fixture
def proxy(endpoint, acr):
    proxy = LocalProxy(endpoint, session=FakeSession(acr))
    yield proxy
    proxy.close()


def call(proxy, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", int(proxy.url.rsplit(":", 1)[1]), timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    connection.request(method, path, body=data, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    if response.getheader("Content-Type") == "application/json":
        return response.status, json.loads(payload)
    return response.status, payload


def control(proxy, method, path, body=None):
    return call(proxy, method, path, body, {"X-API-Key": proxy.api_key, "Content-Type": "application/json"})


def create(proxy, **body):
    status, sandbox = control(proxy, "POST", "/v2/sandboxes", {"templateID": ARN, **body})
    assert status == 201, sandbox
    return sandbox


def data(proxy, sandbox, method, path, *, port=49983, token=True):
    headers = {"E2b-Sandbox-Id": sandbox["sandboxID"], "E2b-Sandbox-Port": str(port)}
    if token:
        headers["X-Access-Token"] = sandbox["envdAccessToken"]
    return call(proxy, method, path, headers=headers)


def test_create_starts_a_session_and_initializes_envd(proxy, acr, endpoint):
    sandbox = create(proxy, envVars={"A": "b"}, metadata={"run": "test"})

    assert sandbox["templateID"] == ARN
    kwargs = acr.invoke_agent_runtime.call_args.kwargs
    assert kwargs["agentRuntimeArn"] == ARN
    assert kwargs["runtimeSessionId"] == sandbox["sandboxID"] and len(sandbox["sandboxID"]) >= 33
    assert json.loads(kwargs["payload"]) == {"action": "start"}
    assert proxy.session.region_name == "us-east-1"

    init = endpoint.requests[0]
    assert (init["method"], init["path"]) == ("POST", "/init")
    assert proxy.upstream_hosts[0] == f"10983--{ENDPOINT_HOST}"
    body = json.loads(init["body"])
    assert body["accessToken"] == sandbox["envdAccessToken"]
    assert (body["envVars"], body["defaultUser"], body["defaultWorkdir"]) == ({"A": "b"}, "user", "/home/user")
    assert "/us-east-1/bedrock-agentcore/aws4_request" in init["headers"]["Authorization"]


def test_control_plane_requires_the_api_key(proxy, acr):
    status, body = call(proxy, "POST", "/v2/sandboxes", {"templateID": ARN}, {"X-API-Key": "wrong"})
    assert status == 401
    acr.invoke_agent_runtime.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [{"templateID": "base"}, {"templateID": ARN, "autoPause": True}],
)
def test_create_rejects_what_agentcore_cannot_honor(proxy, acr, body):
    status, error = control(proxy, "POST", "/v2/sandboxes", body)
    assert status == 400 and error["code"] == 400
    acr.invoke_agent_runtime.assert_not_called()


def test_create_validates_the_request_before_starting_a_session(proxy, acr):
    status, _ = control(proxy, "POST", "/v2/sandboxes", {"templateID": ARN, "timeout": "soon"})
    assert status == 400
    acr.invoke_agent_runtime.assert_not_called()


def test_create_passes_agentcore_throttling_through(proxy, acr):
    acr.invoke_agent_runtime.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}, "ResponseMetadata": {"HTTPStatusCode": 429}},
        "InvokeAgentRuntime",
    )
    status, error = control(proxy, "POST", "/v2/sandboxes", {"templateID": ARN})
    assert status == 429
    assert "ThrottlingException" in error["message"]


def test_create_stops_the_session_when_envd_init_fails(proxy, acr, endpoint):
    endpoint.init_status = 500
    status, error = control(proxy, "POST", "/v2/sandboxes", {"templateID": ARN})
    assert status == 502 and "/init" in error["message"]
    acr.stop_runtime_session.assert_called_once()


def test_create_requires_a_session_endpoint(proxy, acr):
    acr.invoke_agent_runtime.return_value = invoke_response(url=None)
    status, error = control(proxy, "POST", "/v2/sandboxes", {"templateID": ARN})
    assert status == 500 and "allowedPorts" in error["message"]
    acr.stop_runtime_session.assert_called_once()


def test_data_plane_routes_envd_traffic_with_sigv4(proxy, endpoint):
    sandbox = create(proxy)
    status, _ = call(
        proxy,
        "GET",
        "/files?path=%2Fhome%2Fuser%2Fa.txt",
        headers={
            "E2b-Sandbox-Id": sandbox["sandboxID"],
            "E2b-Sandbox-Port": "49983",
            "X-Access-Token": sandbox["envdAccessToken"],
        },
    )
    assert status == 200
    request = endpoint.requests[-1]
    assert request["path"] == "/files?path=%2Fhome%2Fuser%2Fa.txt"
    assert proxy.upstream_hosts[-1] == f"10983--{ENDPOINT_HOST}"
    headers = {k.lower(): v for k, v in request["headers"].items()}
    assert headers["authorization"].startswith("AWS4-HMAC-SHA256")
    assert headers["x-access-token"] == sandbox["envdAccessToken"]
    assert "e2b-sandbox-id" not in headers and "e2b-sandbox-port" not in headers
    assert "x-amz-content-sha256" not in headers


def test_data_plane_forwards_other_ports_without_envd_token(proxy):
    sandbox = create(proxy)
    status, _ = data(proxy, sandbox, "GET", "/", port=8080, token=False)
    assert status == 200
    assert proxy.upstream_hosts[-1] == f"8080--{ENDPOINT_HOST}"


def test_envd_traffic_requires_the_sandbox_access_token(proxy, endpoint):
    sandbox = create(proxy)
    assert data(proxy, sandbox, "GET", "/files?path=a", token=False)[0] == 401
    wrong = {**sandbox, "envdAccessToken": "nope"}
    assert data(proxy, wrong, "GET", "/files?path=a")[0] == 401
    # Signed file URLs are verified by envd.
    assert data(proxy, sandbox, "GET", "/files?path=a&signature=v1_x", token=False)[0] == 200
    assert endpoint.requests[-1]["path"] == "/files?path=a&signature=v1_x"


def test_envd_requests_naming_another_user_fail(proxy, endpoint):
    sandbox = create(proxy)

    def as_user(user):
        return call(
            proxy,
            "POST",
            "/process.Process/List",
            headers={
                "E2b-Sandbox-Id": sandbox["sandboxID"],
                "X-Access-Token": sandbox["envdAccessToken"],
                "Authorization": "Basic " + base64.b64encode(f"{user}:".encode()).decode(),
            },
        )

    status, error = as_user("root")
    assert status == 400 and error["code"] == "invalid_argument" and "'root'" in error["message"]
    assert all(r["path"] != "/process.Process/List" for r in endpoint.requests)

    assert as_user("user")[0] == 200  # the default user is what envd applies anyway
    assert endpoint.requests[-1]["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256")


def test_data_plane_streams_responses_as_they_arrive(proxy, endpoint):
    sandbox = create(proxy)
    connection = http.client.HTTPConnection("127.0.0.1", int(proxy.url.rsplit(":", 1)[1]), timeout=10)
    connection.request(
        "POST",
        "/process.Process/Start",
        body=b"{}",
        headers={
            "E2b-Sandbox-Id": sandbox["sandboxID"],
            "X-Access-Token": sandbox["envdAccessToken"],
            "Content-Type": "application/connect+json",
        },
    )
    response = connection.getresponse()
    assert response.status == 200
    assert response.read1(64) == b"first"  # arrives before the upstream sends the rest
    endpoint.release_stream.set()
    assert response.read() == b"second"
    connection.close()


def test_platform_session_not_found_reads_as_a_stopped_sandbox(proxy):
    sandbox = create(proxy)
    status, error = data(proxy, sandbox, "GET", "/gone")
    assert status == 502 and "SessionNotFound" in error["message"]


def test_kill_stops_the_session(proxy, acr):
    sandbox = create(proxy)
    assert control(proxy, "DELETE", f"/sandboxes/{sandbox['sandboxID']}")[0] == 204
    kwargs = acr.stop_runtime_session.call_args.kwargs
    assert (kwargs["agentRuntimeArn"], kwargs["runtimeSessionId"]) == (ARN, sandbox["sandboxID"])
    assert control(proxy, "DELETE", f"/sandboxes/{sandbox['sandboxID']}")[0] == 404
    assert data(proxy, sandbox, "GET", "/health")[0] == 502


def test_timeout_stops_the_session(proxy, acr):
    create(proxy, timeout=0)
    deadline = time.monotonic() + 5
    while not acr.stop_runtime_session.called and time.monotonic() < deadline:
        time.sleep(0.05)
    acr.stop_runtime_session.assert_called_once()


def test_concurrent_timeout_updates_leave_one_timer(proxy, acr, monkeypatch):
    class SlowCancelTimer(threading.Timer):
        def cancel(self):
            time.sleep(0.01)  # widens the window between cancelling a timer and replacing it
            super().cancel()

    monkeypatch.setattr(threading, "Timer", SlowCancelTimer)
    sandbox = create(proxy, timeout=600)
    path = f"/sandboxes/{sandbox['sandboxID']}/timeout"
    threads = [threading.Thread(target=proxy._control, args=("POST", path, {"timeout": 1})) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert proxy._control("POST", path, {"timeout": 600})[0] == 204
    time.sleep(1.5)  # a timer the last update did not replace would stop the session now
    acr.stop_runtime_session.assert_not_called()


def test_list_get_timeout_and_connect(proxy):
    sandbox = create(proxy, metadata={"run": "a"})
    sandbox_id = sandbox["sandboxID"]

    status, listed = control(proxy, "GET", "/v2/sandboxes")
    assert status == 200 and [s["sandboxID"] for s in listed] == [sandbox_id]

    status, detail = control(proxy, "GET", f"/sandboxes/{sandbox_id}")
    assert status == 200 and (detail["state"], detail["metadata"]) == ("running", {"run": "a"})

    assert control(proxy, "POST", f"/sandboxes/{sandbox_id}/timeout", {"timeout": 3600})[0] == 204
    _, extended = control(proxy, "GET", f"/sandboxes/{sandbox_id}")
    assert extended["endAt"] > detail["endAt"]

    status, connected = control(proxy, "POST", f"/v2/sandboxes/{sandbox_id}/connect", {"timeout": 600})
    assert status == 200 and connected["envdAccessToken"] == sandbox["envdAccessToken"]


def test_unsupported_sandbox_actions_fail_loudly(proxy):
    sandbox = create(proxy)
    status, error = control(proxy, "PUT", f"/sandboxes/{sandbox['sandboxID']}/network", {})
    assert status == 400 and "not supported" in error["message"]


def test_close_stops_running_sandboxes(endpoint, acr):
    proxy = LocalProxy(endpoint, session=FakeSession(acr))
    create(proxy)
    proxy.close()
    acr.stop_runtime_session.assert_called_once()
