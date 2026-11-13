"""Verify the signed bytes a session proxy receives, including credential refresh."""

import hashlib
import hmac
from unittest.mock import AsyncMock

import httpx
import pytest
from botocore.credentials import ReadOnlyCredentials

from agentcore_rl_toolkit.aws_tools import session_endpoints


def verify_signature(request: httpx.Request, secret: str):
    authorization = request.headers["Authorization"]
    fields = dict(field.split("=", 1) for field in authorization.removeprefix("AWS4-HMAC-SHA256 ").split(", "))
    _, scope = fields["Credential"].split("/", 1)
    date, region, service, terminator = scope.split("/")
    names = fields["SignedHeaders"]
    canonical_headers = "".join(f"{name}:{request.headers[name].strip()}\n" for name in names.split(";"))
    payload_hash = request.headers["X-Amz-Content-Sha256"]
    assert payload_hash == hashlib.sha256(request.content).hexdigest()
    canonical = "\n".join(
        [request.method, request.url.path, request.url.query.decode(), canonical_headers, names, payload_hash]
    )
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", request.headers["X-Amz-Date"], scope, hashlib.sha256(canonical.encode()).hexdigest()]
    )
    key = f"AWS4{secret}".encode()
    for value in (date, region, service, terminator):
        key = hmac.digest(key, value.encode(), "sha256")
    assert fields["Signature"] == hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    assert (region, service) == ("us-east-1", "bedrock-agentcore")
    assert "x-amz-content-sha256" in names.split(";")


@pytest.mark.asyncio
async def test_signed_requests_and_refresh(monkeypatch):
    credentials = AsyncMock()
    credentials.get_frozen_credentials.side_effect = [
        ReadOnlyCredentials(f"access-{i}", f"secret-{i}", f"token-{i}") for i in range(3)
    ]
    session = AsyncMock()
    session.get_credentials.return_value = credentials
    monkeypatch.setattr(session_endpoints, "get_async_session", AsyncMock(return_value=session))
    received = []

    def handle(request):
        index = len(received)
        verify_signature(request, f"secret-{index}")
        assert request.headers["X-Amz-Security-Token"] == f"token-{index}"
        if not request.content:
            assert request.headers["Content-Length"] == "0"
        received.append(request)
        return httpx.Response(200, json={})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        session_endpoints.httpx,
        "AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handle), **kwargs),
    )
    async with session_endpoints.session_endpoint_client("https://session.example", "us-east-1") as client:
        await client.get("/.well-known/agent-env.json")
        await client.post("/agentenv", json={"message": "你好"})
        await client.delete("/mcp")
        assert not client.follow_redirects
    assert client.is_closed
    assert len(received) == 3


@pytest.mark.asyncio
async def test_streamed_request_is_hashed_after_buffering():
    credentials = AsyncMock()
    credentials.get_frozen_credentials.return_value = ReadOnlyCredentials("access", "secret", "token")

    async def body():
        yield b'{"message":'
        yield b'"hello"}'

    request = httpx.Request("POST", "https://session.example/agentenv", content=body())
    auth = session_endpoints._SessionEndpointAuth(credentials, "us-east-1", "bedrock-agentcore")
    async for signed in auth.async_auth_flow(request):
        verify_signature(signed, "secret")
        assert signed.content == b'{"message":"hello"}'
