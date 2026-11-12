"""Authenticated HTTP access to an existing AgentCore Runtime session endpoint."""

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from .boto3_tools import SigV4HttpxAuth, get_async_session


class _SessionEndpointAuth(SigV4HttpxAuth):
    async def async_auth_flow(self, request: httpx.Request):
        # The proxy assumes the empty payload hash when this header is absent,
        # even for POSTs. Supply the actual hash before the shared signer runs.
        body = await request.aread()
        request.headers["X-Amz-Content-Sha256"] = hashlib.sha256(body).hexdigest()
        if not body:
            request.headers["Content-Length"] = "0"
        async for signed in super().async_auth_flow(request):
            yield signed


@asynccontextmanager
async def session_endpoint_client(url: str, region: str, *, timeout: float = 60) -> AsyncIterator[httpx.AsyncClient]:
    """Open a SigV4 client for a URL returned by Runtime.

    Requests are buffered to hash their bodies; responses can stream. Credentials
    refresh per request. Redirects are returned to the caller, since a different
    destination requires a new signature. Closing the client leaves the Runtime
    session alive; its owner must stop it separately.

    See https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-session-endpoints.html
    for the URL, allowed ports, and ``InvokeAgentRuntimeProxy`` permission.
    """
    credentials = await (await get_async_session()).get_credentials()
    if credentials is None:
        raise RuntimeError("No AWS credentials found.")
    async with httpx.AsyncClient(
        base_url=url,
        auth=_SessionEndpointAuth(credentials, region, "bedrock-agentcore"),
        timeout=timeout,
        follow_redirects=False,
    ) as client:
        yield client
