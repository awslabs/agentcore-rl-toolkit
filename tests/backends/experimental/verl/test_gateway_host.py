"""gateway_host tests: threaded serving, singleton semantics, and one
end-to-end OpenAI-wire turn captured through the verl sampling backend."""

import urllib.request

import pytest

from agentcore_rl_toolkit.backends.experimental.verl.gateway_host import _url_host, get_or_start_gateway

from .conftest import FakeLLMServerClient, FakeTokenizer


def _start(client=None):
    return get_or_start_gateway(
        server_manager=client or FakeLLMServerClient(),
        tokenizer=FakeTokenizer(),
        host="127.0.0.1",
        public_host="127.0.0.1",
    )


def test_starts_and_serves_healthz():
    handle = _start()
    assert handle.base_url.startswith("http://127.0.0.1:")
    with urllib.request.urlopen(f"{handle.base_url}/healthz", timeout=5) as r:
        assert r.status == 200


def test_singleton_returns_same_handle():
    h1 = _start()
    h2 = _start()
    assert h1 is h2


@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", "127.0.0.1"),  # IPv4 untouched
        ("10.4.132.156", "10.4.132.156"),
        ("gateway.internal", "gateway.internal"),  # hostname untouched
        ("::1", "[::1]"),  # IPv6 literals bracketed...
        ("2001:db8::1", "[2001:db8::1]"),  # ...or the :port suffix is ambiguous
    ],
)
def test_url_host_brackets_ipv6(host, expected):
    assert _url_host(host) == expected


def test_ipv6_public_host_yields_parseable_base_url():
    """An IPv6 node/public host must produce a URL clients can parse: without
    brackets, http://2001:db8::1:1234 has an ambiguous port."""
    from urllib.parse import urlparse

    url = f"http://{_url_host('2001:db8::1')}:1234"
    parsed = urlparse(url)
    assert parsed.hostname == "2001:db8::1"
    assert parsed.port == 1234


@pytest.mark.asyncio
async def test_end_to_end_turn_capture():
    """One OpenAI-wire chat turn with the sid in the Bearer slot, then
    finish_session yields a TraceRecord trained on the fake backend's tokens."""
    import aiohttp

    client = FakeLLMServerClient()
    handle = _start(client)
    sid = "test-session-0000-0000-000000000000000"

    handle.gateway.create_session(sid, sampling_defaults={"max_new_tokens": 32})

    async with aiohttp.ClientSession() as http:
        resp = await http.post(
            f"{handle.base_url}/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {sid}"},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["choices"][0]["message"]["content"] == "hello world"

    # backend saw the sid as the sticky request_id
    assert client.calls and client.calls[0]["request_id"] == sid

    records = await handle.gateway.finish_session(sid, reward=1.0)
    assert len(records) == 1
    r = records[0]
    assert r.reward == 1.0
    # response region == the fake backend's two generated tokens, all trained
    assert r.token_ids[-2:] == [101, 102]
    assert r.loss_mask == [1, 1]
    assert r.logprobs == [-0.5, -0.6]
