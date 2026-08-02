"""Detect upstream chat-template changes against the vendored hash registry.

Detection in ``_TEMPLATE_HASHES`` is byte-exact, and the offline unit tests bypass
it (test_response_schemas.py assigns schemas directly) — so a mistranscribed hash,
or a vendor revising a template on the hub, would silently drop real models into
the no-schema rejection. These tests fetch each family's actual chat template from
the hub (a few KB per repo, never model weights) and assert it still resolves to
the expected schema. A failure means the upstream template changed or a registry
hash is stale: re-sync per the workflow in response_schemas.py.
"""

import hashlib
import json

import pytest
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from agentcore_rl_toolkit.rollout_gateway.response_schemas import _TEMPLATE_HASHES

# One representative repo per registered template hash (not per schema: e.g. the
# Qwen3.5 "think" and "nothink" template variants are two distinct hashes).
REPO_EXPECTATIONS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen3"),
    ("Qwen/Qwen3-0.6B", "qwen3"),
    ("Qwen/Qwen3-4B-Instruct-2507", "qwen3"),
    ("Qwen/Qwen3-VL-2B-Instruct", "qwen3"),
    ("Qwen/Qwen3.5-0.8B", "qwen3_5"),  # nothink template variant
    ("Qwen/Qwen3.5-27B", "qwen3_5"),  # think template variant
    ("Qwen/Qwen3.6-27B", "qwen3_5"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "qwen3_5"),
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "qwen3_5"),
    ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "qwen3_5"),
    ("zai-org/GLM-4.5-Air", "glm4moe"),
    ("openai/gpt-oss-20b", "gptoss"),
]


def _fetch_chat_template(repo: str) -> str:
    try:
        with open(hf_hub_download(repo, "chat_template.jinja")) as f:
            return f.read()
    except EntryNotFoundError:
        # No standalone template file — the template lives in tokenizer_config.json.
        with open(hf_hub_download(repo, "tokenizer_config.json")) as f:
            return json.load(f)["chat_template"]


@pytest.mark.parametrize("repo,expected", REPO_EXPECTATIONS)
def test_hub_template_hash_resolves(repo, expected):
    template = _fetch_chat_template(repo)
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    assert _TEMPLATE_HASHES.get(digest) == expected, (
        f"{repo}'s chat template (sha256 {digest[:16]}…) does not resolve to "
        f"{expected!r} — the upstream template changed or the registry hash is stale; "
        f"re-sync per response_schemas.py"
    )
