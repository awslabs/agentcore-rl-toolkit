"""Detect upstream chat-template changes against the vendored hash registry.

Detection in ``_TEMPLATE_HASHES`` is byte-exact, and the offline unit tests bypass
it (test_response_templates.py assigns templates directly) — so a mistranscribed hash,
or a vendor revising a template on the hub, would silently drop real models into
the no-template rejection. These tests fetch each family's actual chat template from
the hub (a few KB per repo, never model weights) and assert it still resolves to
the expected template. A failure means the upstream template changed or a registry
hash is stale: re-sync per the workflow in response_templates.py. The Qwen3.5-family
test renders a tool call with each hub template and asserts that parsing it with
QWEN3_5_TEMPLATE recovers the exact arguments.
"""

import hashlib
import json

import pytest
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from transformers.utils.chat_parsing import parse_response
from transformers.utils.chat_template_utils import render_jinja_template

from agentcore_rl_toolkit.rollout_gateway.response_templates import _TEMPLATE_HASHES, RESPONSE_TEMPLATES

# One representative repo per registered template hash (not per template: e.g. the
# Qwen3.5 "think" and "nothink" template variants are two distinct hashes).
REPO_EXPECTATIONS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen3"),
    ("Qwen/Qwen3-0.6B", "qwen3"),
    ("Qwen/Qwen3-4B-Instruct-2507", "qwen3"),
    ("Qwen/Qwen3-VL-2B-Instruct", "qwen3"),
    ("Qwen/Qwen3.5-0.8B", "qwen3_5"),  # nothink template variant
    ("Qwen/Qwen3.5-27B", "qwen3_5"),  # think template variant
    ("Qwen/Qwen3.6-27B", "qwen3_5"),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_5"),  # XML tool calls despite Qwen3 era
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
        f"re-sync per response_templates.py"
    )


QWEN3_5_REPOS = [repo for repo, expected in REPO_EXPECTATIONS if expected == "qwen3_5"]
SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
}


@pytest.mark.parametrize("repo", QWEN3_5_REPOS)
@pytest.mark.parametrize("value", ["ls -la", "cat <<'EOF' > f\nhi\nEOF\n", "  echo indented", "echo a\n\n"])
def test_qwen3_5_tool_call_round_trips_through_hub_template(repo, value):
    # Parsing the template's own rendering of a tool call must recover the exact
    # argument, or the replayed history stops extending the sampled tokens.
    template = _fetch_chat_template(repo)
    call = {"type": "function", "function": {"name": "shell", "arguments": {"command": value}}}
    user = [{"role": "user", "content": "q"}]
    (prompt,), _ = render_jinja_template([user], tools=[SHELL_TOOL], chat_template=template, add_generation_prompt=True)
    (full,), _ = render_jinja_template(
        [user + [{"role": "assistant", "content": "", "tool_calls": [call]}]],
        tools=[SHELL_TOOL],
        chat_template=template,
    )
    assert full.startswith(prompt)
    generated = full[len(prompt) : full.rindex("<|im_end|>") + len("<|im_end|>")]
    parsed = parse_response(generated, RESPONSE_TEMPLATES["qwen3_5"], prefix=prompt, tools=[SHELL_TOOL])
    assert parsed["tool_calls"][0]["function"]["arguments"] == {"command": value}
