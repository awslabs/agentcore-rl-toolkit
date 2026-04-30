"""Verify the SGLang v0.5.9 token_ids patch.

Launches a local SGLang server with the given model, issues greedy-decoded
chat completion requests in both streaming and non-streaming modes, and
checks that:

1. Non-streaming responses expose prompt_token_ids (on the response) and
   token_ids (per choice), and decoding token_ids matches the content.
2. Streaming responses expose prompt_token_ids on the first chunk and
   token_ids on content chunks, and decoding the concatenation matches
   the concatenated deltas.
3. Under greedy decoding, the prompt and completion token sequences are
   identical between streaming and non-streaming modes.
4. Streaming with tools active yields the same completion token sequence
   as streaming without tools, validating the _pending_token_ids buffering.

Usage:
    python -m agentcore_rl_toolkit.backends.slime.patches.verify_sglang_token_ids \
        --model-path /path/to/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

GREEDY_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "seed": 42,
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression to evaluate, e.g. '7*6'.",
                }
            },
            "required": ["expression"],
        },
    },
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(base_url: str, timeout_s: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_s
    last_err: Optional[str] = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/health_generate", timeout=5.0)
            if r.status_code == 200:
                return True
            last_err = f"status={r.status_code}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(1.0)
    sys.stderr.write(f"server did not become ready in {timeout_s}s; last={last_err}\n")
    return False


def _post_chat(
    base_url: str,
    messages: list[dict],
    *,
    stream: bool,
    tools: Optional[list[dict]] = None,
    model: str = "default",
) -> httpx.Response:
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        **GREEDY_PARAMS,
    }
    if tools is not None:
        payload["tools"] = tools
    return httpx.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120.0,
    )


def _parse_sse(text: str) -> list[dict]:
    chunks: list[dict] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :].strip()
        if data == "[DONE]":
            continue
        chunks.append(json.loads(data))
    return chunks


def _collect_stream(chunks: list[dict]) -> dict:
    """Collect streaming chunks for index 0 into a single aggregate."""
    prompt_token_ids: Optional[list[int]] = None
    prompt_seen_count = 0
    completion_token_ids: list[int] = []
    content_parts: list[str] = []
    tool_call_parts: list[str] = []
    finish_reason: Optional[str] = None

    for chunk in chunks:
        if chunk.get("prompt_token_ids") is not None:
            prompt_seen_count += 1
            if prompt_token_ids is None:
                prompt_token_ids = chunk["prompt_token_ids"]
        choices = chunk.get("choices") or []
        for choice in choices:
            if choice.get("index", 0) != 0:
                continue
            tids = choice.get("token_ids")
            if tids:
                completion_token_ids.extend(tids)
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if fn.get("name"):
                    tool_call_parts.append(fn["name"])
                if fn.get("arguments"):
                    tool_call_parts.append(fn["arguments"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    return {
        "prompt_token_ids": prompt_token_ids,
        "prompt_seen_count": prompt_seen_count,
        "completion_token_ids": completion_token_ids,
        "content": "".join(content_parts),
        "tool_call_text": "".join(tool_call_parts),
        "finish_reason": finish_reason,
    }


def _diverge_at(a: list[int], b: list[int]) -> Optional[int]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def _snippet(ids: list[int], at: int, window: int = 3) -> str:
    lo = max(0, at - window)
    hi = min(len(ids), at + window + 1)
    return "... " + ", ".join(str(x) for x in ids[lo:hi]) + " ..."


def check_non_streaming_basic(base_url: str, tokenizer) -> tuple[CheckResult, dict]:
    messages = [{"role": "user", "content": "What is 2+2?"}]
    r = _post_chat(base_url, messages, stream=False)
    if r.status_code != 200:
        return CheckResult("non_streaming_basic", False, f"http {r.status_code}: {r.text[:200]}"), {}
    data = r.json()
    choice = data["choices"][0]

    if choice.get("finish_reason") == "length":
        return CheckResult(
            "non_streaming_basic",
            False,
            "response truncated at max_tokens; increase limit or pick a shorter prompt",
        ), {}

    prompt_ids = data.get("prompt_token_ids")
    if not prompt_ids:
        return CheckResult("non_streaming_basic", False, "prompt_token_ids missing or empty"), {}

    token_ids = choice.get("token_ids")
    if not token_ids:
        return CheckResult("non_streaming_basic", False, "choices[0].token_ids missing or empty"), {}

    content = choice["message"]["content"] or ""
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    if decoded.strip() != content.strip():
        return CheckResult(
            "non_streaming_basic",
            False,
            f"decoded token_ids != content\n  decoded: {decoded[:120]!r}\n  content: {content[:120]!r}",
        ), {}

    return CheckResult("non_streaming_basic", True), {
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": token_ids,
        "content": content,
    }


def check_streaming_basic(base_url: str, tokenizer) -> tuple[CheckResult, dict]:
    messages = [{"role": "user", "content": "What is 2+2?"}]
    r = _post_chat(base_url, messages, stream=True)
    if r.status_code != 200:
        return CheckResult("streaming_basic", False, f"http {r.status_code}: {r.text[:200]}"), {}

    chunks = _parse_sse(r.text)
    agg = _collect_stream(chunks)

    if agg["finish_reason"] == "length":
        return CheckResult(
            "streaming_basic",
            False,
            "stream truncated at max_tokens; increase limit",
        ), {}

    if agg["prompt_token_ids"] is None:
        return CheckResult("streaming_basic", False, "prompt_token_ids not found on any chunk"), {}
    if agg["prompt_seen_count"] != 1:
        return CheckResult(
            "streaming_basic",
            False,
            f"prompt_token_ids appeared on {agg['prompt_seen_count']} chunks, expected exactly 1",
        ), {}

    if not agg["completion_token_ids"]:
        return CheckResult("streaming_basic", False, "no completion token_ids collected across chunks"), {}

    decoded = tokenizer.decode(agg["completion_token_ids"], skip_special_tokens=True)
    if decoded.strip() != agg["content"].strip():
        return CheckResult(
            "streaming_basic",
            False,
            (
                "decoded token_ids != concatenated delta content\n"
                f"  decoded: {decoded[:120]!r}\n"
                f"  content: {agg['content'][:120]!r}"
            ),
        ), {}

    return CheckResult("streaming_basic", True), agg


def check_cross_mode_consistency(non_stream: dict, stream: dict) -> CheckResult:
    if not non_stream or not stream:
        return CheckResult(
            "cross_mode_consistency",
            False,
            "prerequisite check(s) failed, skipping comparison",
            skipped=True,
        )

    if non_stream["prompt_token_ids"] != stream["prompt_token_ids"]:
        at = _diverge_at(non_stream["prompt_token_ids"], stream["prompt_token_ids"])
        return CheckResult(
            "cross_mode_consistency",
            False,
            (
                f"prompt_token_ids differ (len ns={len(non_stream['prompt_token_ids'])}, "
                f"len s={len(stream['prompt_token_ids'])}, diverge at {at})"
            ),
        )

    ns_ids = non_stream["completion_token_ids"]
    s_ids = stream["completion_token_ids"]
    if ns_ids != s_ids:
        at = _diverge_at(ns_ids, s_ids)
        return CheckResult(
            "cross_mode_consistency",
            False,
            (
                f"completion_token_ids diverge at index {at}\n"
                f"  non-stream: {_snippet(ns_ids, at or 0)}\n"
                f"  streaming:  {_snippet(s_ids, at or 0)}\n"
                f"  lengths: ns={len(ns_ids)} s={len(s_ids)}"
            ),
        )

    if non_stream["content"].strip() != stream["content"].strip():
        return CheckResult(
            "cross_mode_consistency",
            False,
            "content text differs between streaming and non-streaming",
        )

    return CheckResult("cross_mode_consistency", True)


def check_tool_call_consistency(base_url: str) -> CheckResult:
    """Same cross-mode consistency check as basic, but with tools enabled.

    Exercises the _pending_token_ids buffering on the tool-call streaming path:
    the tool-call parser may yield 0, 1, or multiple chunks per engine yield,
    and the patch must still produce a completion_token_ids sequence equal to
    the non-streaming response for the same prompt under greedy decoding.
    """
    messages = [
        {
            "role": "user",
            "content": "What is 7 times 6? Use the calculator tool.",
        }
    ]

    r_ns = _post_chat(base_url, messages, stream=False, tools=[CALCULATOR_TOOL])
    if r_ns.status_code != 200:
        return CheckResult(
            "tool_call_consistency",
            False,
            f"non-streaming request failed: http {r_ns.status_code}: {r_ns.text[:200]}",
        )
    ns_data = r_ns.json()
    ns_choice = ns_data["choices"][0]
    if ns_choice.get("finish_reason") == "length":
        return CheckResult(
            "tool_call_consistency",
            False,
            "non-streaming tool-call truncated at max_tokens",
        )
    ns_prompt = ns_data.get("prompt_token_ids")
    ns_completion = ns_choice.get("token_ids")
    if not ns_prompt or not ns_completion:
        return CheckResult(
            "tool_call_consistency",
            False,
            "non-streaming response missing prompt_token_ids or token_ids",
        )

    r_s = _post_chat(base_url, messages, stream=True, tools=[CALCULATOR_TOOL])
    if r_s.status_code != 200:
        return CheckResult(
            "tool_call_consistency",
            False,
            f"streaming request failed: http {r_s.status_code}: {r_s.text[:200]}",
        )
    agg = _collect_stream(_parse_sse(r_s.text))
    if agg["finish_reason"] == "length":
        return CheckResult(
            "tool_call_consistency",
            False,
            "streaming tool-call truncated at max_tokens",
        )
    if agg["prompt_token_ids"] is None:
        return CheckResult(
            "tool_call_consistency",
            False,
            "streaming response missing prompt_token_ids",
        )
    if not agg["completion_token_ids"]:
        return CheckResult(
            "tool_call_consistency",
            False,
            "streaming response collected no token_ids",
        )

    if ns_prompt != agg["prompt_token_ids"]:
        at = _diverge_at(ns_prompt, agg["prompt_token_ids"])
        return CheckResult(
            "tool_call_consistency",
            False,
            (
                f"prompt_token_ids differ (len ns={len(ns_prompt)}, "
                f"len s={len(agg['prompt_token_ids'])}, diverge at {at})"
            ),
        )
    if ns_completion != agg["completion_token_ids"]:
        at = _diverge_at(ns_completion, agg["completion_token_ids"])
        return CheckResult(
            "tool_call_consistency",
            False,
            (
                f"completion_token_ids diverge at index {at}\n"
                f"  non-stream: {_snippet(ns_completion, at or 0)}\n"
                f"  streaming:  {_snippet(agg['completion_token_ids'], at or 0)}\n"
                f"  lengths: ns={len(ns_completion)} s={len(agg['completion_token_ids'])}"
            ),
        )

    return CheckResult("tool_call_consistency", True)


def run_checks(base_url: str, model_path: str) -> list[CheckResult]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    results: list[CheckResult] = []
    r1, ns_data = check_non_streaming_basic(base_url, tokenizer)
    results.append(r1)
    r2, s_data = check_streaming_basic(base_url, tokenizer)
    results.append(r2)

    if r1.passed and r2.passed:
        results.append(check_cross_mode_consistency(ns_data, s_data))
    else:
        results.append(
            CheckResult(
                "cross_mode_consistency",
                False,
                "skipped: basic checks did not both pass",
                skipped=True,
            )
        )

    results.append(check_tool_call_consistency(base_url))
    return results


def _print_report(results: list[CheckResult], base_url: str, model_path: str) -> int:
    print("SGLang token_ids patch verification")
    print(f"Model:  {model_path}")
    print(f"Server: {base_url}")
    print()
    failed = 0
    for i, r in enumerate(results, 1):
        status = "PASS" if r.passed else ("SKIP" if r.skipped else "FAIL")
        print(f"[{i}/{len(results)}] {r.name:<30} {status}")
        if r.detail:
            for line in r.detail.splitlines():
                print(f"        {line}")
        if not r.passed and not r.skipped:
            failed += 1
    print()
    if failed:
        print(f"FAILED: {failed}/{len(results)} checks")
    else:
        print(f"OK: {len(results)}/{len(results)} checks passed")
    return 1 if failed else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the model (local dir or HF repo id).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the SGLang server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port for the SGLang server (default: auto).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for /health_generate (default: 180).",
    )
    args = parser.parse_args(argv)

    if not Path(args.model_path).exists() and "/" not in args.model_path:
        print(
            f"warning: model path {args.model_path!r} does not exist locally; " "assuming HF repo id", file=sys.stderr
        )

    port = args.port or _free_port()
    base_url = f"http://{args.host}:{port}"

    stderr_file = tempfile.NamedTemporaryFile(prefix="sglang-verify-", suffix=".log", delete=False)
    print(f"launching sglang server on {base_url} (stderr -> {stderr_file.name})")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model_path,
            "--host",
            args.host,
            "--port",
            str(port),
        ],
        stdout=stderr_file,
        stderr=subprocess.STDOUT,
    )

    try:
        if not _wait_for_server(base_url, timeout_s=args.startup_timeout):
            tail = _tail(stderr_file.name, 40)
            print(f"\nserver startup failed. stderr tail:\n{tail}", file=sys.stderr)
            return 2

        results = run_checks(base_url, args.model_path)
        return _print_report(results, base_url, args.model_path)
    finally:
        print("\nshutting down sglang server...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        stderr_file.close()


def _tail(path: str, lines: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read = min(size, 64 * 1024)
            f.seek(size - read)
            tail = f.read().decode(errors="replace")
        return "\n".join(tail.splitlines()[-lines:])
    except Exception as e:
        return f"<could not read log: {e}>"


if __name__ == "__main__":
    sys.exit(main())
