"""lab_bench_seqqa: native sequence tools.

A turn that hits the output cap continues only if at least one complete tool call in it parsed; a turn
whose tool calls are all malformed ends the episode.
"""

from __future__ import annotations

import json
import time

import requests

from ..common.episode import MAX_OUTPUT, MAX_TURNS, REQUEST_TIMEOUT, check_request, payload, record, turn_record
from .tools import TOOL_SPECS

TASKS = ("lab_bench_seqqa",)
SYSTEM = {
    "lab_bench_seqqa": (
        "Use the available tools as needed to answer the question. "
        "Follow the answer format requested in the question."
    )
}


def _parse_calls(raw):
    """Separate complete JSON-object calls from truncated or malformed calls."""
    envelopes, parsed, errors = [], [], []
    if not isinstance(raw, list):
        return [], [], [{"index": None, "error": "tool_calls must be a list"}]
    for index, call in enumerate(raw):
        if not isinstance(call, dict):
            errors.append({"index": index, "error": "tool call must be an object"})
            continue
        function = call.get("function")
        call_id = call.get("id")
        if (
            not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"]
            or not isinstance(call_id, str)
            or not call_id
        ):
            errors.append({"index": index, "error": "tool call requires id and function name"})
            continue
        arguments = function.get("arguments")
        envelope = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": function["name"],
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
            },
        }
        try:
            value = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(value, dict):
                raise ValueError("tool arguments must be a JSON object")
            # JSON extensions NaN/Infinity cannot become executable arguments.
            json.dumps(value, allow_nan=False)
            entry = {"index": index, "call": envelope, "arguments": value, "error": None}
            parsed.append({"id": call_id, "name": function["name"], "arguments": value})
        except (ValueError, TypeError) as exc:
            entry = {
                "index": index,
                "call": envelope,
                "arguments": arguments,
                "error": f"Invalid tool arguments: {type(exc).__name__}",
            }
            errors.append({"index": index, "error": entry["error"]})
        envelopes.append(entry)
    return envelopes, parsed, errors


def run_one(client: requests.Session, base_url: str, model: str, row, task: str, system: str, invoke) -> dict:
    prompt = str(row.prompt)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    calls, turns, sent = [], [], []
    final, finish_reason, error = "", None, None
    terminal_reason = "max_turns"
    started = time.monotonic()
    for turn_index in range(MAX_TURNS):
        body = payload(messages, TOOL_SPECS, model)
        sent.append(check_request(body, system, prompt))
        try:
            response = client.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            completion = response.json()
            choice = completion["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise ValueError("Backend message must be an object")
        except requests.ReadTimeout as exc:
            error, terminal_reason = f"{type(exc).__name__}: {exc}", "model_turn_timeout"
            break
        except requests.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
            code = exc.response.status_code if exc.response is not None else 0
            terminal_reason = (
                "model_request_error"
                if 400 <= code < 500 and code not in (401, 403, 408, 429)
                else "model_backend_error"
            )
            break
        except Exception as exc:
            error, terminal_reason = f"{type(exc).__name__}: {exc}", "model_backend_error"
            break
        finish_reason = choice.get("finish_reason")
        usage = completion.get("usage") or {}
        output_limit_hit = finish_reason == "length" or (usage.get("completion_tokens") or 0) >= MAX_OUTPUT
        raw_calls = message.get("tool_calls") or []
        envelopes, parsed, parse_errors = _parse_calls(raw_calls)
        turns.append(turn_record(turn_index, message, finish_reason, usage, raw_calls, parse_errors, output_limit_hit))
        if output_limit_hit and not parsed:
            terminal_reason = "model_output_length"
            break
        if not raw_calls:
            final = message.get("content") or ""
            terminal_reason = "final" if final else "empty_final"
            break
        if not envelopes:
            terminal_reason = "malformed_tool_calls"
            break
        # Only the public content and structured calls enter later requests.
        messages.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [entry["call"] for entry in envelopes],
            }
        )
        for entry in envelopes:
            call = entry["call"]
            if entry["error"]:
                result = {"status": "invalid_arguments", "error": entry["error"]}
            else:
                result = invoke(call["function"]["name"], entry["arguments"])
            calls.append(
                {
                    "turn": turn_index + 1,
                    "name": call["function"]["name"],
                    "arguments": entry["arguments"],
                    "status": result.get("status"),
                    "result": result,
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)}
            )
    return record(
        task,
        row,
        system,
        final=final,
        terminal=terminal_reason,
        finish_reason=finish_reason,
        error=error,
        turns=turns,
        calls=calls,
        requests=sent,
        started=started,
    )
