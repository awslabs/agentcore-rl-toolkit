"""screen_gene_retrieval: the official question, ORCS SQL without the hit flag, and related online tools.

The model must choose the source, find the screen and judge significance from its scores and the
screen's stated criteria. A turn that hits the output cap continues only if at least one tool call in it
parsed. SQLite authorization denials of the model's own ORCS SQL are not infrastructure.
"""

from __future__ import annotations

import json
import time

import requests

from ..common.episode import (
    MAX_OUTPUT,
    MAX_TURNS,
    REQUEST_TIMEOUT,
    ProtocolError,
    check_request,
    payload,
    record,
    turn_record,
)
from .tools import TOOL_SPECS

TASKS = ("screen_gene_retrieval",)
SYSTEM = {
    "screen_gene_retrieval": (
        "Use the available tools as needed to answer the question. "
        "Put exactly one candidate gene symbol inside <solution></solution> at the end of your response."
    )
}


def sql_denial(call):
    error = call["result"].get("error", "")
    return call["name"] == "query_orcs" and (
        error == "not authorized" or error.startswith("access to ") and error.endswith(" is prohibited")
    )


def run_one(client, base_url, model, row, task, system, invoke):
    prompt = str(row.prompt)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    calls, turns, sent = [], [], []
    final, error, finish_reason = "", None, None
    terminal = "max_turns"
    started = time.monotonic()
    for index in range(MAX_TURNS):
        body = payload(messages, TOOL_SPECS, model)
        sent.append(check_request(body, system, prompt))
        try:
            response = client.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            completion = response.json()
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            terminal = "model_request_error" if 400 <= code < 500 and code != 429 else "model_backend_error"
            error = f"{type(exc).__name__}: {exc}"
            break
        except requests.Timeout as exc:
            terminal, error = "model_turn_timeout", f"{type(exc).__name__}: {exc}"
            break
        except requests.RequestException as exc:
            terminal, error = "model_backend_error", f"{type(exc).__name__}: {exc}"
            break
        try:
            choice = completion["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")
            usage = completion.get("usage") or {}
            capped = finish_reason == "length" or usage.get("completion_tokens", 0) >= MAX_OUTPUT
            requested = message.get("tool_calls") or []
            parsed, invalid = [], []
            for call in requested:
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be an object")
                    if not call.get("id") or not call["function"].get("name"):
                        raise ValueError("missing tool ID or name")
                    parsed.append((call, arguments))
                except (ValueError, TypeError, KeyError) as exc:
                    invalid.append({"call": call, "error": str(exc)})
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise ProtocolError("Malformed completion response") from exc
        turns.append(turn_record(index, message, finish_reason, usage, requested, invalid, capped))
        if capped and not parsed:
            final = message.get("content") or ""
            terminal = "model_output_length"
            break
        if not requested:
            final = message.get("content") or ""
            terminal = "final" if final else "empty_final"
            break
        # Invalid fragments never reach a tool. Outside the capped/no-call
        # branch, errors can be returned for structurally complete call IDs.
        valid_ids = {c["id"] for c, _ in parsed}
        usable_invalid = [
            x
            for x in invalid
            if isinstance(x["call"], dict)
            and x["call"].get("id")
            and isinstance(x["call"].get("function"), dict)
            and x["call"]["function"].get("name")
        ]
        replay_calls = [c for c, _ in parsed] + [x["call"] for x in usable_invalid]
        if not replay_calls:
            raise ProtocolError("Tool parser returned structurally invalid calls")
        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": replay_calls})
        for call in replay_calls:
            name = call["function"]["name"]
            if call["id"] in valid_ids:
                arguments = next(a for c, a in parsed if c["id"] == call["id"])
                try:
                    result = invoke(name, arguments)
                except (ValueError, TypeError) as exc:
                    result = {"status": "invalid_arguments", "error": str(exc)}
            else:
                arguments = call["function"]["arguments"]
                result = {"status": "invalid_arguments", "error": "Tool arguments were not valid JSON objects"}
            calls.append(
                {
                    "turn": index + 1,
                    "name": name,
                    "arguments": arguments,
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
        terminal=terminal,
        finish_reason=finish_reason,
        error=error,
        turns=turns,
        calls=calls,
        requests=sent,
        started=started,
        denial=sql_denial,
    )
