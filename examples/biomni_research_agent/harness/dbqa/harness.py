"""lab_bench_dbqa: local evidence SQL, sequence tools and direct-query online databases.

The model writes every query; nothing is matched or answered for it. SQLite authorization denials of the
model's own SQL are the model's error, not infrastructure. A turn that hits the output cap still runs the
tool calls it made.
"""

from __future__ import annotations

import json
import time

import requests

from ..common.episode import MAX_TURNS, REQUEST_TIMEOUT, check_request, payload, record, turn_record
from .tools import TOOL_SPECS

TASKS = ("lab_bench_dbqa",)
SYSTEM = {
    "lab_bench_dbqa": (
        "Use the available evidence tools as needed to answer the question. "
        "A missing record is not proof of absence. "
        "Submit exactly one uppercase option letter as your final response."
    )
}
LOCAL_SQL_TOOLS = {"query_gene_sets", "query_interactions", "query_disease_evidence", "query_clinvar_records"}


def sql_denial(call: dict) -> bool:
    if call["name"] not in LOCAL_SQL_TOOLS:
        return False
    text = call["result"].get("error", "")
    return text.startswith("access to ") and text.endswith(" is prohibited") or text == "not authorized"


def run_one(client: requests.Session, base_url: str, model: str, row, task: str, system: str, invoke) -> dict:
    prompt = str(row.prompt)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    calls, turns, sent = [], [], []
    final, finish_reason, error = "", None, None
    terminal_reason = "max_turns"
    started = time.monotonic()
    for index in range(MAX_TURNS):
        body = payload(messages, TOOL_SPECS, model)
        sent.append(check_request(body, system, prompt))
        try:
            response = client.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            completion = response.json()
        except requests.ConnectTimeout as exc:
            error, terminal_reason = f"{type(exc).__name__}: {exc}", "model_backend_error"
            break
        except requests.ReadTimeout as exc:
            error, terminal_reason = f"{type(exc).__name__}: {exc}", "model_turn_timeout"
            break
        except requests.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
            code = exc.response.status_code if exc.response is not None else 0
            terminal_reason = "model_request_error" if 400 <= code < 500 and code != 429 else "model_backend_error"
            break
        except Exception as exc:
            error, terminal_reason = f"{type(exc).__name__}: {exc}", "model_backend_error"
            break
        choice = completion["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")
        requested = message.get("tool_calls") or []
        turns.append(
            turn_record(
                index, message, finish_reason, completion.get("usage"), requested, [], finish_reason == "length"
            )
        )
        if not requested:
            if finish_reason == "length":
                terminal_reason = "model_output_length"
            else:
                final = message.get("content") or ""
                terminal_reason = "final" if final else "empty_final"
            break
        messages.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]},
                    }
                    for call in requested
                ],
            }
        )
        for call in requested:
            try:
                arguments = json.loads(call["function"]["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                result = invoke(call["function"]["name"], arguments)
            except (ValueError, TypeError) as exc:
                arguments = call["function"]["arguments"]
                result = {"status": "invalid_arguments", "error": str(exc)}
            calls.append(
                {
                    "turn": index + 1,
                    "name": call["function"]["name"],
                    "arguments": arguments,
                    "status": result.get("status"),
                    "result": result,
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)}
            )
        if finish_reason == "length":
            terminal_reason = "model_output_length"
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
        denial=sql_denial,
    )
