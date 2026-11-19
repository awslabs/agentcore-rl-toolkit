"""The three GWAS tasks: Open Targets causal gene, Open Targets variant prioritization, GWAS Catalog causal gene.

Each task's system prompt and tools come from the upstream GWAS agent: its prompt text minus the agent's
role sentences and output contract (for Open Targets, the official SDL printed without descriptions), ending
with our one-line answer instruction. The GWAS Catalog task also mounts the Open Targets tool. The episode
protocol is screen's: a turn that hits the output cap continues only if at least one tool call in it parsed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

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
from ..common.paths import DATA
from . import tools

HERE = Path(__file__).resolve().parent
GWAS_TASK = "gwas_causal_gene_gwas_catalog"
TASKS = ("gwas_causal_gene_gwas_catalog", "gwas_causal_gene_opentargets", "gwas_variant_prioritization")
SYSTEM = {
    "gwas_causal_gene_opentargets": (HERE / "prompts/causal_gene_opentargets.txt").read_text(),
    "gwas_variant_prioritization": (HERE / "prompts/variant_prioritization.txt").read_text(),
    GWAS_TASK: (HERE / "prompts/causal_gene_gwas_catalog.txt").read_text(),
}
TASK_TOOLS = {
    "gwas_causal_gene_opentargets": ["query_open_targets_graphql"],
    "gwas_variant_prioritization": ["query_open_targets_graphql"],
    GWAS_TASK: ["query_gwas_catalog_sql", "query_open_targets_graphql"],
}
_BY_NAME = {t["function"]["name"]: t for t in (tools.OT_TOOL, tools.GWAS_TOOL)}
TOOLS = {task: [_BY_NAME[name] for name in TASK_TOOLS[task]] for task in TASKS}
GWAS_CATALOG = "gwas_catalog/2026-03-17"
os.environ.setdefault("GWAS_CATALOG_SQL_DATA_ROOT", str(DATA / GWAS_CATALOG))
os.environ.setdefault("TMPDIR", "/tmp")


def sql_backend():
    # One backend per episode, kept by the tool service; it verifies the three parquet files and the prompt schema.
    backend = tools.SnapshotSQLBackend()
    if any(backend.schema not in SYSTEM[t] for t in TASKS if "query_gwas_catalog_sql" in TASK_TOOLS[t]):
        backend.close()
        raise ProtocolError("GWAS backend schema differs from the system prompt")
    return backend


def run_one(client, base_url, model, row, task, system, invoke):
    if str(getattr(row, "task_name", task)) != task:
        raise ProtocolError(f"row is {row.task_name}, not {task}")
    prompt = str(row.prompt)
    tool_specs = TOOLS[task]
    names = {t["function"]["name"] for t in tool_specs}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    calls, turns, sent = [], [], []
    final, error, finish_reason = "", None, None
    terminal = "max_turns"
    started = time.monotonic()
    for index in range(MAX_TURNS):
        body = payload(messages, tool_specs, model)
        sent.append(check_request(body, system, prompt))
        try:
            response = client.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            completion = response.json()
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            terminal = "model_request_error" if 400 <= code < 500 and code != 429 else "model_backend_error"
            body = exc.response.text[:2000] if exc.response is not None else ""
            error = f"{type(exc).__name__}: {exc} {body}"
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
            if call["id"] not in valid_ids:
                arguments = call["function"]["arguments"]
                status, content = "invalid_arguments", "Error: tool arguments were not a valid JSON object"
                result = {"error": content}
            elif name not in names:
                arguments = next(a for c, a in parsed if c["id"] == call["id"])
                status, content = "invalid_arguments", f"Error: unknown tool {name}"
                result = {"error": content}
            else:
                arguments = next(a for c, a in parsed if c["id"] == call["id"])
                t0 = time.monotonic()
                status, content, result = invoke(name, arguments)
                result = {
                    "status": status,
                    "seconds": round(time.monotonic() - t0, 3),
                    "content_chars": len(content),
                    "result": result,
                }
            calls.append({"turn": index + 1, "name": name, "arguments": arguments, "status": status, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
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
    )
