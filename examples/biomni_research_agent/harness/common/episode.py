"""What every task's episode shares: the served model, the sampling, the per-episode budget, the request
check, and the episode record.

Sampling is Qwen3.6's non-thinking mode with presence penalty 0, sent as is with every request; the
reasoning a model returns is recorded but never enters a later turn. Each turn may produce at most
8192 tokens, an episode has at most 20 model turns, and a request that takes longer than 3600 s is cut
(a hang guard; there is no episode deadline).

The record of one episode, the same for every task:
  task_name, task_instance_id (str), subtype (when the item has one), prompt_sha256, system_sha256;
  final_response, parsed_answer, score (the metric's score of the final response);
  terminal_reason, finish_reason (of the last turn), infrastructure_error, error;
  model_turns  [{turn, content, reasoning_content, finish_reason, usage, tool_calls, parse_errors, output_limit}];
  tool_calls   [{turn, name, arguments, status, result}];
  requests     [{the sampling fields, model, message_count}], one per model request; seconds.
"""

from __future__ import annotations

import hashlib
import time

from . import metric

SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
    "max_tokens": 8192,
}
MAX_OUTPUT = SAMPLING["max_tokens"]
MAX_TURNS = 20
REQUEST_TIMEOUT = 3600
CONTEXT_LENGTH = 65536
INFRA_ENDINGS = {"model_backend_error", "model_turn_timeout"}


class ProtocolError(RuntimeError):
    """The harness built a request it must not send; the run stops."""


def payload(messages: list, tools: list, model: str) -> dict:
    return {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto", **SAMPLING}


def check_request(body: dict, system: str, prompt: str) -> dict:
    """Refuse a request whose sampling, system prompt or question is not the pinned one, or into
    which earlier reasoning leaked; return what the record keeps of it."""
    actual = {k: body.get(k) for k in SAMPLING}
    msgs = body["messages"]
    if actual != SAMPLING:
        raise ProtocolError("request sampling differs from the pinned one")
    if msgs[0] != {"role": "system", "content": system} or msgs[1] != {"role": "user", "content": prompt}:
        raise ProtocolError("request system prompt or question differs")
    if any("reasoning_content" in m or "reasoning" in m for m in msgs):
        raise ProtocolError("earlier reasoning entered the request")
    return {**actual, "model": body["model"], "message_count": len(msgs)}


def turn_record(index: int, message: dict, finish_reason, usage, requested, parse_errors, output_limit) -> dict:
    return {
        "turn": index + 1,
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content") or message.get("reasoning"),
        "finish_reason": finish_reason,
        "usage": usage,
        "tool_calls": requested,
        "parse_errors": parse_errors,
        "output_limit": output_limit,
    }


def record(
    task: str,
    row,
    system: str,
    *,
    final: str,
    terminal: str,
    finish_reason,
    error,
    turns: list,
    calls: list,
    requests: list,
    started: float,
    denial=lambda call: False,
) -> dict:
    """The episode record. Infrastructure: a model backend failure or turn timeout, or a tool whose backend
    failed (except the task's `denial`, a database refusing the model's own SQL)."""
    prompt = str(row.prompt)
    infra = terminal in INFRA_ENDINGS or any(c["status"] == "backend_error" and not denial(c) for c in calls)
    out = {"task_name": task, "task_instance_id": str(row.task_instance_id)}
    if getattr(row, "subtype", None) is not None:
        out["subtype"] = row.subtype
    out.update(
        {
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "final_response": final,
            "parsed_answer": metric.extract(task, final, prompt=prompt),
            "score": float(metric.score(task, final, row.answer, prompt=prompt)),
            "terminal_reason": terminal,
            "finish_reason": finish_reason,
            "infrastructure_error": infra,
            "error": error,
            "model_turns": turns,
            "tool_calls": calls,
            "requests": requests,
            "seconds": round(time.monotonic() - started, 3),
        }
    )
    return out
