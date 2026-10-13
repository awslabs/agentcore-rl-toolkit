"""TMax's Vanillux2 loop, using OpenAI Chat Completions and sandboxd exec."""

import math
import time

from openai import APIConnectionError, APIStatusError, BadRequestError, OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt
from vanillux import (
    _SYSTEM_TEMPLATE,
    SUBMIT_MARKER,
    TOOL_SCHEMAS,
    _extract_tool_call,
    _format_error_message,
    _render_instance,
    _truncate_observation,
)

from agentcore_rl_toolkit.sandbox import ExecTimeoutError

STATE = '"$TMPDIR/.vanillux2"'


class Bash:
    def __init__(self, sandbox):
        self.sandbox = sandbox

    def exec(self, command, timeout=120):
        try:
            return self.sandbox.exec(["/bin/bash", "-c", command], timeout=timeout)
        except ExecTimeoutError as exc:
            return exc.result

    def setup(self, timeout=120):
        result = self.exec(f"mkdir -p {STATE} && pwd > {STATE}/cwd && export -p > {STATE}/env", timeout=timeout)
        if result.exit_code:
            raise RuntimeError(f"Shell setup failed: {result.stderr}")

    def run(self, command, timeout=120):
        # Match Vanillux2's cwd/export persistence; every exec is still a new shell.
        # Anchor before sourcing exports, so changing TMPDIR does not move state.
        state = '"$_vanillux2_state"'
        return self.exec(
            f"_vanillux2_state={STATE}\n"
            f'cd "$(cat {state}/cwd)" 2>/dev/null || true\n'
            f". {state}/env 2>/dev/null || true\n"
            f"{command}\n"
            "_vanillux2_ec=$?\n"
            f"pwd > {state}/cwd\n"
            f"export -p > {state}/env\n"
            "exit $_vanillux2_ec",
            timeout=timeout,
        )


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Agent time budget exhausted")
    return seconds


def _retryable(exc):
    return isinstance(exc, APIConnectionError) or (
        isinstance(exc, APIStatusError) and (exc.status_code in (408, 409, 429) or exc.status_code >= 500)
    )


@retry(
    retry=retry_if_exception(_retryable),
    stop=stop_after_attempt(5),
    wait=lambda state: min(2**state.attempt_number, remaining(state.kwargs["deadline"])),
    reraise=True,
)
def query(client, config, messages, *, deadline):
    response = client.chat.completions.create(
        model=config["model_id"],
        messages=messages,
        tools=TOOL_SCHEMAS,
        max_tokens=16384,
        temperature=0.7,
        top_p=0.95,
        timeout=remaining(deadline),
    )
    remaining(deadline)
    return response


def run(instruction, bash, config, *, timeout=600, max_steps=64, record=None):
    deadline = time.monotonic() + timeout
    messages = [
        {"role": "system", "content": _SYSTEM_TEMPLATE},
        {"role": "user", "content": _render_instance(instruction.strip())},
    ]
    result = record if record is not None else {}
    result.update(trajectory=messages, timing=[], usage={}, stop_reason="max_steps")
    format_errors = 0
    # sandboxd accepts whole seconds; check the deadline again after each call.
    bash.setup(timeout=math.ceil(min(120, remaining(deadline))))
    remaining(deadline)
    with OpenAI(
        base_url=config["base_url"], api_key=config.get("api_key", "EMPTY"), max_retries=0, timeout=600
    ) as client:
        for step in range(max_steps):
            start = time.monotonic()
            try:
                response = query(client, config, messages, deadline=deadline)
            except BadRequestError as exc:
                if "context" not in str(exc).lower():
                    raise
                result["stop_reason"] = "context_limit"
                break
            llm_s = time.monotonic() - start
            if response.usage:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    result["usage"][key] = result["usage"].get(key, 0) + getattr(response.usage, key)
            msg = response.choices[0].message.model_dump(exclude_none=True)
            action = _extract_tool_call(msg)
            # Vanillux2 accepts the first bash call. Retain only that call so the
            # next request has exactly one tool result for every assistant call.
            if action["type"] == "no_tool_call":
                msg.pop("tool_calls", None)
                msg["content"] = msg.get("content") or ""
                messages.extend(
                    [
                        msg,
                        {
                            "role": "user",
                            "content": _format_error_message(
                                "Your last response did not include a valid `bash` tool call."
                            ),
                        },
                    ]
                )
                format_errors += 1
                result["timing"].append({"step": step + 1, "llm_s": llm_s, "format_error": True})
                if format_errors >= 64:
                    result["stop_reason"] = "format_errors"
                    break
                continue
            msg["tool_calls"] = msg["tool_calls"][:1]
            messages.append(msg)
            format_errors = 0
            start = time.monotonic()
            execution = bash.run(action["command"], timeout=math.ceil(min(120, remaining(deadline))))
            output = "\n".join(part for part in (execution.stdout, execution.stderr) if part).rstrip()
            content = (
                f"{_truncate_observation(output) if output else '(no output)'}\n\n(exit_code={execution.exit_code})"
            )
            messages.append({"role": "tool", "tool_call_id": action["tool_call_id"], "content": content})
            result["timing"].append(
                {
                    "step": step + 1,
                    "llm_s": llm_s,
                    "bash_s": time.monotonic() - start,
                    "exit_code": execution.exit_code,
                    "timed_out": execution.timed_out,
                    "output_truncated": execution.stdout_truncated or execution.stderr_truncated,
                }
            )
            remaining(deadline)
            if action["type"] == "done" or SUBMIT_MARKER in content:
                result["stop_reason"] = "submitted"
                break
    return result
