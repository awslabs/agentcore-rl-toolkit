"""Versioned start/get messages carried through the HTTP invocation endpoint."""

import re
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

ENVELOPE_KEY = "_agentcore_runtime"
VERSION = 1

Status = Literal["in_progress", "completed", "interrupted", "not_found"]


class InvocationState(TypedDict):
    version: int
    invocation_id: str
    status: Status
    result: NotRequired[Any]
    error: NotRequired[str]


def state(invocation_id: str, status: Status) -> InvocationState:
    return {"version": VERSION, "invocation_id": invocation_id, "status": status}


@dataclass(frozen=True)
class InvocationRequest:
    operation: Literal["start", "get"]
    invocation_id: str
    background: bool = False

    @classmethod
    def parse(cls, envelope: Any) -> "InvocationRequest":
        if not isinstance(envelope, dict):
            raise ValueError(f"{ENVELOPE_KEY} must be an object")
        if set(envelope) - {"version", "operation", "invocation_id", "background"}:
            raise ValueError("Unknown runtime invocation field")
        version = envelope.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version != VERSION:
            raise ValueError(f"Runtime invocation version must be {VERSION}")
        operation = envelope.get("operation")
        if operation not in ("start", "get"):
            raise ValueError("operation must be 'start' or 'get'")
        invocation_id = envelope.get("invocation_id")
        if not isinstance(invocation_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", invocation_id):
            raise ValueError(
                "invocation_id must be 1–128 letters, digits, underscores or hyphens, starting alphanumeric"
            )
        background = envelope.get("background", False)
        if not isinstance(background, bool):
            raise ValueError("background must be a boolean")
        if operation == "get" and "background" in envelope:
            raise ValueError("background is only valid for start")
        return cls(operation=operation, invocation_id=invocation_id, background=background)
