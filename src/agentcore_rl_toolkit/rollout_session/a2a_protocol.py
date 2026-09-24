"""A2A wire conventions shared by the rollout executor and client.

A rollout is one A2A ``Task``: setup parks it at ``input-required``; a second
``message/send`` on the same ``taskId`` runs the rollout to a terminal state. Task input
rides in one data ``Part`` (``{"task_input": {...}}``); the result rides back as one data
``Part`` in one ``Artifact``.
"""

from __future__ import annotations

from a2a.types import Message, Part, Role
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse

TASK_INPUT_KEY = "task_input"
# distinguishes the rollout result from any incidental artifacts a harness attaches
ROLLOUT_ARTIFACT_NAME = "rollout"


def _data_part(data: dict) -> Part:
    """A ``Part`` carrying ``data`` as its protobuf struct content."""
    value = Value()
    value.struct_value.update(data)
    return Part(data=value)


def _part_data(part: Part) -> dict | None:
    """The dict in a data ``Part``, or ``None`` if it is not one."""
    if part.WhichOneof("content") != "data":
        return None
    return json_format.MessageToDict(part.data)


def build_task_input_message(
    task_input: dict,
    *,
    context_id: str | None = None,
    task_id: str | None = None,
    message_id: str,
) -> Message:
    """A user ``Message`` carrying ``task_input`` as its one data ``Part``.

    ``task_id`` is unset on setup (server mints it) and set on the rollout message to
    resume. Proto strings cannot be ``None``, so unset ids are omitted from kwargs.
    """
    kwargs: dict = {
        "message_id": message_id,
        "role": Role.ROLE_USER,
        "parts": [_data_part({TASK_INPUT_KEY: task_input})],
    }
    if context_id is not None:
        kwargs["context_id"] = context_id
    if task_id is not None:
        kwargs["task_id"] = task_id
    return Message(**kwargs)


def extract_task_input(message: Message) -> dict:
    """The ``task_input`` dict from a message's first data part.

    Raises ``ValueError`` on a wire contract violation (no data part / no ``task_input``),
    not a rollout failure.
    """
    for part in message.parts:
        data = _part_data(part)
        if data is None:
            continue
        if TASK_INPUT_KEY not in data:
            raise ValueError(f"message data part has no {TASK_INPUT_KEY!r} key: {list(data)}")
        value = data[TASK_INPUT_KEY]
        if not isinstance(value, dict):
            raise ValueError(f"{TASK_INPUT_KEY!r} must be a dict, got {type(value).__name__}")
        return value
    raise ValueError("message carries no data part")


def dump_to_parts(dump: RolloutDumpResponse) -> list[Part]:
    """The rollout result as one data ``Part``. Drops ``response_type`` (data, not envelope)."""
    data = dump.model_dump(mode="json")
    data.pop("response_type", None)
    return [_data_part(data)]


def dump_from_artifacts(artifacts) -> RolloutDumpResponse | None:
    """The :class:`RolloutDumpResponse` in the rollout artifact, or ``None`` if absent.

    ``None`` (terminal task, no result) is treated by the caller as a failed rollout.
    """
    for artifact in artifacts or []:
        if artifact.name != ROLLOUT_ARTIFACT_NAME:
            continue
        for part in artifact.parts:
            data = _part_data(part)
            if data is not None:
                return RolloutDumpResponse.model_validate(data)
    return None


def status_message_text(message: Message | None) -> str | None:
    """The concatenated text of a status ``Message`` (e.g. a ``failed`` reason), or ``None``.

    An unset proto status message is an empty ``Message`` with no parts, yielding ``None`` too.
    """
    if message is None:
        return None
    texts = [part.text for part in message.parts if part.WhichOneof("content") == "text"]
    return "\n".join(texts) if texts else None
