"""Filesystem records; one serving process owns each session namespace."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .protocol import InvocationState, state


class InvocationStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else Path(tempfile.gettempdir()) / ".agentcore_runtime"
        self.root.mkdir(parents=True, exist_ok=True)
        # Fail at startup if an existing root is not writable.
        with tempfile.TemporaryFile(dir=self.root):
            pass

    def _directory(self, session_id: str, invocation_id: str) -> Path:
        # Session headers are opaque strings, never filesystem path components.
        session_key = hashlib.sha256(session_id.encode()).hexdigest()
        return self.root / session_key / invocation_id

    def read(self, session_id: str, invocation_id: str) -> InvocationState | None:
        directory = self._directory(session_id, invocation_id)
        for name in ("result.json", "started.json"):
            try:
                return json.loads((directory / name).read_text())
            except FileNotFoundError:
                continue
        return None

    def start(self, session_id: str, invocation_id: str) -> None:
        directory = self._directory(session_id, invocation_id)
        directory.mkdir(parents=True, exist_ok=True)
        self._write(directory / "started.json", state(invocation_id, "in_progress"))

    def finish(self, session_id: str, result: InvocationState) -> None:
        self._write(self._directory(session_id, result["invocation_id"]) / "result.json", result)

    @staticmethod
    def _write(path: Path, record: InvocationState) -> None:
        data = json.dumps(record, allow_nan=False)
        # Stage beside the destination so replace is atomic on the selected filesystem.
        file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
        try:
            with file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(file.name, path)
        finally:
            Path(file.name).unlink(missing_ok=True)
