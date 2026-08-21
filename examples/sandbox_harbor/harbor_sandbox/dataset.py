"""ensure_dataset — pull a Harbor benchmark's tasks to a local dir on demand.

A Harbor benchmark ("org/name") is a dataset in the hosted Harbor registry
(hub.harborframework.com, public read). ``ensure_dataset`` downloads it into a
task_root where each task is a subdirectory (instruction.md, environment/,
tests/, task.toml) — the layout both the image builder and bench.py read.

Runs launcher-side, never in the harness container (the container gets a task's
instruction + tests via the rollout payload, and its image from ECR). Needs the
``harbor`` package, imported lazily so importing this module stays cheap.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _local_task_count(task_root: Path) -> int:
    if not task_root.is_dir():
        return 0
    return sum(1 for d in task_root.iterdir() if (d / "instruction.md").is_file())


def ensure_dataset(benchmark: str, task_root: str | Path, *, ref: str = "latest", overwrite: bool = False) -> Path:
    """Ensure benchmark's tasks live under task_root (one subdir per task),
    pulling them from the Harbor registry if absent. Returns task_root.

    Idempotent: when task_root already holds tasks the download is skipped
    entirely (pass overwrite=True to force a re-pull). ``ref`` selects a version
    ("latest", a tag, or "sha256:...").
    """
    task_root = Path(task_root)
    have = _local_task_count(task_root)
    if have and not overwrite:
        logger.info(f"{benchmark}: {have} tasks already in {task_root}")
        return task_root

    try:
        from harbor.cli.utils import run_async
        from harbor.registry.client.package import PackageDatasetClient
    except ImportError as e:
        raise RuntimeError(
            "ensure_dataset needs the 'harbor' package importable on the launcher "
            "host — install it into this venv: `uv sync --extra harbor` (or "
            "`uv pip install harbor`)"
        ) from e

    task_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"pulling {benchmark}@{ref} -> {task_root}")
    # download the client directly (not the CLI) so tasks land straight under
    # task_root without the CLI's extra <dataset-name>/ wrapper directory.
    items = run_async(
        PackageDatasetClient().download_dataset(
            f"{benchmark}@{ref}", overwrite=overwrite, output_dir=task_root, export=True
        )
    )
    logger.info(f"{benchmark}: {len(items)} tasks ready in {task_root}")
    return task_root
