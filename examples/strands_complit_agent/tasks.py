"""CompLit tasks as invocation payloads, from the Hugging Face dataset at a pinned revision.

Splits: the settings `standard` (350), `implicit` (220) and `unmet`, plus `pilot50` and `smoke2`, 50 and 2 Implicit
tasks drawn with seed 0.
"""

import random

from datasets import load_dataset

REPO, REVISION = "dayoon/CompLit", "7561ad63854fd21575fe785d9c078101567a1a05"
SETTINGS = ("standard", "implicit", "unmet")


def load_split(name: str) -> list[dict]:
    """[{"task_id", "question", "gold", "answerable"}] of one split."""
    setting = "implicit" if name in ("pilot50", "smoke2") else name
    if setting not in SETTINGS:
        raise ValueError(f"split must be one of {SETTINGS + ('pilot50', 'smoke2')}, got {name!r}")
    tasks = [
        {
            "task_id": r["id"],
            "question": r["question"],
            "gold": (r["paper"] or {}).get("arxiv_id") or "",
            "answerable": bool(r["answerable"]),
        }
        for r in load_dataset(REPO, setting, split="test", revision=REVISION)
    ]
    if name in ("pilot50", "smoke2"):
        keep = set(random.Random(0).sample([t["task_id"] for t in tasks], 50 if name == "pilot50" else 2))
        tasks = [t for t in tasks if t["task_id"] in keep]
    return tasks
