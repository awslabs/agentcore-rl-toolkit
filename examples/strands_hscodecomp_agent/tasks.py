"""HSCodeComp tasks as invocation payloads, from the Hugging Face dataset at a pinned revision.

Splits: `all` (632 tasks); `pilot100`, 100 tasks drawn with seed 0; `smoke2`, the first two of them.
"""

import random

from datasets import load_dataset

REPO, REVISION = "AIDC-AI/HSCodeComp", "ce9119795acef4ca537b2175e10a3feb7a0ecae9"


def load_split(name: str) -> list[dict]:
    """[{"task_id", "question", "gold"}] of one split."""
    rows = load_dataset(REPO, split="test", revision=REVISION)
    tasks = {
        r["task_id"]: {"task_id": r["task_id"], "question": r["question"], "gold": str(r["hs_code"]).zfill(10)}
        for r in rows
    }
    if name == "all":
        return [tasks[t] for t in sorted(tasks)]
    pilot = sorted(random.Random(0).sample(sorted(tasks), 100))
    if name in ("pilot100", "smoke2"):
        return [tasks[t] for t in (pilot if name == "pilot100" else pilot[:2])]
    raise ValueError(f"split must be all, pilot100 or smoke2, got {name!r}")
