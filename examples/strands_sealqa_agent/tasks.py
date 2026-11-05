"""Seal-Hard tasks as invocation payloads: seal-hard.parquet of the Hugging Face dataset vtllms/sealqa at a pinned
revision. task_id is the row index.

Splits: `all` (254 tasks); `pilot50` and `smoke2`, 50 and 2 tasks drawn with seed 0.
"""

import random

from datasets import load_dataset

REPO, REVISION = "vtllms/sealqa", "0c298f5"


def load_split(name: str) -> list[dict]:
    """[{"task_id", "question", "gold", "freshness"}] of one split."""
    rows = load_dataset(REPO, data_files="seal-hard.parquet", revision=REVISION, split="train")
    tasks = [
        {"task_id": i, "question": r["question"], "gold": r["answer"], "freshness": r["freshness"]}
        for i, r in enumerate(rows)
    ]
    if name == "all":
        return tasks
    subsets = {"pilot50": 50, "smoke2": 2}
    if name in subsets:
        keep = set(random.Random(0).sample(range(len(tasks)), subsets[name]))
        return [t for t in tasks if t["task_id"] in keep]
    raise ValueError(f"split must be all, pilot50 or smoke2, got {name!r}")
