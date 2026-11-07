"""WideSearch tasks as invocation payloads: the 100 English tasks of the Hugging Face dataset at a pinned revision,
with their gold tables.

Splits: `all` (100 tasks); `pilot50` and `smoke2`, 50 and 2 tasks drawn with seed 0.
"""

import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO, REVISION = "ByteDance-Seed/WideSearch", "6531a7e"


def download(name: str) -> str:
    return Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)).read_text()


def load_split(name: str) -> list[dict]:
    """[{"task_id", "instance_id", "question", "evaluation", "gold_csv"}] of one split."""
    rows = [json.loads(line) for line in download("widesearch.jsonl").splitlines() if line.strip()]
    tasks = []
    for r in rows:
        if r["language"] != "en":
            continue
        evaluation = json.loads(r["evaluation"]) if isinstance(r["evaluation"], str) else r["evaluation"]
        tasks.append(
            {
                "task_id": r["instance_id"],
                "instance_id": r["instance_id"],
                "question": r["query"],
                "evaluation": evaluation,
                "gold_csv": download(f"widesearch_gold/{r['instance_id']}.csv"),
            }
        )
    if name == "all":
        return tasks
    subsets = {"pilot50": 50, "smoke2": 2}
    if name in subsets:
        keep = set(random.Random(0).sample([t["task_id"] for t in tasks], subsets[name]))
        return [t for t in tasks if t["task_id"] in keep]
    raise ValueError(f"split must be all, pilot50 or smoke2, got {name!r}")
