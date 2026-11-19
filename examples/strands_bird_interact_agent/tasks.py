"""BIRD-Interact Lite tasks as invocation payloads: the instance ids of the Hugging Face dataset
birdsql/bird-interact-lite at a pinned revision (the environment service holds the rest, with the gold).

Splits: `all` (300 tasks); `pilot50` and `smoke2`, 50 and 2 tasks drawn with seed 0.
"""

import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO, REVISION = "birdsql/bird-interact-lite", "f7881a9c2b9630cc4fc13b0c39279740b0a2fd87"


def load_split(name: str) -> list[dict]:
    """[{"task_id", "instance_id"}] of one split."""
    path = hf_hub_download(REPO, "bird_interact_data.jsonl", repo_type="dataset", revision=REVISION)
    ids = [json.loads(line)["instance_id"] for line in Path(path).read_text().splitlines() if line.strip()]
    tasks = [{"task_id": i, "instance_id": i} for i in ids]
    if name == "all":
        return tasks
    subsets = {"pilot50": 50, "smoke2": 2}
    if name in subsets:
        keep = set(random.Random(0).sample(ids, subsets[name]))
        return [t for t in tasks if t["task_id"] in keep]
    raise ValueError(f"split must be all, pilot50 or smoke2, got {name!r}")
