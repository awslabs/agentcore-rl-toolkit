"""CompLit tasks as invocation payloads, from the Hugging Face dataset at a pinned revision.

Splits: the settings `standard` (350), `implicit` (220), `unmet` (300) and `cumulative` (220: the Implicit queries
revealed over 3-5 researcher turns), and subsets drawn with seed 0: `pilot50` and `smoke2` (50 and 2 Implicit
tasks), `cumulative_pilot50` and `cumulative_smoke2`.
"""

import random

from datasets import load_dataset

REPO, REVISION = "dayoon/CompLit", "7561ad63854fd21575fe785d9c078101567a1a05"
SETTINGS = ("standard", "implicit", "unmet", "cumulative")
PAPER_TYPE = {"content": "content", "figural": "vision", "comparative": "multi_doc", "metadata": "metadata"}


def payload(r: dict) -> dict:
    task = {
        "task_id": r["id"],
        "question": r["question"],
        "gold": (r["paper"] or {}).get("arxiv_id") or "",
        "answerable": bool(r["answerable"]),
    }
    if r.get("conversation"):  # Cumulative, mapped as upstream's complit/data.py
        task["intention"] = r["conversation"]["intention"]
        task["turns"] = [
            {"utterance": t["utterance"], "revealed": t["revealed_constraints"]} for t in r["conversation"]["turns"]
        ]
        # the simulator's prompts name constraint types as in the paper's code
        task["constraints"] = [
            {"type": PAPER_TYPE[c["source"]], "field": c["field"], "value": c["value"]} for c in r["constraints"]
        ]
    return task


SUBSETS = {"pilot50": 50, "smoke2": 2}


def load_split(name: str) -> list[dict]:
    """[{"task_id", "question", "gold", "answerable", ...}] of one split."""
    setting, _, subset = name.rpartition("_") if name.rpartition("_")[2] in SUBSETS else (name, "", "")
    if name in SUBSETS:
        setting, subset = "implicit", name
    if setting not in SETTINGS:
        raise ValueError(f"split must be a setting {SETTINGS}, optionally with _pilot50 or _smoke2; got {name!r}")
    tasks = [payload(r) for r in load_dataset(REPO, setting, split="test", revision=REVISION)]
    if subset:
        keep = set(random.Random(0).sample([t["task_id"] for t in tasks], SUBSETS[subset]))
        tasks = [t for t in tasks if t["task_id"] in keep]
    return tasks
