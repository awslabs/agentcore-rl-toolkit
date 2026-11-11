"""ExCyTIn tasks as invocation payloads: the questions of upstream's repository at a pinned commit
(secgym/questions/opus5/, the latest_cleaned set). task_id is <incident>/<split>/<index>, e.g. incident_39/test/12.

Training data must not touch an evaluated database, so 3 of the 8 incidents, drawn with seed 0, are held out:
`eval` is the test questions of those 3 incidents; `pilot50` and `smoke2` are 50 and 2 of them, drawn with seed 0;
`train` is the train questions of the other 5 incidents.
"""

import json
import random
import urllib.request

REPO, COMMIT = "microsoft/SecRL", "a1b234f7a846133f08b7b5e1614c5c63b96f3b5c"
INCIDENTS = [5, 34, 38, 39, 55, 134, 166, 322]


def questions(split: str, incident: int) -> list[dict]:
    url = (
        f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/secgym/questions/opus5/{split}/"
        f"incident_{incident}_qa_opus5_cleaned.json"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())


def load_split(name: str) -> list[dict]:
    """[{"task_id", "incident", "context", "question", "gold", "solution"}] of one split."""
    held = set(random.Random(0).sample(INCIDENTS, 3))
    split = "train" if name == "train" else "test"
    tasks = [
        {
            "task_id": f"incident_{inc}/{split}/{i}",
            "incident": inc,
            "context": q.get("context", ""),
            "question": q["question"],
            "gold": str(q["answer"]),
            "solution": [str(s) for s in q.get("solution") or []],
        }
        for inc in sorted(INCIDENTS, key=lambda i: f"incident_{i}")  # upstream's file order
        if (inc in held) == (split == "test")
        for i, q in enumerate(questions(split, inc))
    ]
    if name in ("eval", "train"):
        return tasks
    subsets = {"pilot50": 50, "smoke2": 2}
    if name in subsets:
        keep = set(random.Random(0).sample([t["task_id"] for t in tasks], subsets[name]))
        return [t for t in tasks if t["task_id"] in keep]
    raise ValueError(f"split must be eval, pilot50, smoke2 or train, got {name!r}")
