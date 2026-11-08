"""HERB questions as invocation payloads, from the downloaded data (HERB_DIR; scripts/fetch_herb.py).

Splits: `answerable` (815 questions), `unanswerable` (699), `all`; `pilot50`, 25 answerable and 25 unanswerable
questions (the first 25 of 100 and of 50 drawn with seed 0); `smoke2`, one of each.
"""

import json
import random

from upstream import HERB_DIR


def _questions(kind: str) -> dict[str, dict]:
    """{task_id: payload} of the answerable or unanswerable questions, task_id = <kind>/<product>#<i>."""
    out = {}
    for p in sorted((HERB_DIR / "data" / "products").glob("*.json")):
        for i, q in enumerate(json.loads(p.read_text())[f"{kind}_questions"]):
            q = q if isinstance(q, dict) else {"question": q}
            gold = q.get("ground_truth", "") if kind == "answerable" else ""
            out[f"{kind}/{p.stem}#{i}"] = {
                "question": q["question"],
                "type": q["type"] if kind == "answerable" else "unanswerable",
                "gold": gold if isinstance(gold, str) else [str(x) for x in gold],
            }
    return out


def load_split(name: str) -> list[dict]:
    """[{"task_id", "question", "type", "gold"}] of one split."""
    ans, unans = _questions("answerable"), _questions("unanswerable")
    pick = lambda qs, n: sorted(random.Random(0).sample(sorted(qs), n))  # noqa: E731
    keys = {
        "answerable": sorted(ans),
        "unanswerable": sorted(unans),
        "all": sorted(ans) + sorted(unans),
        "pilot50": pick(ans, 100)[:25] + pick(unans, 50)[:25],
        "smoke2": pick(ans, 100)[:1] + pick(unans, 50)[:1],
    }
    if name not in keys:
        raise ValueError(f"split must be one of {sorted(keys)}, got {name!r}")
    qs = {**ans, **unans}
    return [{"task_id": k, **qs[k]} for k in keys[name]]
