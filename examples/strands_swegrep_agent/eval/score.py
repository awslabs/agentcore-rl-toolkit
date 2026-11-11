"""Score a benchmark run: the line precision, recall and F1 of each task's answer against the published
gold, averaged over all tasks (an unanswered task scores 0).

  python eval/score.py OUT     (reads OUT/swe-explore and OUT/contextbench-lite; prints the scores)

SWE-Explore's gold is its read_core_regions, and these scores equal its official evaluator's.
ContextBench's gold is its gold_context; this scores the final answer, not ContextBench's own metric,
which scores everything the agent looked at. data/build_benchmarks.py writes both as
data/benchmarks/<benchmark>.gold.json.
Writes OUT/<benchmark>/scores.jsonl, one row per task.
"""

import json
import statistics
import sys
from pathlib import Path

from scoring.reward import lines

BENCHMARKS = Path(__file__).resolve().parents[1] / "data/benchmarks"
NAMES = {"swe-explore": "SWE-Explore", "contextbench-lite": "ContextBench Lite"}


def score_run(run: Path, gold: dict[str, list]) -> dict[str, float]:
    assert (run / "COMPLETED").exists(), f"{run} has not finished"
    results = {row["instance_id"]: row for row in map(json.loads, (run / "results.jsonl").read_text().splitlines())}
    assert set(results) == set(gold), f"{run} does not cover the benchmark's {len(gold)} tasks"
    scores = []
    for instance_id, spans in gold.items():
        cited, expected = lines(results[instance_id]["regions"]), lines(spans)
        hits = len(cited & expected)
        scores.append({
            "instance_id": instance_id,
            "precision": hits / len(cited) if cited else 0.0,
            "recall": hits / len(expected),
            "f1": 2 * hits / (len(cited) + len(expected)),
            "cited_lines": len(cited),
            "gold_lines": len(expected),
        })  # fmt: skip
    (run / "scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores))
    return {key: statistics.mean(s[key] for s in scores) for key in ("precision", "recall", "f1")}


def main():
    out = Path(sys.argv[1])
    for benchmark, name in NAMES.items():
        gold = json.loads((BENCHMARKS / f"{benchmark}.gold.json").read_text())
        s = score_run(out / benchmark, gold)
        f1, precision, recall = 100 * s["f1"], 100 * s["precision"], 100 * s["recall"]
        print(f"{name} line F1 {f1:.2f} (P {precision:.1f} R {recall:.1f})")


if __name__ == "__main__":
    main()
