"""The training reward: F-beta over the answer's lines against the gold lines, less a small penalty
per invalid region. An answer with no valid region scores -0.3, and -1 if the rollout also crashed."""

import json
import os

from scoring.answer import Answer, extract_answer

BETA = float(os.environ.get("SWEGREP_BETA", "2"))  # recall weight; the final recipe uses 2
NO_ANSWER_REWARD = -0.3  # every real answer must score above this, or the policy learns to stay silent
INVALID_REGION_PENALTY = 0.05  # per distinct invalid region, at most 0.3 in total
MAX_INVALID_PENALTY = 0.3


def lines(spans) -> set[tuple[str, int]]:
    """The (path, line) pairs covered by (path, start, end) spans, ends inclusive."""
    return {(path, n) for path, start, end in spans for n in range(start, end + 1)}


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def compute_reward(messages: list[dict], gold_lines: dict | str, repo_root: str, crashed=False) -> tuple[dict, Answer]:
    """Score a rollout's messages against gold_lines, {path: [[start, end], ...]} or its JSON.
    Returns the metrics, "reward" among them, and the parsed answer."""
    if isinstance(gold_lines, str):
        gold_lines = json.loads(gold_lines)
    answer = extract_answer(messages, repo_root)
    cited = lines((r.path, r.start_line, r.end_line) for r in answer.regions)
    gold = lines((path, start, end) for path, spans in gold_lines.items() for start, end in spans)
    hits = len(cited & gold)
    precision = hits / len(cited) if cited else 0.0
    recall = hits / len(gold)
    if not answer.regions:
        reward = -1.0 if crashed else NO_ANSWER_REWARD
    else:
        penalty = min(MAX_INVALID_PENALTY, INVALID_REGION_PENALTY * len({r.strip() for r in answer.invalid_regions}))
        reward = fbeta(precision, recall, BETA) - penalty
    metrics = {
        "reward": reward,
        "submitted": float(answer.submitted),
        "precision": precision,
        "recall": recall,
        "f1": fbeta(precision, recall, 1),
        "cited_lines": float(len(cited)),
        "gold_lines": float(len(gold)),
        "invalid_regions": float(len(answer.invalid_regions)),
        "search_turns": float(answer.search_turns),
        "crashed": float(crashed),
    }
    return metrics, answer
