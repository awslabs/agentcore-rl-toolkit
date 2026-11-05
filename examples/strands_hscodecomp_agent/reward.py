"""HSCodeComp's scoring, the official rule (code/eval.py, eval/test_llm.py of AIDC-AI/Marco-Search-Agent@2be90e40,
Apache-2.0): take the first \\boxed{...} whose content has at least 10 characters, keep its digits, and compare
prefixes with the zero-padded gold code."""

import re

from agentcore_rl_toolkit import RewardFunction


def extract(text: str) -> str:
    for m in re.findall(r"\\boxed\{([^}]+)\}", text or ""):
        if len(m) >= 10:
            return re.sub(r"\D", "", m)
    return ""


def prefix_matches(response_text: str, gold: str) -> dict:
    """{2, 4, 6, 8, 10}-digit matches with the gold code."""
    pred, gold = extract(response_text), str(gold).zfill(10)
    return {k: len(pred) >= k and pred[:k] == gold[:k] for k in (2, 4, 6, 8, 10)}


class HSCodeReward(RewardFunction):
    def __call__(self, response_text: str = "", gold: str = "", **kwargs) -> float:
        """1.0 iff the 10-digit code matches (the reported accuracy)."""
        return float(prefix_matches(response_text, gold)[10])
