"""HERB's scoring (its evaluate.py, with Claude Haiku 5.5 as the judge in place of GPT-4o; upstream.py):

- content: the judge's 0-100 rating of the answer against the reference, divided by 100;
- person, company, url, pr: the judge extracts the items the answer names; set F1 against the reference;
- unanswerable: 1 if the answer does not answer the question (judge B: HERB's prompt plus one rule, upstream.py);
  an episode without an answer scores 0.

An answerable episode without an answer is scored as "I don't know.", as HERB does.
"""

import upstream

from agentcore_rl_toolkit import RewardFunction

EXTRACT = {
    "person": "answer_to_employee_ids",
    "company": "answer_to_companies",
    "url": "answer_to_urls",
    "pr": "answer_to_prs",
}


class HERBReward(RewardFunction):
    def __call__(self, response_text: str = "", question: str = "", type: str = "", gold="", **kwargs) -> float:  # noqa: A002
        ev = upstream.evaluation()
        if type == "unanswerable":
            if not response_text.strip():
                return 0.0
            return float(ev.unanswerability_eval(question, response_text) is False)
        text = response_text.split("</think>")[-1].strip() or "I don't know."
        if type == "content":
            return int(ev.answer_likert_score(question, gold, text)) / 100
        return float(ev.f1_score_sets(getattr(ev, EXTRACT[type])(question, text), list(gold)))
