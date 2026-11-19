"""LiveBrowseComp's scoring, the official judge (arXiv 2605.28721, Appendix B): the BrowseComp grader template
(prompts/judge_template.txt, from openai/simple-evals@652c89d, MIT) with gpt-oss-120b on Bedrock, as in the paper.
The verdict is the template's `correct: yes|no`, as in simple-evals; a reply without one is asked again, up to 5
times, then counts as incorrect. An episode without an answer is incorrect without a judge call."""

import re
from pathlib import Path

import boto3
from botocore.config import Config

from agentcore_rl_toolkit import RewardFunction

JUDGE = "openai.gpt-oss-120b-1:0"
TEMPLATE = (Path(__file__).parent / "prompts/judge_template.txt").read_text()
_bedrock = boto3.client("bedrock-runtime", config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))


def judge(question: str, response: str, gold: str) -> tuple[bool, str]:
    """-> (correct, the judge's last reply)."""
    if not (response or "").strip():
        return False, ""
    prompt = TEMPLATE.format(question=question, response=response, correct_answer=gold)
    reply = ""
    for _ in range(5):
        r = _bedrock.converse(
            modelId=JUDGE,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 4096},
        )
        reply = "".join(b["text"] for b in r["output"]["message"]["content"] if "text" in b).strip()
        m = re.search(r"correct: (yes|no)", reply)
        if m:
            return m.group(1) == "yes", reply
    return False, reply


class LiveBrowseCompReward(RewardFunction):
    def __call__(self, correct: bool = False, **kwargs) -> float:
        """1.0 iff the judge finds the answer correct."""
        return float(correct)
