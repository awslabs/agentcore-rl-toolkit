"""SealQA's scoring, the official grader (the SealQA Colab; the template is also in the paper's Appendix C):
the judge grades the answer A (correct), B (incorrect) or C (not attempted); the grade is the first A, B or C in its
reply, and a reply without one counts as C. The judge is gpt-oss-120b on Bedrock at temperature 0 (officially
GPT-4o-mini). An episode without an answer is C without a judge call."""

import re
from pathlib import Path

import boto3
from botocore.config import Config

from agentcore_rl_toolkit import RewardFunction

JUDGE = "openai.gpt-oss-120b-1:0"
TEMPLATE = (Path(__file__).parent / "prompts/grader_template.txt").read_text()
_bedrock = boto3.client("bedrock-runtime", config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))


def grade(question: str, gold: str, predicted: str) -> tuple[str, str]:
    """-> (A | B | C, the judge's reply)."""
    if not (predicted or "").strip():
        return "C", ""
    r = _bedrock.converse(
        modelId=JUDGE,
        messages=[
            {
                "role": "user",
                "content": [{"text": TEMPLATE.format(question=question, target=gold, predicted_answer=predicted)}],
            }
        ],
        inferenceConfig={"maxTokens": 4096, "temperature": 0},
    )
    reply = "".join(b["text"] for b in r["output"]["message"]["content"] if "text" in b).strip()
    m = re.search(r"(A|B|C)", reply)
    return (m.group(0) if m else "C"), reply


class SealQAReward(RewardFunction):
    def __call__(self, grade: str = "C", **kwargs) -> float:
        """1.0 iff the answer is graded A (correct)."""
        return float(grade == "A")
