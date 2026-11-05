"""ExCyTIn's scoring, upstream's evaluator (secgym/evaluator.py, LLMEvaluator as in experiments/run_exp.py; prompts in
prompts/evaluator_prompts.json). The judge is gpt-oss-120b on Bedrock (upstream: GPT-4o) with its default sampling.

- Answer check: the fuzzy-match prompt, then its reflection; the reflection's Is_Answer_Correct decides. Each call is
  re-asked (up to 10 times) until its decision parses, as upstream.
- Step check, only when the answer is wrong: the judge marks which steps of the gold solution the answer contains
  (JSON), then reflects; the credit counts the correct steps from the second-to-last backwards, weighted 0.4, 0.16,
  ... (capped at 1).

success is the answer check (0 or 1); the reward adds the step credit (upstream's reported metric). An episode
without an answer scores 0 without a judge call.
"""

import json
import re
from pathlib import Path

import boto3
from botocore.config import Config
from models import InvocationRequest

from agentcore_rl_toolkit import RewardFunction

JUDGE = "openai.gpt-oss-120b-1:0"
P = json.loads((Path(__file__).parent / "prompts/evaluator_prompts.json").read_text())
DECISION = re.compile(r"Is_Answer_Correct: (True|False)")
DISCOUNT = 0.4
_bedrock = boto3.client("bedrock-runtime", config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))


def chat(system: str, user: str) -> str:
    r = _bedrock.converse(
        modelId=JUDGE,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": 4096},
    )
    return "".join(b["text"] for b in r["output"]["message"]["content"] if "text" in b).strip()


def decided(system: str, user: str) -> tuple[bool, str]:
    for _ in range(10):
        out = chat(system, user)
        if m := DECISION.search(out):
            return m.group(1) == "True", out
    return False, out


def json_steps(system: str, user: str) -> dict | None:
    for _ in range(10):
        out = chat(system, user)
        text = out.split("```json")[1].split("```")[0] if "```json" in out else out
        try:
            steps = json.loads(text[text.index("{") : text.rindex("}") + 1])
            if all("is_step_correct" in v for v in steps.values()):
                return steps
        except (ValueError, AttributeError, TypeError):
            continue
    return None


def step_credit(steps: dict) -> float:
    """Upstream's discounted credit: the correct steps from the second-to-last backwards (the last is the answer)."""
    flags = [str(v["is_step_correct"]) == "True" for v in steps.values()][::-1]
    total, weight = 0.0, DISCOUNT
    for f in flags[1:]:
        if f:
            total += weight
        if total >= 1:
            return 1.0
        weight *= DISCOUNT
    return round(total, 4)


def evaluate(request: InvocationRequest, answer: str) -> dict:
    """{"success", "credit", "judge"}: the answer check, the reward with step credit, and the judge's replies."""
    if not answer.strip():
        return {"success": 0, "credit": 0.0, "judge": {}}
    question = f"{request.context} {request.question}".strip()
    ans_in = P["EVAL_ANSWER_TEMPLATE"].format(question=question, golden_answer=request.gold, submitted_answer=answer)
    _, first = decided(P["FUZZY_ANSWER_CHECK_PROMPT"], ans_in)
    ok, reflection = decided(P["FUZZY_ANSWER_CHECK_REFLECTION_PROMPT"], ans_in + "\n" + first)
    judge = {"check_answer": first, "check_answer_reflection": reflection}
    if ok or not request.solution:
        return {"success": int(ok), "credit": float(ok), "judge": judge}
    gold = "".join(f"Step {i}: {s}\n" for i, s in enumerate(request.solution))
    sol_in = P["EVAL_SOLUTION_TEMPLATE"].format(question=question, golden_solution=gold, submitted_answer=answer)
    steps = json_steps(P["STEP_CHECK_PROMPT"], sol_in)
    if steps is not None:
        steps = json_steps(P["STEP_CHECK_REFLECTION_PROMPT"], sol_in + "\n" + json.dumps(steps)) or steps
    judge["check_steps"] = steps
    return {"success": 0, "credit": step_credit(steps) if steps else 0.0, "judge": judge}


class ExCyTInReward(RewardFunction):
    def __call__(self, credit: float = 0.0, **kwargs) -> float:
        """The answer check (1.0 if correct), else the partial credit for the solution's steps."""
        return float(credit)
