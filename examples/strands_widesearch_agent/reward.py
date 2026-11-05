"""WideSearch's scoring, the official evaluation (widesearch_eval/, src/evaluation/ of ByteDance-Seed/WideSearch) on
one response: the response's Markdown table is aligned with the gold table by primary key and scored per row and
per item. The reward is item F1, the official primary metric; the result also has row F1 and the success rate (the
whole table correct)."""

import copy
import dataclasses
from io import StringIO

import pandas as pd
from widesearch_eval.data_loader import WideSearchQuery, WideSearchResponse
from widesearch_eval.evaluation import evaluate_single_query
from widesearch_eval.utils import norm_column

from agentcore_rl_toolkit import RewardFunction


def evaluate(instance_id: str, query: str, evaluation: dict, gold_csv: str, response: str) -> dict:
    """The official EvaluationResult of one response, as a dict. The gold columns are normalized and restricted to
    evaluation["required"], as the official data loader does."""
    answer = pd.read_csv(StringIO(gold_csv))
    answer.columns = [norm_column(c.strip()) for c in answer.columns]
    q = WideSearchQuery(
        instance_id=instance_id,
        query=query,
        evaluation=copy.deepcopy(evaluation),
        answer=answer[evaluation["required"]],
        language="en",
    )
    r = WideSearchResponse(instance_id=instance_id, response=response or "")
    return dataclasses.asdict(evaluate_single_query(q, r, result_save_path=None))


class WideSearchReward(RewardFunction):
    def __call__(self, f1_by_item: float = 0.0, **kwargs) -> float:
        """Item F1 of the response's table."""
        return float(f1_by_item)
