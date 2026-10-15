"""Exact-answer reward over labels supplied by the caller."""

from agentcore_rl_toolkit import RewardFunction

from biomedical_db_agent.contracts import DatabaseResponse


class ExactAnswerReward(RewardFunction):
    def __call__(self, response: DatabaseResponse, expected_answers: list[str]) -> float:
        return float(response.status == "completed" and response.answer in expected_answers)
