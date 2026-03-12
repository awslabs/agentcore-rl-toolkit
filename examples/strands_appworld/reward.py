from agentcore_rl_toolkit import RewardFunction


class AppWorldReward(RewardFunction):
    """Reward function for AppWorld tasks.

    Computes len(passes) / num_tests for continuous 0.0-1.0 partial credit.
    """

    def __call__(self, evaluation: dict | None = None, **kwargs) -> float:
        if evaluation is None:
            return 0.0

        num_tests = evaluation.get("num_tests", 0)
        if num_tests == 0:
            return 0.0

        passes = evaluation.get("passes", [])
        return len(passes) / num_tests
