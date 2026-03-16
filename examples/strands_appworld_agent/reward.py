from agentcore_rl_toolkit import RewardFunction


class AppWorldReward(RewardFunction):
    def __call__(self, test_tracker=None, **kwargs) -> float:
        """1.0 if all tests pass, else pass_count / num_tests for partial credit."""
        if test_tracker is None:
            return 0.0
        if test_tracker.success:
            return 1.0
        if test_tracker.num_tests > 0:
            return test_tracker.pass_count / test_tracker.num_tests
        return 0.0
