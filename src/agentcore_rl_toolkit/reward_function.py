"""
Base reward function interface for pure reward computation in RL training.

Reward functions only compute rewards - the app framework handles all validation and formatting.
"""

from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """
    Base class for reward functions focused purely on reward computation.

    Users implement compute_reward() and can return:
    - float: Single reward value
    - list of floats: Per-turn rewards or single-element list for outcome rewards

    The app framework handles all validation, normalization, and formatting automatically.
    Right now, this class mostly defines a contract, but we might add some more shared utilities
    in the future.
    """

    @abstractmethod
    def __call__(self, **kwargs):
        """
        Compute reward(s) for the rollout.

        Args:
            **kwargs: Flexible arguments for reward computation, such as:
                     - response_text: Agent's response text
                     - ground_truth: Correct answer
                     - user_input: Original user input
                     - Any other context needed for reward computation

        Returns:
            float: Single reward value, or
            list[float]: Per-turn rewards or single-element list for outcome rewards
        """
        pass
