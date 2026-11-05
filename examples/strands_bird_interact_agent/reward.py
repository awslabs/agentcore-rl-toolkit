"""BIRD-Interact's reward (upstream's phase rewards, action_handler.py): 0.7 for passing phase 1 (the ambiguous
question), plus 0.3 for passing phase 2 (the follow-up). The environment service runs the tests."""

from agentcore_rl_toolkit import RewardFunction


class BirdInteractReward(RewardFunction):
    def __call__(self, phase1: bool = False, phase2: bool = False, **kwargs) -> float:
        return 0.7 * phase1 + 0.3 * phase2
