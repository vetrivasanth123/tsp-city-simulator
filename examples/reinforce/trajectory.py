
from __future__ import annotations

from copy import deepcopy


class EpisodeTrajectory:
    """Store the state, intended action, and reward for one episode."""

    def __init__(self):
        self.observations: list[dict] = []
        self.available_actions: list[list[int]] = []
        self.intended_actions: list[int] = []
        self.rewards: list[float] = []

    def add_step(
        self,
        observation: dict,
        available_actions: list[int],
        intended_action: int,
        reward: float,
    ) -> None:
        """Record one action and its reward.

        Pass the observation and available actions from BEFORE env.step().
        Pass the reward returned by env.step().
        """

        self.observations.append(deepcopy(observation))
        self.available_actions.append(
            [int(action) for action in available_actions]
        )
        self.intended_actions.append(int(intended_action))
        self.rewards.append(float(reward))

    def clear(self) -> None:
        """Remove all recorded steps before starting another episode."""

        self.observations.clear()
        self.available_actions.clear()
        self.intended_actions.clear()
        self.rewards.clear()

    def __len__(self) -> int:
        """Return the number of recorded actions."""

        return len(self.rewards)
