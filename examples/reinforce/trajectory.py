
from __future__ import annotations

from copy import deepcopy

import numpy as np


class EpisodeTrajectory:
    """Store one episode of experience for REINFORCE."""

    def __init__(self, close_action: int):
        if close_action < 0:
            raise ValueError("close_action must be non-negative.")

        self.close_action = close_action
        self.clear()

    def clear(self) -> None:
        """Reset the stored experience for a new episode."""

        self.observations: list[dict] = []
        self.available_actions: list[list[int]] = []
        self.intended_actions: list[int] = []
        self.rewards: list[float] = []
        self.infos: list[dict] = []

    def add_step(
        self,
        observation: dict,
        available_actions: list[int],
        intended_action: int,
        reward: float,
        info: dict,
    ) -> None:
        """Store one decision and its observed outcome."""

        legal_actions = [int(action) for action in available_actions]
        action = int(intended_action)

        if legal_actions:
            if action not in legal_actions:
                raise ValueError(
                    "intended_action must be one of available_actions."
                )
        elif action != self.close_action:
            raise ValueError(
                "When no city actions remain, intended_action must be CLOSE."
            )

        # Copy observations so later changes cannot alter stored experience.
        stored_observation = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                stored_observation[key] = value.copy()
            else:
                stored_observation[key] = deepcopy(value)

        self.observations.append(stored_observation)
        self.available_actions.append(legal_actions)
        self.intended_actions.append(action)
        self.rewards.append(float(reward))
        self.infos.append(deepcopy(info))

    def __len__(self) -> int:
        """Return the number of stored decisions."""
        return len(self.rewards)

    def is_empty(self) -> bool:
        """Return whether the episode contains no decisions."""
        return len(self) == 0
