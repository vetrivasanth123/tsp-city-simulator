from copy import deepcopy

import numpy as np


class Trajectory:
    """Store the transitions collected during one TSP episode."""

    def __init__(self):
        self.observations = []
        self.available_actions = []
        self.actions = []
        self.rewards = []
        self.infos = []
        self.terminated = []
        self.truncated = []

    def add_step(
        self,
        observation,
        available_actions,
        intended_action,
        reward,
        info=None,
        terminated=False,
        truncated=False,
    ):
        """Record one state-action-reward transition."""

        legal_actions = [int(action) for action in available_actions]
        intended_action = int(intended_action)

        if not legal_actions:
            raise ValueError(
                "available_actions must contain the legal action."
            )

        if intended_action not in legal_actions:
            raise ValueError(
                f"Intended action {intended_action} is not in "
                f"available_actions {legal_actions}."
            )

        if not np.isfinite(reward):
            raise ValueError("Reward must be finite.")

        if terminated and truncated:
            raise ValueError(
                "A step cannot be both terminated and truncated."
            )

        # Copy observations so later environment changes cannot
        # accidentally alter the recorded state.
        saved_observation = {
            key: (
                value.copy()
                if isinstance(value, np.ndarray)
                else deepcopy(value)
            )
            for key, value in observation.items()
        }

        self.observations.append(saved_observation)
        self.available_actions.append(legal_actions)
        self.actions.append(intended_action)
        self.rewards.append(float(reward))
        self.infos.append(deepcopy(info) if info is not None else {})
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))

    def __len__(self):
        """Return the number of recorded transitions."""
        return len(self.rewards)

    @property
    def total_reward(self):
        """Return the sum of rewards collected in the episode."""
        return float(sum(self.rewards))

    @property
    def is_complete(self):
        """Check whether the final recorded step ended the episode."""
        return bool(
            self.terminated
            and (self.terminated[-1] or self.truncated[-1])
        )

    def clear(self):
        """Remove all recorded transitions."""
        self.__init__()
