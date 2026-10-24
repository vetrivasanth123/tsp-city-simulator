
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class SoftmaxPolicy(nn.Module):
    """Dynamic linear softmax policy for the TSP environment."""

    def __init__(self, num_cities: int):
        super().__init__()

        if num_cities < 2:
            raise ValueError("num_cities must be at least 2.")

        self.num_cities = num_cities
        self.close_action = num_cities
        self.num_actions = num_cities + 1
        self.num_features = 2 * num_cities

        self.linear = nn.Linear(
            self.num_features,
            self.num_actions,
        )

    def encode_observation(self, observation: dict) -> torch.Tensor:
        """Convert the environment observation into policy features."""

        current_city = int(observation["current_city"])

        visited_mask = np.asarray(
            observation["visited_mask"],
            dtype=np.float32,
        ).reshape(-1)

        if not 0 <= current_city < self.num_cities:
            raise ValueError("current_city is outside the valid range.")

        if visited_mask.shape != (self.num_cities,):
            raise ValueError("visited_mask has an incorrect length.")

        current_city_one_hot = np.zeros(
            self.num_cities,
            dtype=np.float32,
        )
        current_city_one_hot[current_city] = 1.0

        features = np.concatenate(
            [current_city_one_hot, visited_mask]
        )

        return torch.as_tensor(features, dtype=torch.float32)

    def forward(self, observation: dict) -> torch.Tensor:
        """Calculate unnormalized action preferences (logits)."""

        features = self.encode_observation(observation)
        logits = self.linear(features)

        return logits

    def get_distribution(
        self,
        observation: dict,
        available_actions: list[int],
    ) -> Categorical:
        """Build a categorical policy distribution over legal actions."""

        legal_actions = list(available_actions)

        if not legal_actions:
            legal_actions = [self.close_action]

        if len(set(legal_actions)) != len(legal_actions):
            raise ValueError("available_actions contains duplicates.")

        if any(
            action < 0 or action >= self.num_actions
            for action in legal_actions
        ):
            raise ValueError("available_actions contains an invalid action.")

        if self.close_action in legal_actions:
            if len(legal_actions) != 1:
                raise ValueError(
                    "CLOSE must be the only legal action when available."
                )
        elif any(
            action >= self.num_cities
            for action in legal_actions
        ):
            raise ValueError("An invalid city action was provided.")

        logits = self.forward(observation)

        action_mask = torch.zeros(
            self.num_actions,
            dtype=torch.bool,
            device=logits.device,
        )
        action_mask[legal_actions] = True

        masked_logits = logits.masked_fill(
            ~action_mask,
            float("-inf"),
        )

        return Categorical(logits=masked_logits)

    def select_action(
        self,
        observation: dict,
        available_actions: list[int],
    ) -> int:
        """Sample and return the intended action."""

        distribution = self.get_distribution(
            observation,
            available_actions,
        )

        return int(distribution.sample().item())

    def log_probability(
        self,
        observation: dict,
        available_actions: list[int],
        intended_action: int,
    ) -> torch.Tensor:
        """Calculate log pi(action | state, parameters)."""

        distribution = self.get_distribution(
            observation,
            available_actions,
        )

        action = torch.tensor(
            intended_action,
            dtype=torch.long,
        )

        return distribution.log_prob(action)
