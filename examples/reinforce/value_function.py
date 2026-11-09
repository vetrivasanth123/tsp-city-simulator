
from __future__ import annotations

import torch
from torch import nn


class LinearValueFunction(nn.Module):
    """Linear state-value function."""

    def __init__(self, num_features: int):
        super().__init__()

        if num_features < 1:
            raise ValueError("num_features must be positive.")

        self.network = nn.Linear(num_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class MLPValueFunction(nn.Module):
    """MLP state-value function."""

    def __init__(
        self,
        num_features: int,
        hidden_sizes: tuple[int, int] = (128, 64),
    ):
        super().__init__()

        if num_features < 1:
            raise ValueError("num_features must be positive.")

        if len(hidden_sizes) != 2 or any(
            size < 1 for size in hidden_sizes
        ):
            raise ValueError(
                "hidden_sizes must contain two positive sizes."
            )

        hidden1, hidden2 = hidden_sizes

        self.network = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
