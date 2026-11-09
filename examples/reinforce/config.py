
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReinforceConfig:
    """Configuration for REINFORCE training and evaluation."""

    learning_rate: float = 1e-3
    value_learning_rate: float = 1e-3
    gamma: float = 1.0
    num_episodes: int = 500
    eval_episodes: int = 30
    log_interval: int = 5
    seed: int = 42

    def __post_init__(self) -> None:

        # Policy learning rate (alpha_theta)
        if isinstance(self.learning_rate, bool):
            raise ValueError("learning_rate must be numeric.")

        learning_rate = float(self.learning_rate)

        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")

        # Value-function learning rate (alpha_w)
        if isinstance(self.value_learning_rate, bool):
            raise ValueError("value_learning_rate must be numeric.")

        value_learning_rate = float(self.value_learning_rate)

        if (
            not math.isfinite(value_learning_rate)
            or value_learning_rate <= 0
        ):
            raise ValueError(
                "value_learning_rate must be finite and positive."
            )

        # Discount factor
        if isinstance(self.gamma, bool):
            raise ValueError("gamma must be numeric.")

        gamma = float(self.gamma)

        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")

        # Number of training episodes
        if (
            isinstance(self.num_episodes, bool)
            or not isinstance(self.num_episodes, int)
            or self.num_episodes < 1
        ):
            raise ValueError("num_episodes must be a positive integer.")

        # Number of evaluation episodes
        if (
            isinstance(self.eval_episodes, bool)
            or not isinstance(self.eval_episodes, int)
            or self.eval_episodes < 1
        ):
            raise ValueError("eval_episodes must be a positive integer.")

        # Logging interval
        if (
            isinstance(self.log_interval, bool)
            or not isinstance(self.log_interval, int)
            or self.log_interval < 1
        ):
            raise ValueError("log_interval must be a positive integer.")

        # Random seed
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer.")
