
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReinforceConfig:
    """Configuration for vanilla REINFORCE training and evaluation."""

    learning_rate: float = 1e-3
    gamma: float = 0.9
    num_episodes: int = 500
    eval_episodes: int = 30
    log_interval: int = 25
    seed: int = 42

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.learning_rate))
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive.")

        if (
            not math.isfinite(float(self.gamma))
            or not 0.0 <= self.gamma <= 1.0
        ):
            raise ValueError("gamma must be between 0 and 1.")

        if (
            isinstance(self.num_episodes, bool)
            or not isinstance(self.num_episodes, int)
            or self.num_episodes < 1
        ):
            raise ValueError("num_episodes must be a positive integer.")

        if (
            isinstance(self.eval_episodes, bool)
            or not isinstance(self.eval_episodes, int)
            or self.eval_episodes < 1
        ):
            raise ValueError("eval_episodes must be a positive integer.")

        if (
            isinstance(self.log_interval, bool)
            or not isinstance(self.log_interval, int)
            or self.log_interval < 1
        ):
            raise ValueError("log_interval must be a positive integer.")

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer.")
