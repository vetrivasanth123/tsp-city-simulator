
from __future__ import annotations

import math


def calculate_returns(
    rewards: list[float],
    gamma: float = 0.99,
) -> list[float]:
    """Calculate discounted returns for one complete episode.

    Args:
        rewards: Rewards in chronological order.
        gamma: Discount factor in the range [0, 1].

    Returns:
        One return per reward, in chronological order.
    """

    if isinstance(gamma, bool):
        raise ValueError("gamma must be a numeric value, not a Boolean.")
    
    gamma = float(gamma)

    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and between 0 and 1.")

    reward_values = [float(reward) for reward in rewards]

    if not all(math.isfinite(reward) for reward in reward_values):
        raise ValueError("All rewards must be finite.")

    returns = [0.0] * len(reward_values)
    running_return = 0.0

    for t in range(len(reward_values) - 1, -1, -1):
        running_return = reward_values[t] + gamma * running_return
        returns[t] = running_return

    return returns
