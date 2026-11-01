
from __future__ import annotations

import math
import torch

from tsp.env import TSPEnv
from .policy import SoftmaxPolicy


def evaluate_policy(
    env: TSPEnv,
    policy: SoftmaxPolicy,
    start_city: int,
    num_episodes: int = 30,
    seed: int = 42,
) -> dict[str, int | float | list[float]]:
    """Estimate the episodic return objective without updating the policy."""

    if (
        isinstance(num_episodes, bool)
        or not isinstance(num_episodes, int)
        or num_episodes < 1
    ):
        raise ValueError("num_episodes must be a positive integer.")

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer.")

    if not 0 <= start_city < env.instance.num_cities:
        raise ValueError("start_city is outside the valid city range.")

    # Make stochastic policy action sampling reproducible.
    # Environment transition randomness is controlled separately
    # by the seed used when constructing the environment.
    torch.manual_seed(seed)

    episode_returns: list[float] = []
    episode_costs: list[float] = []

    was_training = policy.training
    policy.eval()

    try:
        # Evaluation uses no gradients and never updates parameters.
        with torch.no_grad():
            for _ in range(num_episodes):
                observation, info = env.reset(start_city=start_city)
                terminated = False
                total_return = 0.0

                while not terminated:
                    available_actions = info["available_actions"]

                    intended_action = policy.select_action(
                        observation,
                        available_actions,
                    )

                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(intended_action)

                    if truncated:
                        raise RuntimeError(
                            "The evaluation episode was truncated "
                            "before termination."
                        )

                    total_return += float(reward)
                    observation = next_observation

                if not math.isfinite(total_return):
                    raise ValueError(
                        "Evaluation produced a non-finite episode return."
                    )

                episode_returns.append(total_return)
                episode_costs.append(-total_return)

    finally:
        # Restore the policy's previous training/evaluation mode.
        policy.train(was_training)

    mean_return = sum(episode_returns) / num_episodes
    mean_cost = sum(episode_costs) / num_episodes

    variance_cost = sum(
        (cost - mean_cost) ** 2 for cost in episode_costs
    ) / num_episodes

    std_cost = math.sqrt(variance_cost)

    return {
        "num_episodes": num_episodes,
        "estimated_J_theta": mean_return,
        "mean_return": mean_return,
        "mean_cost": mean_cost,
        "std_cost": std_cost,
        "episode_returns": episode_returns,
        "episode_costs": episode_costs,
    }
