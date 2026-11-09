
from __future__ import annotations

import math
import torch

from tsp.env import TSPEnv
from .policy import BaseSoftmaxPolicy
from .returns import calculate_returns


def evaluate_policy(
    env: TSPEnv,
    policy: BaseSoftmaxPolicy,
    start_city: int,
    num_episodes: int = 30,
    seed: int = 42,
    gamma: float = 1.0,
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

    if isinstance(gamma, bool):
        raise ValueError("gamma must be numeric, not Boolean.")
    
    gamma = float(gamma)
    
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and between 0 and 1.")

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
                episode_rewards: list[float] = []

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

                    episode_rewards.append(float(reward))
                    observation = next_observation

                discounted_returns = calculate_returns(
                    episode_rewards,
                    gamma,
                )
                
                if not discounted_returns:
                    raise RuntimeError("Evaluation produced an empty episode.")
                
                episode_return = float(discounted_returns[0])
                
                if not math.isfinite(episode_return):
                    raise ValueError(
                        "Evaluation produced a non-finite discounted return."
                    )
                
                # Textbook episodic objective: G_0
                episode_returns.append(episode_return)
                episode_costs.append(-episode_return)

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
