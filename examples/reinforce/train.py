
from __future__ import annotations

import random
import numpy as np
import torch

from tsp.env import TSPEnv

from .config import ReinforceConfig
from .policy import SoftmaxPolicy, MLPSoftmaxPolicy
from .trajectory import EpisodeTrajectory
from .reinforce import reinforce_update


def train_reinforce(
    env: TSPEnv,
    config: ReinforceConfig,
    start_city: int,
    policy_type: str = "linear",
) -> tuple[SoftmaxPolicy | MLPSoftmaxPolicy, list[dict[str, int | float]]]:
    """Train a selected policy architecture using REINFORCE."""

    if policy_type not in {"linear", "mlp"}:
        raise ValueError(
            "policy_type must be either 'linear' or 'mlp'."
        )

    if not 0 <= start_city < env.instance.num_cities:
        raise ValueError("start_city is outside the valid range.")

    # Seed training-side randomness before creating the policy.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    num_cities = env.instance.num_cities

    # Select the policy architecture without changing the training loop.
    if policy_type == "linear":
        policy = SoftmaxPolicy(num_cities)
    else:
        policy = MLPSoftmaxPolicy(
            num_cities=num_cities,
            cost_matrix=env.instance.cost_matrix,
            start_city=start_city,
            hidden_sizes=(128, 64),
        )

    optimizer = torch.optim.SGD(
        policy.parameters(),
        lr=config.learning_rate,
    )

    history: list[dict[str, int | float]] = []

    for episode in range(1, config.num_episodes + 1):
        observation, info = env.reset(start_city=start_city)
        trajectory = EpisodeTrajectory()
        terminated = False

        while not terminated:
            available_actions = info["available_actions"]

            intended_action = policy.select_action(
                observation,
                available_actions,
            )

            next_observation, reward, terminated, truncated, info = (
                env.step(intended_action)
            )

            if truncated:
                raise RuntimeError(
                    "The TSP episode was truncated unexpectedly; "
                    "REINFORCE requires a complete episode here."
                )

            trajectory.add_step(
                observation,
                available_actions,
                intended_action,
                reward,
            )

            observation = next_observation

        # Keep the existing sequential REINFORCE update unchanged.
        diagnostics = reinforce_update(
            policy=policy,
            trajectory=trajectory,
            optimizer=optimizer,
            gamma=config.gamma,
        )

        total_reward = float(sum(trajectory.rewards))
        total_cost = -total_reward

        episode_record: dict[str, int | float] = {
            "episode": episode,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "mean_loss": diagnostics["mean_loss"],
            "num_updates": diagnostics["num_updates"],
        }
        history.append(episode_record)

        if (
            episode % config.log_interval == 0
            or episode == 1
            or episode == config.num_episodes
        ):
            print(
                f"Policy: {policy_type:6s} | "
                f"Episode {episode:4d}/{config.num_episodes} | "
                f"Cost: {total_cost:10.4f} | "
                f"Reward: {total_reward:10.4f} | "
                f"Mean loss: {diagnostics['mean_loss']:10.4f} | "
                f"Updates: {int(diagnostics['num_updates'])}"
            )

    return policy, history
