
from __future__ import annotations

import random
import numpy as np
import torch

from tsp.env import TSPEnv

from .config import ReinforceConfig
from .policy import SoftmaxPolicy
from .trajectory import EpisodeTrajectory
from .reinforce import reinforce_update


def train_reinforce(
    env: TSPEnv,
    config: ReinforceConfig,
    start_city: int,
) -> tuple[SoftmaxPolicy, list[dict[str, int | float]]]:
    """Train a REINFORCE policy using an existing TSP environment."""

    # Seed training-side randomness before creating the policy.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Reuse the number of cities from the existing environment's instance.
    policy = SoftmaxPolicy(env.instance.num_cities)

    # Vanilla REINFORCE uses gradient ascent on expected return.
    # SGD minimizes the equivalent negative-return loss.
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
            # Legal actions are supplied by the existing environment.
            available_actions = info["available_actions"]

            # Sample the intended action from the current policy.
            intended_action = policy.select_action(
                observation,
                available_actions,
            )

            # Let the existing TSP environment handle transitions and costs.
            next_observation, reward, terminated, truncated, info = env.step(
                intended_action
            )

            if truncated:
                raise RuntimeError(
                    "The TSP episode was truncated unexpectedly; "
                    "REINFORCE requires a complete episode here."
                )

            # Store the pre-action state, legal actions, intended action,
            # and the reward returned by the environment.
            trajectory.add_step(
                observation,
                available_actions,
                intended_action,
                reward,
            )

            observation = next_observation

        # Update only after the complete episode has been collected.
        diagnostics = reinforce_update(
            policy=policy,
            trajectory=trajectory,
            optimizer=optimizer,
            gamma=config.gamma,
        )

        # Environment rewards are negative costs, including the CLOSE step.
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
                f"Episode {episode:4d}/{config.num_episodes} | "
                f"Cost: {total_cost:10.4f} | "
                f"Reward: {total_reward:10.4f} | "
                f"Mean loss: {diagnostics['mean_loss']:10.4f} | "
                f"Updates: {int(diagnostics['num_updates'])}"
            )

    return policy, history
