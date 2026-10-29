
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



# ============================================================
# DEBUG CODE STARTED
# Purpose: Record actual training episodes and connect policy
# hooks and per-decision REINFORCE update diagnostics.
# ============================================================

def debug_train_reinforce(
    env: TSPEnv,
    config: ReinforceConfig,
    start_city: int,
    policy_type: str,
    debug_report,
) -> tuple[SoftmaxPolicy | MLPSoftmaxPolicy, list[dict[str, int | float]]]:
    """
    Run the existing REINFORCE training procedure with diagnostics.

    This function deliberately mirrors train_reinforce() so the
    ordinary training entry point and its behavior remain available.
    It does not add action samples, environment steps, or optimizer
    updates. The runner must call this function only for debug runs.
    """
    import copy

    from .policy import debug_attach_policy_hooks
    from .reinforce import debug_reinforce_update

    if policy_type not in {"linear", "mlp"}:
        raise ValueError("policy_type must be either 'linear' or 'mlp'.")

    if not 0 <= start_city < env.instance.num_cities:
        raise ValueError("start_city is outside the valid range.")

    # Preserve the original training-side seed setup and order.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    num_cities = env.instance.num_cities

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

    # Record run configuration and the initial network before training.
    debug_report.record(
        "experiment",
        {
            "policy_type": policy_type,
            "seed": config.seed,
            "start_city": start_city,
            "num_cities": num_cities,
            "num_episodes": config.num_episodes,
            "eval_episodes": config.eval_episodes,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "optimizer": type(optimizer).__name__,
            "optimizer_parameters": {
                "learning_rate": config.learning_rate,
            },
            "environment_instance": {
                "cost_matrix": env.instance.cost_matrix,
            },
        },
    )

    debug_report.record(
        "network_initialization",
        debug_report.snapshot_network(policy),
    )

    # Hooks observe existing network calls; they do not call forward().
    debug_attach_policy_hooks(policy, debug_report)

    history: list[dict[str, int | float]] = []

    for episode in range(1, config.num_episodes + 1):
        observation, info = env.reset(start_city=start_city)
        trajectory = EpisodeTrajectory()
        terminated = False
        episode_steps = []

        while not terminated:
            # Snapshot the actual state and legal actions before selection.
            observation_before = copy.deepcopy(observation)
            available_actions = list(info["available_actions"])

            # Call the existing sampler exactly once.
            intended_action = policy.select_action(
                observation,
                available_actions,
            )

            # Call the environment exactly once for this action.
            next_observation, reward, terminated, truncated, next_info = (
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

            # Store the real transition, not a reconstructed transition.
            episode_steps.append(
                {
                    "timestep": len(episode_steps),
                    "observation": observation_before,
                    "available_actions": available_actions,
                    "intended_action": int(intended_action),
                    "reward": float(reward),
                    "next_observation": copy.deepcopy(next_observation),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info_after_step": copy.deepcopy(next_info),
                }
            )

            observation = next_observation
            info = next_info

        total_reward = float(sum(trajectory.rewards))
        total_cost = -total_reward

        debug_report.record(
            "training_episodes",
            {
                "episode": episode,
                "start_city": start_city,
                "steps": episode_steps,
                "rewards": list(trajectory.rewards),
                "total_reward": total_reward,
                "total_cost": total_cost,
                "num_steps": len(trajectory),
            },
        )

        # This records the actual returns, losses, gradients and parameter
        # changes around each real optimizer update.
        diagnostics = debug_reinforce_update(
            policy=policy,
            trajectory=trajectory,
            optimizer=optimizer,
            gamma=config.gamma,
            debug_report=debug_report,
            episode_index=episode,
        )

        episode_record: dict[str, int | float] = {
            "episode": episode,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "mean_loss": diagnostics["mean_loss"],
            "num_updates": diagnostics["num_updates"],
        }
        history.append(episode_record)

        # Keep the original training progress output unchanged.
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

    # Record the final parameters after all training episodes.
    debug_report.record(
        "final_network",
        debug_report.snapshot_network(policy),
    )

    return policy, history


# ============================================================
# DEBUG CODE ENDED
# ============================================================

