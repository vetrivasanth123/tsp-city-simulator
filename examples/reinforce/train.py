
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
        # ============================================================
                # TEMPORARY DIAGNOSTIC: GRID-DIAGONAL COST NORMALIZATION
                # Purpose: Pass the actual environment dimensions to the MLP.
                # ============================================================
        
                grid_width = env.instance.width
                grid_height = env.instance.height
        
                if grid_width is None or grid_height is None:
                    raise ValueError(
                        "Grid width and height must be specified in TSPInstance "
                        "for the normalization experiment."
                    )
        
                policy = MLPSoftmaxPolicy(
                    num_cities=num_cities,
                    cost_matrix=env.instance.cost_matrix,
                    start_city=start_city,
                    hidden_sizes=(128, 64),
                    grid_width=grid_width,
                    grid_height=grid_height,
                    normalize_cost_matrix=False,  # Run A: original input
                )
        
                # ============================================================
                # END TEMPORARY DIAGNOSTIC
                # ============================================================
                
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
            # TEMPORARY DIAGNOSTIC — inspect MLP input and activations once.
            if policy_type == "mlp" and episode == 1 and len(trajectory.rewards) == 0:
                with torch.no_grad():
                    x = policy.encode_observation(observation)
            
                    z1 = policy.network[0](x)
                    a1 = policy.network[1](z1)
            
                    z2 = policy.network[2](a1)
                    a2 = policy.network[3](z2)
            
                    logits = policy.network[4](a2)
            
                    def show_stats(name, tensor):
                        print(
                            f"{name:20s} | "
                            f"min={tensor.min().item():12.4f} | "
                            f"max={tensor.max().item():12.4f} | "
                            f"mean={tensor.mean().item():12.4f} | "
                            f"std={tensor.std().item():12.4f} | "
                            f"L2={torch.linalg.vector_norm(tensor).item():12.4f}"
                        )
            
                    print("\n========== MLP SCALE DIAGNOSTICS ==========")
            
                    show_stats("Input features", x)
                    show_stats("Current city", x[:num_cities])
                    show_stats("Starting city", x[num_cities:2*num_cities])
                    show_stats("Visited mask", x[2*num_cities:3*num_cities])
                    show_stats("Cost matrix", x[3*num_cities:])
            
                    show_stats("Layer 1 pre-ReLU", z1)
                    show_stats("Layer 1 activation", a1)
                    show_stats("Layer 2 pre-ReLU", z2)
                    show_stats("Layer 2 activation", a2)
                    show_stats("Output logits", logits)
            
                    print("===========================================\n")

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
# Purpose: Record actual training transitions and connect
# policy hooks to per-decision REINFORCE diagnostics.
# ============================================================

def debug_train_reinforce(
    env: TSPEnv,
    config: ReinforceConfig,
    start_city: int,
    policy_type: str,
    debug_report,
) -> tuple[SoftmaxPolicy | MLPSoftmaxPolicy, list[dict[str, int | float]]]:
    """Debug-enabled training entry point; preserves the training sequence."""
    import copy

    from .policy import debug_attach_policy_hooks
    from .reinforce import debug_reinforce_update

    if policy_type not in {"linear", "mlp"}:
        raise ValueError("policy_type must be either 'linear' or 'mlp'.")

    if not 0 <= start_city < env.instance.num_cities:
        raise ValueError("start_city is outside the valid range.")

    # Preserve the original training seed setup.
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

    # Attach observers before training; no diagnostic forward pass is run.
    policy._debug_phase = "training"
    debug_attach_policy_hooks(policy, debug_report)

    history: list[dict[str, int | float]] = []

    for episode in range(1, config.num_episodes + 1):
        observation, info = env.reset(start_city=start_city)
        trajectory = EpisodeTrajectory()
        terminated = False
        episode_steps = []

        while not terminated:
            observation_before = copy.deepcopy(observation)
            available_actions = list(info["available_actions"])

            # Exactly one call to the existing action sampler.
            intended_action = policy.select_action(
                observation,
                available_actions,
            )

            # Exactly one environment transition.
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

            # Extract actual transition details when the environment
            # provides them. Do not infer a slip from the reward.
            transition = next_info.get("transition")

            if isinstance(transition, dict):
                actual_action = transition.get("actual_action")
                slipped = transition.get("slipped")
                transition_probabilities = transition.get(
                    "transition_probabilities"
                )
                slip_probabilities = transition.get("slip_probabilities")
            else:
                actual_action = None
                slipped = None
                transition_probabilities = None
                slip_probabilities = None

            episode_steps.append(
                {
                    "timestep": len(episode_steps),
                    "observation": observation_before,
                    "available_actions": available_actions,
                    "intended_action": int(intended_action),
                    "actual_action": (
                        int(actual_action)
                        if actual_action is not None
                        else None
                    ),
                    "slipped": (
                        bool(slipped)
                        if slipped is not None
                        else None
                    ),
                    "transition_probabilities": transition_probabilities,
                    "slip_probabilities": slip_probabilities,
                    "reward": float(reward),
                    "next_observation": copy.deepcopy(next_observation),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                },
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

        # Preserve the original progress-message format.
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

    debug_report.record(
        "final_network",
        debug_report.snapshot_network(policy),
    )

    return policy, history


# ============================================================
# DEBUG CODE ENDED
# ============================================================
