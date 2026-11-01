
from __future__ import annotations

import random
import numpy as np
import torch

from tsp.env import TSPEnv

from .config import ReinforceConfig
from .policy import SoftmaxPolicy, MLPSoftmaxPolicy
from .trajectory import EpisodeTrajectory
from .reinforce import reinforce_update
from .returns import calculate_returns


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
                    normalize_cost_matrix=True,  # Run B: normalized input
                )
        
                # ============================================================
                # END TEMPORARY DIAGNOSTIC
                # ============================================================
    # TEMPORARY DIAGNOSTIC: save initial policy parameters
    import copy
    initial_policy_state = copy.deepcopy(policy.state_dict())
    # END TEMPORARY DIAGNOSTIC  
                           
    optimizer = torch.optim.SGD(
        policy.parameters(),
        lr=config.learning_rate,
    )
    
    history: list[dict[str, int | float]] = []
    
    print("\n========== TRAINING CONFIG DEBUG ==========")
    print(f"num_episodes : {config.num_episodes}")
    print(f"log_interval : {config.log_interval}")
    print(f"learning_rate: {config.learning_rate}")
    print(f"seed         : {config.seed}")
    print("===========================================\n", flush=True)
    
    for episode in range(1, config.num_episodes + 1):
    
        observation, info = env.reset(start_city=start_city)
        trajectory = EpisodeTrajectory()
        terminated = False
        # ============================================================
        # DIAGNOSTIC: POLICY ENTROPY DURING EPISODE GENERATION
        # ============================================================
        
        episode_entropies = []

        while not terminated:
            available_actions = info["available_actions"]
            # ============================================================
            # DIAGNOSTIC: MEASURE ENTROPY BEFORE SAMPLING THE ACTION
            # ============================================================
            
            distribution = policy.get_distribution(
                observation,
                available_actions,
            )
            
            # Exclude forced decisions (only one legal action).
            if len(available_actions) > 1:
                episode_entropies.append(
                    float(distribution.entropy().item())
                )
            
            intended_action = int(distribution.sample().item())

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

        # ============================================================
        # TEMPORARY DIAGNOSTIC: RETURN-TO-COST RELATIONSHIP
        # Purpose:
        #   Compare each discounted return G_t against the actual
        #   remaining cost from that timestep.
        #   Does NOT modify rewards, returns, or policy updates.
        # ============================================================

        diagnostic_returns = calculate_returns(
            trajectory.rewards,
            config.gamma,
        )

        print(f"\n[RETURN-COST DIAGNOSTIC] Episode {episode}")

        for t, (reward_t, return_t) in enumerate(
            zip(trajectory.rewards, diagnostic_returns)
        ):
            remaining_cost = -sum(trajectory.rewards[t:])

            print(
                f"t={t} | "
                f"reward={reward_t:9.4f} | "
                f"G_t={return_t:9.4f} | "
                f"undiscounted_remaining_cost={remaining_cost:9.4f}"
            )

        print("================================================\n")

        # ============================================================
        # END TEMPORARY DIAGNOSTIC
        # ============================================================

        # Keep the existing sequential REINFORCE update unchanged.
        diagnostics = reinforce_update(
            policy=policy,
            trajectory=trajectory,
            optimizer=optimizer,
            gamma=config.gamma,
        )

        # ============================================================
        # DIAGNOSTIC: EPISODE RETURN STATISTICS
        # ============================================================
        
        episode_returns = np.asarray(
            calculate_returns(trajectory.rewards, config.gamma),
            dtype=np.float64,
        )
        
        return_mean = float(np.mean(episode_returns))
        return_std = float(np.std(episode_returns))
        return_min = float(np.min(episode_returns))
        return_max = float(np.max(episode_returns))
        
        mean_entropy = (
            float(np.mean(episode_entropies))
            if episode_entropies
            else 0.0
        )
        total_reward = float(sum(trajectory.rewards))
        total_cost = -total_reward

        episode_record: dict[str, int | float] = {
            "episode": episode,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "mean_loss": diagnostics["mean_loss"],
            "num_updates": diagnostics["num_updates"],
        
            # Diagnostic measurements
            "return_mean": return_mean,
            "return_std": return_std,
            "return_min": return_min,
            "return_max": return_max,
            "policy_entropy": mean_entropy,
        }
        history.append(episode_record)

        should_log = (
            episode % config.log_interval == 0
            or episode == 1
            or episode == config.num_episodes
        )
        
        if should_log:
            print(
                f"Policy: {policy_type:<6} | "
                f"Episode {episode:4d}/{config.num_episodes} | "
                f"Cost: {total_cost:9.4f} | "
                f"Reward: {total_reward:9.4f} | "
                f"Mean loss: {diagnostics['mean_loss']:9.4f} | "
                f"Updates: {int(diagnostics['num_updates'])} | "
                f"Return μ: {return_mean:9.4f} | "
                f"Return σ: {return_std:9.4f} | "
                f"Entropy: {mean_entropy:.4f}",
                flush=True,
            )
    # FINAL TRAINING SUMMARY
    # ============================================================
        
    print("\n========== TRAINING SUMMARY ==========")
        
    for entry in history:
        print(
            f"Policy: {policy_type:<6} | "
            f"Episode {entry['episode']:4d}/{config.num_episodes} | "
            f"Cost: {entry['total_cost']:9.4f} | "
            f"Reward: {entry['total_reward']:9.4f} | "
            f"Mean loss: {entry['mean_loss']:9.4f} | "
            f"Updates: {int(entry['num_updates'])} | "
            f"Return μ: {entry['return_mean']:9.4f} | "
            f"Return σ: {entry['return_std']:9.4f} | "
            f"Entropy: {entry['policy_entropy']:.4f}"
        )
        
    print("======================================\n", flush=True)      

    # TEMPORARY DIAGNOSTIC: attach initial parameters for comparison
    policy.initial_policy_state = initial_policy_state
    # END TEMPORARY DIAGNOSTIC

    return policy, history


