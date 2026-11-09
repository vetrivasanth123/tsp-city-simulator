
from __future__ import annotations

import random
import numpy as np
import torch
import copy

from tsp.env import TSPEnv

from .config import ReinforceConfig
from .policy import SoftmaxPolicy, MLPSoftmaxPolicy
from .trajectory import EpisodeTrajectory
from .reinforce import (
    reinforce_update,
    reinforce_baseline_update,
)

from .value_function import (
    LinearValueFunction,
    MLPValueFunction,
)
from .returns import calculate_returns


def train_reinforce(
    env: TSPEnv,
    config: ReinforceConfig,
    start_city: int,
    policy_type: str = "linear",
    algorithm: str = "reinforce",
):
    """Train a selected policy architecture using REINFORCE."""
    if algorithm not in {"reinforce", "reinforce-baseline"}:
        raise ValueError(
            "algorithm must be 'reinforce' or 'reinforce-baseline'."
        )
    if policy_type not in {"linear", "mlp"}:
        raise ValueError(
            "policy_type must be either 'linear' or 'mlp'."
        )

    if not 0 <= start_city < env.instance.num_cities:
        raise ValueError("start_city is outside the valid range.")
    # ============================================================
    # INITIALIZE VALUE FUNCTION ONLY FOR REINFORCE WITH BASELINE
    # ============================================================
    
    value_function = None
    
    if algorithm == "reinforce-baseline":
    
        if policy_type == "linear":
            value_function = LinearValueFunction(
                num_features=policy.num_features
            )
    
        else:
            value_function = MLPValueFunction(
                num_features=policy.num_features,
                hidden_sizes=(128, 64),
            )
    # Save initial value-function parameters before training.
    initial_value_state = (
        copy.deepcopy(value_function.state_dict())
        if value_function is not None
        else None
    )
    # ============================================================
    # END VALUE FUNCTION INITIALIZATION
    # ============================================================

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
    initial_policy_state = copy.deepcopy(policy.state_dict())
    # END TEMPORARY DIAGNOSTIC  
    
    history: list[dict[str, int | float]] = []
    
    print("\n========== TRAINING CONFIG DEBUG ==========")
    print(f"num_episodes : {config.num_episodes}")
    print(f"log_interval : {config.log_interval}")
    print(f"learning_rate: {config.learning_rate}")
    print(f"value_learning_rate: {config.value_learning_rate}")
    print(f"algorithm    : {algorithm}")
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
        # SELECT THE POLICY UPDATE ALGORITHM
        # ============================================================
        
        if algorithm == "reinforce":
        
            diagnostics = reinforce_update(
                policy=policy,
                trajectory=trajectory,
                step_size=config.learning_rate,
                gamma=config.gamma,
            )
        
        else:
        
            diagnostics = reinforce_baseline_update(
                policy=policy,
                value_function=value_function,
                trajectory=trajectory,
                policy_step_size=config.learning_rate,
                value_step_size=config.value_learning_rate,
                gamma=config.gamma,
            )
        
        # ============================================================
        # END ALGORITHM SELECTION
        # ============================================================

        # ============================================================
        # DIAGNOSTIC: EPISODE RETURN STATISTICS
        # ============================================================
        
        episode_returns = calculate_returns(
            trajectory.rewards,
            config.gamma,
        )
        
        if not episode_returns:
            raise RuntimeError("Training produced an empty episode.")
        
        episode_return = float(episode_returns[0])
        
        mean_entropy = (
            float(np.mean(episode_entropies))
            if episode_entropies
            else 0.0
        )
        total_reward = episode_return
        total_cost = -episode_return

        episode_record: dict[str, int | float] = {
            "episode": episode,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "mean_gradient_norm": diagnostics["mean_gradient_norm"],
            "num_updates": diagnostics["num_updates"],
        
            # Episodic objective
            "episode_return": episode_return,
        
            # Diagnostic measurement
            "policy_entropy": mean_entropy,
        }
        
        if algorithm == "reinforce-baseline":
            episode_record["mean_value_gradient_norm"] = (
                diagnostics["mean_value_gradient_norm"]
            )
            episode_record["mean_absolute_advantage"] = (
                diagnostics["mean_absolute_advantage"]
            )
         
        history.append(episode_record)

        should_log = (
            episode % config.log_interval == 0
            or episode == 1
            or episode == config.num_episodes
        )
        
        if should_log:
            print(
                print(
                f"Algorithm: {algorithm:<18} | "
                f"Policy: {policy_type:<6} | "
                f"Episode {episode:4d}/{config.num_episodes} | "
                f"Cost: {total_cost:9.4f} | "
                f"Reward: {total_reward:9.4f} | "
                f"Mean grad norm: {diagnostics['mean_gradient_norm']:9.4f} | "
                f"Updates: {int(diagnostics['num_updates'])} | "
                f"G0: {episode_return:9.4f} | "
                f"Entropy: {mean_entropy:.4f}",
                flush=True,
            )
        if algorithm == "reinforce-baseline":
            print(
                f"Value grad norm: "
                f"{diagnostics['mean_value_gradient_norm']:.4f} | "
                f"Mean |Advantage|: "
                f"{diagnostics['mean_absolute_advantage']:.4f}",
                flush=True,
            )           
    # FINAL TRAINING SUMMARY
    # ============================================================
        
    print("\n========== TRAINING SUMMARY ==========")
        
    for entry in history:
        print(
            f"Algorithm: {algorithm:<18} | "
            f"Policy: {policy_type:<6} | "
            f"Episode {entry['episode']:4d}/{config.num_episodes} | "
            f"Cost: {entry['total_cost']:9.4f} | "
            f"Reward: {entry['total_reward']:9.4f} | "
            f"Mean grad norm: {entry['mean_gradient_norm']:9.4f} | "
            f"Updates: {int(entry['num_updates'])} | "
            f"G0: {entry['episode_return']:9.4f} | "
            f"Entropy: {entry['policy_entropy']:.4f}"
        )
        
    print("======================================\n", flush=True)      

    # ============================================================
    # SAVE INITIAL AND FINAL MODEL PARAMETERS
    # ============================================================
    
    policy.initial_policy_state = initial_policy_state
    policy.final_policy_state = copy.deepcopy(policy.state_dict())
    
    if algorithm == "reinforce-baseline":
        policy.initial_value_state = initial_value_state
        policy.final_value_state = copy.deepcopy(
            value_function.state_dict()
        )
    
    # ============================================================
    # END DIAGNOSTIC
    # ============================================================

    return policy, history
