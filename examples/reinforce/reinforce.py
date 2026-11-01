
from __future__ import annotations

import math
import numpy as np
import torch
from torch.optim import Optimizer

from .policy import SoftmaxPolicy
from .trajectory import EpisodeTrajectory
from .returns import calculate_returns



def reinforce_update(
    policy: SoftmaxPolicy,
    trajectory: EpisodeTrajectory,
    optimizer: Optimizer,
    gamma: float = 1.0,
) -> dict[str, float]:

    gamma = float(gamma)

    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and between 0 and 1.")

    num_steps = len(trajectory)

    if num_steps == 0:
        raise ValueError("Cannot update the policy from an empty episode.")

    if not (
        len(trajectory.observations)
        == len(trajectory.available_actions)
        == len(trajectory.intended_actions)
        == len(trajectory.rewards)
    ):
        raise ValueError("Trajectory data lists have inconsistent lengths.")

    returns = calculate_returns(trajectory.rewards, gamma)

    if not all(math.isfinite(float(r)) for r in returns):
        raise FloatingPointError("Non-finite episode returns")

    # TEMPORARY DIAGNOSTIC:
    # Optimal decisions for the known five-city instance,
    # with fixed start city 0 and best tour 0 -> 3 -> 2 -> 1 -> 4 -> 0.
    optimal_action_by_state = {
        (0, (0,)): 3,
        (3, (0, 3)): 2,
        (2, (0, 2, 3)): 1,
    }

    losses = []

    # Textbook REINFORCE: one update per decision, chronologically.
    for t in range(num_steps):

        observation = trajectory.observations[t]
        available_actions = trajectory.available_actions[t]
        intended_action = trajectory.intended_actions[t]
        G_t = returns[t]

        # Identify whether this is a prefix of the known optimal tour.
        current_city = int(observation["current_city"])
        visited = tuple(
            np.flatnonzero(
                np.asarray(observation["visited_mask"], dtype=bool)
            ).tolist()
        )

        state_key = (current_city, visited)
        optimal_action = optimal_action_by_state.get(state_key)

        # Probability distribution BEFORE this decision's update.
        distribution_before = policy.get_distribution(
            observation, available_actions
        )

        probs_before = distribution_before.probs.detach()

        if not torch.isfinite(probs_before).all():
            raise FloatingPointError(
                f"Non-finite action probabilities at timestep {t}"
            )

        selected_probability_before = float(
            probs_before[intended_action].item()
        )

        optimal_probability_before = None

        if optimal_action is not None:
            optimal_probability_before = float(
                probs_before[optimal_action].item()
            )

        # Calculate log probability of the sampled/intended action.
        log_prob = policy.log_probability(
            observation,
            available_actions,
            intended_action,
        )

        if not torch.isfinite(log_prob).all():
            raise FloatingPointError(
                f"Non-finite log probability at timestep {t}"
            )

        # Keep the existing textbook objective unchanged.
        loss = -(gamma ** t) * G_t * log_prob

        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                f"Non-finite loss at timestep {t}"
            )

        optimizer.zero_grad()
        loss.backward()

        # TEMPORARY DIAGNOSTIC: gradient magnitude.
        gradient_norm_squared = 0.0

        for name, parameter in policy.named_parameters():
            if parameter.grad is not None:

                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(
                        f"Non-finite gradient at timestep {t}: {name}"
                    )

                gradient_norm_squared += (
                    torch.linalg.vector_norm(parameter.grad).item() ** 2
                )

        gradient_norm = math.sqrt(gradient_norm_squared)

        # Save parameters only to measure the actual update magnitude.
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in policy.named_parameters()
        }

        optimizer.step()

        # Probability distribution AFTER this decision's update,
        # evaluated at exactly the same state and legal-action mask.
        with torch.no_grad():

            distribution_after = policy.get_distribution(
                observation, available_actions
            )

            probs_after = distribution_after.probs

            selected_probability_after = float(
                probs_after[intended_action].item()
            )

            optimal_probability_after = None

            if optimal_action is not None:
                optimal_probability_after = float(
                    probs_after[optimal_action].item()
                )

        # TEMPORARY DIAGNOSTIC: actual parameter-update magnitude.
        update_norm_squared = 0.0

        for name, parameter in policy.named_parameters():

            delta = parameter.detach() - parameters_before[name]

            update_norm_squared += (
                torch.linalg.vector_norm(delta).item() ** 2
            )

        update_norm = math.sqrt(update_norm_squared)

        # Print sampled-action and optimal-action probabilities separately.
        print(
            f"[REINFORCE CHECK] "
            f"t={t} | state={state_key} | "
            f"sampled={intended_action} | "
            f"G={float(G_t):.4f} | "
            f"P(sampled): "
            f"{selected_probability_before:.6f} -> "
            f"{selected_probability_after:.6f} | "
            f"grad={gradient_norm:.4e} | "
            f"update={update_norm:.4e}",
            flush=True,
        )

        if optimal_action is not None:
            print(
                f"[OPTIMAL ACTION CHECK] "
                f"t={t} | "
                f"optimal_action={optimal_action} | "
                f"P(optimal): "
                f"{optimal_probability_before:.6f} -> "
                f"{optimal_probability_after:.6f}",
                flush=True,
            )

        losses.append(float(loss.detach().item()))

    return {
        "mean_loss": sum(losses) / len(losses),
        "num_updates": float(num_steps),
    }
