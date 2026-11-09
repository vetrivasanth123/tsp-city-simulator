
from __future__ import annotations

import math

from .policy import BaseSoftmaxPolicy
from .trajectory import EpisodeTrajectory
from .returns import calculate_returns



def reinforce_update(
    policy: BaseSoftmaxPolicy,
    trajectory: EpisodeTrajectory,
    step_size: float,
    gamma: float = 1.0,
) -> dict[str, float]:
    """Perform one textbook REINFORCE update over a completed episode.

    Each recorded decision gets its own parameter update, in chronological
    order. Returns are calculated from the episode's rewards.

    Args:
        policy: Differentiable parameterized policy.
        trajectory: Completed episode recorded by trajectory.py.
        step_size: Step size alpha used in the textbook update.
        gamma: Discount factor in the range [0, 1].

    Returns:
        Basic training information: mean gradient norm and number of updates.
    """

    gamma = float(gamma)
    step_size = float(step_size)

    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive.")

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

    # File 3: calculate the return for each decision.
    returns = calculate_returns(trajectory.rewards, gamma)

    if not all(math.isfinite(float(r)) for r in returns):
        raise FloatingPointError("Non-finite episode returns")
    
    
    # ========================================================
    # DEBUG CODE 1 ENDED
    # ========================================================

    # Textbook REINFORCE: update once per decision, in chronological order.
    parameters = tuple(policy.parameters())
    gradient_norms = []
    for t in range(num_steps):
        observation = trajectory.observations[t]
        available_actions = trajectory.available_actions[t]
        intended_action = trajectory.intended_actions[t]
        G_t = returns[t]

        
        
        # File 1: calculate log pi_theta(A_t | S_t).
        log_prob = policy.log_probability(
            observation,
            available_actions,
            intended_action,
        )
        if not torch.isfinite(log_prob).all():
            raise FloatingPointError(
                f"Non-finite log probability at timestep {t}"
            )
                 
        # Calculate the eligibility vector:
        # grad_theta log pi(A_t | S_t, theta)
        gradients = torch.autograd.grad(
            outputs=log_prob,
            inputs=parameters,
        )

        gradient_norm_squared = sum(
            torch.sum(gradient.detach() ** 2).item()
            for gradient in gradients
        )

        gradient_norm = math.sqrt(gradient_norm_squared)

        # Textbook REINFORCE parameter update:
        # theta <- theta + alpha * gamma^t * G_t
        #                  * grad_theta log pi(A_t | S_t, theta)

        update_factor = step_size * (gamma ** t) * float(G_t)

        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter.add_(gradient, alpha=update_factor)

        gradient_norms.append(gradient_norm)

    return {
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "num_updates": float(num_steps),
    }
