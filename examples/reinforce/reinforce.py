
from __future__ import annotations

import math
import torch
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

def reinforce_baseline_update(
    policy: BaseSoftmaxPolicy,
    value_function: torch.nn.Module,
    trajectory: EpisodeTrajectory,
    policy_step_size: float,
    value_step_size: float,
    gamma: float = 1.0,
) -> dict[str, float]:
    """Perform one textbook REINFORCE with Baseline update.

    Both policy and value-function parameters are updated directly
    using autograd gradients, without a loss or optimizer.
    """

    policy_step_size = float(policy_step_size)
    value_step_size = float(value_step_size)
    gamma = float(gamma)

    if (
        not math.isfinite(policy_step_size)
        or policy_step_size <= 0
    ):
        raise ValueError(
            "policy_step_size must be finite and positive."
        )

    if (
        not math.isfinite(value_step_size)
        or value_step_size <= 0
    ):
        raise ValueError(
            "value_step_size must be finite and positive."
        )

    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "gamma must be finite and between 0 and 1."
        )

    num_steps = len(trajectory)

    if num_steps == 0:
        raise ValueError(
            "Cannot update from an empty episode."
        )

    if not (
        len(trajectory.observations)
        == len(trajectory.available_actions)
        == len(trajectory.intended_actions)
        == len(trajectory.rewards)
    ):
        raise ValueError(
            "Trajectory data lists have inconsistent lengths."
        )

    # Calculate Monte Carlo returns for the completed episode.
    returns = calculate_returns(
        trajectory.rewards, gamma
    )

    if not all(math.isfinite(float(r)) for r in returns):
        raise FloatingPointError(
            "Non-finite episode returns."
        )

    policy_parameters = tuple(policy.parameters())
    value_parameters = tuple(value_function.parameters())

    if not policy_parameters or not value_parameters:
        raise ValueError(
            "Policy and value function must have parameters."
        )

    # Ensure the two models have independent parameters.
    if {
        id(parameter) for parameter in policy_parameters
    } & {
        id(parameter) for parameter in value_parameters
    }:
        raise ValueError(
            "Policy and value function must not share parameters."
        )

    policy_gradient_norms = []
    value_gradient_norms = []
    absolute_advantages = []
    # DIAGNOSTIC CODE 1: Track value-function learning
    value_squared_errors_before = []
    value_squared_errors_after = []

    # Textbook updates: chronological order.
    for t in range(num_steps):

        observation = trajectory.observations[t]
        available_actions = trajectory.available_actions[t]
        intended_action = trajectory.intended_actions[t]
        G_t = float(returns[t])

        # --------------------------------------------------
        # 1. Encode the state using the policy's encoder.
        # --------------------------------------------------

        features = policy.encode_observation(observation)

        # --------------------------------------------------
        # 2. Estimate the state value: v_hat(S_t, w).
        # --------------------------------------------------

        value_estimate = value_function(features)

        if value_estimate.numel() != 1:
            raise ValueError(
                "Value function must return a single scalar."
            )

        if not torch.isfinite(value_estimate).all():
            raise FloatingPointError(
                f"Non-finite value estimate at timestep {t}."
            )

        # --------------------------------------------------
        # 3. Calculate delta = G_t - v_hat(S_t, w).
        #    Calculate it BEFORE updating value parameters.
        # --------------------------------------------------

        delta = (
            G_t - value_estimate.detach().item()
        )
        # DIAGNOSTIC CODE 2: Prediction error before value update
        value_before = value_estimate.detach().item()
        
        squared_error_before = (G_t - value_before) ** 2

        if not math.isfinite(delta):
            raise FloatingPointError(
                f"Non-finite advantage at timestep {t}."
            )

        # --------------------------------------------------
        # 4. Value-function parameter update.
        #
        # w <- w + alpha_w * delta * grad_w v_hat(S_t,w)
        # --------------------------------------------------

        value_gradients = torch.autograd.grad(
            outputs=value_estimate,
            inputs=value_parameters,
        )

        value_gradient_norm_squared = sum(
            torch.sum(gradient.detach() ** 2).item()
            for gradient in value_gradients
        )

        value_gradient_norm = math.sqrt(
            value_gradient_norm_squared
        )

        with torch.no_grad():
            for parameter, gradient in zip(
                value_parameters, value_gradients
            ):
                parameter.add_(
                    gradient,
                    alpha=value_step_size * delta,
                )
                
        # DIAGNOSTIC CODE 3: Prediction error after value update
        with torch.no_grad():
            value_after = value_function(features).item()
        
        squared_error_after = (G_t - value_after) ** 2
        
        value_squared_errors_before.append(squared_error_before)
        value_squared_errors_after.append(squared_error_after)

        # --------------------------------------------------
        # 5. Policy parameter update.
        #
        # theta <- theta + alpha_theta * gamma^t * delta
        #          * grad_theta log pi(A_t | S_t, theta)
        # --------------------------------------------------

        log_prob = policy.log_probability(
            observation,
            available_actions,
            intended_action,
        )

        if not torch.isfinite(log_prob).all():
            raise FloatingPointError(
                f"Non-finite log probability at timestep {t}."
            )

        policy_gradients = torch.autograd.grad(
            outputs=log_prob,
            inputs=policy_parameters,
        )

        policy_gradient_norm_squared = sum(
            torch.sum(gradient.detach() ** 2).item()
            for gradient in policy_gradients
        )

        policy_gradient_norm = math.sqrt(
            policy_gradient_norm_squared
        )

        update_factor = (
            policy_step_size * (gamma ** t) * delta
        )

        with torch.no_grad():
            for parameter, gradient in zip(
                policy_parameters, policy_gradients
            ):
                parameter.add_(
                    gradient,
                    alpha=update_factor,
                )

        policy_gradient_norms.append(
            policy_gradient_norm
        )
        value_gradient_norms.append(
            value_gradient_norm
        )
        absolute_advantages.append(abs(delta))

    return {
        "mean_gradient_norm": (
            sum(policy_gradient_norms)
            / len(policy_gradient_norms)
        ),
        "mean_value_gradient_norm": (
            sum(value_gradient_norms)
            / len(value_gradient_norms)
        ),
        "mean_absolute_advantage": (
            sum(absolute_advantages)
            / len(absolute_advantages)
        ),
        "num_updates": float(num_steps),
        "mean_value_mse_before": (
            sum(value_squared_errors_before)
            / len(value_squared_errors_before)
        ),
        "mean_value_mse_after": (
            sum(value_squared_errors_after)
            / len(value_squared_errors_after)
        ),
        "value_mse_reduction": (
            sum(value_squared_errors_before)
            / len(value_squared_errors_before)
            -
            sum(value_squared_errors_after)
            / len(value_squared_errors_after)
        ),
    }
    
    
