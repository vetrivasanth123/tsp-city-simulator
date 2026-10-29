
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .policy import SoftmaxPolicy
from .trajectory import EpisodeTrajectory
from .returns import calculate_returns


def reinforce_update(
    policy: SoftmaxPolicy,
    trajectory: EpisodeTrajectory,
    optimizer: Optimizer,
    gamma: float = 0.99,
) -> dict[str, float]:
    """Perform one textbook REINFORCE update over a completed episode.

    Each recorded decision gets its own parameter update, in chronological
    order. Returns are calculated from the episode's rewards.

    Args:
        policy: The trainable policy defined in policy.py.
        trajectory: The completed episode recorded by trajectory.py.
        optimizer: A persistent PyTorch optimizer for the policy parameters.
        gamma: Discount factor in the range [0, 1].

    Returns:
        Basic training diagnostics: mean loss and number of updates.
    """

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

    # File 3: calculate the return for each decision.
    returns = calculate_returns(trajectory.rewards, gamma)

    losses = []

    # Textbook REINFORCE: update once per decision, in chronological order.
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

        # PyTorch minimizes losses, so negate the policy-gradient objective.
        loss = -(gamma ** t) * G_t * log_prob

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().item()))

    return {
        "mean_loss": sum(losses) / len(losses),
        "num_updates": float(num_steps),
    }



# ============================================================
# DEBUG CODE STARTED
# Purpose: Capture compact REINFORCE update diagnostics.
# ============================================================

def debug_reinforce_update(
    policy,
    trajectory,
    optimizer,
    gamma=1.0,
    debug_report=None,
    episode_index=None,
):
    """
    Debug-enabled equivalent of reinforce_update().

    Preserves the original chronological per-decision update order,
    loss equation, optimizer, and number of optimizer steps.

    Stores actual returns, log probabilities, losses, gradients,
    parameter deltas, and update norms without repeated parameter
    snapshots at every timestep.
    """
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
    losses = []

    def record(section, details):
        if debug_report is not None:
            debug_report.record(section, details)

    def snapshot_parameters():
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in policy.named_parameters()
        }

    def tensor_norm(tensor):
        return float(torch.linalg.vector_norm(tensor.detach()).item())

    # Record the actual rewards and returns used by this episode.
    record(
        "training_returns",
        {
            "episode_index": episode_index,
            "gamma": gamma,
            "rewards": [float(r) for r in trajectory.rewards],
            "returns": [float(value) for value in returns],
            "num_steps": num_steps,
        },
    )

    # One optimizer update per decision, in chronological order.
    for t in range(num_steps):
        observation = trajectory.observations[t]
        available_actions = trajectory.available_actions[t]
        intended_action = trajectory.intended_actions[t]
        G_t = returns[t]
        discount_factor = gamma ** t

        # Capture parameters immediately before the real update.
        parameters_before = snapshot_parameters()

        log_prob = policy.log_probability(
            observation,
            available_actions,
            intended_action,
        )

        # Keep the original loss expression and operation order.
        loss = -(gamma ** t) * G_t * log_prob

        optimizer.zero_grad()
        loss.backward()

        # Record actual gradients from this backward pass.
        gradients = {}
        gradient_norm_squared = 0.0

        for name, parameter in policy.named_parameters():
            if parameter.grad is None:
                gradients[name] = None
                continue

            gradient = parameter.grad.detach().cpu().clone()
            gradients[name] = gradient

            gradient_norm = tensor_norm(parameter.grad)
            gradient_norm_squared += gradient_norm ** 2

        total_gradient_norm = math.sqrt(gradient_norm_squared)

        # Exactly one real optimizer step, as in reinforce_update().
        optimizer.step()

        # Measure the actual change caused by this optimizer step.
        parameter_deltas = {}
        update_norm_squared = 0.0

        for name, parameter in policy.named_parameters():
            before = parameters_before[name]
            after = parameter.detach().cpu()
            delta = after - before

            parameter_deltas[name] = delta

            delta_norm = tensor_norm(delta)
            update_norm_squared += delta_norm ** 2

        total_update_norm = math.sqrt(update_norm_squared)

        record(
            "training_updates",
            {
                "episode_index": episode_index,
                "timestep": t,
                "observation": observation,
                "available_actions": list(available_actions),
                "intended_action": int(intended_action),
                "reward": float(trajectory.rewards[t]),
                "return_G_t": float(G_t),
                "gamma": gamma,
                "discount_factor_gamma_power_t": float(discount_factor),
                "log_probability": float(log_prob.detach().item()),
                "loss": float(loss.detach().item()),
                "gradients": gradients,
                "gradient_global_norm": total_gradient_norm,
                "parameter_deltas": parameter_deltas,
                "update_global_norm": total_update_norm,
            },
        )

        losses.append(float(loss.detach().item()))

    result = {
        "mean_loss": sum(losses) / len(losses),
        "num_updates": float(num_steps),
    }

    record(
        "training_episode_updates",
        {
            "episode_index": episode_index,
            "mean_loss": result["mean_loss"],
            "num_updates": int(num_steps),
        },
    )

    return result


# ============================================================
# DEBUG CODE ENDED
# ============================================================
