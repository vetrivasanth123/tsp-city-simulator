
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
    gamma: float = 1.0,
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

        # Measure gradients before changing parameters.
        grads = [
            p.grad.detach()
            for p in policy.parameters()
            if p.grad is not None
        ]

        grad_norm = math.sqrt(
            sum(g.pow(2).sum().item() for g in grads)
        )
        max_grad = max(
            (g.abs().max().item() for g in grads),
            default=0.0,
        )
        max_param_before = max(
            p.detach().abs().max().item()
            for p in policy.parameters()
        )

        optimizer.step()

        max_param_after = max(
            p.detach().abs().max().item()
            for p in policy.parameters()
        )

        print(
            f"t={t:02d} "
            f"G_t={G_t:.3f} "
            f"loss={loss.item():.3e} "
            f"grad_norm={grad_norm:.3e} "
            f"max_grad={max_grad:.3e} "
            f"param_before={max_param_before:.3e} "
            f"param_after={max_param_after:.3e}"
        )

        losses.append(float(loss.detach().item()))
        print(
            f"reward range=({min(trajectory.rewards):.3f}, "
            f"{max(trajectory.rewards):.3f}), "
            f"return range=({min(returns):.3f}, "
            f"{max(returns):.3f})"
        )
