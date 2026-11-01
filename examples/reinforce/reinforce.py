
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

    if not all(math.isfinite(float(r)) for r in returns):
        raise FloatingPointError("Non-finite episode returns")
    
    losses = []
    # ========================================================
    # DEBUG CODE 1 ENDED
    # ========================================================

    # Textbook REINFORCE: update once per decision, in chronological order.
    for t in range(num_steps):
        observation = trajectory.observations[t]
        available_actions = trajectory.available_actions[t]
        intended_action = trajectory.intended_actions[t]
        G_t = returns[t]

        
        # TEMPORARY DIAGNOSTIC 1: check parameters before log_prob.
        for name, parameter in policy.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(
                    f"Non-finite parameter before log_prob at timestep {t}: {name}"
                )   

            # TEMPORARY: detect parameters that are finite but exploding.
            max_abs = parameter.detach().abs().max().item()

            if max_abs > 1e6:
                print(f"\n[WARNING] Very large parameter at timestep {t}")
                print("Parameter:", name)
                print("Maximum absolute value:", max_abs)

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
        
        selected_action_probability = float(log_prob.detach().exp().item())
        # ========================================================
        # DIAGNOSTIC: SELECTED ACTION PROBABILITY
        # ========================================================
        
                
        # TEMPORARY DIAGNOSTIC 2: check log probability.
        if not torch.isfinite(log_prob).all():
            print(f"\n[FAIL] Invalid log probability at timestep {t}")
            print("Available actions:", available_actions)
            print("Intended action:", intended_action)
            print("Return G_t:", float(G_t))
            print("Log probability:", log_prob.detach().cpu())
            raise FloatingPointError(
                f"Invalid log probability at timestep {t}"
            )
        
    
        # TEMPORARY DIAGNOSTIC: ACTION GRADIENT SIGNAL
        with torch.no_grad():
            dist_before = policy.get_distribution(
                observation, available_actions
            )
            probs_before = dist_before.probs.detach().cpu().numpy()
        
        # END TEMPORARY DIAGNOSTIC
        # PyTorch minimizes losses, so negate the policy-gradient objective.
        loss = -(gamma ** t) * G_t * log_prob

        
        # TEMPORARY DIAGNOSTIC 3: check loss.
        if not torch.isfinite(loss).all():
            print(f"\n[FAIL] Invalid loss at timestep {t}")
            print("Return G_t:", float(G_t))
            print("Log probability:", float(log_prob.detach()))
            print("Loss:", float(loss.detach()))
            raise FloatingPointError(
                f"Invalid loss at timestep {t}"
            )


        optimizer.zero_grad()
        loss.backward()
        
        gradient_norm_squared = 0.0
        max_gradient = 0.0
        
        for name, parameter in policy.named_parameters():
            if parameter.grad is not None:
                grad = parameter.grad.detach()
        
                if not torch.isfinite(grad).all():
                    raise FloatingPointError(
                        f"Non-finite gradient at timestep {t}: {name}"
                    )
        
                gradient_norm_squared += (
                    torch.linalg.vector_norm(grad).item() ** 2
                )
                max_gradient = max(
                    max_gradient,
                    grad.abs().max().item(),
                )
        
        global_gradient_norm = math.sqrt(gradient_norm_squared)

        # ====================================================
        # DEBUG CODE 3 ENDED
        # ====================================================

        # ========================================================
        # DIAGNOSTIC: PARAMETERS BEFORE UPDATE
        # ========================================================
        
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in policy.named_parameters()
        }

        # ====================================================
        # DEBUG CODE 4 ENDED
        # ====================================================
        # TEMPORARY DIAGNOSTIC: PER-ACTION LOGIT GRADIENT
        if hasattr(policy, "linear"):
            output_layer = policy.linear
        elif hasattr(policy, "network"):
            output_layer = policy.network[-1]
        else:
            output_layer = None
        
        if output_layer is not None and output_layer.bias.grad is not None:
            print(
                f"[GRADIENT SIGNAL] t={t} | "
                f"action={intended_action} | "
                f"G={float(G_t):.4f} | "
                f"probabilities={np.round(probs_before, 4)} | "
                f"logit_bias_grad={output_layer.bias.grad.detach().cpu().numpy()}"
            )
        # END TEMPORARY DIAGNOSTIC
        optimizer.step()

        # DIAGNOSTIC: probability of the SAME state-action pair after update
        with torch.no_grad():
            distribution_after = policy.get_distribution(
                observation,
                available_actions,
            )

            action_index = int(intended_action)
            if not 0 <= action_index < distribution_after.probs.numel():
                raise ValueError(
                    f"Action index {action_index} outside probability vector "
                    f"of size {distribution_after.probs.numel()}"
                )
            probability_after = float(
                distribution_after.probs[action_index].item()
            )

        print(
            f"[ACTION PROBABILITY CHANGE] "
            f"t={t} | "
            f"before={selected_action_probability:.8f} | "
            f"after={probability_after:.8f} | "
            f"change={probability_after - selected_action_probability:+.8f}"
        )

        # ========================================================
        # DIAGNOSTIC: ACTUAL PARAMETER UPDATE MAGNITUDE
        # ========================================================
        
        update_norm_squared = 0.0
        maximum_parameter_update = 0.0
        maximum_parameter = 0.0
        
        for name, parameter in policy.named_parameters():
            before = parameters_before[name]
            after = parameter.detach()
        
            delta = after - before
        
            update_norm_squared += (
                torch.linalg.vector_norm(delta).item() ** 2
            )
        
            maximum_parameter_update = max(
                maximum_parameter_update,
                delta.abs().max().item(),
            )
        
            maximum_parameter = max(
                maximum_parameter,
                after.abs().max().item(),
            )
        
        global_update_norm = math.sqrt(update_norm_squared)
        # ====================================================
        # DEBUG CODE 5 ENDED
        # ====================================================
        

        # ========================================================
        # DIAGNOSTIC: COMPACT PER-STEP UPDATE SUMMARY
        # ========================================================
        
        print(
            f"[STEP {t:02d}] "
            f"G={float(G_t):.4f} | "
            f"gamma^t={gamma ** t:.4f} | "
            f"logP={float(log_prob.detach()):.4f} | "
            f"P(action)={selected_action_probability:.4f} | "
            f"grad={global_gradient_norm:.4e} | "
            f"update={global_update_norm:.4e} | "
            f"max|theta|={maximum_parameter:.4e}",
            flush=True,
        )
        # TEMPORARY DIAGNOSTIC 5: check parameters after update.
        for name, parameter in policy.named_parameters():
            if not torch.isfinite(parameter).all():
                print(f"\n[FAIL] Non-finite parameter AFTER optimizer.step()")
                print("Timestep:", t)
                print("Parameter:", name)
                print("NaN:", torch.isnan(parameter).sum().item())
                print("+Inf:", torch.isposinf(parameter).sum().item())
                print("-Inf:", torch.isneginf(parameter).sum().item())
                print("Return G_t:", float(G_t))
                print("Loss:", float(loss.detach()))
                raise FloatingPointError(
                    f"Non-finite parameter after update at timestep {t}"
                )


        losses.append(float(loss.detach().item()))

    return {
        "mean_loss": sum(losses) / len(losses),
        "num_updates": float(num_steps),
    }

