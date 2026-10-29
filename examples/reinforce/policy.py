
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class BaseSoftmaxPolicy(nn.Module):
    """Shared action selection and masking for policy architectures."""

    def __init__(self, num_cities: int):
        super().__init__()

        if num_cities < 2:
            raise ValueError("num_cities must be at least 2.")

        self.num_cities = num_cities
        self.close_action = num_cities
        self.num_actions = num_cities + 1

    def encode_observation(self, observation: dict) -> torch.Tensor:
        """Encode current city and visited mask."""

        current_city = int(observation["current_city"])
        visited_mask = np.asarray(
            observation["visited_mask"], dtype=np.float32
        ).reshape(-1)

        if not 0 <= current_city < self.num_cities:
            raise ValueError("current_city is outside the valid range.")

        if visited_mask.shape != (self.num_cities,):
            raise ValueError("visited_mask has an incorrect length.")

        current_city_one_hot = np.zeros(
            self.num_cities, dtype=np.float32
        )
        current_city_one_hot[current_city] = 1.0

        return torch.as_tensor(
            np.concatenate([current_city_one_hot, visited_mask]),
            dtype=torch.float32,
        )

    def forward(self, observation: dict) -> torch.Tensor:
        raise NotImplementedError

    def get_distribution(
        self,
        observation: dict,
        available_actions: list[int],
    ) -> Categorical:
        """Create a categorical distribution over legal actions."""

        legal_actions = list(available_actions)

        if not legal_actions:
            legal_actions = [self.close_action]

        if len(set(legal_actions)) != len(legal_actions):
            raise ValueError("available_actions contains duplicates.")

        if any(
            not isinstance(action, (int, np.integer))
            or action < 0
            or action >= self.num_actions
            for action in legal_actions
        ):
            raise ValueError("available_actions contains an invalid action.")

        if self.close_action in legal_actions:
            if legal_actions != [self.close_action]:
                raise ValueError(
                    "CLOSE must be the only legal action when available."
                )

        logits = self.forward(observation)

        action_mask = torch.zeros(
            self.num_actions,
            dtype=torch.bool,
            device=logits.device,
        )
        action_mask[legal_actions] = True

        masked_logits = logits.masked_fill(
            ~action_mask, float("-inf")
        )

        
        # TEMPORARY MLP DIAGNOSTIC — remove after testing.
        if not torch.isfinite(logits).all():
            print("\n========== INVALID RAW LOGITS ==========")
            print("Policy class:", type(self).__name__)
            print("Phase:", getattr(self, "_debug_phase", "normal"))
            print("Observation:", observation)
            print("Available actions:", legal_actions)
            print("Raw logits:", logits.detach().cpu())
            print("Masked logits:", masked_logits.detach().cpu())

            for name, parameter in self.named_parameters():
                if not torch.isfinite(parameter).all():
                    print("NON-FINITE PARAMETER:", name)
                    print("NaN:", torch.isnan(parameter).sum().item())
                    print("+Inf:", torch.isposinf(parameter).sum().item())
                    print("-Inf:", torch.isneginf(parameter).sum().item())

            print("========================================")
            raise FloatingPointError("Non-finite raw policy logits")

        if not torch.isfinite(logits[action_mask]).all():
            print("Non-finite logits among legal actions:", legal_actions)
            raise FloatingPointError("Non-finite legal-action logits")

        return Categorical(logits=masked_logits)


    def select_action(
        self,
        observation: dict,
        available_actions: list[int],
    ) -> int:
        """Sample an intended action from the policy."""

        return int(
            self.get_distribution(
                observation, available_actions
            ).sample().item()
        )

    def log_probability(
        self,
        observation: dict,
        available_actions: list[int],
        intended_action: int,
    ) -> torch.Tensor:
        """Return log pi(action | state)."""

        distribution = self.get_distribution(
            observation, available_actions
        )

        action = torch.tensor(
            intended_action,
            dtype=torch.long,
            device=distribution.logits.device,
        )

        return distribution.log_prob(action)


class SoftmaxPolicy(BaseSoftmaxPolicy):
    """Original linear softmax policy: 2N inputs -> N+1 outputs."""

    def __init__(self, num_cities: int):
        super().__init__(num_cities)

        self.num_features = 2 * num_cities
        self.linear = nn.Linear(
            self.num_features, self.num_actions
        )

    def forward(self, observation: dict) -> torch.Tensor:
        features = self.encode_observation(observation)
        return self.linear(features)


class MLPSoftmaxPolicy(BaseSoftmaxPolicy):
    """Instance-specific MLP policy using the supplied cost matrix."""

    def __init__(
        self,
        num_cities: int,
        cost_matrix,
        start_city: int,
        hidden_sizes: tuple[int, int] = (128, 64),
    ):
        super().__init__(num_cities)

        if not 0 <= start_city < num_cities:
            raise ValueError("start_city is outside the valid range.")

        matrix = np.asarray(cost_matrix, dtype=np.float32)

        if matrix.shape != (num_cities, num_cities):
            raise ValueError(
                "cost_matrix must have shape (num_cities, num_cities)."
            )

        if not np.isfinite(matrix).all():
            raise ValueError("cost_matrix contains non-finite values.")

        if len(hidden_sizes) != 2 or any(
            size < 1 for size in hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain two positive sizes.")

        self.start_city = start_city
        self.num_features = num_cities**2 + 3 * num_cities

        # Store the existing instance matrix once; do not recalculate it.
        self.register_buffer(
            "cost_matrix_flat",
            torch.as_tensor(matrix.reshape(-1), dtype=torch.float32),
        )

        start_one_hot = np.zeros(num_cities, dtype=np.float32)
        start_one_hot[start_city] = 1.0
        self.register_buffer(
            "start_city_one_hot",
            torch.as_tensor(start_one_hot, dtype=torch.float32),
        )

        hidden1, hidden2 = hidden_sizes

        self.network = nn.Sequential(
            nn.Linear(self.num_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, self.num_actions),
        )

    def encode_observation(self, observation: dict) -> torch.Tensor:
        """Combine dynamic state features and fixed instance features."""

        state_features = super().encode_observation(observation)

        features = torch.cat(
            [
                state_features[:self.num_cities],  # Current city
                self.start_city_one_hot,            # Starting city
                state_features[self.num_cities:],   # Visited mask
                self.cost_matrix_flat,               # N x N cost matrix
            ]
        )

        if features.numel() != self.num_features:
            raise RuntimeError("Unexpected MLP input feature dimension.")

        return features

    def forward(self, observation: dict) -> torch.Tensor:
        features = self.encode_observation(observation)
        return self.network(features)




# ============================================================
# DEBUG CODE STARTED
# Purpose: Record compact policy decisions without extra
# forward passes or action samples.
# ============================================================

def debug_attach_policy_hooks(policy, debug_report):
    """Attach compact, observational debug wrappers to one policy."""
    import types

    debug_remove_policy_hooks(policy)

    policy._debug_report = debug_report

    original_methods = {
        "get_distribution": policy.get_distribution,
        "select_action": policy.select_action,
        "log_probability": policy.log_probability,
    }
    policy._debug_original_methods = original_methods

    def record(event):
        event["policy_type"] = debug_report.policy_type
        event["phase"] = getattr(policy, "_debug_phase", "training")
        debug_report.record("policy_events", event)

    def debug_get_distribution(
        self,
        observation,
        available_actions,
    ):
        # Preserve the original legality checks.
        legal_actions = list(available_actions)

        if not legal_actions:
            legal_actions = [self.close_action]

        if len(set(legal_actions)) != len(legal_actions):
            raise ValueError("available_actions contains duplicates.")

        if any(
            not isinstance(action, (int, np.integer))
            or action < 0
            or action >= self.num_actions
            for action in legal_actions
        ):
            raise ValueError("available_actions contains an invalid action.")

        if self.close_action in legal_actions:
            if legal_actions != [self.close_action]:
                raise ValueError(
                    "CLOSE must be the only legal action when available."
                )

        # Exactly one forward pass, as in the original method.
        logits = self.forward(observation)

        action_mask = torch.zeros(
            self.num_actions,
            dtype=torch.bool,
            device=logits.device,
        )
        action_mask[legal_actions] = True

        masked_logits = logits.masked_fill(
            ~action_mask, float("-inf")
        )
        distribution = Categorical(logits=masked_logits)

        # Record only the values needed to audit the distribution.
        record(
            {
                "event": "action_distribution",
                "observation": observation,
                "available_actions": legal_actions,
                "raw_logits": logits.detach().cpu().clone(),
                "action_mask": action_mask.detach().cpu().clone(),
                "masked_logits": masked_logits.detach().cpu().clone(),
                "probabilities": distribution.probs.detach().cpu().clone(),
            }
        )

        return distribution

    def debug_select_action(
        self,
        observation,
        available_actions,
    ):
        # The original method still performs exactly one sample.
        action = original_methods["select_action"](
            observation,
            available_actions,
        )

        record(
            {
                "event": "sampled_action",
                "observation": observation,
                "available_actions": list(available_actions),
                "intended_action": int(action),
            }
        )

        return action

    def debug_log_probability(
        self,
        observation,
        available_actions,
        intended_action,
    ):
        # Preserve the original log-probability computation.
        log_prob = original_methods["log_probability"](
            observation,
            available_actions,
            intended_action,
        )

        record(
            {
                "event": "log_probability",
                "observation": observation,
                "available_actions": list(available_actions),
                "intended_action": int(intended_action),
                "log_probability": log_prob.detach().cpu().clone(),
            }
        )

        return log_prob

    # Patch this policy instance only.
    policy.get_distribution = types.MethodType(
        debug_get_distribution, policy
    )
    policy.select_action = types.MethodType(
        debug_select_action, policy
    )
    policy.log_probability = types.MethodType(
        debug_log_probability, policy
    )

    return policy


def debug_remove_policy_hooks(policy):
    """Restore normal method lookup for the policy instance."""
    original_methods = getattr(policy, "_debug_original_methods", {})

    for name in original_methods:
        if name in policy.__dict__:
            delattr(policy, name)

    if hasattr(policy, "_debug_original_methods"):
        delattr(policy, "_debug_original_methods")

    if hasattr(policy, "_debug_report"):
        delattr(policy, "_debug_report")

    if hasattr(policy, "_debug_hook_handles"):
        policy._debug_hook_handles = []


# ============================================================
# DEBUG CODE ENDED
# ============================================================
