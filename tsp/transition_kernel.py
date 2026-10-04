from __future__ import annotations

from typing import Any
import random
import math

from .instance import TSPInstance


class TSPTransitionKernel:
    """Stochastic transition model for TSP city-to-city movement."""

    def __init__(
        self,
        instance: TSPInstance,
        rng: random.Random,
        kappa: float = 0.9,
        beta: float = 0.9,
    ) -> None:
        self.instance = instance
        self._rng = rng
        self.kappa = kappa
        self.beta = beta

        if not 0 < self.kappa <= 1:
            raise ValueError("kappa must satisfy 0 < kappa <= 1.")

        if not 0 <= self.beta <= 1:
            raise ValueError("beta must satisfy 0 <= beta <= 1.")

    def sample(
        self,
        current_city: int,
        available_actions: list[int],
        intended_action: int,
    ) -> tuple[int, dict[str, Any]]:
        """Sample the actual next city using the stochastic transition model."""
        print(">>> USING TSPTransitionKernel.sample() FROM transition_kernel.py")

        if intended_action not in available_actions:
            raise ValueError(
                f"Intended action {intended_action} is not available. "
                f"Available actions: {available_actions}"
            )

        current_location = self.instance.cities[
            current_city
        ]["facility"]["location"]

        # Calculate slip weights only for alternatives to the intended action.
        slip_candidates = [
            candidate
            for candidate in available_actions
            if candidate != intended_action
        ]

        weights = {}

        squared_distances = {}

        for candidate in slip_candidates:
            candidate_location = self.instance.cities[
                candidate
            ]["facility"]["location"]

            squared_distance = sum(
                (candidate_location[i] - current_location[i]) ** 2
                for i in range(len(current_location))
            )

            squared_distances[candidate] = squared_distance

        if slip_candidates:
            min_squared_distance = min(squared_distances.values())

            for candidate in slip_candidates:
                weights[candidate] = math.exp(
                    -self.beta
                    * (
                        squared_distances[candidate]
                        - min_squared_distance
                    )
                )

        if slip_candidates:
            denominator = sum(weights.values())

            slip_probabilities = {
                candidate: weights[candidate] / denominator
                for candidate in slip_candidates
            }
        else:
            slip_probabilities = {}

        # Complete transition distribution.
        if not slip_candidates:
            transition_probabilities = {
                intended_action: 1.0
            }
        else:
            transition_probabilities = {
                candidate: (
                    self.kappa
                    if candidate == intended_action
                    else (1.0 - self.kappa)
                    * slip_probabilities[candidate]
                )
                for candidate in available_actions
            }

        probability_sum = sum(
            transition_probabilities.values()
        )

        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"Transition probabilities must sum to 1. "
                f"Got {probability_sum}."
            )

        # Sample the actual action from the transition distribution.
        actual_action = self._rng.choices(
            population=list(transition_probabilities.keys()),
            weights=list(transition_probabilities.values()),
            k=1,
        )[0]

        transition_info = {
            "current_city": current_city,
            "available_actions": list(available_actions),
            "intended_action": intended_action,
            "slip_probabilities": slip_probabilities,
            "transition_probabilities": transition_probabilities,
            "actual_action": actual_action,
            "slipped": actual_action != intended_action,
        }

        return actual_action, transition_info
