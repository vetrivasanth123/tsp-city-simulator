from __future__ import annotations

from typing import Any
import random
import math

from .instance import TSPInstance


class TSPSimulator:
    """Simulator for constructing tours using edge costs."""

    def __init__(
        self,
        instance: TSPInstance,
        seed: int | None = None,
        kappa: float = 0.9,
        beta: float = 0.9,
    ) -> None:
        self.instance = instance
        self._rng = random.Random(seed)
        self.kappa = kappa
        self.beta = beta
        if not 0 < self.kappa <= 1:
            raise ValueError("kappa must satisfy 0 < kappa <= 1.")
        
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must satisfy 0 <= beta <= 1.")

    def reset(self, start_city: int | None = None) -> dict[str, Any]:
        """Start a new episode at the specified start city."""
    
        if start_city is None:
            raise ValueError("start_city must be provided.")
    
        self._validate_city(start_city)
        self.start_city = start_city
        self.tour = [self.start_city]
        self.current_city = self.start_city
        self.total_cost = 0.0
        self.done = False

        return self.state()

    @property
    def total_distance(self) -> float:
        """Backward-compatible alias for total_cost."""
        return self.total_cost

    def available_actions(self) -> list[int]:
        """Return unvisited cities that can be selected."""

        if self.done:
            return []

        visited = set(self.tour)

        return [
            city
            for city in range(self.instance.num_cities)
            if city not in visited
        ]
        
    def transition_kernel(
        self,
        intended_action: int,
    ) -> tuple[int, dict[str, Any]]:
        """Sample the actual next city using the stochastic transition kernel."""

        if self.done:
            raise RuntimeError(
                "Episode is already complete. Call reset()."
            )

        available = self.available_actions()

        if intended_action not in available:
            raise ValueError(
                f"Intended action {intended_action} is not available. "
                f"Available actions: {available}"
            )

        current_location = self.instance.cities[
            self.current_city
        ]["facility"]["location"]

        # Calculate slip weights only for alternatives to the intended action.
        slip_candidates = [
            candidate for candidate in available
            if candidate != intended_action
        ]
        
        weights = {}
        
        for candidate in slip_candidates:
            candidate_location = self.instance.cities[
                candidate
            ]["facility"]["location"]
        
            squared_distance = sum(
                (candidate_location[i] - current_location[i]) ** 2
                for i in range(len(current_location))
            )
        
            weights[candidate] = math.exp(
                -self.beta * squared_distance
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
                    else (1.0 - self.kappa) * slip_probabilities[candidate]
                )
                for candidate in available
            }
        
        probability_sum = sum(transition_probabilities.values())
        
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
            "current_city": self.current_city,
            "available_actions": list(available),
            "intended_action": intended_action,
            "slip_probabilities": slip_probabilities,
            "transition_probabilities": transition_probabilities,
            "actual_action": actual_action,
            "slipped": actual_action != intended_action,
        }

        return actual_action, transition_info  
        
    def step(self, intended_action: int) -> dict[str, Any]:
        """Execute an intended action through the stochastic transition kernel."""
    
        if self.done:
            raise RuntimeError(
                "Episode is already complete. Call reset()."
            )
    
        actual_action, transition_info = self.transition_kernel(
            intended_action
        )
    
        step_cost = self.instance.cost(
            self.current_city,
            actual_action,
        )
    
        self.total_cost += step_cost
    
        self.tour.append(actual_action)
        self.current_city = actual_action
    
        state = self.state()
        state["transition_info"] = transition_info
    
        return state
        
    def close_tour(self) -> dict[str, Any]:
        """Return to the starting city and complete the tour."""

        if not self.tour:
            raise ValueError("Cannot close an empty tour.")

        if self.done:
            return self.state()

        if self.available_actions():
            raise RuntimeError(
                "Cannot close the tour while valid actions remain."
            )

        if len(self.tour) > 1:
            return_cost = self.instance.cost(
                self.current_city,
                self.start_city,
            )
            
            self.total_cost += return_cost
            self.current_city = self.start_city
        
        self.done = True

        return self.state()

    def state(self) -> dict[str, Any]:
        """Return the current simulator state."""

        return {
            "tour": list(self.tour),
            "start_city": self.start_city,
            "current_city": self.current_city,
            "visited": list(self.tour),
            "available_actions": self.available_actions(),
            "total_cost": self.total_cost,
            "total_distance": self.total_distance,
            "done": self.done,
        }

    def _validate_city(self, city: int) -> None:
        if not isinstance(city, int):
            raise TypeError("City index must be an integer.")

        if not 0 <= city < self.instance.num_cities:
            raise IndexError(
                f"City index {city} is out of range for "
                f"{self.instance.num_cities} cities."
            )
