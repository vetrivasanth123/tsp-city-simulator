from __future__ import annotations

from typing import Any
import random

from .instance import TSPInstance
from .transition_kernel import TSPTransitionKernel


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
    
        self.transition_model = TSPTransitionKernel(
            instance=self.instance,
            rng=self._rng,
            kappa=self.kappa,
            beta=self.beta,
        )

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
        
    def step(self, intended_action: int) -> dict[str, Any]:
        """Execute an intended action through the stochastic transition kernel."""
    
        if self.done:
            raise RuntimeError(
                "Episode is already complete. Call reset()."
            )
    
        actual_action, transition_info = self.transition_model.sample(
            current_city=self.current_city,
            available_actions=self.available_actions(),
            intended_action=intended_action,
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
