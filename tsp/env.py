
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .instance import TSPInstance
from .simulator import TSPSimulator
from .cost import transition_cost, city_internal_cost


class TSPEnv(gym.Env):
    """Gymnasium environment for TSP."""

    metadata = {"render_modes": []}

    def __init__(self, instance: TSPInstance, seed: int | None = None):
        super().__init__()

        self.instance = instance
        self.simulator = TSPSimulator(instance, seed=seed)
        self.total_cost = 0.0

        # Select the reward experiment:
        # "transition"       : transition cost only
        # "current_internal" : transition + current city internal cost
        # "next_internal"    : transition + destination city internal cost
        self.reward_mode = "next_internal"

        n = instance.num_cities

        self.close_action = n
        self.action_space = spaces.Discrete(n + 1)
        self.observation_space = spaces.Dict({
            "current_city": spaces.Discrete(n),
            "visited_mask": spaces.MultiBinary(n),
            "total_cost": spaces.Box(
                0.0, np.inf, shape=(1,), dtype=np.float32
            ),
        })

    def reset(self, *, seed=None, options=None, start_city=None):
        super().reset(seed=seed)

        if seed is not None:
            self.simulator._rng.seed(seed)

        state = self.simulator.reset(start_city=start_city)
        self.total_cost = 0.0

        return self._observation(state), self._info(state)

    def step(self, action):
        action = int(action)

        # --------------------------------------------------
        # CLOSE ACTION: return from final city to start city
        # --------------------------------------------------
        if action == self.close_action:

            if self.simulator.available_actions():
                raise ValueError(
                    "Cannot close the tour while valid actions remain."
                )

            current_city = self.simulator.current_city
            start_city = self.simulator.start_city

            state = self.simulator.close_tour()

            if current_city != start_city:

                transition_step_cost = transition_cost(
                    self.instance,
                    current_city,
                    start_city,
                )

                if self.reward_mode == "transition":

                    step_cost = transition_step_cost

                elif self.reward_mode == "current_internal":

                    internal_step_cost = city_internal_cost(
                        self.instance,
                        current_city,
                    )

                    step_cost = (
                        transition_step_cost + internal_step_cost
                    )

                elif self.reward_mode == "next_internal":

                    internal_step_cost = city_internal_cost(
                        self.instance,
                        start_city,
                    )

                    step_cost = (
                        transition_step_cost + internal_step_cost
                    )

                else:
                    raise ValueError(
                        f"Unknown reward mode: {self.reward_mode}"
                    )

            else:
                step_cost = 0.0

            self.total_cost += step_cost
            reward = -step_cost

            return (
                self._observation(state),
                float(reward),
                True,
                False,
                self._info(state),
            )

        # --------------------------------------------------
        # NORMAL CITY-TO-CITY MOVEMENT
        # --------------------------------------------------
        if action not in self.simulator.available_actions():
            raise ValueError(f"Invalid action: {action}")

        current_city = self.simulator.current_city

        state = self.simulator.step(action)

        # Use actual city reached, accounting for action slips.
        actual_city = state["current_city"]

        transition_step_cost = transition_cost(
            self.instance,
            current_city,
            actual_city,
        )

        if self.reward_mode == "transition":

            step_cost = transition_step_cost

        elif self.reward_mode == "current_internal":

            internal_step_cost = city_internal_cost(
                self.instance,
                current_city,
            )

            step_cost = (
                transition_step_cost + internal_step_cost
            )

        elif self.reward_mode == "next_internal":

            internal_step_cost = city_internal_cost(
                self.instance,
                actual_city,
            )

            step_cost = (
                transition_step_cost + internal_step_cost
            )

        else:
            raise ValueError(
                f"Unknown reward mode: {self.reward_mode}"
            )

        self.total_cost += step_cost
        reward = -step_cost

        info = self._info(state)
        info["transition"] = state["transition_info"]

        return (
            self._observation(state),
            float(reward),
            False,
            False,
            info,
        )

    def _observation(self, state):
        mask = np.zeros(self.instance.num_cities, dtype=np.int8)
        mask[state["visited"]] = 1

        return {
            "current_city": state["current_city"],
            "visited_mask": mask,
            "total_cost": np.array(
                [self.total_cost], dtype=np.float32
            ),
        }

    def _info(self, state):
        return {
            "tour": list(state["tour"]),
            "start_city": state["start_city"],
            "current_city": state["current_city"],
            "available_actions": list(state["available_actions"]),
            "total_cost": float(self.total_cost),
        }
