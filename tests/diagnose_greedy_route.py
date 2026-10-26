
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tsp.city_generator import CityLocationGenerator
from tsp.instance import TSPInstance
from tsp.env import TSPEnv
from tsp.cost import transition_cost, city_internal_cost

from examples.reinforce.config import ReinforceConfig
from examples.reinforce.train import train_reinforce


OPTIMAL_TOUR = [0, 3, 2, 1, 4, 0]
REVERSE_OPTIMAL_TOUR = [0, 4, 1, 2, 3, 0]


def build_instance(args):
    generator = CityLocationGenerator(
        width=args.width,
        height=args.height,
        n_cities=args.n_cities,
        seed=args.seed,
        user_nodes_per_city=args.user_nodes,
        city_shape=args.city_shape,
        city_radius=args.city_radius,
        city_size=args.city_size,
        city_width=args.city_width,
        city_height=args.city_height,
    )

    coordinates = generator.generate()
    cities = generator.get_cities()

    return TSPInstance(
        coordinates,
        name="generated_tsp",
        width=args.width,
        height=args.height,
        cities=cities,
    )


def get_cost_matrix(instance):
    n = instance.num_cities
    return np.array([
        [
            transition_cost(instance, i, j)
            for j in range(n)
        ]
        for i in range(n)
    ], dtype=float)


def get_internal_cost(instance):
    return sum(
        city_internal_cost(instance, city)
        for city in range(instance.num_cities)
    )


def inspect_greedy_route(policy, cost_matrix, internal_cost, start_city):
    n = policy.num_cities
    visited = {start_city}
    current_city = start_city
    route = [start_city]
    route_cost = 0.0

    print("\n" + "=" * 65)
    print("GREEDY POLICY ROUTE")
    print("=" * 65)
    print("Selection rule: highest-probability legal action")
    print("This analysis does not simulate environment slips.")

    policy.eval()

    with torch.no_grad():
        while len(visited) < n:
            observation = {
                "current_city": current_city,
                "visited_mask": [
                    int(city in visited) for city in range(n)
                ],
            }

            legal_actions = [
                city for city in range(n)
                if city not in visited
            ]

            distribution = policy.get_distribution(
                observation,
                legal_actions,
            )
            probabilities = distribution.probs.cpu().numpy()

            # Choose the highest-probability legal city.
            next_city = max(
                legal_actions,
                key=lambda action: float(probabilities[action]),
            )

            print(f"\nCurrent city: {current_city}")
            print("Visited cities:", sorted(visited))
            print("Action probabilities:")

            for action in legal_actions:
                marker = " <-- GREEDY CHOICE" if action == next_city else ""
                print(
                    f"  City {action}: "
                    f"{float(probabilities[action]):.6f}{marker}"
                )

            edge_cost = float(cost_matrix[current_city, next_city])
            route_cost += edge_cost

            print(
                f"Chosen transition: {current_city} -> {next_city}; "
                f"travel cost = {edge_cost:.6f}"
            )

            route.append(next_city)
            visited.add(next_city)
            current_city = next_city

        # Close the tour by returning to the starting city.
        close_observation = {
            "current_city": current_city,
            "visited_mask": [
                int(city in visited) for city in range(n)
            ],
        }

        close_distribution = policy.get_distribution(
            close_observation,
            [policy.close_action],
        )

        close_probability = float(
            close_distribution.probs[policy.close_action].item()
        )

        closing_cost = float(cost_matrix[current_city, start_city])
        route_cost += closing_cost
        route.append(start_city)

        print(f"\nCLOSE probability: {close_probability:.6f}")
        print(
            f"Closing transition: {current_city} -> {start_city}; "
            f"travel cost = {closing_cost:.6f}"
        )

    total_cost = route_cost + internal_cost

    print("\n" + "-" * 65)
    print("GREEDY ROUTE RESULTS")
    print("-" * 65)
    print("Closed route:", " -> ".join(map(str, route)))
    print(f"Travel cost: {route_cost:.6f}")
    print(f"Internal city cost: {internal_cost:.6f}")
    print(f"Deterministic total cost: {total_cost:.6f}")

    if route in [OPTIMAL_TOUR, REVERSE_OPTIMAL_TOUR]:
        print("Route matches one of the known optimal tours.")
    else:
        print("Route does not match either known optimal tour.")

    print(
        "\nNote: this is the deterministic cost of the greedy route. "
        "The stochastic environment can realize a different route "
        "when actions slip."
    )

    return route, total_cost


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh REINFORCE policy and inspect its greedy route "
            "without modifying existing project files."
        )
    )

    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--n-cities", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--city-shape",
        choices=["circle", "square", "rectangle"],
        required=True,
    )
    parser.add_argument("--user-nodes", type=int, required=True)
    parser.add_argument("--city-radius", type=float)
    parser.add_argument("--city-size", type=float)
    parser.add_argument("--city-width", type=float)
    parser.add_argument("--city-height", type=float)

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)

    args = parser.parse_args()

    if args.n_cities != 5:
        parser.error(
            "The known optimal tours in this script require five cities."
        )

    # Seed policy/training randomness, following the existing experiment.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    instance = build_instance(args)
    cost_matrix = get_cost_matrix(instance)
    internal_cost = get_internal_cost(instance)

    start_city = random.Random(args.seed).randrange(
        instance.num_cities
    )

    print("=" * 65)
    print("GREEDY ROUTE DIAGNOSTIC")
    print("=" * 65)
    print("Project root:", PROJECT_ROOT)
    print("Seed:", args.seed)
    print("Start city:", start_city)
    print("Training episodes:", args.episodes)
    print("Learning rate:", args.learning_rate)
    print("Gamma:", args.gamma)
    print("\nCost matrix:")
    print(cost_matrix)

    if start_city != 0:
        print(
            "\nWARNING: The known reference optimal tours start at city 0. "
            "This run starts elsewhere."
        )

    env = TSPEnv(instance)

    config = ReinforceConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        num_episodes=args.episodes,
        eval_episodes=30,
        log_interval=25,
        seed=args.seed,
    )

    print("\nTraining a fresh policy with the existing training function...")
    policy, history = train_reinforce(
        env=env,
        config=config,
        start_city=start_city,
    )

    inspect_greedy_route(
        policy=policy,
        cost_matrix=cost_matrix,
        internal_cost=internal_cost,
        start_city=start_city,
    )

    print("\nNo existing files were modified.")


if __name__ == "__main__":
    main()
