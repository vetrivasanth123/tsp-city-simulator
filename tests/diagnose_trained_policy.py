
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
from tsp.cost import transition_cost

from examples.reinforce.config import ReinforceConfig
from examples.reinforce.train import train_reinforce
from examples.reinforce.evaluate import evaluate_policy


# These tours are optimal for the previously examined five-city
# cost matrix. We verify the matrix before treating them as optimal.
EXPECTED_COST_MATRIX = np.array([
    [0.0, 5.79520087, 7.87832558, 2.00174506, 2.68592220],
    [5.79520087, 0.0, 4.62742818, 4.22168022, 3.77103595],
    [7.87832558, 4.62742818, 0.0, 5.88326092, 7.26202349],
    [2.00174506, 4.22168022, 5.88326092, 0.0, 2.51588378],
    [2.68592220, 3.77103595, 7.26202349, 2.51588378, 0.0],
])

REFERENCE_TOURS = [
    [0, 3, 2, 1, 4, 0],
    [0, 4, 1, 2, 3, 0],
]


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

    instance = TSPInstance(
        coordinates,
        name="generated_tsp",
        width=args.width,
        height=args.height,
        cities=cities,
    )

    return instance


def get_cost_matrix(instance):
    n = instance.num_cities
    return np.array([
        [transition_cost(instance, i, j) for j in range(n)]
        for i in range(n)
    ], dtype=float)


def inspect_tour_probabilities(policy, tour):
    """Inspect the trained policy along one reference tour."""
    n = policy.num_cities
    close_action = policy.close_action
    visited = {tour[0]}

    optimal_actions_are_top = []
    optimal_action_probabilities = []

    print("\n" + "=" * 65)
    print("Reference tour:", " -> ".join(map(str, tour)))
    print("=" * 65)

    policy.eval()

    with torch.no_grad():
        # Inspect each city-selection decision.
        for i in range(1, n):
            current_city = tour[i - 1]
            optimal_next_city = tour[i]

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
                observation, legal_actions
            )
            probabilities = distribution.probs.cpu().numpy()

            optimal_probability = float(
                probabilities[optimal_next_city]
            )
            max_probability = max(
                float(probabilities[action])
                for action in legal_actions
            )

            is_top_action = np.isclose(
                optimal_probability, max_probability, atol=1e-7
            )

            optimal_actions_are_top.append(bool(is_top_action))
            optimal_action_probabilities.append(optimal_probability)

            print(
                f"\nCurrent city: {current_city}"
                f" | Visited: {sorted(visited)}"
            )
            print(f"Reference next city: {optimal_next_city}")

            for action in legal_actions:
                probability = float(probabilities[action])
                marker = ""

                if action == optimal_next_city:
                    marker = " <-- reference next action"
                elif np.isclose(
                    probability, max_probability, atol=1e-7
                ):
                    marker = " <-- tied for highest probability"

                print(f"  Action {action}: {probability:.6f}{marker}")

            print(
                "Reference action has highest probability:",
                "YES" if is_top_action else "NO",
            )

            visited.add(optimal_next_city)

        # Once all cities are visited, CLOSE should be the only legal action.
        final_city = tour[n - 1]
        observation = {
            "current_city": final_city,
            "visited_mask": [
                int(city in visited) for city in range(n)
            ],
        }

        close_distribution = policy.get_distribution(
            observation, [close_action]
        )
        close_probability = float(
            close_distribution.probs[close_action].item()
        )

        print(f"\nCLOSE action probability: {close_probability:.6f}")

    print("\nTour probability summary")
    print("------------------------")
    print(
        "Reference actions with highest probability:",
        f"{sum(optimal_actions_are_top)}/{len(optimal_actions_are_top)}",
    )
    print(
        "Mean probability of reference next action:",
        f"{np.mean(optimal_action_probabilities):.6f}",
    )
    print(
        "Reference action probabilities:",
        [round(p, 6) for p in optimal_action_probabilities],
    )

    return {
        "top_action_count": sum(optimal_actions_are_top),
        "decision_count": len(optimal_actions_are_top),
        "mean_reference_probability": float(
            np.mean(optimal_action_probabilities)
        ),
        "reference_probabilities": optimal_action_probabilities,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh REINFORCE policy and inspect its action "
            "probabilities along reference TSP tours."
        )
    )

    # Keep instance-generation arguments consistent with run_reinforce.py.
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
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)

    args = parser.parse_args()

    if args.n_cities != 5:
        parser.error(
            "These reference tours and expected cost matrix require "
            "--n-cities 5."
        )

    # Reproduce the same project instance and start-city rule.
    instance = build_instance(args)
    cost_matrix = get_cost_matrix(instance)

    start_city = random.Random(args.seed).randrange(
        instance.num_cities
    )

    print("\n" + "=" * 65)
    print("REINFORCE TRAINED-POLICY DIAGNOSTIC")
    print("=" * 65)
    print("Project root:", PROJECT_ROOT)
    print("Number of cities:", instance.num_cities)
    print("Seed:", args.seed)
    print("Start city:", start_city)
    print("Training episodes:", args.episodes)
    print("Evaluation episodes:", args.eval_episodes)
    print("Learning rate:", args.learning_rate)
    print("Gamma:", args.gamma)

    same_matrix = np.allclose(
        cost_matrix,
        EXPECTED_COST_MATRIX,
        rtol=1e-6,
        atol=1e-6,
    )

    print("\nMatches previously examined cost matrix:", same_matrix)

    if not same_matrix:
        print(
            "WARNING: The generated cost matrix differs from the "
            "reference matrix. The listed tours may not be optimal "
            "for this instance. Use the exact instance-generation "
            "arguments from your earlier experiment."
        )

    if start_city != 0:
        print(
            "\nWARNING: The selected start city is not 0. "
            "The reference tours start at city 0, so their full "
            "closed-tour interpretation does not match this run."
        )

    env = TSPEnv(instance)

    config = ReinforceConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        num_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        log_interval=25,
        seed=args.seed,
    )

    print("\nTraining a fresh policy using existing project code...")

    policy, history = train_reinforce(
        env=env,
        config=config,
        start_city=start_city,
    )

    print("\nEvaluating the trained policy...")

    evaluation = evaluate_policy(
        env=env,
        policy=policy,
        start_city=start_city,
        num_episodes=config.eval_episodes,
        seed=config.seed,
    )

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print("Estimated J(theta):", evaluation["estimated_J_theta"])
    print("Mean episode return:", evaluation["mean_return"])
    print("Mean episode cost:", evaluation["mean_cost"])
    print(
        "Episode-cost standard deviation:",
        evaluation["std_cost"],
    )

    if same_matrix and start_city == 0:
        results = []

        for tour in REFERENCE_TOURS:
            results.append(
                inspect_tour_probabilities(policy, tour)
            )

        print("\n" + "=" * 65)
        print("FINAL DIAGNOSTIC SUMMARY")
        print("=" * 65)

        for tour, result in zip(REFERENCE_TOURS, results):
            print(
                f"Tour {' -> '.join(map(str, tour))}: "
                f"{result['top_action_count']}/"
                f"{result['decision_count']} reference actions "
                "have the highest probability; "
                f"mean reference-action probability = "
                f"{result['mean_reference_probability']:.6f}"
            )

        print(
            "\nInterpretation: this measures the policy's preferences "
            "at the reference tour states. It does not guarantee that "
            "the full tour will be executed, because action sampling "
            "and environment slips can change the realized route."
        )
    else:
        print(
            "\nSkipped the optimal-tour comparison because the "
            "instance or start city does not match the reference setup."
        )

    print("\nDiagnostic finished. No existing project files were modified.")


if __name__ == "__main__":
    main()
