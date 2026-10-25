
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tsp.city_generator import CityLocationGenerator
from tsp.instance import TSPInstance
from tsp.env import TSPEnv
from tsp.cost import transition_cost, city_internal_cost
import tsp.visualization as visualization

from examples.reinforce.config import ReinforceConfig
from examples.reinforce.train import train_reinforce
from examples.reinforce.evaluate import evaluate_policy


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate REINFORCE on the TSP environment."
    )

    # Reuse the Random runner's instance-generation arguments.
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

    # REINFORCE-specific settings.
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)

    args = parser.parse_args()

    # 1. Generate the instance using the existing project module.
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

    # Keep the same environment-construction behavior as the Random runner.
    env = TSPEnv(instance)
    print("\nREINFORCE experiment")
    print("--------------------")
    print_instance_details(instance, cities, PROJECT_ROOT)
    
    # Match the Random runner's start-city selection rule.
    start_city = random.Random(args.seed).randrange(instance.num_cities)

    config = ReinforceConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        num_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        log_interval=25,
        seed=args.seed,
    )

    # 2. Train on the existing TSP environment.
    policy, history = train_reinforce(
        env=env,
        config=config,
        start_city=start_city,
    )

    # 3. Evaluate the trained policy without updating its parameters.
    evaluation = evaluate_policy(
        env=env,
        policy=policy,
        start_city=start_city,
        num_episodes=config.eval_episodes,
        seed=config.seed,
    )

    print("\nEvaluation results")
    print("------------------")
    print("Estimated J(theta):", evaluation["estimated_J_theta"])
    print("Mean episode return:", evaluation["mean_return"])
    print("Mean episode cost:", evaluation["mean_cost"])
    print("Episode-cost standard deviation:", evaluation["std_cost"])

    # 4. Generate one complete policy rollout for visualization.
    # This is one tour, not the mean evaluation result.
    torch.manual_seed(config.seed + 1)
    policy.eval()

    observation, info = env.reset(start_city=start_city)
    trajectory = []
    terminated = False

    while not terminated:
        current_city = int(info["current_city"])
        coordinate = instance.coordinates[current_city]
        available_actions = list(info["available_actions"])

        intended_action = policy.select_action(
            observation,
            available_actions,
        )

        next_observation, reward, terminated, truncated, next_info = (
            env.step(intended_action)
        )

        if truncated:
            raise RuntimeError("The visualization episode was truncated.")

        if intended_action == env.close_action:
            # Preserve the existing runner's CLOSE record structure.
            trajectory.append({
                "current_city": current_city,
                "current_coordinate": [
                    float(coordinate[0]),
                    float(coordinate[1]),
                ],
                "city": {
                    **cities[current_city],
                    "pickup": {
                        "location": [
                            float(cities[current_city]["pickup"]["location"][0]),
                            float(cities[current_city]["pickup"]["location"][1]),
                        ],
                        "type": cities[current_city]["pickup"]["type"],
                    },
                },
                "action": "CLOSE",
                "transition_cost": float(
                    transition_cost(
                        instance,
                        current_city,
                        env.simulator.start_city,
                    )
                ),
                "internal_cost": float(
                    city_internal_cost(instance, current_city)
                ),
                "step_cost": float(-reward),
                "reward": float(reward),
            })
        else:
            # Reuse the transition details supplied by the existing environment.
            transition = next_info["transition"]
            actual_action = int(transition["actual_action"])

            trajectory.append({
                "current_city": current_city,
                "current_coordinate": [
                    float(coordinate[0]),
                    float(coordinate[1]),
                ],
                "city": {
                    **cities[current_city],
                    "pickup": {
                        "location": [
                            float(cities[current_city]["pickup"]["location"][0]),
                            float(cities[current_city]["pickup"]["location"][1]),
                        ],
                        "type": cities[current_city]["pickup"]["type"],
                    },
                },
                "available_actions": list(available_actions),
                "intended_action": int(intended_action),
                "transition_probabilities": {
                    str(k): float(v)
                    for k, v in transition[
                        "transition_probabilities"
                    ].items()
                },
                "slip_probabilities": {
                    str(k): float(v)
                    for k, v in transition["slip_probabilities"].items()
                },
                "actual_action": actual_action,
                "slipped": bool(transition["slipped"]),
                "transition_cost": float(
                    transition_cost(instance, current_city, actual_action)
                ),
                "internal_cost": float(
                    city_internal_cost(instance, current_city)
                ),
                "step_cost": float(-reward),
                "reward": float(reward),
                "visited_mask": next_observation["visited_mask"].tolist(),
            })

        observation, info = next_observation, next_info

    simulator = env.simulator
    closed_tour = simulator.tour + [simulator.start_city]

    print("\nSaved visualization rollout")
    print("---------------------------")
    print("Start city:", simulator.start_city)
    print("Tour:", closed_tour)
    print("Total cost:", env.total_cost)
    print("Total reward:", sum(step["reward"] for step in trajectory))

    
    print("\nInitial state")
    print("-------------")
    print("Start city:", simulator.start_city)
    print("Current city:", simulator.start_city)
    print("Visited:", [simulator.start_city])
    print(
        "Available actions:",
        list(range(instance.num_cities))[:simulator.start_city]
        + [
            i for i in range(instance.num_cities)
            if i != simulator.start_city
        ],
    )

    print("\nAction sequence")
    print("---------------")
    for step in trajectory:
        current = step["current_city"]
        x, y = step["current_coordinate"]

        if step.get("action") == "CLOSE":
            print(f"Current city: {current} ({x:.4f}, {y:.4f}) | Action: CLOSE")
        else:
            print(
                f"Current city: {current} ({x:.4f}, {y:.4f}) | "
                f"Available actions: {step['available_actions']} | "
                f"Intended: {step['intended_action']} | "
                f"Actual: {step['actual_action']} | "
                f"Slip: {step['slipped']} | "
                f"P(actual): "
                f"{step['transition_probabilities'].get(str(step['actual_action']), 0.0):.4f}"
            )

    print("\nFinal result")
    print("------------")
    print("Start city:", simulator.start_city)
    print("Tour:", closed_tour)
    print("Closed:", closed_tour[0] == closed_tour[-1])
    print("Total cost:", env.total_cost)
    print("Total reward:", sum(step["reward"] for step in trajectory))

    print("\nInternal cost details")
    print("---------------------")
    total_internal_cost = 0.0

    for city in cities:
        city_id = city["city_id"]
        pickup = city["pickup"]["location"]
        internal_cost = city_internal_cost(instance, city_id)
        total_internal_cost += internal_cost

        print(
            f"City {city_id} | Pickup: {pickup} | "
            f"Total internal cost: {internal_cost:.6f}"
        )

        # Show each user's distance to the pickup point.
        for user_idx, node in enumerate(city["user_nodes"]):
            distance = float(np.linalg.norm(
                np.asarray(node, dtype=float)
                - np.asarray(pickup, dtype=float)
            ))
            print(f"  User {user_idx} → Pickup: {distance:.6f}")

    print(f"Total internal cost: {total_internal_cost}")


    # 5. Use the existing visualization serializer and schema.
    visualization.save_simulation(
        simulator,
        PROJECT_ROOT,
        trajectory,
        env.total_cost,
    )

    print("\nSimulation saved:", PROJECT_ROOT / ".simulation.json")


def print_instance_details(instance, cities, project_root):
    import numpy as np

    print(f"Project root: {project_root}")
    print(f"Instance: {instance.name}")
    print(f"Grid: {instance.width} x {instance.height}")
    print(f"Number of cities: {instance.num_cities}")
    print(f"City shape: {cities[0]['shape']}")
    print(f"User nodes per city: {len(cities[0]['user_nodes'])}")

    print("\nCities:")
    for city in cities:
        print(city)

    cost_matrix = np.array([
        [
            transition_cost(instance, i, j)
            for j in range(instance.num_cities)
        ]
        for i in range(instance.num_cities)
    ])

    print("\nCost matrix:")
    print(cost_matrix)

if __name__ == "__main__":
    main()
