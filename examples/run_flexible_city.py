import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tsp.city_generator import CityLocationGenerator
from tsp.instance import TSPInstance
from tsp.env import TSPEnv
import tsp.visualization as visualization


def main():
    parser = argparse.ArgumentParser(description="Run a flexible-city TSP simulation.")

    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--n-cities", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)

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

    args = parser.parse_args()

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

    env = TSPEnv(instance)
    rng = random.Random(args.seed)
    start_city = rng.randrange(instance.num_cities)
    obs, info = env.reset(start_city=start_city)
    
    
    print("\nProject root:", PROJECT_ROOT)
    print("Instance:", instance.name)
    print("Grid:", f"{args.width} x {args.height}")
    print("Number of cities:", instance.num_cities)
    print("City shape:", args.city_shape)
    print("User nodes per city:", args.user_nodes)

    print("\nCities:")
    for city in cities:
        print(city)

    print("\nCost matrix:")
    print(instance.cost_matrix)

    print("\nInitial state")
    print("-------------")
    print("Start city:", info["start_city"])
    print("Current city:", info["current_city"])
    print("Visited:", info["tour"])
    print("Available actions:", info["available_actions"])

    trajectory = []

    print("\nAction sequence")
    print("---------------")

    while info["available_actions"]:
        current_city = info["current_city"]
        current_coordinate = instance.coordinates[current_city]
        available_actions = info["available_actions"]
    
        action = rng.choice(available_actions)

        obs, reward, terminated, truncated, info = env.step(action)
        
        transition = info["transition"]
        
        actual_action = transition["actual_action"]
        slipped = transition["slipped"]
        actual_probability = transition["transition_probabilities"][actual_action]
        
        print(
            f"Current city: {current_city} "
            f"({current_coordinate[0]:.4f}, {current_coordinate[1]:.4f}) | "
            f"Available actions: {available_actions} | "
            f"Intended: {action} | "
            f"Actual: {actual_action} | "
            f"Slip: {slipped} | "
            f"P(actual): {actual_probability:.4f}"
        )
        transition = info["transition"]
        
        trajectory.append({
            "current_city": int(current_city),
            "current_coordinate": [
                float(current_coordinate[0]),
                float(current_coordinate[1]),
            ],
            "city": {
                **cities[current_city],
                "pickup": {
                    "location": [
                        float(cities[current_city]["center"][0]),
                        float(cities[current_city]["center"][1]),
                    ],
                    "type": "temporary_center",
                },
            },
            "available_actions": list(transition["available_actions"]),
            "intended_action": int(transition["intended_action"]),
            "transition_probabilities": {
                str(k): float(v)
                for k, v in transition["transition_probabilities"].items()
            },
            "slip_probabilities": {
                str(k): float(v)
                for k, v in transition["slip_probabilities"].items()
            },
            "actual_action": int(transition["actual_action"]),
            "slipped": bool(transition["slipped"]),
            "step_cost": float(-reward),
            "reward": float(reward),
            "visited_mask": obs["visited_mask"].tolist(),
        })
    current_city = info["current_city"]
    current_coordinate = instance.coordinates[current_city]
    available_actions = info["available_actions"]

    _, reward, terminated, truncated, info = env.step(env.close_action)

    trajectory.append({
        "current_city": int(current_city),
        "current_coordinate": [
            float(current_coordinate[0]),
            float(current_coordinate[1]),
        ],
        "city": {
            **cities[current_city],
            "pickup": {
                "location": [
                    float(cities[current_city]["center"][0]),
                    float(cities[current_city]["center"][1]),
                ],
                "type": "temporary_center",
            },
        },
        "action": "CLOSE",
        "step_cost": float(-reward),
        "reward": float(reward),
    })

    print(
        f"Current city: {current_city} "
        f"({current_coordinate[0]:.4f}, {current_coordinate[1]:.4f}) | "
        f"Available actions: {available_actions} | "
        f"Action: CLOSE"
    )
    simulator = env.simulator
    closed_tour = simulator.tour + [simulator.start_city]

    print("\nFinal result")
    print("------------")
    print("Start city:", simulator.start_city)
    print("Tour:", closed_tour)
    print("Closed:", simulator.done)
    print("Total cost:", simulator.total_cost)
    print("Total reward:", sum(x["reward"] for x in trajectory))

    visualization.save_simulation(
        simulator,
        PROJECT_ROOT,
        trajectory,
    )

    print("\nSimulation saved:", PROJECT_ROOT / ".simulation.json")


if __name__ == "__main__":
    main()
