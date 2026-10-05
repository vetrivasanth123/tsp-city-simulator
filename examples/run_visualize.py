from pathlib import Path
import sys
import importlib

from IPython.display import HTML, display

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsp.instance import TSPInstance
import tsp.visualization as visualization

importlib.reload(visualization)


def main():
    saved = visualization.load_saved_simulation(PROJECT_ROOT)

    if saved is None:
        raise RuntimeError(
            "No simulation exists. Run run_flexible_city.py first."
        )

    instance_data = saved["instance"]
    trajectory = saved["trajectory"]
    summary = saved["summary"]

    instance = TSPInstance(
        instance_data["coordinates"],
        name=instance_data["name"],
        cost_matrix=instance_data["cost_matrix"],
        width=instance_data["width"],
        height=instance_data["height"],
        cities=instance_data.get("cities"),
    )
    actions = [
        step["actual_action"] if "actual_action" in step else step["action"]
        for step in trajectory
    ]
    
    rewards = [step["reward"] for step in trajectory]

    print("Visualizing saved simulation:")
    print("Saved instance:", instance.name)
    print("Grid:", f"{instance.width} x {instance.height}")
    print("Start city:", summary["start_city"])
    print("Actions:", actions)
    print("Tour:", summary["tour"])
    print("Total cost:", summary["total_cost"])
    print("Total reward:", summary["total_reward"])

    print("\nVisualization cost matrix:")
    print(instance.cost_matrix)

    _, animation = visualization.animate_simulation(
        instance,
        actions,
        summary["start_city"],
        rewards,
        trajectory,
    )

    display(HTML(animation.to_html5_video()))


if __name__ == "__main__":
    main()
