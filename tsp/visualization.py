from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

from .utils import tour_cost


def plot_cities(instance, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    # New spatial city visualization
    if instance.cities is not None:
        for city in instance.cities:
            center = np.asarray(city["center"], dtype=float)
            shape = city["shape"]
            dimensions = city["dimensions"]

            if shape == "circle":
                patch = Circle(
                    center,
                    dimensions["radius"],
                    fill=False,
                    linewidth=2,
                )

            elif shape in ("square", "rectangle"):
                width = dimensions.get("width", dimensions.get("size"))
                height = dimensions.get("height", dimensions.get("size"))

                patch = Rectangle(
                    (center[0] - width / 2, center[1] - height / 2),
                    width,
                    height,
                    fill=False,
                    linewidth=2,
                )

            else:
                raise ValueError(f"Unsupported city shape: {shape}")

            ax.add_patch(patch)
            ax.annotate(
                str(city["city_id"]),
                center,
                xytext=(7, 7),
                textcoords="offset points",
            )

            # User nodes
            user_nodes = city.get("user_nodes", [])
            if user_nodes:
                user_xy = np.asarray(user_nodes, dtype=float)
                ax.scatter(
                    user_xy[:, 0],
                    user_xy[:, 1],
                    s=35,
                    marker="o",
                    zorder=3,
                )

            # Pickup
            pickup = city.get("pickup")
            if pickup is not None:
                fx, fy = pickup["location"]
                ax.scatter(
                    fx, fy,
                    s=130,
                    marker="*",
                    zorder=6,
                )

    else:
        # Backward-compatible old point-city visualization
        xy = np.asarray(instance.coordinates, dtype=float)
        ax.scatter(xy[:, 0], xy[:, 1], s=100, zorder=3)

        for i, (x, y) in enumerate(xy):
            ax.annotate(
                str(i),
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")

    if instance.width is not None:
        ax.set_xlim(0, instance.width)
    if instance.height is not None:
        ax.set_ylim(0, instance.height)

    ax.grid(True, alpha=0.3)
    return ax

def plot_tour(instance, tour, ax=None, title="TSP Tour"):
    tour = list(tour)

    if len(tour) != instance.num_cities:
        raise ValueError("Tour must contain every city exactly once.")

    ax = plot_cities(instance, ax)
    route = tour + [tour[0]]
    if instance.cities is not None:
        xy = np.asarray(
            [
                instance.cities[i]["pickup"]["location"]
                for i in route
            ],
            dtype=float,
        )
    else:
        xy = np.asarray(instance.coordinates)[route]

    ax.plot(xy[:, 0], xy[:, 1], marker="o", linewidth=2)
    ax.set_title(f"{title} — Cost: {tour_cost(tour, instance):.4f}")

    return ax


def save_simulation(simulator, project_root, trajectory, total_cost):
    path = Path(project_root) / ".simulation.json"
    instance = simulator.instance
    rewards = [float(step["reward"]) for step in trajectory]

    data = { 
        "instance": {
            "name": instance.name,
            "coordinates": instance.coordinates.tolist(),
            "cost_matrix": instance.cost_matrix.tolist(),
            "width": instance.width,
            "height": instance.height,
            "cities": instance.cities,
        },

        "trajectory": trajectory,
        "summary": {
            "start_city": int(simulator.start_city),
            "tour": simulator.tour + [simulator.start_city],
            "total_cost": float(total_cost),
            "total_reward": float(sum(rewards)),
        },
    }

    def format_json(obj, level=0):
        indent = "  " * level
        child_indent = "  " * (level + 1)

        if isinstance(obj, dict):
            if not obj:
                return "{}"

            lines = ["{"]
            items = list(obj.items())

            for i, (key, value) in enumerate(items):
                formatted_value = format_json(value, level + 1)
                comma = "," if i < len(items) - 1 else ""

                lines.append(
                    f'{child_indent}{json.dumps(key)}: '
                    f'{formatted_value}{comma}'
                )

            lines.append(f"{indent}}}")
            return "\n".join(lines)

        if isinstance(obj, list):
            if not obj:
                return "[]"

            # Keep simple lists such as visited_mask and tour on one line.
            if all(not isinstance(item, (list, dict)) for item in obj):
                return json.dumps(obj, separators=(", ", ": "))

            # Keep coordinate pairs and matrix rows compact.
            if all(
                isinstance(item, list)
                and all(not isinstance(x, (list, dict)) for x in item)
                for item in obj
            ):
                lines = ["["]

                for i, item in enumerate(obj):
                    comma = "," if i < len(obj) - 1 else ""
                    lines.append(
                        f"{child_indent}"
                        f"{json.dumps(item, separators=(', ', ': '))}"
                        f"{comma}"
                    )

                lines.append(f"{indent}]")
                return "\n".join(lines)

            # General nested-list handling.
            lines = ["["]

            for i, item in enumerate(obj):
                comma = "," if i < len(obj) - 1 else ""
                lines.append(
                    f"{child_indent}{format_json(item, level + 1)}{comma}"
                )

            lines.append(f"{indent}]")
            return "\n".join(lines)

        return json.dumps(obj)

    path.write_text(
        format_json(data),
        encoding="utf-8",
    )



def load_saved_simulation(project_root):
    path = Path(project_root) / ".simulation.json"

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def animate_simulation(
    instance,
    actions,
    start_city,
    rewards,
    trajectory,
    interval=180,
    frames_per_move=12,
):
    if instance.cities is not None:
        xy = np.asarray(
            [
                city["pickup"]["location"]
                for city in instance.cities
            ],
            dtype=float,
        )
    else:
        xy = np.asarray(instance.coordinates, dtype=float)

    fig, (summary_ax, ax) = plt.subplots(
        1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [1, 3]}
    )
    summary_ax.axis("off")
    plot_cities(instance, ax)

    blue, green, red, gray, edge = (
        "#2563EB", "#16A34A", "#DC2626", "#D1D5DB", "#374151"
    )
    final_city = next((a for a in reversed(actions) if a != "CLOSE"), start_city)

    ax.scatter([xy[start_city, 0]], [xy[start_city, 1]], s=280, marker="*", color=green, edgecolors=edge, linewidths=1.2, zorder=8)

    if final_city != start_city:
        ax.scatter([xy[final_city, 0]], [xy[final_city, 1]], s=280, marker="*", color=red, edgecolors=edge, linewidths=1.2, zorder=8)

    line, = ax.plot([], [], linewidth=2.8, color=blue)
    current, = ax.plot([], [], "o", markersize=12, color=green, zorder=7)
    summary = summary_ax.text(
        0.02, 0.95, "", transform=summary_ax.transAxes, va="top", ha="left"
    )

    route = [start_city]
    drawn = [False] * len(actions)
    labels = []

    def add_arrow(a, b):
        p = 0.5 * (xy[a] + xy[b])
        d = xy[b] - xy[a]
        q1, q2 = p - 0.06 * d, p + 0.06 * d

        ax.plot(
            [xy[a, 0], xy[b, 0]],
            [xy[a, 1], xy[b, 1]],
            color=blue,
            linewidth=2.8,
            zorder=4,
        )
        ax.annotate(
            "",
            xy=q2,
            xytext=q1,
            arrowprops={
                "arrowstyle": "->",
                "color": "orange",
                "linewidth": 2,
            },
            zorder=6,
        )
        labels.append(
            ax.text(
                p[0], p[1],
                f"{instance.cost(a, b):.2f}",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="black"
            )
        )

    def update(frame):
        step, sub = divmod(frame, frames_per_move)

        if step == 0:
            current.set_data([xy[start_city, 0]], [xy[start_city, 1]])
            summary.set_text(
                f"SUMMARY\n\nStart city: {start_city}\n"
                "Total cost: 0.0000\nTotal reward: 0.0000"
            )
            return line, current, summary, *labels

        i = min(step - 1, len(actions) - 1)
        action = actions[i]
        a = route[-1]
        b = start_city if action == "CLOSE" else action

        p = (sub + 1) / frames_per_move
        pos = xy[a] + p * (xy[b] - xy[a])
        current.set_data([pos[0]], [pos[1]])

        if sub == frames_per_move - 1 and not drawn[i]:
            add_arrow(a, b)
            drawn[i] = True
            if action != "CLOSE":
                route.append(action)

        step_data = trajectory[i]

        summary.set_text(
            f"SUMMARY\n\n"
            f"{'Returning to start' if action == 'CLOSE' else f'Current city: {b}'}\n"
            f"Action: {action}\n"
            f"Transition cost: {step_data['transition_cost']:.4f}\n"
            f"Internal cost: {step_data['internal_cost']:.4f}\n"
            f"Step cost: {step_data['step_cost']:.4f}\n"
            f"Reward: {step_data['reward']:.4f}\n"
            f"Total cost: {-sum(rewards[:i + 1]):.4f}\n"
            f"Total reward: {sum(rewards[:i + 1]):.4f}"
        )

        if step > len(actions):
            current.set_data([xy[start_city, 0]], [xy[start_city, 1]])
            route_plot = route + [start_city]
            line.set_data(xy[route_plot, 0], xy[route_plot, 1])
            summary.set_text(
                f"SUMMARY\n\nTour complete\n"
                f"Total cost: {-sum(rewards):.4f}\n"
                f"Total reward: {sum(rewards):.4f}"
            )

        return line, current, summary, *labels

    animation = FuncAnimation(
        fig,
        update,
        frames=(len(actions) + 1) * frames_per_move,
        interval=interval,
        repeat=False,
        blit=False,
    )

    return fig, animation
