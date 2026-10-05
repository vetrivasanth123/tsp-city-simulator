from __future__ import annotations

from .instance import TSPInstance
import numpy as np


def transition_cost(
    instance: TSPInstance,
    current_city: int,
    next_city: int,
) -> float:
    """Return the cost of one city-to-city transition."""

    return instance.cost(current_city, next_city)


def tour_cost(
    instance: TSPInstance,
    tour: list[int],
) -> float:
    """Return the total cost of a closed tour."""

    if len(tour) < 2:
        raise ValueError("Tour must contain at least two cities.")

    total = 0.0

    for i in range(len(tour) - 1):
        total += transition_cost(
            instance,
            tour[i],
            tour[i + 1],
        )

    # CLOSE: return from the final city to the starting city.
    total += transition_cost(
        instance,
        tour[-1],
        tour[0],
    )

    return total

def internal_cost_details(
    instance: TSPInstance,
) -> dict:
    """Return detailed internal user-to-pickup distances and costs."""

    if instance.cities is None:
        raise ValueError(
            "Internal cost requires complete city data with user nodes "
            "and pickup locations."
        )

    city_details = []
    total_internal_cost = 0.0

    for city_id, city in enumerate(instance.cities):
        pickup_location = city["pickup"]["location"]

        user_distances = []

        for user_id, user_node in enumerate(city["user_nodes"]):
            distance = float(
                np.linalg.norm(
                    np.asarray(user_node, dtype=float)
                    - np.asarray(pickup_location, dtype=float)
                )
            )

            user_distances.append(
                {
                    "user_id": user_id,
                    "distance": distance,
                }
            )

        city_total = sum(
            item["distance"]
            for item in user_distances
        )

        total_internal_cost += city_total

        city_details.append(
            {
                "city_id": city_id,
                "pickup_location": list(pickup_location),
                "user_distances": user_distances,
                "total_cost": city_total,
            }
        )

    return {
        "cities": city_details,
        "total_internal_cost": total_internal_cost,
    }

def objective_cost(
    instance: TSPInstance,
    tour: list[int],
) -> dict:
    """Return the complete objective and its cost components."""

    tour_total = tour_cost(instance, tour)

    internal_details = internal_cost_details(instance)
    internal_total = internal_details["total_internal_cost"]

    total_cost = tour_total + internal_total

    return {
        "tour_cost": tour_total,
        "internal_cost": internal_total,
        "total_cost": total_cost,
        "internal_details": internal_details,
    }

