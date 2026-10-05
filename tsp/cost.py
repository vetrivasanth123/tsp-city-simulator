from __future__ import annotations

from .instance import TSPInstance


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
