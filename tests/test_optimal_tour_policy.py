
"""Inspect policy action probabilities along the two optimal TSP tours.

Run from the project root:
    python -m pytest -s tests/test_optimal_tour_policy.py

This test does not train or modify the policy.
"""

import pytest
import torch

from examples.reinforce.policy import SoftmaxPolicy


NUM_CITIES = 5

OPTIMAL_TOURS = [
    [0, 3, 2, 1, 4, 0],
    [0, 4, 1, 2, 3, 0],
]


def make_observation(current_city, visited):
    mask = [int(city in visited) for city in range(NUM_CITIES)]
    return {
        "current_city": current_city,
        "visited_mask": mask,
    }


@pytest.mark.parametrize("tour", OPTIMAL_TOURS)
def test_print_optimal_tour_action_probabilities(tour):
    torch.manual_seed(42)
    policy = SoftmaxPolicy(num_cities=NUM_CITIES)

    visited = {tour[0]}
    print(f"\n\nTour: {' -> '.join(map(str, tour))}")

    for next_city in tour[1:-1]:
        current_city = tour[tour.index(next_city) - 1]
        observation = make_observation(current_city, visited)

        legal_actions = [
            city for city in range(NUM_CITIES)
            if city not in visited
        ]

        distribution = policy.get_distribution(
            observation, legal_actions
        )
        probabilities = distribution.probs.detach()

        print(
            f"Current city {current_city}; visited={sorted(visited)}"
        )
        for action in legal_actions:
            marker = " <-- optimal next city" if action == next_city else ""
            print(
                f"  action {action}: "
                f"{probabilities[action].item():.6f}{marker}"
            )

        assert next_city in legal_actions
        assert probabilities[next_city].item() > 0.0
        assert torch.isclose(
            probabilities.sum(), torch.tensor(1.0), atol=1e-6
        )

        visited.add(next_city)

    # After all cities are visited, CLOSE must be the only legal action.
    observation = make_observation(tour[-2], visited)
    close_action = NUM_CITIES
    distribution = policy.get_distribution(
        observation, [close_action]
    )
    assert distribution.probs[close_action].item() == 1.0
