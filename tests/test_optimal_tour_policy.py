
"""Inspect policy probabilities along the optimal five-city TSP tours.

Run from the project root:
    python -m pytest -s tests/test_optimal_tour_policy.py
"""

import pytest
import torch

from examples.reinforce.policy import SoftmaxPolicy


NUM_CITIES = 5
CLOSE_ACTION = NUM_CITIES

OPTIMAL_TOURS = [
    [0, 3, 2, 1, 4, 0],
    [0, 4, 1, 2, 3, 0],
]


def make_observation(current_city, visited):
    """Build the observation format expected by SoftmaxPolicy."""
    visited_mask = [
        int(city in visited)
        for city in range(NUM_CITIES)
    ]

    return {
        "current_city": current_city,
        "visited_mask": visited_mask,
    }


@pytest.mark.parametrize("tour", OPTIMAL_TOURS)
def test_optimal_tour_action_probabilities(tour):
    """Print action probabilities at each decision along an optimal tour."""
    torch.manual_seed(42)
    policy = SoftmaxPolicy(num_cities=NUM_CITIES)
    policy.eval()

    visited = {tour[0]}

    print("\n")
    print("=" * 55)
    print(f"Optimal tour: {' -> '.join(map(str, tour))}")
    print("=" * 55)

    with torch.no_grad():
        # There are NUM_CITIES - 1 city-selection decisions.
        for i in range(1, NUM_CITIES):
            current_city = tour[i - 1]
            next_city = tour[i]

            observation = make_observation(current_city, visited)

            legal_actions = [
                city
                for city in range(NUM_CITIES)
                if city not in visited
            ]

            distribution = policy.get_distribution(
                observation,
                legal_actions,
            )
            probabilities = distribution.probs

            print(
                f"\nCurrent city: {current_city}"
                f" | Visited cities: {sorted(visited)}"
            )
            print(f"Optimal next city: {next_city}")
            print("Action probabilities:")

            for action in legal_actions:
                probability = probabilities[action].item()
                marker = " <-- optimal next action" if action == next_city else ""

                print(
                    f"  City {action}: {probability:.6f}{marker}"
                )

            # Verify that the optimal next city is legal.
            assert next_city in legal_actions

            # Verify all legal-action probabilities sum to one.
            assert torch.isclose(
                probabilities.sum(),
                torch.tensor(1.0),
                atol=1e-6,
            )

            # Verify illegal actions have zero probability.
            for action in range(NUM_CITIES):
                if action not in legal_actions:
                    assert probabilities[action].item() == 0.0

            # The selected legal action must have a valid probability.
            assert 0.0 < probabilities[next_city].item() <= 1.0

            visited.add(next_city)

        # After every city has been visited, CLOSE is the only legal action.
        final_city = tour[NUM_CITIES - 1]
        observation = make_observation(final_city, visited)

        close_distribution = policy.get_distribution(
            observation,
            [CLOSE_ACTION],
        )

        print("\nAll cities visited.")
        print(
            "CLOSE action probability: "
            f"{close_distribution.probs[CLOSE_ACTION].item():.6f}"
        )

        assert close_distribution.probs[CLOSE_ACTION].item() == 1.0

    print("\nAll probability checks passed.")
