
import numpy as np
import pytest

from tsp.city_generator import CityLocationGenerator
from tsp.instance import TSPInstance
from tsp.cost import transition_cost, tour_cost


def make_instance():
    generator = CityLocationGenerator(
        width=10,
        height=10,
        n_cities=5,
        seed=42,
        city_shape="circle",
        city_radius=1.0,
    )

    coordinates = np.asarray(generator.generate(), dtype=float)
    cities = generator.get_cities()

    return TSPInstance(
        coordinates,
        width=10,
        height=10,
        cities=cities,
    )


def test_transition_cost():
    instance = make_instance()

    current_city = 0
    next_city = 1

    expected = instance.cost(current_city, next_city)

    assert transition_cost(
        instance,
        current_city,
        next_city,
    ) == pytest.approx(expected)


def test_tour_cost():
    instance = make_instance()

    tour = list(range(instance.num_cities))

    expected = sum(
        instance.cost(tour[i], tour[i + 1])
        for i in range(len(tour) - 1)
    )

    expected += instance.cost(tour[-1], tour[0])

    assert tour_cost(instance, tour) == pytest.approx(expected)


def test_tour_cost_uses_instance_cost_matrix():
    instance = make_instance()

    tour = list(range(instance.num_cities))

    expected = sum(
        instance.cost(tour[i], tour[i + 1])
        for i in range(len(tour) - 1)
    )

    expected += instance.cost(tour[-1], tour[0])

    assert tour_cost(instance, tour) == pytest.approx(expected)


def test_tour_cost_rejects_short_tour():
    instance = make_instance()

    with pytest.raises(ValueError):
        tour_cost(instance, [0])
