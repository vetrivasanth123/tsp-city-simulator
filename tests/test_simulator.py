"""Tests for the TSP simulator."""

import numpy as np
import pytest

from tsp.city_generator import CityLocationGenerator
from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator


def make_instance(n_cities=5, seed=42, custom_cost=False):
    width = 10
    height = 10
    city_radius = 1.0

    generator = CityLocationGenerator(
        width,
        height,
        n_cities,
        seed=seed,
        city_shape="circle",
        city_radius=city_radius,
    )

    coordinates = np.asarray(
        generator.generate(),
        dtype=float,
    )

    cities = generator.get_cities()

    if custom_cost:
        rng = np.random.default_rng(seed)
        cost_matrix = rng.uniform(
            1.0, 100.0, (n_cities, n_cities)
        )
        cost_matrix = (cost_matrix + cost_matrix.T) / 2.0
        np.fill_diagonal(cost_matrix, 0.0)

        return TSPInstance(
            coordinates,
            cost_matrix=cost_matrix,
            width=width,
            height=height,
            cities=cities,
        )

    return TSPInstance(
        coordinates,
        width=width,
        height=height,
        cities=cities,
    )


@pytest.fixture
def instance():
    return make_instance()


@pytest.fixture
def simulator(instance):
    simulator = TSPSimulator(instance, seed=42)
    simulator.reset(start_city=0)
    return simulator

def test_simulator_initial_state(simulator):
    assert simulator.tour == [simulator.start_city]
    assert simulator.current_city == simulator.start_city
    assert simulator.tour == [simulator.start_city]
    assert simulator.done is False


def test_reset_uses_specified_start(instance):
    simulator = TSPSimulator(instance, seed=42)

    state = simulator.reset(start_city=2)

    assert simulator.start_city == 2
    assert simulator.current_city == 2
    assert simulator.tour == [2]
    assert state["start_city"] == 2
    assert state["current_city"] == 2

    assert len(simulator.tour) == 1
    assert simulator.tour[0] == simulator.start_city
    assert state["current_city"] == simulator.start_city

def test_reset_requires_start_city(instance):
    simulator = TSPSimulator(instance, seed=42)

    with pytest.raises(ValueError, match="start_city must be provided"):
        simulator.reset()
        
def test_available_actions_excludes_visited_city(simulator):
    available = simulator.available_actions()

    assert simulator.start_city not in available
    assert len(available) == simulator.instance.num_cities - 1
    assert all(city not in simulator.tour for city in available)


def test_step_adds_city(simulator):
    next_city = simulator.available_actions()[0]
    old_city = simulator.current_city

    simulator.step(next_city)

    assert simulator.tour == [simulator.start_city, next_city]
    assert simulator.current_city == next_city


def test_multiple_steps_build_tour(simulator):
    actions = simulator.available_actions()[:3]

    for city in actions:
        simulator.step(city)

    assert len(simulator.tour) == 4
    assert len(set(simulator.tour)) == 4
    assert simulator.current_city == actions[-1]


def test_close_tour_returns_to_start(simulator):
    while len(simulator.tour) < simulator.instance.num_cities:
        simulator.step(simulator.available_actions()[0])

    simulator.close_tour()

    assert simulator.done is True


def test_close_tour_rejects_incomplete_tour(simulator):
    simulator.step(simulator.available_actions()[0])

    with pytest.raises(RuntimeError):
        simulator.close_tour()

    assert simulator.done is False


def test_complete_tour(simulator):
    n_cities = simulator.instance.num_cities

    while len(simulator.tour) < n_cities:
        simulator.step(simulator.available_actions()[0])

    assert len(simulator.tour) == n_cities
    assert len(set(simulator.tour)) == n_cities
    assert sorted(simulator.tour) == list(range(n_cities))

    simulator.close_tour()

    assert simulator.done is True


def test_invalid_city_index_is_rejected(simulator):
    with pytest.raises((ValueError, IndexError)):
        simulator.step(simulator.instance.num_cities)


def test_negative_city_index_is_rejected(simulator):
    with pytest.raises((ValueError, IndexError)):
        simulator.step(-1)


def test_duplicate_city_is_rejected(simulator):
    next_city = simulator.available_actions()[0]
    simulator.step(next_city)

    with pytest.raises(ValueError):
        simulator.step(next_city)


def test_close_empty_tour_is_not_allowed(instance):
    simulator = TSPSimulator(instance, seed=42)
    simulator.tour = []
    simulator.current_city = None

    with pytest.raises(ValueError):
        simulator.close_tour()


def test_step_after_completion_is_rejected(simulator):
    while len(simulator.tour) < simulator.instance.num_cities:
        simulator.step(simulator.available_actions()[0])

    simulator.close_tour()

    with pytest.raises(RuntimeError):
        simulator.step(simulator.start_city)

def test_state_contains_simulator_state(simulator):
    state = simulator.state()

    required = {
        "tour",
        "current_city",
        "start_city",
        "visited",
        "available_actions",
        "done",
    }

    assert required.issubset(state)
    assert state["available_actions"] == simulator.available_actions()
    
def test_stochastic_multi_seed_valid_tours(instance):
    for seed in range(10):
        simulator = TSPSimulator(instance, seed=seed)
        simulator.reset(start_city=0)

        while simulator.available_actions():
            action = simulator.available_actions()[0]
            simulator.step(action)

        simulator.close_tour()

        closed_tour = simulator.tour + [simulator.start_city]

        assert len(closed_tour) == instance.num_cities + 1
        assert len(set(closed_tour[:-1])) == instance.num_cities
        assert closed_tour[0] == closed_tour[-1]
        assert simulator.done
        
def test_transition_probabilities_are_valid(instance):
    simulator = TSPSimulator(instance, seed=42)
    simulator.reset(start_city=0)

    for _ in range(instance.num_cities - 1):
        action = simulator.available_actions()[0]
        state = simulator.step(action)

        transition_info = state["transition_info"]
        probabilities = transition_info["transition_probabilities"]

        assert set(probabilities) == set(
            map(int, transition_info["available_actions"])
        )
        assert abs(sum(probabilities.values()) - 1.0) < 1e-9
        assert all(0.0 <= p <= 1.0 for p in probabilities.values())
