"""Tests for the TSP Gymnasium environment."""

import numpy as np
import pytest

from tsp.city_generator import CityLocationGenerator
from tsp.env import TSPEnv
from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.cost import city_internal_cost


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
def env():
    return TSPEnv(make_instance(), seed=42)


def test_reset(env):
    obs, info = env.reset(seed=42, start_city=0)

    assert env.observation_space.contains(obs)
    assert obs["visited_mask"].sum() == 1
    assert obs["current_city"] == info["current_city"]
    assert obs["total_cost"][0] == 0.0
    assert len(info["available_actions"]) == env.instance.num_cities - 1


def test_action_space(env):
    assert env.action_space.n == env.instance.num_cities + 1
    assert env.close_action == env.instance.num_cities


def test_step(env):
    _, info = env.reset(seed=42, start_city=0)
    action = info["available_actions"][0]

    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert reward < 0
    assert not terminated and not truncated
    assert action in info["tour"]
    assert obs["visited_mask"].sum() == 2


def test_state_continuity(env):
    obs, info = env.reset(seed=42, start_city=0)

    for _ in range(3):
        action = info["available_actions"][0]
        obs, _, _, _, info = env.step(action)
        state = env.simulator.state()

        assert info["tour"] == state["tour"]
        assert info["current_city"] == state["current_city"]
        assert info["start_city"] == state["start_city"]
        assert info["available_actions"] == state["available_actions"]

        assert obs["current_city"] == state["current_city"]

        # Cost is owned by the environment, not the simulator.
        assert np.isclose(info["total_cost"], env.total_cost)
        assert np.isclose(obs["total_cost"][0], env.total_cost)


def test_invalid_action(env):
    env.reset(seed=42, start_city=0)

    with pytest.raises(ValueError):
        env.step(env.simulator.current_city)


def test_invalid_action_range(env):
    env.reset(seed=42, start_city=0)

    with pytest.raises(ValueError):
        env.step(env.action_space.n)


def test_complete_tour(env):
    obs, info = env.reset(seed=42, start_city=0)
    n_cities = env.instance.num_cities

    while info["available_actions"]:
        obs, _, terminated, truncated, info = env.step(
            info["available_actions"][0]
        )

    assert not terminated and not truncated
    assert len(info["tour"]) == n_cities
    assert len(set(info["tour"])) == n_cities
    assert obs["visited_mask"].sum() == n_cities
    assert not info["available_actions"]

    obs, reward, terminated, truncated, info = env.step(
        env.close_action
    )

    assert terminated and not truncated
    assert reward < 0
    assert len(info["tour"]) == n_cities
    assert obs["visited_mask"].sum() == n_cities
    assert not info["available_actions"]
    assert info["total_cost"] > 0
    assert info["current_city"] == info["start_city"]


def test_reward_matches_tour_cost(env):
    _, info = env.reset(seed=42, start_city=0)
    total_reward = 0.0

    while info["available_actions"]:
        _, reward, _, _, info = env.step(
            info["available_actions"][0]
        )
        total_reward += reward

    _, reward, terminated, truncated, info = env.step(
        env.close_action
    )
    total_reward += reward

    assert terminated and not truncated
    assert np.isclose(-total_reward, info["total_cost"])


def test_seed_reproducibility(env):
    _, info1 = env.reset(seed=123, start_city=0)

    action = info1["available_actions"][0]
    _, _, _, _, step_info1 = env.step(action)

    _, info2 = env.reset(seed=123, start_city=0)

    _, _, _, _, step_info2 = env.step(action)

    assert step_info1["transition"] == step_info2["transition"]

def test_environment_owns_simulator(env):
    assert isinstance(env.simulator, TSPSimulator)
    assert env.simulator.instance is env.instance


def test_env_uses_same_simulator(env):
    _, info = env.reset(seed=42, start_city=0)
    action = info["available_actions"][0]

    obs, _, _, _, info = env.step(action)
    state = env.simulator.state()

    assert info["tour"] == state["tour"]
    assert obs["current_city"] == state["current_city"]
    assert np.isclose(obs["total_cost"][0], env.total_cost)


def test_premature_close_is_rejected(env):
    env.reset(seed=42, start_city=0)

    with pytest.raises(ValueError):
        env.step(env.close_action)


def test_available_actions_match_visited_mask(env):
    obs, info = env.reset(seed=42, start_city=0)

    for city in range(env.instance.num_cities):
        if city in info["tour"]:
            assert obs["visited_mask"][city] == 1
        else:
            assert obs["visited_mask"][city] == 0
            assert city in info["available_actions"]


def test_custom_cost_environment():
    instance = make_instance(custom_cost=True)
    env = TSPEnv(instance, seed=42)

    _, info = env.reset(seed=42, start_city=0)
    action = info["available_actions"][0]
    current_city = info["current_city"]

    _, reward, _, _, _ = env.step(action)

    transition_cost = instance.cost(current_city, action)
    internal_cost = city_internal_cost(instance, current_city)
    
    expected_reward = -(transition_cost + internal_cost)
    
    assert reward == pytest.approx(expected_reward)

def test_dynamic_pickup_changes_environment_cost_and_reward():
    instance = make_instance()

    changed_city = 0

    original_pickup = np.asarray(
        instance.cities[changed_city]["pickup"]["location"],
        dtype=float,
    )

    new_pickup = original_pickup + np.array([0.2, 0.1])

    instance.cities[changed_city]["pickup"]["location"] = (
        new_pickup.tolist()
    )

    instance.rebuild_dynamic_data()

    env = TSPEnv(instance, seed=42)

    start_city = 0
    _, info = env.reset(start_city=start_city)

    tour = list(range(instance.num_cities))

    total_reward = 0.0

    for next_city in tour[1:]:
        _, reward, terminated, truncated, info = env.step(next_city)

        total_reward += reward

        assert not terminated
        assert not truncated

    _, reward, terminated, truncated, info = env.step(
        env.close_action
    )

    total_reward += reward

    assert terminated
    assert not truncated

    tour_cost_value = tour_cost(
        instance,
        tour,
    )

    internal_details = internal_cost_details(instance)
    internal_cost_value = internal_details["total_internal_cost"]

    expected_total_cost = (
        tour_cost_value
        + internal_cost_value
    )

    assert env.total_cost == pytest.approx(
        expected_total_cost
    )

    assert total_reward == pytest.approx(
        -expected_total_cost
    )
