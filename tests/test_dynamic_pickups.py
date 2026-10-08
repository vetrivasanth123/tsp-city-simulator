
"""Tests for dynamic pickup-location behaviour."""

import numpy as np
import pytest

from tsp.city_generator import CityLocationGenerator
from tsp.instance import TSPInstance
from tsp.cost import city_internal_cost, internal_cost_details


def make_instance():
    generator = CityLocationGenerator(
        width=10,
        height=10,
        n_cities=5,
        seed=42,
        user_nodes_per_city=10,
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


def pickup_coordinates(instance):
    return np.asarray(
        [city["pickup"]["location"] for city in instance.cities],
        dtype=float,
    )


def expected_distance_matrix(pickups):
    return np.linalg.norm(
        pickups[:, None, :] - pickups[None, :, :],
        axis=2,
    )


def test_single_pickup_change_rebuilds_distance_matrix():
    instance = make_instance()
    original = instance.distance_matrix.copy()

    city_id = 0
    new_pickup = pickup_coordinates(instance)[city_id] + np.array([0.2, 0.1])

    instance.cities[city_id]["pickup"]["location"] = new_pickup.tolist()
    instance.rebuild_dynamic_data()

    expected = expected_distance_matrix(pickup_coordinates(instance))

    np.testing.assert_allclose(instance.distance_matrix, expected)
    assert not np.allclose(
        instance.distance_matrix[city_id],
        original[city_id],
    )


def test_unaffected_distances_remain_unchanged():
    instance = make_instance()
    original = instance.distance_matrix.copy()

    changed_city = 0
    pickups = pickup_coordinates(instance)
    pickups[changed_city] += np.array([0.2, 0.1])

    instance.cities[changed_city]["pickup"]["location"] = (
        pickups[changed_city].tolist()
    )
    instance.rebuild_dynamic_data()

    for a in range(instance.num_cities):
        for b in range(instance.num_cities):
            if a != changed_city and b != changed_city:
                assert instance.distance_matrix[a, b] == pytest.approx(
                    original[a, b]
                )


def test_default_cost_matrix_follows_dynamic_distance_matrix():
    instance = make_instance()

    city_id = 0
    new_pickup = pickup_coordinates(instance)[city_id] + np.array([0.2, 0.1])

    instance.cities[city_id]["pickup"]["location"] = new_pickup.tolist()
    instance.rebuild_dynamic_data()

    np.testing.assert_allclose(
        instance.cost_matrix,
        instance.distance_matrix,
    )


def test_single_pickup_change_updates_only_its_internal_cost():
    instance = make_instance()

    original = np.array([
        city_internal_cost(instance, i)
        for i in range(instance.num_cities)
    ])

    changed_city = 0
    new_pickup = pickup_coordinates(instance)[changed_city] + np.array([0.2, 0.1])

    instance.cities[changed_city]["pickup"]["location"] = new_pickup.tolist()
    instance.rebuild_dynamic_data()

    updated = np.array([
        city_internal_cost(instance, i)
        for i in range(instance.num_cities)
    ])

    assert updated[changed_city] != pytest.approx(original[changed_city])

    for city_id in range(instance.num_cities):
        if city_id != changed_city:
            assert updated[city_id] == pytest.approx(original[city_id])


def test_internal_cost_details_uses_current_pickups():
    instance = make_instance()

    city_id = 0
    new_pickup = pickup_coordinates(instance)[city_id] + np.array([0.2, 0.1])

    instance.cities[city_id]["pickup"]["location"] = new_pickup.tolist()
    instance.rebuild_dynamic_data()

    details = internal_cost_details(instance)
    recorded = np.asarray(
        details["cities"][city_id]["pickup_location"],
        dtype=float,
    )

    np.testing.assert_allclose(recorded, new_pickup)

    expected = sum(
        np.linalg.norm(np.asarray(user, dtype=float) - new_pickup)
        for user in instance.cities[city_id]["user_nodes"]
    )

    assert details["cities"][city_id]["total_cost"] == pytest.approx(expected)


def test_multiple_pickups_rebuild_from_complete_design():
    instance = make_instance()

    pickups = pickup_coordinates(instance)
    changed = [0, 2, 4]

    for city_id in changed:
        pickups[city_id] += np.array([0.2, 0.1])
        instance.cities[city_id]["pickup"]["location"] = (
            pickups[city_id].tolist()
        )

    instance.rebuild_dynamic_data()

    expected = expected_distance_matrix(pickups)

    np.testing.assert_allclose(instance.distance_matrix, expected)
    np.testing.assert_allclose(instance.cost_matrix, expected)


def test_all_pickups_can_be_changed_and_rebuilt():
    instance = make_instance()

    pickups = pickup_coordinates(instance)

    for city_id in range(instance.num_cities):
        pickups[city_id] += np.array([
            0.1 * (city_id + 1),
            -0.05 * (city_id + 1),
        ])
        instance.cities[city_id]["pickup"]["location"] = (
            pickups[city_id].tolist()
        )

    instance.rebuild_dynamic_data()

    expected = expected_distance_matrix(pickups)

    np.testing.assert_allclose(instance.distance_matrix, expected)
    np.testing.assert_allclose(instance.cost_matrix, expected)

    details = internal_cost_details(instance)

    expected_internal = sum(
        np.linalg.norm(
            np.asarray(user, dtype=float) - pickups[city_id]
        )
        for city_id, city in enumerate(instance.cities)
        for user in city["user_nodes"]
    )

    assert details["total_internal_cost"] == pytest.approx(
        expected_internal
    )

