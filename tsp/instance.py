
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import euclidean_distance_matrix


class TSPInstance:
    """TSP instance with configurable edge costs, spatial bounds, and city data."""

    def __init__(
        self,
        coordinates: np.ndarray,
        name: str = "tsp_instance",
        cost_matrix: np.ndarray | None = None,
        width: float | None = None,
        height: float | None = None,
        cities: list | None = None,
    ) -> None:
        coordinates = np.asarray(coordinates, dtype=float)

        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must be an N x 2 array.")
        if len(coordinates) < 2:
            raise ValueError("A TSP instance must contain at least two cities.")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must contain only finite values.")

        if width is not None and width <= 0:
            raise ValueError("width must be positive.")
        if height is not None and height <= 0:
            raise ValueError("height must be positive.")

        self.name = name
        self.coordinates = coordinates
        self.num_cities = len(coordinates)
        self.width = float(width) if width is not None else None
        self.height = float(height) if height is not None else None

        if self.width is not None and np.any(coordinates[:, 0] > self.width):
            raise ValueError("coordinates exceed the specified width.")
        if self.height is not None and np.any(coordinates[:, 1] > self.height):
            raise ValueError("coordinates exceed the specified height.")

        # Complete city-region information.
        self.cities = cities
        if cities is not None:
            if not isinstance(cities, list):
                raise ValueError("cities must be a list.")

            if len(cities) != self.num_cities:
                raise ValueError(
                    "Number of city definitions must match number of coordinates."
                )

            for i, city in enumerate(cities):
                if not isinstance(city, dict):
                    raise ValueError("Each city definition must be a dictionary.")

                if "center" not in city:
                    raise ValueError(f"City {i} is missing 'center'.")

                center = np.asarray(city["center"], dtype=float)

                if center.shape != (2,) or not np.all(np.isfinite(center)):
                    raise ValueError(f"City {i} has an invalid center.")

                if not np.allclose(center, coordinates[i]):
                    raise ValueError(
                        f"City {i} center does not match its coordinate."
                    )

                # Pickup node  information.
                if "pickup" not in city:
                    raise ValueError(
                        f"City {i} is missing 'pickup'."
                    )

                pickup = city["pickup"]

                if not isinstance(pickup, dict):
                    raise ValueError(
                        f"City {i} pickup must be a dictionary."
                    )

                if "location" not in pickup:
                    raise ValueError(
                        f"City {i} pickup is missing 'location'."
                    )

                pickup_location = np.asarray(
                    pickup["location"],
                    dtype=float,
                )

                if (
                    pickup_location.shape != (2,)
                    or not np.all(np.isfinite(pickup_location))
                ):
                    raise ValueError(
                        f"City {i} has an invalid pickup location."
                    )

        if self.cities is not None:
            pickup_coordinates = self._get_current_pickup_coordinates()
            self.distance_matrix = euclidean_distance_matrix(pickup_coordinates)
        else:
            self.distance_matrix = euclidean_distance_matrix(coordinates)

        if cost_matrix is None:
            self._cost_matrix_is_distance_based = True
            cost_matrix = self.distance_matrix.copy()
        else:
            self._cost_matrix_is_distance_based = False
            cost_matrix = np.asarray(cost_matrix, dtype=float)

        if cost_matrix.shape != (self.num_cities, self.num_cities):
            raise ValueError("cost_matrix must be N x N.")
        if not np.all(np.isfinite(cost_matrix)):
            raise ValueError("cost_matrix must contain finite values.")
        if np.any(cost_matrix < 0):
            raise ValueError("cost_matrix cannot contain negative values.")

        self.cost_matrix = cost_matrix

    @classmethod
    def from_json(cls, path: str | Path) -> "TSPInstance":
        """Load a TSP instance from JSON."""

        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "cities" not in data or not isinstance(data["cities"], list):
            raise ValueError("JSON file must contain a 'cities' list.")

        city_data = data["cities"]

        # Support both old coordinate-only city data
        # and the new complete city structure.
        if all("coordinates" in city for city in city_data):
            coordinates = np.asarray(
                [city["coordinates"] for city in city_data],
                dtype=float,
            )
            cities = None
        elif all("center" in city for city in city_data):
            coordinates = np.asarray(
                [city["center"] for city in city_data],
                dtype=float,
            )
            cities = city_data
        else:
            raise ValueError(
                "Each city must contain either 'coordinates' or 'center'."
            )

        return cls(
            coordinates=coordinates,
            name=data.get("name", path.stem),
            cost_matrix=data.get("cost_matrix"),
            width=data.get("width"),
            height=data.get("height"),
            cities=cities,
        )

    def cost(self, city_a: int, city_b: int) -> float:
        self._validate_city_index(city_a)
        self._validate_city_index(city_b)
        return float(self.cost_matrix[city_a, city_b])

    def distance(self, city_a: int, city_b: int) -> float:
        self._validate_city_index(city_a)
        self._validate_city_index(city_b)
        return float(self.distance_matrix[city_a, city_b])

    def _get_current_pickup_coordinates(self) -> np.ndarray:
        """Return the current pickup location of every city as an N x 2 array."""
    
        if self.cities is None:
            return self.coordinates.copy()
    
        return np.asarray(
            [city["pickup"]["location"] for city in self.cities],
            dtype=float,
        )

    def rebuild_dynamic_data(self) -> None:
        """Rebuild quantities derived from the current pickup locations."""
    
        pickup_coordinates = self._get_current_pickup_coordinates()
    
        self.distance_matrix = euclidean_distance_matrix(
            pickup_coordinates
        )
    
        if self._cost_matrix_is_distance_based:
            self.cost_matrix = self.distance_matrix.copy()

    def _validate_city_index(self, city_index: int) -> None:
        if not isinstance(city_index, (int, np.integer)):
            raise TypeError("city index must be an integer.")

        if not 0 <= city_index < self.num_cities:
            raise IndexError(
                f"City index {city_index} is out of range "
                f"for {self.num_cities} cities."
            )

