import math
import random


class CityLocationGenerator:
    """Generate non-overlapping city regions and user nodes."""

    def __init__(
        self,
        width,
        height,
        n_cities,
        seed=None,
        user_nodes_per_city=1,
        city_shape="circle",
        city_radius=None,
        city_size=None,
        city_width=None,
        city_height=None,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("Grid width and height must be positive.")
        if n_cities < 2:
            raise ValueError("n_cities must be at least 2.")
        if user_nodes_per_city < 1:
            raise ValueError("user_nodes_per_city must be at least 1.")

        self.width = float(width)
        self.height = float(height)
        self.n_cities = int(n_cities)
        self.user_nodes_per_city = int(user_nodes_per_city)
        self.city_shape = city_shape.lower()

        self.city_radius = city_radius
        self.city_size = city_size
        self.city_width = city_width
        self.city_height = city_height

        self.rng = random.Random(seed)

        # Kept for backward compatibility with Phase 2.
        self.coordinates = []

        # New complete city information.
        self.cities = []

        self._validate_city_parameters()

    def _calculate_city_area(self):
        if self.city_shape == "circle":
            return math.pi * self.city_radius ** 2

        if self.city_shape == "square":
            return self.city_size ** 2

        if self.city_shape == "rectangle":
            return self.city_width * self.city_height

        raise ValueError(
            "city_shape must be 'circle', 'square', or 'rectangle'."
        )

    def _validate_city_parameters(self):
        if self.city_shape == "circle":
            if self.city_radius is None or self.city_radius <= 0:
                raise ValueError(
                    "Circle requires --city-radius > 0."
                )

        elif self.city_shape == "square":
            if self.city_size is None or self.city_size <= 0:
                raise ValueError(
                    "Square requires --city-size > 0."
                )

        elif self.city_shape == "rectangle":
            if (
                self.city_width is None
                or self.city_height is None
                or self.city_width <= 0
                or self.city_height <= 0
            ):
                raise ValueError(
                    "Rectangle requires --city-width and --city-height > 0."
                )

        else:
            raise ValueError(
                "city_shape must be 'circle', 'square', or 'rectangle'."
            )

        area = self._calculate_city_area()

        if area < 2:
            raise ValueError(
                f"City area is {area:.4f}, but minimum allowed area is 2."
            )

        grid_area = self.width * self.height
        total_city_area = area * self.n_cities
        allowed_area = 0.75 * grid_area

        if total_city_area > allowed_area:
            raise ValueError(
                f"City area validation failed: "
                f"{area:.4f} × {self.n_cities} = {total_city_area:.4f} "
                f"> 0.75 × {grid_area:.4f} = {allowed_area:.4f}. "
                f"Decrease city dimensions, reduce n_cities, "
                f"or increase grid dimensions."
            )

    def _generate_center(self):
        if self.city_shape == "circle":
            margin_x = self.city_radius
            margin_y = self.city_radius

        elif self.city_shape == "square":
            margin_x = self.city_size / 2
            margin_y = self.city_size / 2

        else:
            margin_x = self.city_width / 2
            margin_y = self.city_height / 2

        if 2 * margin_x > self.width or 2 * margin_y > self.height:
            raise ValueError(
                "City dimensions do not fit inside the grid."
            )

        return (
            self.rng.uniform(margin_x, self.width - margin_x),
            self.rng.uniform(margin_y, self.height - margin_y),
        )

    def _overlap(self, a, b):
        if a["shape"] == "circle" and b["shape"] == "circle":
            dx = a["center"][0] - b["center"][0]
            dy = a["center"][1] - b["center"][1]
            r = a["dimensions"]["radius"] + b["dimensions"]["radius"]
            return dx * dx + dy * dy < r * r

        if a["shape"] != "circle" and b["shape"] != "circle":
            ax, ay = a["center"]
            bx, by = b["center"]

            aw = a["dimensions"]["width"]
            ah = a["dimensions"]["height"]
            bw = b["dimensions"]["width"]
            bh = b["dimensions"]["height"]

            return (
                abs(ax - bx) < (aw + bw) / 2
                and abs(ay - by) < (ah + bh) / 2
            )

        if a["shape"] == "circle":
            circle, rect = a, b
        else:
            circle, rect = b, a

        cx, cy = circle["center"]
        rx, ry = rect["center"]

        rw = rect["dimensions"]["width"] / 2
        rh = rect["dimensions"]["height"] / 2
        r = circle["dimensions"]["radius"]

        closest_x = max(rx - rw, min(cx, rx + rw))
        closest_y = max(ry - rh, min(cy, ry + rh))

        dx = cx - closest_x
        dy = cy - closest_y

        return dx * dx + dy * dy < r * r
        
    def _create_city(self, city_id):
        center = self._generate_center()
    
        if self.city_shape == "circle":
            dimensions = {
                "radius": float(self.city_radius)
            }
            area = math.pi * self.city_radius ** 2
    
            geometry = {
                "type": "circle",
                "center": [float(center[0]), float(center[1])],
                "radius": float(self.city_radius),
            }
    
        elif self.city_shape == "square":
            width = float(self.city_size)
            height = float(self.city_size)
            half_w = width / 2
            half_h = height / 2
    
            dimensions = {
                "width": width,
                "height": height,
            }
            area = width * height
    
            geometry = {
                "type": "polygon",
                "boundary": [
                    [center[0] - half_w, center[1] - half_h],
                    [center[0] + half_w, center[1] - half_h],
                    [center[0] + half_w, center[1] + half_h],
                    [center[0] - half_w, center[1] + half_h],
                ],
            }
    
        else:
            width = float(self.city_width)
            height = float(self.city_height)
            half_w = width / 2
            half_h = height / 2
    
            dimensions = {
                "width": width,
                "height": height,
            }
            area = width * height
    
            geometry = {
                "type": "polygon",
                "boundary": [
                    [center[0] - half_w, center[1] - half_h],
                    [center[0] + half_w, center[1] - half_h],
                    [center[0] + half_w, center[1] + half_h],
                    [center[0] - half_w, center[1] + half_h],
                ],
            }
    
        return {
            "city_id": city_id,
            "shape": self.city_shape,
            "area": float(area),
            "dimensions": dimensions,
            "center": [float(center[0]), float(center[1])],
            "rotation": 0.0,
            "geometry": geometry,
            "user_nodes": [],
            "pickup": {
                "location": [float(center[0]), float(center[1])],
                "type": "temporary_center",
            },
        }

    def _point_inside(self, city, point):
        x, y = point
        cx, cy = city["center"]

        if city["shape"] == "circle":
            r = city["dimensions"]["radius"]
            return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2

        w = city["dimensions"]["width"]
        h = city["dimensions"]["height"]

        return (
            abs(x - cx) < w / 2
            and abs(y - cy) < h / 2
        )

    def _generate_user_nodes(self, city):
        nodes = []

        max_attempts = 10000
        attempts = 0
        
        while len(nodes) < self.user_nodes_per_city and attempts < max_attempts:
            attempts += 1
        
            if city["shape"] == "circle":
                r = city["dimensions"]["radius"]
                x = city["center"][0] + self.rng.uniform(-r, r)
                y = city["center"][1] + self.rng.uniform(-r, r)
        
            else:
                w = city["dimensions"]["width"]
                h = city["dimensions"]["height"]
        
                x = city["center"][0] + self.rng.uniform(-w / 2, w / 2)
                y = city["center"][1] + self.rng.uniform(-h / 2, h / 2)
        
            point = [float(x), float(y)]
        
            if self._point_inside(city, point) and point not in nodes:
                nodes.append(point)
        
        if len(nodes) < self.user_nodes_per_city:
            raise RuntimeError(
                f"Could not generate {self.user_nodes_per_city} user nodes "
                f"inside city {city['city_id']} after {max_attempts} attempts. "
                f"Try reducing user_nodes_per_city or increasing city dimensions."
            )
        
        city["user_nodes"] = nodes

    def generate(self):
        """Generate all city regions and user nodes."""
        max_attempts = 10000
        max_restarts = 100
    
        for _ in range(max_restarts):
            self.coordinates = []
            self.cities = []
            success = True
    
            for city_id in range(self.n_cities):
                placed = False
    
                for _ in range(max_attempts):
                    city = self._create_city(city_id)
    
                    if not any(
                        self._overlap(city, existing)
                        for existing in self.cities
                    ):
                        self._generate_user_nodes(city)
                        self.cities.append(city)
                        self.coordinates.append(city["center"])
                        placed = True
                        break
    
                if not placed:
                    success = False
                    break
    
            if success:
                return self.get_coordinates()
    
        raise RuntimeError(
            f"Could not generate {self.n_cities} non-overlapping cities "
            f"after {max_restarts} layout restarts. "
            f"Try reducing city dimensions or n_cities, "
            f"or increasing grid dimensions."
        )

    def get_coordinates(self):
        """Return city center coordinates."""
        return [list(point) for point in self.coordinates]

    def get_cities(self):
        """Return complete city-region and user-node information."""
        return self.cities
