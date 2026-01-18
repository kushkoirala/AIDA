#!/usr/bin/env python3
"""
Generate Route Training Data from Expert Flights

This script generates training data for the route generation transformer by
running expert flights using the GeneralizedXCController and extracting:
- Route specifications (origin/destination airports)
- Waypoint sequences (lat, lon, alt, speed, heading, phase)
- Controller parameters (tp_distance, glideslope, pattern_alt, etc.)

The data is saved in a format ready for transformer training.

Usage:
    python generate_route_training_data.py --num-routes 100 --output route_data.npz

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpu-flight-dynamics' / 'python'))

from generalized_xc_controller import (
    GeneralizedXCController, AirportConfig, XCPhase,
    latlon_to_xy, FT_TO_NM, NM_TO_FT, KTS_TO_FPS, FPS_TO_KTS, M_TO_FT
)
from earth_model import EarthModel

# Flight dynamics imports
try:
    from flight_dynamics import State, StateIndex
    HAS_FLIGHT_DYNAMICS = True
except ImportError:
    HAS_FLIGHT_DYNAMICS = False
    print("Warning: flight_dynamics not available, using simplified simulation")


@dataclass
class WaypointData:
    """Data for a single waypoint in a flight plan."""
    lat: float          # Latitude (degrees)
    lon: float          # Longitude (degrees)
    altitude_ft: float  # Altitude AGL (feet)
    speed_kts: float    # Target speed (knots)
    heading_deg: float  # Heading (degrees true)
    phase: int          # Flight phase (0-5: takeoff, climb, cruise, descent, approach, landing)
    valid: bool = True  # Whether this waypoint is active


@dataclass
class RouteData:
    """Complete route data for training."""
    # Origin airport
    origin_lat: float
    origin_lon: float
    origin_elev_ft: float
    origin_rwy_deg: float

    # Destination airport
    dest_lat: float
    dest_lon: float
    dest_elev_ft: float
    dest_rwy_deg: float

    # Route parameters
    cruise_alt_ft: float
    distance_nm: float

    # Waypoints (list of WaypointData)
    waypoints: List[WaypointData]

    # Controller parameters
    tp_distance_nm: float
    glideslope_deg: float
    pattern_alt_ft: float
    v_approach_kts: float
    v_cruise_kts: float
    turn_bank_deg: float
    use_triangle_pattern: bool


# Extended airport database for training data generation
TRAINING_AIRPORTS = {
    # Kansas region
    "SN65": AirportConfig(
        icao="SN65", name="Lake Waltanna Airport",
        lat=37.594167, lon=-97.615833,
        elevation_ft=1448.0, runway_heading_deg=4.0
    ),
    "KHUT": AirportConfig(
        icao="KHUT", name="Hutchinson Regional Airport",
        lat=38.066167, lon=-97.860500,
        elevation_ft=1542.0, runway_heading_deg=314.0
    ),
    "KICT": AirportConfig(
        icao="KICT", name="Wichita Eisenhower National",
        lat=37.649944, lon=-97.433056,
        elevation_ft=1333.0, runway_heading_deg=14.0
    ),
    "KAAO": AirportConfig(
        icao="KAAO", name="Colonel James Jabara Airport",
        lat=37.747500, lon=-97.221389,
        elevation_ft=1421.0, runway_heading_deg=180.0
    ),
    "KEWK": AirportConfig(
        icao="KEWK", name="Newton City-County Airport",
        lat=38.058333, lon=-97.275556,
        elevation_ft=1533.0, runway_heading_deg=174.0
    ),
    "KSLC": AirportConfig(
        icao="KSLC", name="Augusta Municipal Airport",
        lat=37.670000, lon=-97.077778,
        elevation_ft=1328.0, runway_heading_deg=36.0
    ),
    "KELY": AirportConfig(
        icao="KELY", name="El Dorado Airport",
        lat=37.791389, lon=-96.833889,
        elevation_ft=1378.0, runway_heading_deg=175.0
    ),
    "KMCI": AirportConfig(
        icao="KMCI", name="Kansas City International",
        lat=39.297500, lon=-94.713889,
        elevation_ft=1026.0, runway_heading_deg=19.0
    ),
    # Add some synthetic airports for data augmentation
    "SYN01": AirportConfig(
        icao="SYN01", name="Synthetic Airport 1",
        lat=37.8, lon=-97.5,
        elevation_ft=1400.0, runway_heading_deg=90.0
    ),
    "SYN02": AirportConfig(
        icao="SYN02", name="Synthetic Airport 2",
        lat=38.1, lon=-97.0,
        elevation_ft=1500.0, runway_heading_deg=270.0
    ),
    "SYN03": AirportConfig(
        icao="SYN03", name="Synthetic Airport 3",
        lat=37.4, lon=-97.8,
        elevation_ft=1350.0, runway_heading_deg=45.0
    ),
}


class ExpertFlightSimulator:
    """Simulates expert flights to generate training data."""

    def __init__(self, ref_lat: float = 37.594167, ref_lon: float = -97.615833):
        """Initialize the simulator.

        Args:
            ref_lat: Reference latitude for Earth model (default: SN65)
            ref_lon: Reference longitude for Earth model (default: SN65)
        """
        self.earth = EarthModel(ref_lat, ref_lon)

    def compute_route_distance(self, origin: AirportConfig, dest: AirportConfig) -> float:
        """Compute great circle distance between airports in nm."""
        dist_m = self.earth.great_circle_distance(
            origin.lat, origin.lon, dest.lat, dest.lon
        )
        return dist_m / 1852.0  # meters to nm

    def compute_bearing(self, origin: AirportConfig, dest: AirportConfig) -> float:
        """Compute initial bearing from origin to destination."""
        return self.earth.initial_bearing(origin.lat, origin.lon, dest.lat, dest.lon)

    def destination_point(self, lat: float, lon: float, bearing_deg: float, dist_m: float) -> Tuple[float, float]:
        """Compute destination point given start, bearing, and distance.

        Uses simplified calculation suitable for short distances (<100nm).

        Args:
            lat: Starting latitude (degrees)
            lon: Starting longitude (degrees)
            bearing_deg: Initial bearing (degrees)
            dist_m: Distance (meters)

        Returns:
            (dest_lat, dest_lon) in degrees
        """
        R = 6371000.0  # Earth radius in meters

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        bearing_rad = np.deg2rad(bearing_deg)

        angular_dist = dist_m / R

        dest_lat_rad = np.arcsin(
            np.sin(lat_rad) * np.cos(angular_dist) +
            np.cos(lat_rad) * np.sin(angular_dist) * np.cos(bearing_rad)
        )
        dest_lon_rad = lon_rad + np.arctan2(
            np.sin(bearing_rad) * np.sin(angular_dist) * np.cos(lat_rad),
            np.cos(angular_dist) - np.sin(lat_rad) * np.sin(dest_lat_rad)
        )

        return np.rad2deg(dest_lat_rad), np.rad2deg(dest_lon_rad)

    def generate_waypoints(self, origin: AirportConfig, dest: AirportConfig,
                          cruise_alt_ft: float, tp_distance_nm: float,
                          pattern_alt_ft: float) -> List[WaypointData]:
        """Generate waypoint sequence for a route.

        Args:
            origin: Origin airport
            dest: Destination airport
            cruise_alt_ft: Cruise altitude in feet
            tp_distance_nm: Turn point distance from destination
            pattern_alt_ft: Pattern altitude AGL

        Returns:
            List of waypoints defining the flight plan
        """
        waypoints = []
        distance_nm = self.compute_route_distance(origin, dest)
        bearing = self.compute_bearing(origin, dest)

        # Waypoint 1: Takeoff (origin airport)
        waypoints.append(WaypointData(
            lat=origin.lat,
            lon=origin.lon,
            altitude_ft=0.0,
            speed_kts=0.0,
            heading_deg=origin.runway_heading_deg,
            phase=0  # Takeoff
        ))

        # Waypoint 2: Initial climb point (~1nm from origin)
        climb_pt = self.destination_point(
            origin.lat, origin.lon, origin.runway_heading_deg, 1852.0
        )
        waypoints.append(WaypointData(
            lat=climb_pt[0],
            lon=climb_pt[1],
            altitude_ft=500.0,
            speed_kts=74.0,  # V_climb
            heading_deg=origin.runway_heading_deg,
            phase=1  # Climb
        ))

        # Waypoint 3: Turn to cruise heading (~2nm from origin)
        turn_pt = self.destination_point(
            origin.lat, origin.lon, origin.runway_heading_deg, 3704.0  # 2nm
        )
        waypoints.append(WaypointData(
            lat=turn_pt[0],
            lon=turn_pt[1],
            altitude_ft=1500.0,
            speed_kts=80.0,
            heading_deg=bearing,  # Turn to destination
            phase=1  # Still climbing
        ))

        # Waypoint 4: Top of climb (cruise start)
        # Position: ~1/4 of the way to destination
        toc_dist_m = distance_nm * 0.25 * 1852.0
        toc_pt = self.destination_point(origin.lat, origin.lon, bearing, toc_dist_m)
        waypoints.append(WaypointData(
            lat=toc_pt[0],
            lon=toc_pt[1],
            altitude_ft=cruise_alt_ft,
            speed_kts=110.0,  # V_cruise
            heading_deg=bearing,
            phase=2  # Cruise
        ))

        # Waypoint 5: Turn point (TP) - behind destination on backcourse
        backcourse = (dest.runway_heading_deg + 180) % 360
        tp_dist_m = tp_distance_nm * 1852.0
        tp_pt = self.destination_point(dest.lat, dest.lon, backcourse, tp_dist_m)
        waypoints.append(WaypointData(
            lat=tp_pt[0],
            lon=tp_pt[1],
            altitude_ft=cruise_alt_ft,
            speed_kts=100.0,
            heading_deg=bearing,
            phase=2  # Still cruise
        ))

        # Waypoint 6: Final approach intercept point
        # Position: on final approach course, ~3nm from threshold
        final_dist_m = 3.0 * 1852.0
        final_pt = self.destination_point(dest.lat, dest.lon, backcourse, final_dist_m)
        approach_alt = pattern_alt_ft + dest.elevation_ft
        waypoints.append(WaypointData(
            lat=final_pt[0],
            lon=final_pt[1],
            altitude_ft=approach_alt,
            speed_kts=75.0,  # Slowing down
            heading_deg=dest.runway_heading_deg,  # Aligned with runway
            phase=4  # Approach
        ))

        # Waypoint 7: Short final (~1nm from threshold)
        short_final_pt = self.destination_point(dest.lat, dest.lon, backcourse, 1852.0)
        waypoints.append(WaypointData(
            lat=short_final_pt[0],
            lon=short_final_pt[1],
            altitude_ft=300.0,  # ~300ft AGL
            speed_kts=65.0,  # V_approach
            heading_deg=dest.runway_heading_deg,
            phase=4  # Approach
        ))

        # Waypoint 8: Touchdown (destination airport)
        waypoints.append(WaypointData(
            lat=dest.lat,
            lon=dest.lon,
            altitude_ft=0.0,
            speed_kts=50.0,  # V_touchdown
            heading_deg=dest.runway_heading_deg,
            phase=5  # Landing
        ))

        return waypoints

    def generate_route_data(self, origin: AirportConfig, dest: AirportConfig,
                           randomize_params: bool = True) -> RouteData:
        """Generate complete route data for training.

        Args:
            origin: Origin airport
            dest: Destination airport
            randomize_params: Whether to randomize controller parameters

        Returns:
            RouteData with all fields filled
        """
        distance_nm = self.compute_route_distance(origin, dest)

        # Controller parameters (with optional randomization)
        if randomize_params:
            # Randomize within reasonable ranges
            cruise_alt_ft = random.uniform(2500, 8500)  # 2500-8500 ft
            tp_distance_nm = random.uniform(5, 15)  # 5-15 nm
            pattern_alt_ft = random.uniform(800, 1500)  # 800-1500 ft AGL
            v_approach_kts = random.uniform(60, 80)  # 60-80 kts
            v_cruise_kts = random.uniform(90, 130)  # 90-130 kts
            turn_bank_deg = random.uniform(20, 35)  # 20-35 degrees
            glideslope_deg = random.uniform(2.5, 4.5)  # 2.5-4.5 degrees
            use_triangle = random.random() > 0.3  # 70% use triangle pattern
        else:
            # Default parameters
            cruise_alt_ft = 3500.0
            tp_distance_nm = 10.0
            pattern_alt_ft = 1000.0
            v_approach_kts = 65.0
            v_cruise_kts = 110.0
            turn_bank_deg = 25.0
            glideslope_deg = 3.0
            use_triangle = True

        # Adjust parameters based on distance
        if distance_nm < 15:
            cruise_alt_ft = min(cruise_alt_ft, 3500)
            tp_distance_nm = min(tp_distance_nm, distance_nm * 0.4)
        elif distance_nm > 50:
            cruise_alt_ft = max(cruise_alt_ft, 5500)
            tp_distance_nm = min(tp_distance_nm, 12)

        # Generate waypoints
        waypoints = self.generate_waypoints(
            origin, dest, cruise_alt_ft, tp_distance_nm, pattern_alt_ft
        )

        return RouteData(
            origin_lat=origin.lat,
            origin_lon=origin.lon,
            origin_elev_ft=origin.elevation_ft,
            origin_rwy_deg=origin.runway_heading_deg,
            dest_lat=dest.lat,
            dest_lon=dest.lon,
            dest_elev_ft=dest.elevation_ft,
            dest_rwy_deg=dest.runway_heading_deg,
            cruise_alt_ft=cruise_alt_ft,
            distance_nm=distance_nm,
            waypoints=waypoints,
            tp_distance_nm=tp_distance_nm,
            glideslope_deg=glideslope_deg,
            pattern_alt_ft=pattern_alt_ft,
            v_approach_kts=v_approach_kts,
            v_cruise_kts=v_cruise_kts,
            turn_bank_deg=turn_bank_deg,
            use_triangle_pattern=use_triangle
        )


def generate_training_dataset(num_routes: int = 100,
                              min_distance_nm: float = 10.0,
                              max_distance_nm: float = 100.0,
                              include_reverse: bool = True) -> List[RouteData]:
    """Generate a dataset of routes for training.

    Args:
        num_routes: Number of routes to generate
        min_distance_nm: Minimum route distance
        max_distance_nm: Maximum route distance
        include_reverse: Whether to include reverse routes

    Returns:
        List of RouteData objects
    """
    simulator = ExpertFlightSimulator()
    airports = list(TRAINING_AIRPORTS.values())
    dataset = []

    # Generate all valid airport pairs
    pairs = []
    for i, origin in enumerate(airports):
        for j, dest in enumerate(airports):
            if i != j:
                dist = simulator.compute_route_distance(origin, dest)
                if min_distance_nm <= dist <= max_distance_nm:
                    pairs.append((origin, dest, dist))

    print(f"Found {len(pairs)} valid airport pairs")

    # Generate routes
    routes_per_pair = max(1, num_routes // len(pairs)) if pairs else 0

    for origin, dest, dist in pairs:
        for _ in range(routes_per_pair):
            route = simulator.generate_route_data(origin, dest, randomize_params=True)
            dataset.append(route)

            if len(dataset) >= num_routes:
                break
        if len(dataset) >= num_routes:
            break

    # Fill remaining with random pairs
    while len(dataset) < num_routes and pairs:
        origin, dest, _ = random.choice(pairs)
        route = simulator.generate_route_data(origin, dest, randomize_params=True)
        dataset.append(route)

    return dataset


def convert_to_tensors(dataset: List[RouteData], max_waypoints: int = 8) -> Dict[str, np.ndarray]:
    """Convert RouteData list to numpy arrays for training.

    Args:
        dataset: List of RouteData objects
        max_waypoints: Maximum number of waypoints to include

    Returns:
        Dictionary of numpy arrays ready for PyTorch
    """
    n = len(dataset)

    # Route specifications (normalized)
    route_specs = np.zeros((n, 10), dtype=np.float32)

    # Waypoints: [n, max_waypoints, 6] (lat, lon, alt, speed, heading, phase_progress)
    waypoints = np.zeros((n, max_waypoints, 6), dtype=np.float32)

    # Waypoint validity: [n, max_waypoints, 1]
    validity = np.zeros((n, max_waypoints, 1), dtype=np.float32)

    # Phase labels: [n, max_waypoints] (class indices 0-5)
    phases = np.zeros((n, max_waypoints), dtype=np.int64)

    # Controller parameters: [n, 8]
    controller_params = np.zeros((n, 8), dtype=np.float32)

    for i, route in enumerate(dataset):
        # Route spec (normalized)
        route_specs[i] = [
            route.origin_lat / 90.0,
            route.origin_lon / 180.0,
            route.origin_elev_ft / 10000.0,
            route.origin_rwy_deg / 360.0,
            route.dest_lat / 90.0,
            route.dest_lon / 180.0,
            route.dest_elev_ft / 10000.0,
            route.dest_rwy_deg / 360.0,
            route.cruise_alt_ft / 10000.0,
            route.distance_nm / 200.0,
        ]

        # Waypoints
        num_wps = min(len(route.waypoints), max_waypoints)
        for j, wp in enumerate(route.waypoints[:num_wps]):
            # Normalize waypoint coordinates relative to origin
            lat_delta = (wp.lat - route.origin_lat) / 2.0  # Max ~2 degrees
            lon_delta = (wp.lon - route.origin_lon) / 2.0

            waypoints[i, j] = [
                lat_delta,
                lon_delta,
                wp.altitude_ft / 10000.0,
                wp.speed_kts / 200.0,
                np.deg2rad(wp.heading_deg) / np.pi,  # Normalize to [-1, 1]
                j / max_waypoints,  # Phase progress (position in sequence)
            ]
            validity[i, j, 0] = 1.0 if wp.valid else 0.0
            phases[i, j] = wp.phase

        # Controller parameters (normalized to output ranges)
        controller_params[i] = [
            route.tp_distance_nm / 20.0,  # 0-20 nm -> 0-1
            (route.glideslope_deg - 2.0) / 5.0,  # 2-7 deg -> 0-1
            (route.pattern_alt_ft - 500.0) / 2000.0,  # 500-2500 ft -> 0-1
            (route.cruise_alt_ft - 2000.0) / 10000.0,  # 2000-12000 ft -> 0-1
            (route.v_approach_kts - 60.0) / 50.0,  # 60-110 kts -> 0-1
            (route.v_cruise_kts - 80.0) / 60.0,  # 80-140 kts -> 0-1
            (route.turn_bank_deg - 15.0) / 30.0,  # 15-45 deg -> 0-1
            1.0 if route.use_triangle_pattern else 0.0,
        ]

    return {
        'route_specs': route_specs,
        'waypoints': waypoints,
        'validity': validity,
        'phases': phases,
        'controller_params': controller_params,
    }


def save_dataset(data: Dict[str, np.ndarray], output_path: str):
    """Save dataset to npz file."""
    np.savez_compressed(output_path, **data)
    print(f"Saved dataset to {output_path}")

    # Print statistics
    n = data['route_specs'].shape[0]
    print(f"  Routes: {n}")
    print(f"  Route spec shape: {data['route_specs'].shape}")
    print(f"  Waypoints shape: {data['waypoints'].shape}")
    print(f"  Controller params shape: {data['controller_params'].shape}")


def main():
    parser = argparse.ArgumentParser(description="Generate route training data")
    parser.add_argument("--num-routes", type=int, default=100,
                       help="Number of routes to generate")
    parser.add_argument("--output", type=str, default="route_training_data.npz",
                       help="Output file path")
    parser.add_argument("--min-distance", type=float, default=10.0,
                       help="Minimum route distance in nm")
    parser.add_argument("--max-distance", type=float, default=100.0,
                       help="Maximum route distance in nm")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Route Training Data Generator")
    print("=" * 60)
    print(f"Generating {args.num_routes} routes...")
    print(f"Distance range: {args.min_distance}-{args.max_distance} nm")

    # Generate dataset
    dataset = generate_training_dataset(
        num_routes=args.num_routes,
        min_distance_nm=args.min_distance,
        max_distance_nm=args.max_distance
    )

    print(f"\nGenerated {len(dataset)} routes")

    # Convert to tensors
    tensor_data = convert_to_tensors(dataset)

    # Save
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / args.output

    save_dataset(tensor_data, str(output_path))

    # Also save metadata as JSON
    metadata = {
        'num_routes': len(dataset),
        'airports': list(TRAINING_AIRPORTS.keys()),
        'max_waypoints': 8,
        'waypoint_dim': 6,
        'route_input_dim': 10,
        'controller_param_dim': 8,
        'generated_at': datetime.now().isoformat(),
        'seed': args.seed,
    }

    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    print("\n" + "=" * 60)
    print("Data generation complete!")


if __name__ == "__main__":
    main()
