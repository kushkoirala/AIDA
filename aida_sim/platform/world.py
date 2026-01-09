"""
World Coordinate System for AIDA Flight Simulation

Provides coordinate transforms between:
- Geodetic (lat/lon/alt)
- Local NED (North-East-Down) frames
- Runway-aligned coordinates

The simulation uses a local tangent plane (LTP) with origin at a reference
airport. All positions are in meters relative to this origin.

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .airports import Airport, Runway, RunwayEnd, get_airport


# WGS84 Earth parameters
WGS84_A = 6378137.0           # Semi-major axis (m)
WGS84_B = 6356752.314245      # Semi-minor axis (m)
WGS84_E2 = 0.00669437999014   # First eccentricity squared
WGS84_F = 1 / 298.257223563   # Flattening


@dataclass
class WorldOrigin:
    """
    Reference point for the local tangent plane (LTP).

    All simulation coordinates are relative to this point.
    The X-axis points North, Y-axis points East, Z-axis points Down (NED).
    """
    lat: float          # Latitude (degrees)
    lon: float          # Longitude (degrees)
    alt_msl: float      # Altitude MSL (meters)

    # Cached transform parameters
    _lat_rad: float = None
    _lon_rad: float = None
    _cos_lat: float = None
    _sin_lat: float = None
    _meters_per_deg_lat: float = None
    _meters_per_deg_lon: float = None

    def __post_init__(self):
        """Precompute transform constants."""
        self._lat_rad = np.radians(self.lat)
        self._lon_rad = np.radians(self.lon)
        self._cos_lat = np.cos(self._lat_rad)
        self._sin_lat = np.sin(self._lat_rad)

        # Meters per degree at this latitude
        # Using simplified spherical approximation for small distances
        R = 6371000.0  # Earth radius in meters
        self._meters_per_deg_lat = R * np.pi / 180.0
        self._meters_per_deg_lon = R * np.cos(self._lat_rad) * np.pi / 180.0


class World:
    """
    World coordinate system manager.

    Provides transforms between geodetic coordinates and the local NED frame.
    The origin is typically set at the departure airport.
    """

    def __init__(self, origin_airport: str = None, origin: WorldOrigin = None):
        """
        Initialize world with origin at specified airport.

        Args:
            origin_airport: ICAO code of origin airport (e.g., "SN65")
            origin: Direct WorldOrigin specification (overrides origin_airport)
        """
        if origin is not None:
            self.origin = origin
        elif origin_airport is not None:
            apt = get_airport(origin_airport)
            if apt is None:
                raise ValueError(f"Airport not found: {origin_airport}")
            self.origin = WorldOrigin(
                lat=apt.lat,
                lon=apt.lon,
                alt_msl=apt.elevation_m
            )
        else:
            raise ValueError("Must specify origin_airport or origin")

        # Cache airports in local coordinates
        self._airport_positions = {}

    def geodetic_to_ned(self, lat: float, lon: float, alt_msl: float = 0.0) -> Tuple[float, float, float]:
        """
        Convert geodetic coordinates to local NED.

        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            alt_msl: Altitude above mean sea level (meters)

        Returns:
            (north, east, down) in meters relative to origin
        """
        dlat = lat - self.origin.lat
        dlon = lon - self.origin.lon
        dalt = alt_msl - self.origin.alt_msl

        north = dlat * self.origin._meters_per_deg_lat
        east = dlon * self.origin._meters_per_deg_lon
        down = -dalt  # NED: positive down

        return north, east, down

    def ned_to_geodetic(self, north: float, east: float, down: float) -> Tuple[float, float, float]:
        """
        Convert local NED to geodetic coordinates.

        Args:
            north: North position (meters)
            east: East position (meters)
            down: Down position (meters, positive = below origin)

        Returns:
            (lat, lon, alt_msl) in degrees and meters
        """
        dlat = north / self.origin._meters_per_deg_lat
        dlon = east / self.origin._meters_per_deg_lon
        dalt = -down

        lat = self.origin.lat + dlat
        lon = self.origin.lon + dlon
        alt_msl = self.origin.alt_msl + dalt

        return lat, lon, alt_msl

    def get_airport_position(self, icao: str) -> Tuple[float, float, float]:
        """
        Get airport reference point in local NED coordinates.

        Returns:
            (north, east, down) of airport center
        """
        if icao in self._airport_positions:
            return self._airport_positions[icao]

        apt = get_airport(icao)
        if apt is None:
            raise ValueError(f"Airport not found: {icao}")

        pos = self.geodetic_to_ned(apt.lat, apt.lon, apt.elevation_m)
        self._airport_positions[icao] = pos
        return pos

    def get_runway_geometry(self, icao: str, runway_designator: str) -> dict:
        """
        Get runway geometry in local NED coordinates.

        Args:
            icao: Airport ICAO code
            runway_designator: Runway designator (e.g., "35", "13")

        Returns:
            Dictionary with:
                - threshold: (north, east, down) of runway threshold
                - opposite: (north, east, down) of opposite end
                - heading: True heading (radians)
                - length: Runway length (meters)
                - width: Runway width (meters)
                - centerline_start: Position at start of centerline
                - centerline_dir: Unit vector along centerline
        """
        apt = get_airport(icao)
        if apt is None:
            raise ValueError(f"Airport not found: {icao}")

        result = apt.get_runway_by_designator(runway_designator)
        if result is None:
            raise ValueError(f"Runway {runway_designator} not found at {icao}")

        runway, end = result
        opposite = runway.get_opposite_end(runway_designator)

        # Get threshold positions
        threshold = self.geodetic_to_ned(end.lat, end.lon, end.elevation_ft * 0.3048)
        opposite_pos = self.geodetic_to_ned(opposite.lat, opposite.lon, opposite.elevation_ft * 0.3048)

        # Calculate centerline direction
        dx = opposite_pos[0] - threshold[0]
        dy = opposite_pos[1] - threshold[1]
        length = np.sqrt(dx**2 + dy**2)

        centerline_dir = np.array([dx/length, dy/length, 0.0])

        return {
            'threshold': threshold,
            'opposite': opposite_pos,
            'heading_rad': np.radians(end.heading_true),
            'heading_deg': end.heading_true,
            'length': runway.length_m,
            'width': runway.width_m,
            'centerline_start': np.array(threshold),
            'centerline_dir': centerline_dir,
            'surface': runway.surface.value,
            'traffic_pattern': end.traffic_pattern.value,
            'glide_slope_deg': end.glide_slope_deg,
            'has_ils': end.has_ils,
            'elevation_ft': end.elevation_ft,
        }

    def get_pattern_waypoints(self, icao: str, runway_designator: str,
                               pattern_altitude_agl_ft: float = 1000.0) -> dict:
        """
        Calculate traffic pattern waypoints for a runway.

        Standard traffic pattern legs:
        - Upwind: Departure leg, climbing
        - Crosswind: 90-degree turn after departure
        - Downwind: Parallel to runway, opposite direction
        - Base: 90-degree turn toward runway
        - Final: Aligned with runway for landing

        Args:
            icao: Airport ICAO code
            runway_designator: Landing runway (e.g., "35")
            pattern_altitude_agl_ft: Pattern altitude above ground

        Returns:
            Dictionary with waypoints for each leg
        """
        rwy = self.get_runway_geometry(icao, runway_designator)
        apt = get_airport(icao)

        # Pattern altitude (MSL)
        pattern_alt_m = apt.elevation_m + pattern_altitude_agl_ft * 0.3048
        pattern_down = -(pattern_alt_m - self.origin.alt_msl)

        # Runway parameters
        threshold = np.array(rwy['threshold'])
        opposite = np.array(rwy['opposite'])
        heading = rwy['heading_rad']
        length = rwy['length']

        # Determine pattern direction (left or right)
        is_left_pattern = rwy['traffic_pattern'] == 'left'
        turn_dir = -1 if is_left_pattern else 1  # -1 for left turns

        # Pattern dimensions (standard)
        downwind_offset = 800.0  # meters from runway centerline
        pattern_leg_length = 1500.0  # meters

        # Unit vectors
        along_rwy = rwy['centerline_dir'][:2]  # North, East
        perp_rwy = np.array([-along_rwy[1] * turn_dir, along_rwy[0] * turn_dir])

        # Calculate waypoints
        waypoints = {}

        # Departure end (lift off point, mid-runway)
        departure_end = (threshold[:2] + opposite[:2]) / 2
        waypoints['departure'] = np.array([departure_end[0], departure_end[1], 0.0])

        # Upwind (end of upwind leg)
        upwind = opposite[:2] + along_rwy * pattern_leg_length
        waypoints['upwind'] = np.array([upwind[0], upwind[1], pattern_down])

        # Crosswind turn point
        crosswind = upwind + perp_rwy * downwind_offset
        waypoints['crosswind'] = np.array([crosswind[0], crosswind[1], pattern_down])

        # Downwind (abeam threshold)
        downwind_abeam = threshold[:2] + perp_rwy * downwind_offset
        waypoints['downwind_abeam'] = np.array([downwind_abeam[0], downwind_abeam[1], pattern_down])

        # Base turn point (past threshold)
        base_start = threshold[:2] - along_rwy * pattern_leg_length * 0.5 + perp_rwy * downwind_offset
        waypoints['base'] = np.array([base_start[0], base_start[1], pattern_down])

        # Final approach (short final)
        final = threshold[:2] - along_rwy * 500.0  # 500m from threshold
        final_alt = apt.elevation_m + 150 * 0.3048  # ~500 ft AGL
        waypoints['final'] = np.array([final[0], final[1], -(final_alt - self.origin.alt_msl)])

        # Threshold (landing point)
        waypoints['threshold'] = np.array([threshold[0], threshold[1], 0.0])

        return {
            'waypoints': waypoints,
            'pattern_altitude_m': pattern_alt_m,
            'is_left_pattern': is_left_pattern,
            'runway_heading_deg': rwy['heading_deg'],
        }

    def get_cross_country_route(self, dep_icao: str, dep_runway: str,
                                 arr_icao: str, arr_runway: str,
                                 cruise_altitude_ft: float = 3500.0) -> dict:
        """
        Calculate waypoints for a cross-country flight.

        Args:
            dep_icao: Departure airport
            dep_runway: Departure runway
            arr_icao: Arrival airport
            arr_runway: Arrival runway
            cruise_altitude_ft: Cruise altitude MSL

        Returns:
            Dictionary with route waypoints and metadata
        """
        from .airports import distance_between_airports, bearing_between_airports

        # Get airport and runway info
        dep_apt = get_airport(dep_icao)
        arr_apt = get_airport(arr_icao)

        dep_rwy = self.get_runway_geometry(dep_icao, dep_runway)
        arr_rwy = self.get_runway_geometry(arr_icao, arr_runway)

        # Positions
        dep_pos = self.get_airport_position(dep_icao)
        arr_pos = self.get_airport_position(arr_icao)

        # Route metadata
        distance_nm = distance_between_airports(dep_icao, arr_icao)
        course = bearing_between_airports(dep_icao, arr_icao)

        # Cruise altitude in local NED
        cruise_alt_m = cruise_altitude_ft * 0.3048
        cruise_down = -(cruise_alt_m - self.origin.alt_msl)

        # Waypoints
        waypoints = {}

        # Departure
        waypoints['dep_threshold'] = np.array(dep_rwy['threshold'])

        # Top of climb (10 NM from departure or halfway, whichever is less)
        toc_dist = min(10.0 * 1852, distance_nm * 1852 / 3)  # meters
        course_rad = np.radians(course)
        toc_offset = np.array([toc_dist * np.cos(course_rad), toc_dist * np.sin(course_rad)])
        waypoints['top_of_climb'] = np.array([
            dep_pos[0] + toc_offset[0],
            dep_pos[1] + toc_offset[1],
            cruise_down
        ])

        # Top of descent (10 NM from arrival or 2/3 of way)
        tod_dist = max(10.0 * 1852, distance_nm * 1852 / 3)
        tod_offset = np.array([tod_dist * np.cos(course_rad), tod_dist * np.sin(course_rad)])
        waypoints['top_of_descent'] = np.array([
            arr_pos[0] - tod_offset[0] + dep_pos[0],
            arr_pos[1] - tod_offset[1] + dep_pos[1],
            cruise_down
        ])

        # Initial approach fix (5 NM from runway)
        iaf_dist = 5.0 * 1852
        arr_heading_rad = np.radians(arr_rwy['heading_deg'] + 180)  # Inbound heading
        waypoints['initial_approach'] = np.array([
            arr_rwy['threshold'][0] + iaf_dist * np.cos(arr_heading_rad),
            arr_rwy['threshold'][1] + iaf_dist * np.sin(arr_heading_rad),
            -(arr_apt.elevation_m + 500 * 0.3048 - self.origin.alt_msl)  # 500 ft AGL
        ])

        # Final approach
        waypoints['final_approach'] = np.array([
            arr_rwy['threshold'][0] + 1000 * np.cos(arr_heading_rad),
            arr_rwy['threshold'][1] + 1000 * np.sin(arr_heading_rad),
            -(arr_apt.elevation_m + 100 * 0.3048 - self.origin.alt_msl)  # 100 ft AGL
        ])

        # Arrival threshold
        waypoints['arr_threshold'] = np.array(arr_rwy['threshold'])

        return {
            'waypoints': waypoints,
            'distance_nm': distance_nm,
            'course_true': course,
            'cruise_altitude_ft': cruise_altitude_ft,
            'dep_elevation_ft': dep_apt.elevation_ft,
            'arr_elevation_ft': arr_apt.elevation_ft,
        }


if __name__ == "__main__":
    # Test the world coordinate system
    print("AIDA World Coordinate System Test")
    print("=" * 50)

    # Create world centered at SN65
    world = World(origin_airport="SN65")
    print(f"\nWorld origin: SN65 (Lake Waltanna)")
    print(f"  Lat: {world.origin.lat:.4f}")
    print(f"  Lon: {world.origin.lon:.4f}")
    print(f"  Elev: {world.origin.alt_msl:.1f} m MSL")

    # Get KHUT position relative to SN65
    khut_pos = world.get_airport_position("KHUT")
    print(f"\nKHUT position relative to SN65:")
    print(f"  North: {khut_pos[0]/1000:.2f} km")
    print(f"  East: {khut_pos[1]/1000:.2f} km")
    print(f"  Down: {khut_pos[2]:.1f} m")

    # Get runway geometry
    rwy = world.get_runway_geometry("SN65", "35")
    print(f"\nSN65 Runway 35 geometry:")
    print(f"  Threshold: N={rwy['threshold'][0]:.1f}m, E={rwy['threshold'][1]:.1f}m")
    print(f"  Heading: {rwy['heading_deg']:.0f} deg")
    print(f"  Length: {rwy['length']:.0f} m")
    print(f"  Surface: {rwy['surface']}")

    # Get traffic pattern
    pattern = world.get_pattern_waypoints("SN65", "35")
    print(f"\nTraffic pattern for SN65 RWY 35:")
    print(f"  Pattern altitude: {pattern['pattern_altitude_m']:.0f} m MSL")
    print(f"  Left pattern: {pattern['is_left_pattern']}")
    for name, pos in pattern['waypoints'].items():
        print(f"  {name}: N={pos[0]:.0f}m, E={pos[1]:.0f}m, D={pos[2]:.0f}m")

    # Get cross-country route
    route = world.get_cross_country_route("SN65", "35", "KHUT", "13")
    print(f"\nCross-country SN65 -> KHUT:")
    print(f"  Distance: {route['distance_nm']:.1f} NM")
    print(f"  Course: {route['course_true']:.0f} deg")
    print(f"  Cruise: {route['cruise_altitude_ft']:.0f} ft MSL")
