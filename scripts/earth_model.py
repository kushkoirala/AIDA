#!/usr/bin/env python3
"""
Earth Model Utilities for Flight Simulation

Provides Earth curvature corrections and geodetic transformations for
flight dynamics simulation. Supports both simple curvature correction
(for regional flights) and full WGS84 geodetic transformations.

Author: Kushal Koirala (with Claude Code)
Date: January 2026

Coordinate Systems:
- LLA: Latitude, Longitude, Altitude (geodetic)
- ECEF: Earth-Centered Earth-Fixed (Cartesian)
- NED: North-East-Down (local tangent plane)

WGS84 Ellipsoid Parameters:
- Semi-major axis (a): 6,378,137.0 m
- Semi-minor axis (b): 6,356,752.314245 m
- Flattening (f): 1/298.257223563
- First eccentricity squared (e²): 0.00669437999014
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0                    # Semi-major axis [m]
WGS84_B = 6356752.314245               # Semi-minor axis [m]
WGS84_F = 1.0 / 298.257223563          # Flattening
WGS84_E2 = 0.00669437999014            # First eccentricity squared
WGS84_E_PRIME2 = 0.00673949674228      # Second eccentricity squared

# Mean Earth radius for simple calculations
EARTH_RADIUS_MEAN = 6371000.0          # Mean radius [m]
EARTH_RADIUS_EQUATORIAL = WGS84_A      # Equatorial radius [m]

# Unit conversions
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
FT_TO_M = 0.3048
M_TO_FT = 3.28084
NM_TO_M = 1852.0
M_TO_NM = 1.0 / 1852.0


@dataclass
class GeoPoint:
    """Geographic point in LLA coordinates."""
    lat_deg: float          # Latitude [degrees]
    lon_deg: float          # Longitude [degrees]
    alt_m: float = 0.0      # Altitude above ellipsoid [m]

    @property
    def lat_rad(self) -> float:
        return self.lat_deg * DEG_TO_RAD

    @property
    def lon_rad(self) -> float:
        return self.lon_deg * DEG_TO_RAD


def earth_radius_at_latitude(lat_rad: float) -> float:
    """
    Calculate Earth's radius at a given latitude using WGS84 ellipsoid.

    Args:
        lat_rad: Latitude in radians

    Returns:
        Earth radius at that latitude [m]
    """
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)

    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)

    # Geocentric radius
    R = np.sqrt(
        (WGS84_A**2 * cos_lat)**2 + (WGS84_B**2 * sin_lat)**2
    ) / np.sqrt(
        (WGS84_A * cos_lat)**2 + (WGS84_B * sin_lat)**2
    )

    return R


def curvature_altitude_correction(distance_m: float, earth_radius: float = EARTH_RADIUS_MEAN) -> float:
    """
    Calculate altitude correction due to Earth's curvature.

    On a flat plane, a point at distance d from origin would be at the same
    altitude. On a curved Earth, the actual ground drops below the tangent plane.

    Args:
        distance_m: Horizontal distance from reference point [m]
        earth_radius: Local Earth radius [m]

    Returns:
        Altitude correction (positive = ground is lower than flat model) [m]

    Formula: h = R - sqrt(R² - d²) ≈ d² / (2R) for small d

    Examples:
        - At 10 nm (18.5 km): ~27 m correction
        - At 36 nm (67 km):   ~350 m correction
        - At 100 nm (185 km): ~2700 m correction
    """
    if distance_m < 1.0:
        return 0.0

    # For numerical stability, use exact formula for large distances
    if distance_m > earth_radius * 0.1:
        # Exact: h = R - sqrt(R² - d²)
        return earth_radius - np.sqrt(earth_radius**2 - distance_m**2)
    else:
        # Approximation: h ≈ d² / (2R)
        return distance_m**2 / (2.0 * earth_radius)


def ground_elevation_with_curvature(
    x_m: float,
    y_m: float,
    origin_elevation_m: float,
    dest_elevation_m: float,
    dest_x_m: float,
    dest_y_m: float,
    earth_radius: float = EARTH_RADIUS_MEAN
) -> float:
    """
    Calculate ground elevation at a point, accounting for Earth curvature.

    This provides a ground elevation that can be used in the flat-Earth
    simulator to approximate curved-Earth effects.

    Args:
        x_m, y_m: Position in local NED frame [m]
        origin_elevation_m: Origin airport elevation [m]
        dest_elevation_m: Destination airport elevation [m]
        dest_x_m, dest_y_m: Destination position in NED frame [m]
        earth_radius: Local Earth radius [m]

    Returns:
        Ground elevation at this point [m MSL]
    """
    # Distance from origin
    dist_from_origin = np.sqrt(x_m**2 + y_m**2)

    # Distance from destination
    dist_from_dest = np.sqrt((x_m - dest_x_m)**2 + (y_m - dest_y_m)**2)

    # Total route distance
    route_dist = np.sqrt(dest_x_m**2 + dest_y_m**2)

    # Progress along route (0 = at origin, 1 = at destination)
    if route_dist > 0:
        progress = 1.0 - (dist_from_dest / route_dist)
        progress = np.clip(progress, 0.0, 1.0)
    else:
        progress = 0.0

    # Interpolate base elevation
    base_elevation = origin_elevation_m + progress * (dest_elevation_m - origin_elevation_m)

    # Add curvature correction
    # The ground "drops" below our tangent plane reference
    curvature_drop = curvature_altitude_correction(dist_from_origin, earth_radius)

    # Effective ground elevation in the flat-Earth frame
    return base_elevation - curvature_drop


class EarthModel:
    """
    Earth model for flight simulation with curvature support.

    Provides methods to:
    1. Convert between geodetic (LLA) and local (NED) coordinates
    2. Calculate Earth curvature corrections
    3. Compute great circle distances and bearings

    Usage:
        # Create Earth model centered at origin airport
        earth = EarthModel(origin_lat=38.065, origin_lon=-97.860)

        # Get NED coordinates for destination
        x, y = earth.lla_to_ned(dest_lat, dest_lon, dest_alt)

        # Get ground elevation at a point (with curvature)
        ground = earth.get_ground_elevation(x, y)
    """

    def __init__(
        self,
        origin_lat_deg: float,
        origin_lon_deg: float,
        origin_alt_m: float = 0.0
    ):
        """
        Initialize Earth model with origin point.

        Args:
            origin_lat_deg: Origin latitude [degrees]
            origin_lon_deg: Origin longitude [degrees]
            origin_alt_m: Origin altitude above ellipsoid [m]
        """
        self.origin_lat_deg = origin_lat_deg
        self.origin_lon_deg = origin_lon_deg
        self.origin_alt_m = origin_alt_m

        self.origin_lat_rad = origin_lat_deg * DEG_TO_RAD
        self.origin_lon_rad = origin_lon_deg * DEG_TO_RAD

        # Earth radius at origin
        self.earth_radius = earth_radius_at_latitude(self.origin_lat_rad)

        # Precompute ECEF origin
        self.origin_ecef = self.lla_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m)

        # Precompute rotation matrix (ECEF to NED)
        self._compute_rotation_matrix()

    def _compute_rotation_matrix(self):
        """Compute ECEF to NED rotation matrix at origin."""
        lat = self.origin_lat_rad
        lon = self.origin_lon_rad

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        # Rotation matrix from ECEF to NED
        self.R_ecef_to_ned = np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
            [-sin_lon,            cos_lon,            0.0    ],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
        ])

        # Inverse rotation (NED to ECEF)
        self.R_ned_to_ecef = self.R_ecef_to_ned.T

    @staticmethod
    def lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        """
        Convert geodetic (LLA) to ECEF coordinates.

        Args:
            lat_deg: Latitude [degrees]
            lon_deg: Longitude [degrees]
            alt_m: Altitude above WGS84 ellipsoid [m]

        Returns:
            ECEF coordinates [x, y, z] in meters
        """
        lat = lat_deg * DEG_TO_RAD
        lon = lon_deg * DEG_TO_RAD

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        # Radius of curvature in prime vertical
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)

        x = (N + alt_m) * cos_lat * cos_lon
        y = (N + alt_m) * cos_lat * sin_lon
        z = (N * (1 - WGS84_E2) + alt_m) * sin_lat

        return np.array([x, y, z])

    @staticmethod
    def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert ECEF to geodetic (LLA) coordinates using Bowring's method.

        Args:
            x, y, z: ECEF coordinates [m]

        Returns:
            (lat_deg, lon_deg, alt_m)
        """
        # Longitude is straightforward
        lon_rad = np.arctan2(y, x)

        # Iterative solution for latitude and altitude
        p = np.sqrt(x**2 + y**2)

        # Initial estimate using spherical approximation
        lat_rad = np.arctan2(z, p * (1 - WGS84_E2))

        # Iterate (usually converges in 2-3 iterations)
        for _ in range(10):
            sin_lat = np.sin(lat_rad)
            N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)
            lat_rad_new = np.arctan2(z + WGS84_E2 * N * sin_lat, p)

            if abs(lat_rad_new - lat_rad) < 1e-12:
                break
            lat_rad = lat_rad_new

        # Altitude
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)

        if abs(cos_lat) > 1e-10:
            alt_m = p / cos_lat - N
        else:
            alt_m = abs(z) / abs(sin_lat) - N * (1 - WGS84_E2)

        return lat_rad * RAD_TO_DEG, lon_rad * RAD_TO_DEG, alt_m

    def lla_to_ned(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> Tuple[float, float, float]:
        """
        Convert LLA to local NED coordinates relative to origin.

        Args:
            lat_deg: Latitude [degrees]
            lon_deg: Longitude [degrees]
            alt_m: Altitude above ellipsoid [m]

        Returns:
            (north_m, east_m, down_m) - NED coordinates relative to origin
        """
        # Convert to ECEF
        ecef = self.lla_to_ecef(lat_deg, lon_deg, alt_m)

        # Compute difference from origin
        delta_ecef = ecef - self.origin_ecef

        # Rotate to NED
        ned = self.R_ecef_to_ned @ delta_ecef

        return ned[0], ned[1], ned[2]

    def ned_to_lla(self, north_m: float, east_m: float, down_m: float) -> Tuple[float, float, float]:
        """
        Convert local NED coordinates to LLA.

        Args:
            north_m, east_m, down_m: NED coordinates relative to origin [m]

        Returns:
            (lat_deg, lon_deg, alt_m)
        """
        # NED to ECEF
        ned = np.array([north_m, east_m, down_m])
        delta_ecef = self.R_ned_to_ecef @ ned
        ecef = self.origin_ecef + delta_ecef

        # ECEF to LLA
        return self.ecef_to_lla(ecef[0], ecef[1], ecef[2])

    def great_circle_distance(self, lat1_deg: float, lon1_deg: float,
                               lat2_deg: float, lon2_deg: float) -> float:
        """
        Calculate great circle distance between two points (Haversine formula).

        Args:
            lat1_deg, lon1_deg: First point [degrees]
            lat2_deg, lon2_deg: Second point [degrees]

        Returns:
            Distance [m]
        """
        lat1 = lat1_deg * DEG_TO_RAD
        lon1 = lon1_deg * DEG_TO_RAD
        lat2 = lat2_deg * DEG_TO_RAD
        lon2 = lon2_deg * DEG_TO_RAD

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return self.earth_radius * c

    def initial_bearing(self, lat1_deg: float, lon1_deg: float,
                        lat2_deg: float, lon2_deg: float) -> float:
        """
        Calculate initial bearing from point 1 to point 2.

        Args:
            lat1_deg, lon1_deg: Starting point [degrees]
            lat2_deg, lon2_deg: Ending point [degrees]

        Returns:
            Initial bearing [degrees, 0-360]
        """
        lat1 = lat1_deg * DEG_TO_RAD
        lon1 = lon1_deg * DEG_TO_RAD
        lat2 = lat2_deg * DEG_TO_RAD
        lon2 = lon2_deg * DEG_TO_RAD

        dlon = lon2 - lon1

        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing_rad = np.arctan2(y, x)
        bearing_deg = bearing_rad * RAD_TO_DEG

        return (bearing_deg + 360) % 360

    def get_curvature_correction(self, north_m: float, east_m: float) -> float:
        """
        Get Earth curvature altitude correction at a NED position.

        Args:
            north_m, east_m: NED position relative to origin [m]

        Returns:
            Curvature correction (add to flat-Earth Z to get true altitude) [m]
        """
        distance = np.sqrt(north_m**2 + east_m**2)
        return curvature_altitude_correction(distance, self.earth_radius)


def print_curvature_table():
    """Print a table of Earth curvature corrections at various distances."""
    print("\n" + "="*60)
    print("Earth Curvature Altitude Corrections")
    print("="*60)
    print(f"{'Distance':>12} {'Correction':>12} {'Notes'}")
    print("-"*60)

    distances_nm = [1, 5, 10, 20, 36, 50, 100, 200]

    for d_nm in distances_nm:
        d_m = d_nm * NM_TO_M
        corr_m = curvature_altitude_correction(d_m)
        corr_ft = corr_m * M_TO_FT

        notes = ""
        if d_nm == 36:
            notes = "KHUT-KAAO distance"
        elif d_nm == 100:
            notes = "~1hr flight"

        print(f"{d_nm:>10} nm {corr_ft:>10.1f} ft  {notes}")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Demo: KHUT to KAAO route
    print("\n" + "="*60)
    print("Earth Model Demo: KHUT -> KAAO")
    print("="*60)

    # Airport coordinates (approximate)
    KHUT_LAT = 38.0655
    KHUT_LON = -97.8606
    KHUT_ELEV_FT = 1542.0

    KAAO_LAT = 37.7472
    KAAO_LON = -97.2211
    KAAO_ELEV_FT = 1421.0

    # Create Earth model centered at KHUT
    earth = EarthModel(KHUT_LAT, KHUT_LON, KHUT_ELEV_FT * FT_TO_M)

    # Get KAAO position in NED
    north, east, down = earth.lla_to_ned(KAAO_LAT, KAAO_LON, KAAO_ELEV_FT * FT_TO_M)

    print(f"\nKHUT (Origin):")
    print(f"  Lat: {KHUT_LAT:.4f}°  Lon: {KHUT_LON:.4f}°")
    print(f"  Elevation: {KHUT_ELEV_FT:.0f} ft")

    print(f"\nKAAO (Destination):")
    print(f"  Lat: {KAAO_LAT:.4f}°  Lon: {KAAO_LON:.4f}°")
    print(f"  Elevation: {KAAO_ELEV_FT:.0f} ft")

    print(f"\nKAAO in NED coordinates (from KHUT):")
    print(f"  North: {north:.1f} m ({north*M_TO_FT:.1f} ft)")
    print(f"  East:  {east:.1f} m ({east*M_TO_FT:.1f} ft)")
    print(f"  Down:  {down:.1f} m ({down*M_TO_FT:.1f} ft)")

    # Great circle distance
    gc_dist = earth.great_circle_distance(KHUT_LAT, KHUT_LON, KAAO_LAT, KAAO_LON)
    print(f"\nGreat Circle Distance: {gc_dist:.1f} m ({gc_dist*M_TO_NM:.1f} nm)")

    # Initial bearing
    bearing = earth.initial_bearing(KHUT_LAT, KHUT_LON, KAAO_LAT, KAAO_LON)
    print(f"Initial Bearing: {bearing:.1f}°")

    # Curvature correction at destination
    curvature_corr = earth.get_curvature_correction(north, east)
    print(f"\nCurvature Correction at KAAO:")
    print(f"  {curvature_corr:.1f} m ({curvature_corr*M_TO_FT:.1f} ft)")
    print(f"  (Ground is this much lower than flat-Earth model)")

    # Print curvature table
    print_curvature_table()
