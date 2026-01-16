#!/usr/bin/env python3
"""
AIDA Mission Planner

High-level planning module that:
1. Takes a destination command (e.g., "fly to KHUT")
2. Gets weather/wind information
3. Determines active runway based on wind
4. Selects approach pattern (triangle vs teardrop)
5. Generates waypoint sequence for the NN flight controller

This is the "brain" that decides WHERE to fly.
The NN flight controller executes HOW to fly there.

Usage:
    planner = MissionPlanner()
    waypoints = planner.plan_flight(
        origin='SN65',
        destination='KHUT',
        wind_direction=270,  # Wind from west
        wind_speed=10,       # 10 knots
    )

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum, IntEnum


class WaypointType(IntEnum):
    """Types of waypoints."""
    FLYOVER = 0
    FLYBY = 1
    HOLD = 2
    LAND = 3
    TAKEOFF = 4


class ApproachPattern(Enum):
    """Types of approach patterns."""
    TRIANGLE = "triangle"      # Standard intercept approach
    TEARDROP = "teardrop"      # Course reversal approach
    STRAIGHT_IN = "straight_in"  # Direct approach (wind aligned)
    OVERHEAD = "overhead"      # Overhead break pattern


@dataclass
class Waypoint:
    """Waypoint definition."""
    x: float              # Position X (meters)
    y: float              # Position Y (meters)
    altitude: float       # Target altitude (meters)
    speed: float          # Target airspeed (m/s)
    wp_type: WaypointType # Waypoint type
    name: str = ""        # Optional name for debugging
    heading: Optional[float] = None  # Required heading at waypoint (radians)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for NN input."""
        return np.array([
            self.x, self.y, self.altitude, self.speed, float(self.wp_type)
        ], dtype=np.float32)


@dataclass
class Airport:
    """Airport definition."""
    name: str
    icao: str
    x: float              # Position X (meters from origin)
    y: float              # Position Y (meters from origin)
    elevation: float      # Field elevation (meters)
    runways: List[Dict]   # List of runway definitions


# Airport Database
AIRPORTS = {
    'SN65': Airport(
        name="SN65 Private",
        icao="SN65",
        x=0.0,
        y=0.0,
        elevation=0.0,
        runways=[
            {'id': '36', 'heading': 0.0, 'length': 1000.0, 'width': 30.0},
            {'id': '18', 'heading': 180.0, 'length': 1000.0, 'width': 30.0},
        ]
    ),
    'KHUT': Airport(
        name="Hutchinson Municipal",
        icao="KHUT",
        x=52800.0,
        y=-21300.0,
        elevation=467.0,  # ~1532 ft
        runways=[
            {'id': '31', 'heading': 314.0, 'length': 2134.0, 'width': 23.0},
            {'id': '13', 'heading': 134.0, 'length': 2134.0, 'width': 23.0},
        ]
    ),
}


class MissionPlanner:
    """
    High-level mission planning for autonomous flight.

    Takes destination commands and generates waypoint sequences
    that the NN flight controller can follow.
    """

    def __init__(self):
        self.airports = AIRPORTS

        # Default aircraft performance
        self.cruise_altitude = 1676.0   # 5500 ft
        self.pattern_altitude = 457.0   # 1500 ft AGL
        self.cruise_speed = 56.0        # 110 KIAS in m/s
        self.approach_speed = 33.0      # 65 KIAS in m/s
        self.climb_speed = 38.0         # 75 KIAS in m/s

        # Approach parameters
        self.glideslope_angle = 3.0     # degrees
        self.final_approach_dist = 5556.0  # 3 nm in meters
        self.pattern_entry_dist = 9260.0   # 5 nm in meters

    def plan_flight(self,
                    origin: str,
                    destination: str,
                    wind_direction: float = 0.0,
                    wind_speed: float = 0.0,
                    current_position: Optional[Tuple[float, float, float]] = None,
                    ) -> List[Waypoint]:
        """
        Plan a complete flight from origin to destination.

        Args:
            origin: Origin airport ICAO code
            destination: Destination airport ICAO code
            wind_direction: Wind FROM direction in degrees (0=north)
            wind_speed: Wind speed in knots
            current_position: Current (x, y, alt) if already airborne

        Returns:
            List of Waypoint objects for the NN to follow
        """
        orig_airport = self.airports.get(origin)
        dest_airport = self.airports.get(destination)

        if not orig_airport or not dest_airport:
            raise ValueError(f"Unknown airport: {origin} or {destination}")

        # Determine active runway at destination based on wind
        active_runway = self._select_runway(dest_airport, wind_direction, wind_speed)

        print(f"Mission: {origin} → {destination}")
        print(f"Wind: {wind_direction:.0f}° at {wind_speed:.0f} kts")
        print(f"Active runway: {active_runway['id']} (heading {active_runway['heading']:.0f}°)")

        # Determine approach pattern based on approach direction
        if current_position:
            approach_from = (current_position[0], current_position[1])
        else:
            approach_from = (orig_airport.x, orig_airport.y)

        pattern = self._select_approach_pattern(
            dest_airport, active_runway, approach_from
        )
        print(f"Approach pattern: {pattern.value}")

        # Generate waypoint sequence
        waypoints = []

        # 1. Departure waypoints (if starting from ground)
        if current_position is None:
            waypoints.extend(self._generate_departure(orig_airport))

        # 2. Enroute waypoints
        waypoints.extend(self._generate_enroute(
            orig_airport if current_position is None else None,
            dest_airport,
            current_position,
        ))

        # 3. Approach waypoints based on selected pattern
        if pattern == ApproachPattern.TRIANGLE:
            waypoints.extend(self._generate_triangle_approach(
                dest_airport, active_runway
            ))
        elif pattern == ApproachPattern.TEARDROP:
            waypoints.extend(self._generate_teardrop_approach(
                dest_airport, active_runway
            ))
        elif pattern == ApproachPattern.STRAIGHT_IN:
            waypoints.extend(self._generate_straight_in_approach(
                dest_airport, active_runway
            ))

        # 4. Landing waypoints
        waypoints.extend(self._generate_landing(dest_airport, active_runway))

        return waypoints

    def _select_runway(self, airport: Airport, wind_dir: float, wind_speed: float) -> Dict:
        """
        Select active runway based on wind.

        Prefers runway most aligned with wind (land INTO wind).
        """
        if wind_speed < 5:
            # Light wind - use default (first runway)
            return airport.runways[0]

        best_runway = None
        best_headwind = -float('inf')

        for runway in airport.runways:
            # Calculate headwind component
            rwy_heading = np.deg2rad(runway['heading'])
            wind_from = np.deg2rad(wind_dir)

            # Headwind = wind_speed * cos(angle between runway and wind)
            angle_diff = wind_from - rwy_heading
            headwind = wind_speed * np.cos(angle_diff)

            if headwind > best_headwind:
                best_headwind = headwind
                best_runway = runway

        return best_runway

    def _select_approach_pattern(self,
                                  airport: Airport,
                                  runway: Dict,
                                  approach_from: Tuple[float, float]) -> ApproachPattern:
        """
        Select approach pattern based on approach direction relative to runway.

        Triangle: Approaching from behind runway (need to fly past and turn back)
        Teardrop: Approaching from front but wrong angle (need course reversal)
        Straight-in: Approaching aligned with runway
        """
        rwy_heading = np.deg2rad(runway['heading'])

        # Vector from approach point to airport
        dx = airport.x - approach_from[0]
        dy = airport.y - approach_from[1]
        approach_bearing = np.arctan2(dy, dx)

        # Angle between approach direction and runway heading
        angle_diff = self._normalize_angle(approach_bearing - rwy_heading)
        angle_diff_deg = abs(np.rad2deg(angle_diff))

        print(f"  Approach bearing: {np.rad2deg(approach_bearing):.0f}°")
        print(f"  Runway heading: {runway['heading']:.0f}°")
        print(f"  Angle difference: {angle_diff_deg:.0f}°")

        if angle_diff_deg < 30:
            # Nearly aligned - straight in approach
            return ApproachPattern.STRAIGHT_IN
        elif angle_diff_deg < 90:
            # Coming from the side - teardrop to align
            return ApproachPattern.TEARDROP
        else:
            # Coming from behind - triangle pattern
            return ApproachPattern.TRIANGLE

    def _generate_departure(self, airport: Airport) -> List[Waypoint]:
        """Generate departure waypoints."""
        waypoints = []

        # Takeoff waypoint (end of runway)
        rwy = airport.runways[0]
        rwy_heading = np.deg2rad(rwy['heading'])

        # Climb-out waypoint (1nm from airport)
        climb_dist = 1852.0  # 1 nm
        waypoints.append(Waypoint(
            x=airport.x + climb_dist * np.cos(rwy_heading),
            y=airport.y + climb_dist * np.sin(rwy_heading),
            altitude=300.0,  # ~1000 ft AGL
            speed=self.climb_speed,
            wp_type=WaypointType.FLYOVER,
            name="CLIMB_OUT",
        ))

        return waypoints

    def _generate_enroute(self,
                          origin: Optional[Airport],
                          destination: Airport,
                          current_pos: Optional[Tuple[float, float, float]]) -> List[Waypoint]:
        """Generate enroute waypoints."""
        waypoints = []

        # Starting point
        if current_pos:
            start_x, start_y, start_alt = current_pos
        elif origin:
            start_x, start_y = origin.x, origin.y
            start_alt = 0.0
        else:
            return waypoints

        # Calculate direct distance
        dx = destination.x - start_x
        dy = destination.y - start_y
        distance = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx)

        # Top of climb (reach cruise altitude)
        toc_dist = min(distance * 0.3, 15000.0)  # 30% of route or 15km max
        waypoints.append(Waypoint(
            x=start_x + toc_dist * np.cos(bearing),
            y=start_y + toc_dist * np.sin(bearing),
            altitude=self.cruise_altitude,
            speed=self.cruise_speed,
            wp_type=WaypointType.FLYOVER,
            name="TOP_OF_CLIMB",
        ))

        # Top of descent (start descent to pattern altitude)
        tod_dist = distance - self.pattern_entry_dist - 5000  # 5km before pattern entry
        if tod_dist > toc_dist + 5000:  # Only if there's cruise segment
            waypoints.append(Waypoint(
                x=start_x + tod_dist * np.cos(bearing),
                y=start_y + tod_dist * np.sin(bearing),
                altitude=self.cruise_altitude,
                speed=self.cruise_speed,
                wp_type=WaypointType.FLYOVER,
                name="TOP_OF_DESCENT",
            ))

        return waypoints

    def _generate_triangle_approach(self,
                                     airport: Airport,
                                     runway: Dict) -> List[Waypoint]:
        """
        Generate triangle intercept approach.

        Aircraft flies PAST the airport, then turns back to intercept final.

                    TP (turning point)
                   /
                  /
        Aircraft → ──────── → Final → Runway
        """
        waypoints = []

        rwy_heading = np.deg2rad(runway['heading'])
        backcourse = rwy_heading + np.pi  # Opposite direction

        # Runway threshold
        threshold_dist = runway['length'] / 2
        threshold_x = airport.x - threshold_dist * np.cos(rwy_heading)
        threshold_y = airport.y - threshold_dist * np.sin(rwy_heading)

        # Turning point: 10nm behind threshold on backcourse
        tp_dist = 10 * 1852.0  # 10 nm
        tp_x = threshold_x + tp_dist * np.cos(backcourse)
        tp_y = threshold_y + tp_dist * np.sin(backcourse)

        waypoints.append(Waypoint(
            x=tp_x,
            y=tp_y,
            altitude=self.cruise_altitude,
            speed=self.cruise_speed,
            wp_type=WaypointType.FLYBY,
            name="TURNING_POINT",
        ))

        # Final approach fix: 3nm from threshold
        faf_dist = 3 * 1852.0
        faf_x = threshold_x - faf_dist * np.cos(rwy_heading)
        faf_y = threshold_y - faf_dist * np.sin(rwy_heading)
        faf_alt = faf_dist * np.tan(np.deg2rad(self.glideslope_angle))

        waypoints.append(Waypoint(
            x=faf_x,
            y=faf_y,
            altitude=self.pattern_altitude,
            speed=self.approach_speed,
            wp_type=WaypointType.FLYOVER,
            name="FINAL_APPROACH_FIX",
            heading=rwy_heading,
        ))

        return waypoints

    def _generate_teardrop_approach(self,
                                     airport: Airport,
                                     runway: Dict) -> List[Waypoint]:
        """
        Generate teardrop course reversal approach.

        Aircraft flies to a fix, turns outbound 30° offset,
        then turns back to intercept final.

                    Outbound leg
                   /
        Fix ──────┘
          \
           \──→ Final → Runway
        """
        waypoints = []

        rwy_heading = np.deg2rad(runway['heading'])
        backcourse = rwy_heading + np.pi

        # Runway threshold
        threshold_dist = runway['length'] / 2
        threshold_x = airport.x - threshold_dist * np.cos(rwy_heading)
        threshold_y = airport.y - threshold_dist * np.sin(rwy_heading)

        # Teardrop fix: 5nm from threshold on final approach course
        fix_dist = 5 * 1852.0
        fix_x = threshold_x - fix_dist * np.cos(rwy_heading)
        fix_y = threshold_y - fix_dist * np.sin(rwy_heading)

        waypoints.append(Waypoint(
            x=fix_x,
            y=fix_y,
            altitude=self.pattern_altitude,
            speed=self.approach_speed,
            wp_type=WaypointType.FLYBY,
            name="TEARDROP_FIX",
        ))

        # Outbound point: 30° offset, 2nm outbound
        outbound_heading = backcourse - np.deg2rad(30.0)  # 30° left of backcourse
        outbound_dist = 2 * 1852.0
        outbound_x = fix_x + outbound_dist * np.cos(outbound_heading)
        outbound_y = fix_y + outbound_dist * np.sin(outbound_heading)

        waypoints.append(Waypoint(
            x=outbound_x,
            y=outbound_y,
            altitude=self.pattern_altitude,
            speed=self.approach_speed,
            wp_type=WaypointType.FLYBY,
            name="TEARDROP_OUTBOUND",
        ))

        # Final approach fix
        faf_dist = 3 * 1852.0
        faf_x = threshold_x - faf_dist * np.cos(rwy_heading)
        faf_y = threshold_y - faf_dist * np.sin(rwy_heading)

        waypoints.append(Waypoint(
            x=faf_x,
            y=faf_y,
            altitude=self.pattern_altitude,
            speed=self.approach_speed,
            wp_type=WaypointType.FLYOVER,
            name="FINAL_APPROACH_FIX",
            heading=rwy_heading,
        ))

        return waypoints

    def _generate_straight_in_approach(self,
                                        airport: Airport,
                                        runway: Dict) -> List[Waypoint]:
        """Generate straight-in approach (already aligned)."""
        waypoints = []

        rwy_heading = np.deg2rad(runway['heading'])

        # Runway threshold
        threshold_dist = runway['length'] / 2
        threshold_x = airport.x - threshold_dist * np.cos(rwy_heading)
        threshold_y = airport.y - threshold_dist * np.sin(rwy_heading)

        # Final approach fix: 5nm out
        faf_dist = 5 * 1852.0
        faf_x = threshold_x - faf_dist * np.cos(rwy_heading)
        faf_y = threshold_y - faf_dist * np.sin(rwy_heading)

        waypoints.append(Waypoint(
            x=faf_x,
            y=faf_y,
            altitude=self.pattern_altitude,
            speed=self.approach_speed,
            wp_type=WaypointType.FLYOVER,
            name="FINAL_APPROACH_FIX",
            heading=rwy_heading,
        ))

        return waypoints

    def _generate_landing(self, airport: Airport, runway: Dict) -> List[Waypoint]:
        """Generate landing waypoints."""
        waypoints = []

        rwy_heading = np.deg2rad(runway['heading'])

        # Runway threshold
        threshold_dist = runway['length'] / 2
        threshold_x = airport.x - threshold_dist * np.cos(rwy_heading)
        threshold_y = airport.y - threshold_dist * np.sin(rwy_heading)

        # Touchdown point (1000ft past threshold)
        td_dist = 300.0  # ~1000 ft
        td_x = threshold_x + td_dist * np.cos(rwy_heading)
        td_y = threshold_y + td_dist * np.sin(rwy_heading)

        waypoints.append(Waypoint(
            x=td_x,
            y=td_y,
            altitude=0.0,
            speed=26.0,  # ~50 KIAS touchdown
            wp_type=WaypointType.LAND,
            name="TOUCHDOWN",
            heading=rwy_heading,
        ))

        return waypoints

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def test_planner():
    """Test the mission planner."""
    planner = MissionPlanner()

    print("\n" + "=" * 70)
    print("  TEST 1: SN65 → KHUT with West wind (favors RWY 31)")
    print("=" * 70)
    waypoints = planner.plan_flight(
        origin='SN65',
        destination='KHUT',
        wind_direction=270,  # Wind from west
        wind_speed=10,
    )
    print(f"\nGenerated {len(waypoints)} waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  {i+1}. {wp.name}: ({wp.x/1000:.1f}km, {wp.y/1000:.1f}km) "
              f"alt={wp.altitude:.0f}m spd={wp.speed:.0f}m/s type={WaypointType(wp.wp_type).name}")

    print("\n" + "=" * 70)
    print("  TEST 2: SN65 → KHUT with SE wind (favors RWY 13 - teardrop!)")
    print("=" * 70)
    waypoints = planner.plan_flight(
        origin='SN65',
        destination='KHUT',
        wind_direction=135,  # Wind from SE
        wind_speed=15,
    )
    print(f"\nGenerated {len(waypoints)} waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  {i+1}. {wp.name}: ({wp.x/1000:.1f}km, {wp.y/1000:.1f}km) "
              f"alt={wp.altitude:.0f}m spd={wp.speed:.0f}m/s type={WaypointType(wp.wp_type).name}")


if __name__ == "__main__":
    test_planner()
