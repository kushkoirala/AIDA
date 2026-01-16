#!/usr/bin/env python3
"""
Trajectory Planner for AIDA

Generates complete 4D trajectories from origin to destination,
including:
- Dubins paths for turns
- Vertical profiles for climb/descent
- Speed profiles for each segment
- B-spline smoothing (optional)

This replaces the hardcoded waypoints in mission_planner.py
with dynamically generated smooth trajectories.

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectory.dubins import compute_dubins_path, sample_dubins_path, compute_turn_radius
from trajectory.trajectory import Trajectory, TrajectoryPoint, FlightSegment


# Unit conversion
FT_TO_M = 0.3048
M_TO_FT = 3.28084
NM_TO_FT = 6076.12
KTS_TO_FPS = 1.68781


@dataclass
class AircraftPerformance:
    """Aircraft performance parameters for trajectory planning."""
    # Speeds (ft/s)
    v_rotate: float = 91.0      # 54 kts
    v_climb: float = 125.0      # 74 kts
    v_cruise: float = 185.0     # 110 kts
    v_approach: float = 110.0   # 65 kts
    v_touchdown: float = 84.0   # 50 kts

    # Climb/descent rates (ft/min)
    climb_rate: float = 500.0
    descent_rate: float = 500.0
    max_descent_rate: float = 1500.0

    # Bank angle for turns
    max_bank_deg: float = 25.0

    @property
    def turn_radius_cruise(self) -> float:
        """Minimum turn radius at cruise speed."""
        return compute_turn_radius(self.v_cruise, self.max_bank_deg)

    @property
    def turn_radius_approach(self) -> float:
        """Minimum turn radius at approach speed."""
        return compute_turn_radius(self.v_approach, self.max_bank_deg)


@dataclass
class Waypoint:
    """Simple waypoint for trajectory planning."""
    x: float  # ft
    y: float  # ft
    altitude: float  # ft
    speed: float = 0.0  # ft/s (0 = use default for segment)
    heading: Optional[float] = None  # radians, None = computed
    name: str = ""
    segment: FlightSegment = FlightSegment.CRUISE


class TrajectoryPlanner:
    """
    Generates smooth trajectories between waypoints.

    Uses Dubins paths for horizontal turns and
    linear interpolation for vertical profiles.
    """

    def __init__(self, performance: Optional[AircraftPerformance] = None):
        """Initialize planner with aircraft performance."""
        self.perf = performance or AircraftPerformance()

        # Glideslope for approach
        self.glideslope_deg = 3.0

    def plan_trajectory(self,
                         waypoints: List[Waypoint],
                         start_heading: float = 0.0,
                         point_spacing_ft: float = 100.0) -> Trajectory:
        """
        Plan a complete trajectory through waypoints.

        Args:
            waypoints: List of waypoints to visit
            start_heading: Initial heading in radians
            point_spacing_ft: Approximate spacing between trajectory points

        Returns:
            Trajectory object with sampled points
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        trajectory = Trajectory(
            name=f"{waypoints[0].name} -> {waypoints[-1].name}",
            origin=waypoints[0].name,
            destination=waypoints[-1].name,
        )

        # Current state
        current_x = waypoints[0].x
        current_y = waypoints[0].y
        current_alt = waypoints[0].altitude
        current_heading = start_heading
        current_time = 0.0
        current_distance = 0.0

        # Process each leg
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]

            # Determine speed for this segment
            if wp2.speed > 0:
                segment_speed = wp2.speed
            elif wp2.segment == FlightSegment.TAKEOFF:
                segment_speed = self.perf.v_climb
            elif wp2.segment == FlightSegment.CLIMB:
                segment_speed = self.perf.v_climb
            elif wp2.segment in (FlightSegment.APPROACH, FlightSegment.LANDING):
                segment_speed = self.perf.v_approach
            else:
                segment_speed = self.perf.v_cruise

            # Determine target heading
            if wp2.heading is not None:
                target_heading = wp2.heading
            else:
                dx = wp2.x - wp1.x
                dy = wp2.y - wp1.y
                target_heading = np.arctan2(dy, dx)

            # Turn radius for this speed
            turn_radius = compute_turn_radius(segment_speed, self.perf.max_bank_deg)

            # Generate horizontal path
            # If heading change is significant, use Dubins path
            heading_change = abs(self._normalize_angle(target_heading - current_heading))

            if heading_change > np.deg2rad(10):
                # Use Dubins path for turn
                dubins = compute_dubins_path(
                    current_x, current_y, current_heading,
                    wp2.x, wp2.y, target_heading,
                    turn_radius
                )
                if dubins:
                    path_points = sample_dubins_path(dubins, max(10, int(dubins.total_length / point_spacing_ft)))
                else:
                    # Fallback to straight line
                    path_points = [(current_x, current_y, current_heading), (wp2.x, wp2.y, target_heading)]
            else:
                # Straight line
                dist = np.sqrt((wp2.x - current_x)**2 + (wp2.y - current_y)**2)
                n_points = max(2, int(dist / point_spacing_ft))
                path_points = []
                for j in range(n_points):
                    t = j / (n_points - 1) if n_points > 1 else 0
                    px = current_x + t * (wp2.x - current_x)
                    py = current_y + t * (wp2.y - current_y)
                    ph = target_heading
                    path_points.append((px, py, ph))

            # Generate vertical profile and add points to trajectory
            for j, (px, py, ph) in enumerate(path_points):
                # Progress along this leg
                if len(path_points) > 1:
                    t = j / (len(path_points) - 1)
                else:
                    t = 1.0

                # Linear altitude interpolation
                pz = current_alt + t * (wp2.altitude - current_alt)

                # Compute climb rate
                if j > 0:
                    prev_x, prev_y, _ = path_points[j-1]
                    horiz_dist = np.sqrt((px - prev_x)**2 + (py - prev_y)**2)
                    prev_alt = current_alt + (j-1)/(len(path_points)-1) * (wp2.altitude - current_alt) if len(path_points) > 1 else current_alt
                    vert_dist = pz - prev_alt
                    if horiz_dist > 0:
                        climb_rate = vert_dist / horiz_dist * segment_speed
                    else:
                        climb_rate = 0.0
                else:
                    climb_rate = 0.0

                # Compute distance from previous point
                if j > 0:
                    prev_x, prev_y, _ = path_points[j-1]
                    prev_alt = current_alt + (j-1)/(len(path_points)-1) * (wp2.altitude - current_alt) if len(path_points) > 1 else current_alt
                    seg_dist = np.sqrt((px - prev_x)**2 + (py - prev_y)**2 + (pz - prev_alt)**2)
                else:
                    seg_dist = 0.0

                current_distance += seg_dist
                current_time += seg_dist / segment_speed if segment_speed > 0 else 0

                # Compute curvature (simple approximation)
                if j > 0 and j < len(path_points) - 1:
                    _, _, h_prev = path_points[j-1]
                    _, _, h_next = path_points[j+1]
                    hdg_change = self._normalize_angle(h_next - h_prev)
                    dist_approx = 2 * point_spacing_ft
                    curvature = hdg_change / dist_approx if dist_approx > 0 else 0.0
                else:
                    curvature = 0.0

                # Add point
                trajectory.add_point(TrajectoryPoint(
                    x=px,
                    y=py,
                    z=pz,
                    t=current_time,
                    s=current_distance,
                    speed=segment_speed,
                    heading=ph,
                    climb_rate=climb_rate,
                    curvature=curvature,
                    segment=wp2.segment,
                    target_airspeed=segment_speed,
                    target_heading=ph,
                ))

            # Update current state
            current_x = wp2.x
            current_y = wp2.y
            current_alt = wp2.altitude
            current_heading = target_heading

        return trajectory

    def plan_approach_trajectory(self,
                                  current_x: float, current_y: float,
                                  current_alt: float, current_heading: float,
                                  runway_x: float, runway_y: float,
                                  runway_heading: float,
                                  runway_elevation: float = 0.0,
                                  final_distance_ft: float = 3 * NM_TO_FT) -> Trajectory:
        """
        Plan an approach trajectory to a runway.

        Creates waypoints for:
        1. Turn to intercept final approach course
        2. Descend on glideslope
        3. Flare and touchdown

        Args:
            current_*: Current aircraft state
            runway_*: Runway parameters
            final_distance_ft: Distance from threshold to start final

        Returns:
            Trajectory for the approach
        """
        waypoints = []

        # Final approach fix
        faf_x = runway_x - final_distance_ft * np.cos(runway_heading)
        faf_y = runway_y - final_distance_ft * np.sin(runway_heading)
        faf_alt = runway_elevation + final_distance_ft * np.tan(np.deg2rad(self.glideslope_deg))

        waypoints.append(Waypoint(
            x=current_x, y=current_y, altitude=current_alt,
            heading=current_heading, name="CURRENT",
            segment=FlightSegment.DESCENT
        ))

        waypoints.append(Waypoint(
            x=faf_x, y=faf_y, altitude=faf_alt,
            heading=runway_heading, name="FAF",
            segment=FlightSegment.APPROACH,
            speed=self.perf.v_approach
        ))

        # Threshold
        waypoints.append(Waypoint(
            x=runway_x, y=runway_y, altitude=runway_elevation + 50,  # 50ft at threshold
            heading=runway_heading, name="THRESHOLD",
            segment=FlightSegment.LANDING,
            speed=self.perf.v_touchdown
        ))

        # Touchdown (300ft past threshold)
        td_x = runway_x + 300 * np.cos(runway_heading)
        td_y = runway_y + 300 * np.sin(runway_heading)

        waypoints.append(Waypoint(
            x=td_x, y=td_y, altitude=runway_elevation,
            heading=runway_heading, name="TOUCHDOWN",
            segment=FlightSegment.LANDING,
            speed=self.perf.v_touchdown
        ))

        return self.plan_trajectory(waypoints, current_heading)

    def plan_departure_trajectory(self,
                                   runway_x: float, runway_y: float,
                                   runway_heading: float,
                                   runway_elevation: float,
                                   cruise_altitude: float,
                                   departure_heading: Optional[float] = None) -> Trajectory:
        """
        Plan a departure trajectory.

        Creates waypoints for:
        1. Takeoff roll
        2. Initial climb
        3. Turn to departure heading
        4. Climb to cruise altitude

        Args:
            runway_*: Runway parameters
            cruise_altitude: Target cruise altitude
            departure_heading: Heading after departure (None = runway heading)

        Returns:
            Trajectory for departure
        """
        if departure_heading is None:
            departure_heading = runway_heading

        waypoints = []

        # Start of takeoff roll
        waypoints.append(Waypoint(
            x=runway_x, y=runway_y, altitude=runway_elevation,
            heading=runway_heading, name="TAKEOFF",
            segment=FlightSegment.TAKEOFF,
            speed=0.0
        ))

        # Liftoff point (1000ft down runway)
        liftoff_x = runway_x + 1000 * np.cos(runway_heading)
        liftoff_y = runway_y + 1000 * np.sin(runway_heading)

        waypoints.append(Waypoint(
            x=liftoff_x, y=liftoff_y, altitude=runway_elevation + 50,
            heading=runway_heading, name="LIFTOFF",
            segment=FlightSegment.TAKEOFF,
            speed=self.perf.v_climb
        ))

        # Initial climb (1nm, 500ft AGL)
        climb1_dist = 1 * NM_TO_FT
        climb1_x = runway_x + climb1_dist * np.cos(runway_heading)
        climb1_y = runway_y + climb1_dist * np.sin(runway_heading)

        waypoints.append(Waypoint(
            x=climb1_x, y=climb1_y, altitude=runway_elevation + 500,
            heading=runway_heading, name="CLIMB1",
            segment=FlightSegment.CLIMB,
            speed=self.perf.v_climb
        ))

        # Turn to departure heading if different (2nm, 1000ft AGL)
        climb2_dist = 2 * NM_TO_FT
        climb2_x = runway_x + climb2_dist * np.cos(departure_heading)
        climb2_y = runway_y + climb2_dist * np.sin(departure_heading)

        waypoints.append(Waypoint(
            x=climb2_x, y=climb2_y, altitude=runway_elevation + 1000,
            heading=departure_heading, name="CLIMB2",
            segment=FlightSegment.CLIMB,
            speed=self.perf.v_climb
        ))

        # Top of climb (continue on departure heading)
        # Compute distance to reach cruise altitude at 500 fpm
        alt_to_gain = cruise_altitude - (runway_elevation + 1000)
        time_to_climb = alt_to_gain / (self.perf.climb_rate / 60)  # seconds
        climb_dist = time_to_climb * self.perf.v_climb

        toc_x = climb2_x + climb_dist * np.cos(departure_heading)
        toc_y = climb2_y + climb_dist * np.sin(departure_heading)

        waypoints.append(Waypoint(
            x=toc_x, y=toc_y, altitude=cruise_altitude,
            heading=departure_heading, name="TOC",
            segment=FlightSegment.CRUISE,
            speed=self.perf.v_cruise
        ))

        return self.plan_trajectory(waypoints, runway_heading)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def test_planner():
    """Test the trajectory planner."""
    print("Trajectory Planner Test")
    print("=" * 60)

    planner = TrajectoryPlanner()

    # Create waypoints for SN65 -> KHUT
    waypoints = [
        Waypoint(x=0, y=0, altitude=1448, name="SN65",
                 heading=np.deg2rad(4), segment=FlightSegment.TAKEOFF),
        Waypoint(x=10000, y=1000, altitude=3000, name="CLIMB",
                 segment=FlightSegment.CLIMB),
        Waypoint(x=100000, y=-40000, altitude=5500, name="CRUISE",
                 segment=FlightSegment.CRUISE),
        Waypoint(x=160000, y=-60000, altitude=5500, name="TP",
                 segment=FlightSegment.CRUISE),
        Waypoint(x=173000, y=-70000, altitude=2000, name="FAF",
                 heading=np.deg2rad(314), segment=FlightSegment.APPROACH),
        Waypoint(x=175000, y=-69000, altitude=1550, name="RUNWAY",
                 heading=np.deg2rad(314), segment=FlightSegment.LANDING),
    ]

    trajectory = planner.plan_trajectory(waypoints, start_heading=np.deg2rad(4))

    print(f"Generated trajectory: {trajectory.name}")
    print(f"Total distance: {trajectory.total_distance / NM_TO_FT:.1f} nm")
    print(f"Total time: {trajectory.total_time / 60:.1f} min")
    print(f"Number of points: {len(trajectory)}")

    # Sample a few points
    print("\nSample points:")
    for i in range(0, len(trajectory), len(trajectory) // 10):
        p = trajectory[i]
        print(f"  {p.s/NM_TO_FT:.1f}nm: ({p.x/1000:.1f}k, {p.y/1000:.1f}k) "
              f"alt={p.z:.0f}ft hdg={np.rad2deg(p.heading):.0f}° "
              f"spd={p.speed * FPS_TO_KTS:.0f}kts seg={p.segment.name}")


if __name__ == "__main__":
    test_planner()
