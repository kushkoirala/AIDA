#!/usr/bin/env python3
"""
Guidance Commands

Defines the interface between guidance laws and the autopilot.
Contains command structures and utilities for computing
guidance commands from trajectory following.

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import IntEnum


class LateralMode(IntEnum):
    """Lateral guidance modes."""
    HEADING = 0      # Hold a heading
    TRACK = 1        # Track a course
    WAYPOINT = 2     # Navigate to waypoint
    ORBIT = 3        # Circle a point


class VerticalMode(IntEnum):
    """Vertical guidance modes."""
    ALTITUDE = 0     # Hold altitude
    VS = 1           # Hold vertical speed
    FPA = 2          # Hold flight path angle
    GLIDESLOPE = 3   # Track glideslope


@dataclass
class GuidanceCommands:
    """
    Guidance commands for the autopilot.

    These are the outputs from guidance laws that
    the autopilot should follow.
    """
    # Lateral
    lateral_mode: LateralMode = LateralMode.HEADING
    target_heading: float = 0.0        # radians
    target_track: float = 0.0          # radians (ground track)
    cross_track_error: float = 0.0     # ft

    # Vertical
    vertical_mode: VerticalMode = VerticalMode.ALTITUDE
    target_altitude: float = 5500.0    # ft
    target_vs: float = 0.0             # ft/s (positive = climb)
    target_fpa: float = 0.0            # radians (flight path angle)
    glideslope_error: float = 0.0      # ft (above/below glideslope)

    # Speed
    target_airspeed: float = 150.0     # ft/s (TAS)

    # Configuration
    target_flaps: float = 0.0          # 0-1
    target_gear: bool = False          # Up/Down

    # Approach flags
    on_approach: bool = False
    on_final: bool = False
    on_glideslope: bool = False

    def copy(self) -> 'GuidanceCommands':
        """Create a copy of the commands."""
        return GuidanceCommands(
            lateral_mode=self.lateral_mode,
            target_heading=self.target_heading,
            target_track=self.target_track,
            cross_track_error=self.cross_track_error,
            vertical_mode=self.vertical_mode,
            target_altitude=self.target_altitude,
            target_vs=self.target_vs,
            target_fpa=self.target_fpa,
            glideslope_error=self.glideslope_error,
            target_airspeed=self.target_airspeed,
            target_flaps=self.target_flaps,
            target_gear=self.target_gear,
            on_approach=self.on_approach,
            on_final=self.on_final,
            on_glideslope=self.on_glideslope,
        )


@dataclass
class FlightState:
    """
    Current flight state for guidance computation.

    All values in imperial units (ft, ft/s).
    """
    # Position
    x: float = 0.0
    y: float = 0.0
    altitude: float = 0.0  # ft AGL or MSL

    # Attitude
    heading: float = 0.0   # radians
    pitch: float = 0.0     # radians
    roll: float = 0.0      # radians

    # Velocities
    airspeed: float = 0.0  # ft/s TAS
    ground_speed: float = 0.0  # ft/s
    vertical_speed: float = 0.0  # ft/s (positive = climb)

    # Rates
    heading_rate: float = 0.0  # rad/s
    pitch_rate: float = 0.0    # rad/s
    roll_rate: float = 0.0     # rad/s

    # Configuration
    flaps: float = 0.0     # 0-1
    gear_down: bool = False


class GuidanceComputer:
    """
    High-level guidance computer that generates
    autopilot commands from flight state and mission.
    """

    def __init__(self):
        """Initialize guidance computer."""
        self.commands = GuidanceCommands()

        # Approach parameters
        self.glideslope_deg = 3.0
        self.final_approach_dist_ft = 3 * 6076  # 3 nm

        # Speed targets (ft/s)
        self.cruise_speed = 185.0    # 110 kts
        self.approach_speed = 110.0  # 65 kts
        self.final_speed = 100.0     # 60 kts

    def compute_enroute_guidance(self,
                                  state: FlightState,
                                  target_x: float, target_y: float,
                                  target_alt: float,
                                  track_heading: float) -> GuidanceCommands:
        """
        Compute guidance for enroute flight.

        Args:
            state: Current flight state
            target_x, target_y: Target waypoint position
            target_alt: Target altitude
            track_heading: Desired track heading

        Returns:
            Guidance commands
        """
        cmd = self.commands.copy()

        # Distance to target
        dx = target_x - state.x
        dy = target_y - state.y
        distance = np.sqrt(dx*dx + dy*dy)

        # Cross-track error
        track_dx = np.cos(track_heading)
        track_dy = np.sin(track_heading)
        cross_track = -dx * track_dy + dy * track_dx

        # Lateral: track mode with cross-track correction
        cmd.lateral_mode = LateralMode.TRACK
        cmd.target_track = track_heading
        cmd.cross_track_error = cross_track

        # Compute intercept heading
        intercept_angle = np.arctan(0.001 * cross_track)
        intercept_angle = np.clip(intercept_angle, np.deg2rad(-30), np.deg2rad(30))
        cmd.target_heading = self._normalize_angle(track_heading - intercept_angle)

        # Vertical: altitude hold
        cmd.vertical_mode = VerticalMode.ALTITUDE
        cmd.target_altitude = target_alt

        # Speed: cruise
        cmd.target_airspeed = self.cruise_speed
        cmd.target_flaps = 0.0

        return cmd

    def compute_approach_guidance(self,
                                   state: FlightState,
                                   runway_x: float, runway_y: float,
                                   runway_heading: float,
                                   runway_elevation: float = 0.0) -> GuidanceCommands:
        """
        Compute guidance for approach and landing.

        Args:
            state: Current flight state
            runway_x, runway_y: Touchdown point position
            runway_heading: Runway heading
            runway_elevation: Runway elevation (ft)

        Returns:
            Guidance commands
        """
        cmd = self.commands.copy()

        # Distance to touchdown point
        dx = runway_x - state.x
        dy = runway_y - state.y
        distance = np.sqrt(dx*dx + dy*dy)

        # Cross-track error (from extended centerline)
        rwy_dx = np.cos(runway_heading)
        rwy_dy = np.sin(runway_heading)
        cross_track = -dx * (-rwy_dy) + dy * rwy_dx

        # Lateral: track runway centerline
        cmd.lateral_mode = LateralMode.TRACK
        cmd.target_track = runway_heading
        cmd.cross_track_error = cross_track

        # Intercept heading
        intercept_angle = np.arctan(0.001 * cross_track)
        intercept_angle = np.clip(intercept_angle, np.deg2rad(-15), np.deg2rad(15))
        cmd.target_heading = self._normalize_angle(runway_heading - intercept_angle)

        # Glideslope target altitude
        glideslope_alt = runway_elevation + distance * np.tan(np.deg2rad(self.glideslope_deg))
        cmd.glideslope_error = state.altitude - glideslope_alt

        # Vertical: glideslope mode
        cmd.vertical_mode = VerticalMode.GLIDESLOPE
        cmd.target_altitude = glideslope_alt
        cmd.target_fpa = np.deg2rad(-self.glideslope_deg)

        # Determine phase
        cmd.on_approach = True
        cmd.on_final = distance < self.final_approach_dist_ft
        cmd.on_glideslope = abs(cmd.glideslope_error) < 100  # Within 100 ft

        # Speed based on phase
        if cmd.on_final:
            cmd.target_airspeed = self.final_speed
            cmd.target_flaps = 0.7  # Full flaps for landing
        else:
            cmd.target_airspeed = self.approach_speed
            cmd.target_flaps = 0.3  # Partial flaps

        return cmd

    def compute_takeoff_guidance(self,
                                  state: FlightState,
                                  runway_heading: float,
                                  target_altitude: float) -> GuidanceCommands:
        """
        Compute guidance for takeoff and initial climb.

        Args:
            state: Current flight state
            runway_heading: Runway heading for departure
            target_altitude: Initial climb target altitude

        Returns:
            Guidance commands
        """
        cmd = self.commands.copy()

        # Lateral: hold runway heading
        cmd.lateral_mode = LateralMode.HEADING
        cmd.target_heading = runway_heading
        cmd.target_track = runway_heading

        # Vertical: climb to target altitude
        cmd.vertical_mode = VerticalMode.ALTITUDE
        cmd.target_altitude = target_altitude

        # Speed: climb speed
        cmd.target_airspeed = 125.0  # 74 kts climb speed
        cmd.target_flaps = 0.0

        return cmd

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
