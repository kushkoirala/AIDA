#!/usr/bin/env python3
"""
Lyapunov Vector Field Guidance (LVFG)

A vector field guidance law for path following that provides
guaranteed convergence to the desired path. Particularly effective
for fixed-wing aircraft as it naturally handles wind and provides
smooth transitions.

Reference:
    Nelson, D.R. et al. (2007). "Vector Field Path Following for
    Miniature Air Vehicles"

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectory.trajectory import Trajectory, TrajectoryPoint


@dataclass
class VectorFieldConfig:
    """Configuration for vector field guidance."""
    # Convergence parameters
    k_path: float = 0.01  # Path following gain (1/ft)
    k_orbit: float = 0.5   # Orbit following gain

    # Transition distance
    transition_distance: float = 500.0  # ft, distance to start turning

    # Limits
    max_course_change: float = 45.0  # degrees


class VectorFieldGuidance:
    """
    Vector field guidance for path following.

    Generates a desired heading command that, when followed,
    will guide the aircraft to converge to and follow a trajectory.

    The vector field is constructed such that:
    1. Far from the path: heading points toward the path
    2. On the path: heading aligns with the path direction
    3. The transition is smooth and Lyapunov-stable
    """

    def __init__(self, config: Optional[VectorFieldConfig] = None):
        """Initialize with configuration."""
        self.config = config or VectorFieldConfig()

    def compute_desired_heading(self,
                                 x: float, y: float,
                                 trajectory: Trajectory,
                                 ground_speed: float = 150.0) -> Tuple[float, float, float]:
        """
        Compute desired heading from vector field.

        Args:
            x, y: Current aircraft position (ft)
            trajectory: Trajectory to follow
            ground_speed: Current ground speed (ft/s)

        Returns:
            Tuple of (desired_heading_rad, cross_track_error_ft, along_track_position)
        """
        # Find closest point on trajectory
        closest_point, dist_to_path, cross_track = trajectory.find_closest_point(x, y)

        # Path heading at closest point
        path_heading = closest_point.heading

        # Compute vector field heading
        # The key insight: we want heading to be a function of cross-track error
        # that smoothly transitions from "toward path" to "along path"

        # Course correction angle (function of cross-track error)
        # Using atan for smooth bounded transition
        correction = np.arctan(self.config.k_path * cross_track)

        # Limit correction
        max_correction_rad = np.deg2rad(self.config.max_course_change)
        correction = np.clip(correction, -max_correction_rad, max_correction_rad)

        # Desired heading = path heading - correction
        # (subtract because positive cross_track means we're right of path,
        #  so we want to turn left, i.e., decrease heading)
        desired_heading = path_heading - correction

        # Normalize to [-pi, pi]
        desired_heading = self._normalize_angle(desired_heading)

        return desired_heading, cross_track, closest_point.s

    def compute_desired_heading_line(self,
                                      x: float, y: float,
                                      line_start: Tuple[float, float],
                                      line_end: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute desired heading to follow a straight line.

        Simpler version for straight segments.

        Args:
            x, y: Current position
            line_start: (x, y) of line start
            line_end: (x, y) of line end

        Returns:
            Tuple of (desired_heading_rad, cross_track_error_ft)
        """
        x1, y1 = line_start
        x2, y2 = line_end

        # Line direction
        dx = x2 - x1
        dy = y2 - y1
        line_length = np.sqrt(dx*dx + dy*dy)

        if line_length < 1e-6:
            return 0.0, 0.0

        # Line heading
        line_heading = np.arctan2(dy, dx)

        # Cross-track error (perpendicular distance, signed)
        # Positive = right of line
        cross_track = ((y - y1) * dx - (x - x1) * dy) / line_length

        # Course correction
        correction = np.arctan(self.config.k_path * cross_track)
        max_correction_rad = np.deg2rad(self.config.max_course_change)
        correction = np.clip(correction, -max_correction_rad, max_correction_rad)

        desired_heading = self._normalize_angle(line_heading - correction)

        return desired_heading, cross_track

    def compute_desired_heading_orbit(self,
                                       x: float, y: float,
                                       center_x: float, center_y: float,
                                       radius: float,
                                       clockwise: bool = True) -> Tuple[float, float]:
        """
        Compute desired heading to orbit around a point.

        Useful for holding patterns and circling.

        Args:
            x, y: Current position
            center_x, center_y: Orbit center
            radius: Desired orbit radius (ft)
            clockwise: True for clockwise orbit

        Returns:
            Tuple of (desired_heading_rad, radius_error_ft)
        """
        # Distance from center
        dx = x - center_x
        dy = y - center_y
        dist = np.sqrt(dx*dx + dy*dy)

        if dist < 1e-6:
            # At center, pick arbitrary direction
            return 0.0, -radius

        # Angle from center to aircraft
        angle_to_aircraft = np.arctan2(dy, dx)

        # Tangent direction (perpendicular to radius)
        # Clockwise: tangent is 90 degrees clockwise from radius
        # Counter-clockwise: tangent is 90 degrees counter-clockwise
        direction = -1 if clockwise else 1
        tangent_heading = angle_to_aircraft + direction * np.pi / 2

        # Radius error (positive = outside orbit)
        radius_error = dist - radius

        # Correction to converge to orbit
        # If outside, turn toward center (reduce tangent heading)
        # If inside, turn away from center (increase tangent heading)
        correction = np.arctan(self.config.k_orbit * radius_error)
        max_correction_rad = np.deg2rad(self.config.max_course_change)
        correction = np.clip(correction, -max_correction_rad, max_correction_rad)

        # Apply correction in direction that reduces radius error
        # For clockwise: positive radius_error -> turn left (more toward center)
        desired_heading = self._normalize_angle(tangent_heading - direction * correction)

        return desired_heading, radius_error

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class StraightLineGuidance:
    """
    Simplified guidance for straight line segments.

    Uses cross-track error and along-track position
    for efficient straight-line following.
    """

    def __init__(self, k_crosstrack: float = 0.001, max_intercept_deg: float = 30.0):
        """
        Initialize straight line guidance.

        Args:
            k_crosstrack: Cross-track error gain (rad/ft)
            max_intercept_deg: Maximum intercept angle
        """
        self.k_crosstrack = k_crosstrack
        self.max_intercept_rad = np.deg2rad(max_intercept_deg)

    def compute_guidance(self,
                          x: float, y: float,
                          target_x: float, target_y: float,
                          track_heading: float) -> Tuple[float, float, float]:
        """
        Compute guidance to follow straight line to target.

        Args:
            x, y: Current position
            target_x, target_y: Target waypoint
            track_heading: Desired track heading (radians)

        Returns:
            Tuple of (desired_heading, cross_track_error, distance_to_target)
        """
        # Vector to target
        dx = target_x - x
        dy = target_y - y
        distance = np.sqrt(dx*dx + dy*dy)

        # Cross-track error
        # Positive = right of desired track
        track_dx = np.cos(track_heading)
        track_dy = np.sin(track_heading)
        cross_track = -dx * track_dy + dy * track_dx

        # Intercept angle based on cross-track error
        intercept = np.arctan(self.k_crosstrack * cross_track)
        intercept = np.clip(intercept, -self.max_intercept_rad, self.max_intercept_rad)

        # Desired heading = track heading - intercept correction
        desired_heading = track_heading - intercept

        # Normalize
        while desired_heading > np.pi:
            desired_heading -= 2 * np.pi
        while desired_heading < -np.pi:
            desired_heading += 2 * np.pi

        return desired_heading, cross_track, distance
