#!/usr/bin/env python3
"""
Pure Pursuit Guidance for Path Following

Classic geometric path following algorithm that computes
the curvature needed to reach a lookahead point on the path.

Simple, robust, and widely used in autonomous vehicles.

Reference:
    Coulter, R.C. (1992). "Implementation of the Pure Pursuit
    Path Tracking Algorithm"

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
class PurePursuitConfig:
    """Configuration for pure pursuit guidance."""
    # Lookahead parameters
    min_lookahead: float = 500.0   # Minimum lookahead distance (ft)
    max_lookahead: float = 3000.0  # Maximum lookahead distance (ft)
    lookahead_ratio: float = 2.0   # Lookahead = ratio * speed (s)

    # Curvature limits
    max_curvature: float = 0.001   # Maximum curvature (1/ft)


class PurePursuitGuidance:
    """
    Pure pursuit path following guidance.

    Computes the steering curvature needed to reach a
    lookahead point on the desired path.
    """

    def __init__(self, config: Optional[PurePursuitConfig] = None):
        """Initialize with configuration."""
        self.config = config or PurePursuitConfig()

    def compute_curvature(self,
                           x: float, y: float, heading: float,
                           trajectory: Trajectory,
                           ground_speed: float = 150.0) -> Tuple[float, TrajectoryPoint, float]:
        """
        Compute steering curvature using pure pursuit.

        Args:
            x, y: Current aircraft position (ft)
            heading: Current heading (radians)
            trajectory: Trajectory to follow
            ground_speed: Current ground speed (ft/s)

        Returns:
            Tuple of (curvature, lookahead_point, cross_track_error)
        """
        # Compute lookahead distance based on speed
        lookahead = ground_speed * self.config.lookahead_ratio
        lookahead = np.clip(lookahead, self.config.min_lookahead, self.config.max_lookahead)

        # Get lookahead point on trajectory
        lookahead_point = trajectory.get_lookahead_point(x, y, lookahead)

        # Vector to lookahead point
        dx = lookahead_point.x - x
        dy = lookahead_point.y - y
        distance = np.sqrt(dx*dx + dy*dy)

        if distance < 1e-6:
            return 0.0, lookahead_point, 0.0

        # Angle to lookahead point (in aircraft frame)
        angle_to_point = np.arctan2(dy, dx)
        alpha = self._normalize_angle(angle_to_point - heading)

        # Pure pursuit curvature formula: k = 2 * sin(alpha) / L
        curvature = 2.0 * np.sin(alpha) / distance

        # Limit curvature
        curvature = np.clip(curvature, -self.config.max_curvature, self.config.max_curvature)

        # Cross-track error for monitoring
        _, _, cross_track = trajectory.find_closest_point(x, y)

        return curvature, lookahead_point, cross_track

    def compute_desired_heading(self,
                                 x: float, y: float, heading: float,
                                 trajectory: Trajectory,
                                 ground_speed: float = 150.0) -> Tuple[float, float, float]:
        """
        Compute desired heading using pure pursuit.

        Alternative interface that returns heading instead of curvature.

        Args:
            x, y: Current position (ft)
            heading: Current heading (radians)
            trajectory: Trajectory to follow
            ground_speed: Ground speed (ft/s)

        Returns:
            Tuple of (desired_heading, cross_track_error, distance_to_lookahead)
        """
        # Compute lookahead
        lookahead = ground_speed * self.config.lookahead_ratio
        lookahead = np.clip(lookahead, self.config.min_lookahead, self.config.max_lookahead)

        # Get lookahead point
        lookahead_point = trajectory.get_lookahead_point(x, y, lookahead)

        # Desired heading is directly toward lookahead point
        dx = lookahead_point.x - x
        dy = lookahead_point.y - y
        distance = np.sqrt(dx*dx + dy*dy)

        desired_heading = np.arctan2(dy, dx)

        # Cross-track error
        _, _, cross_track = trajectory.find_closest_point(x, y)

        return desired_heading, cross_track, distance

    def compute_desired_heading_waypoint(self,
                                          x: float, y: float,
                                          waypoint_x: float, waypoint_y: float,
                                          track_heading: float,
                                          capture_radius: float = 500.0) -> Tuple[float, float, bool]:
        """
        Compute desired heading to reach a waypoint.

        Simplified version for single waypoint tracking.

        Args:
            x, y: Current position
            waypoint_x, waypoint_y: Target waypoint
            track_heading: Inbound track heading
            capture_radius: Distance to consider waypoint captured

        Returns:
            Tuple of (desired_heading, distance_to_waypoint, is_captured)
        """
        dx = waypoint_x - x
        dy = waypoint_y - y
        distance = np.sqrt(dx*dx + dy*dy)

        if distance < capture_radius:
            return track_heading, distance, True

        # Head directly toward waypoint
        desired_heading = np.arctan2(dy, dx)

        return desired_heading, distance, False

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class AdaptivePurePursuit(PurePursuitGuidance):
    """
    Adaptive pure pursuit with variable lookahead.

    Adjusts lookahead based on path curvature and
    cross-track error for better performance.
    """

    def __init__(self, config: Optional[PurePursuitConfig] = None):
        super().__init__(config)
        self.crosstrack_gain = 0.5  # Reduce lookahead when off-track

    def compute_adaptive_lookahead(self,
                                    ground_speed: float,
                                    path_curvature: float,
                                    cross_track_error: float) -> float:
        """
        Compute adaptive lookahead distance.

        - Increase lookahead on straight sections
        - Decrease lookahead in turns
        - Decrease lookahead when off-track
        """
        # Base lookahead from speed
        base_lookahead = ground_speed * self.config.lookahead_ratio

        # Reduce for high curvature (tighter turns)
        curvature_factor = 1.0 / (1.0 + abs(path_curvature) * 1000)

        # Reduce when off-track
        crosstrack_factor = 1.0 / (1.0 + self.crosstrack_gain * abs(cross_track_error) / 100)

        lookahead = base_lookahead * curvature_factor * crosstrack_factor
        return np.clip(lookahead, self.config.min_lookahead, self.config.max_lookahead)
