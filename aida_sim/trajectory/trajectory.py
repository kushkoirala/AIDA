#!/usr/bin/env python3
"""
Trajectory Data Structures

Defines the trajectory representation used throughout AIDA.
A trajectory is a sequence of 4D points (x, y, z, t) with
associated velocity and curvature information.

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import IntEnum


class FlightSegment(IntEnum):
    """Flight segment types for trajectory."""
    TAKEOFF = 0
    CLIMB = 1
    CRUISE = 2
    DESCENT = 3
    APPROACH = 4
    LANDING = 5


@dataclass
class TrajectoryPoint:
    """
    A single point along a trajectory.

    All units in feet and ft/s unless noted.
    """
    # Position (ft)
    x: float
    y: float
    z: float  # Altitude (positive up, unlike sim Z)

    # Timing
    t: float  # Time from start (seconds)
    s: float  # Arc length from start (ft)

    # Velocity
    speed: float  # Ground speed (ft/s)
    heading: float  # Track heading (radians)
    climb_rate: float = 0.0  # Vertical speed (ft/s, positive = climb)

    # Curvature
    curvature: float = 0.0  # Path curvature (1/ft)

    # Segment type
    segment: FlightSegment = FlightSegment.CRUISE

    # Optional target values (for controller)
    target_airspeed: Optional[float] = None
    target_heading: Optional[float] = None

    def position_2d(self) -> Tuple[float, float]:
        """Return (x, y) position."""
        return (self.x, self.y)

    def position_3d(self) -> Tuple[float, float, float]:
        """Return (x, y, z) position."""
        return (self.x, self.y, self.z)


@dataclass
class Trajectory:
    """
    A complete trajectory from origin to destination.

    Trajectories are represented as a sequence of points with
    interpolation between them. The trajectory provides methods
    for querying state at any point along the path.
    """
    points: List[TrajectoryPoint] = field(default_factory=list)
    name: str = ""

    # Metadata
    origin: str = ""
    destination: str = ""
    total_distance: float = 0.0  # Total path length (ft)
    total_time: float = 0.0  # Expected flight time (s)

    # Cache for fast closest point search
    _last_closest_idx: int = field(default=0, repr=False)
    _xy_array: Optional[np.ndarray] = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> TrajectoryPoint:
        return self.points[idx]

    def add_point(self, point: TrajectoryPoint):
        """Add a point to the trajectory."""
        self.points.append(point)
        if len(self.points) > 1:
            self.total_distance = self.points[-1].s
            self.total_time = self.points[-1].t
        # Invalidate cached array when points change
        self._xy_array = None

    def _build_xy_cache(self):
        """Build numpy array cache for fast distance calculations."""
        if self._xy_array is None and self.points:
            self._xy_array = np.array([[p.x, p.y] for p in self.points])

    def get_point_at_distance(self, s: float) -> TrajectoryPoint:
        """
        Get trajectory point at given arc length distance.

        Uses linear interpolation between waypoints.
        """
        if not self.points:
            raise ValueError("Empty trajectory")

        if s <= 0:
            return self.points[0]
        if s >= self.total_distance:
            return self.points[-1]

        # Binary search for segment
        left, right = 0, len(self.points) - 1
        while left < right - 1:
            mid = (left + right) // 2
            if self.points[mid].s <= s:
                left = mid
            else:
                right = mid

        # Interpolate between points[left] and points[right]
        p1, p2 = self.points[left], self.points[right]
        if p2.s - p1.s < 1e-6:
            return p1

        alpha = (s - p1.s) / (p2.s - p1.s)

        return TrajectoryPoint(
            x=p1.x + alpha * (p2.x - p1.x),
            y=p1.y + alpha * (p2.y - p1.y),
            z=p1.z + alpha * (p2.z - p1.z),
            t=p1.t + alpha * (p2.t - p1.t),
            s=s,
            speed=p1.speed + alpha * (p2.speed - p1.speed),
            heading=self._interpolate_angle(p1.heading, p2.heading, alpha),
            climb_rate=p1.climb_rate + alpha * (p2.climb_rate - p1.climb_rate),
            curvature=p1.curvature + alpha * (p2.curvature - p1.curvature),
            segment=p1.segment if alpha < 0.5 else p2.segment,
            target_airspeed=p1.target_airspeed,
            target_heading=p1.target_heading,
        )

    def get_point_at_time(self, t: float) -> TrajectoryPoint:
        """
        Get trajectory point at given time.

        Uses linear interpolation between waypoints.
        """
        if not self.points:
            raise ValueError("Empty trajectory")

        if t <= 0:
            return self.points[0]
        if t >= self.total_time:
            return self.points[-1]

        # Binary search for segment
        left, right = 0, len(self.points) - 1
        while left < right - 1:
            mid = (left + right) // 2
            if self.points[mid].t <= t:
                left = mid
            else:
                right = mid

        # Interpolate between points[left] and points[right]
        p1, p2 = self.points[left], self.points[right]
        if p2.t - p1.t < 1e-6:
            return p1

        alpha = (t - p1.t) / (p2.t - p1.t)

        return TrajectoryPoint(
            x=p1.x + alpha * (p2.x - p1.x),
            y=p1.y + alpha * (p2.y - p1.y),
            z=p1.z + alpha * (p2.z - p1.z),
            t=t,
            s=p1.s + alpha * (p2.s - p1.s),
            speed=p1.speed + alpha * (p2.speed - p1.speed),
            heading=self._interpolate_angle(p1.heading, p2.heading, alpha),
            climb_rate=p1.climb_rate + alpha * (p2.climb_rate - p1.climb_rate),
            curvature=p1.curvature + alpha * (p2.curvature - p1.curvature),
            segment=p1.segment if alpha < 0.5 else p2.segment,
            target_airspeed=p1.target_airspeed,
            target_heading=p1.target_heading,
        )

    def find_closest_point(self, x: float, y: float) -> Tuple[TrajectoryPoint, float, float]:
        """
        Find the closest point on trajectory to given position.

        Uses cached index from last call and local search for O(1) amortized
        performance during sequential traversal (typical flight following).

        Args:
            x, y: Current position in feet

        Returns:
            Tuple of (closest_point, distance_to_path, cross_track_error)
            Cross-track error is positive if right of path.
        """
        if not self.points:
            raise ValueError("Empty trajectory")

        n = len(self.points)
        if n == 1:
            p = self.points[0]
            dist = np.sqrt((x - p.x)**2 + (y - p.y)**2)
            return p, dist, 0.0

        # Build cache if needed
        self._build_xy_cache()

        # Start search from cached index, search nearby first
        # Aircraft typically moves forward along trajectory, so search forward-biased
        search_radius = 50  # Check 50 segments around cached position first
        start_idx = max(0, self._last_closest_idx - 10)
        end_idx = min(n - 1, self._last_closest_idx + search_radius)

        best_dist = float('inf')
        best_idx = start_idx
        best_t = 0.0

        # Local search first
        for i in range(start_idx, end_idx):
            p1, p2 = self.points[i], self.points[i + 1]
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            seg_len_sq = dx*dx + dy*dy

            if seg_len_sq < 1e-6:
                continue

            t = max(0, min(1, ((x - p1.x)*dx + (y - p1.y)*dy) / seg_len_sq))
            px = p1.x + t * dx
            py = p1.y + t * dy
            dist = (x - px)**2 + (y - py)**2  # Skip sqrt for comparison

            if dist < best_dist:
                best_dist = dist
                best_idx = i
                best_t = t

        # If best is near boundary of local search, expand search
        # This handles cases where aircraft is far off course
        if best_idx <= start_idx + 2 or best_idx >= end_idx - 3:
            # Full search using vectorized numpy
            xy = self._xy_array
            # Compute distance to each point (rough approximation)
            dists_sq = (xy[:, 0] - x)**2 + (xy[:, 1] - y)**2
            rough_best = np.argmin(dists_sq)

            # Fine search around that point
            fine_start = max(0, rough_best - 5)
            fine_end = min(n - 1, rough_best + 5)

            for i in range(fine_start, fine_end):
                p1, p2 = self.points[i], self.points[i + 1]
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                seg_len_sq = dx*dx + dy*dy

                if seg_len_sq < 1e-6:
                    continue

                t = max(0, min(1, ((x - p1.x)*dx + (y - p1.y)*dy) / seg_len_sq))
                px = p1.x + t * dx
                py = p1.y + t * dy
                dist = (x - px)**2 + (y - py)**2

                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
                    best_t = t

        # Update cache for next call
        self._last_closest_idx = best_idx

        # Now compute the actual result with sqrt and full interpolation
        best_dist = np.sqrt(best_dist)
        p1, p2 = self.points[best_idx], self.points[best_idx + 1]

        # Interpolate full point
        s_interp = p1.s + best_t * (p2.s - p1.s)
        best_point = self.get_point_at_distance(s_interp)

        # Cross-track error (positive = right of path)
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        seg_heading = np.arctan2(dy, dx)
        to_aircraft = np.arctan2(y - p1.y, x - p1.x)
        angle_diff = to_aircraft - seg_heading
        best_cte = best_dist * np.sin(angle_diff)

        return best_point, best_dist, best_cte

    def get_lookahead_point(self, x: float, y: float, lookahead_dist: float) -> TrajectoryPoint:
        """
        Get a point on the trajectory ahead of current position.

        Used for pure pursuit and similar guidance laws.

        Args:
            x, y: Current position
            lookahead_dist: Distance ahead to look (ft)

        Returns:
            TrajectoryPoint at lookahead distance
        """
        closest, _, _ = self.find_closest_point(x, y)
        target_s = min(closest.s + lookahead_dist, self.total_distance)
        return self.get_point_at_distance(target_s)

    @staticmethod
    def _interpolate_angle(a1: float, a2: float, alpha: float) -> float:
        """Interpolate between two angles, handling wrap-around."""
        # Find shortest path
        diff = a2 - a1
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return a1 + alpha * diff

    def to_numpy(self) -> np.ndarray:
        """Convert trajectory to numpy array [N x 8]."""
        data = np.zeros((len(self.points), 8))
        for i, p in enumerate(self.points):
            data[i] = [p.x, p.y, p.z, p.t, p.s, p.speed, p.heading, p.curvature]
        return data

    @classmethod
    def from_numpy(cls, data: np.ndarray, name: str = "") -> 'Trajectory':
        """Create trajectory from numpy array."""
        traj = cls(name=name)
        for row in data:
            traj.add_point(TrajectoryPoint(
                x=row[0], y=row[1], z=row[2],
                t=row[3], s=row[4], speed=row[5],
                heading=row[6], curvature=row[7]
            ))
        return traj
