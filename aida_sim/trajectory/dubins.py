#!/usr/bin/env python3
"""
Dubins Path Generator for Fixed-Wing Aircraft

Computes minimum-length paths between two poses (position + heading)
with bounded turn radius. Essential for fixed-wing trajectory planning.

Reference:
    Dubins, L.E. (1957). "On Curves of Minimal Length with a Constraint
    on Average Curvature, and with Prescribed Initial and Terminal
    Positions and Tangents"

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional


class DubinsPathType(Enum):
    """Types of Dubins paths."""
    LSL = "LSL"  # Left-Straight-Left
    LSR = "LSR"  # Left-Straight-Right
    RSL = "RSL"  # Right-Straight-Left
    RSR = "RSR"  # Right-Straight-Right
    LRL = "LRL"  # Left-Right-Left (CCC path)
    RLR = "RLR"  # Right-Left-Right (CCC path)


@dataclass
class DubinsPath:
    """
    A Dubins path consisting of up to 3 segments.

    Each segment is either:
    - An arc (turn) with signed curvature
    - A straight line (curvature = 0)
    """
    path_type: DubinsPathType
    turn_radius: float
    segment_lengths: Tuple[float, float, float]  # Length of each segment
    total_length: float

    # Start and end poses
    start_x: float
    start_y: float
    start_heading: float  # radians
    end_x: float
    end_y: float
    end_heading: float  # radians


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 2*pi)."""
    return angle % (2 * np.pi)


def _mod2pi(angle: float) -> float:
    """Normalize angle to [0, 2*pi)."""
    return _normalize_angle(angle)


def _compute_lsl(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """Compute LSL path lengths."""
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(alpha)
    cb = np.cos(beta)
    c_ab = np.cos(alpha - beta)

    p_sq = 2 + d*d - 2*c_ab + 2*d*(sa - sb)
    if p_sq < 0:
        return None

    p = np.sqrt(p_sq)
    theta = np.arctan2(cb - ca, d + sa - sb)
    t = _mod2pi(-alpha + theta)
    q = _mod2pi(beta - theta)

    return (t, p, q)


def _compute_rsr(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """Compute RSR path lengths."""
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(alpha)
    cb = np.cos(beta)
    c_ab = np.cos(alpha - beta)

    p_sq = 2 + d*d - 2*c_ab + 2*d*(sb - sa)
    if p_sq < 0:
        return None

    p = np.sqrt(p_sq)
    theta = np.arctan2(ca - cb, d - sa + sb)
    t = _mod2pi(alpha - theta)
    q = _mod2pi(-beta + theta)

    return (t, p, q)


def _compute_lsr(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """Compute LSR path lengths."""
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(alpha)
    cb = np.cos(beta)
    c_ab = np.cos(alpha - beta)

    p_sq = -2 + d*d + 2*c_ab + 2*d*(sa + sb)
    if p_sq < 0:
        return None

    p = np.sqrt(p_sq)
    theta = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
    t = _mod2pi(-alpha + theta)
    q = _mod2pi(-_mod2pi(beta) + theta)

    return (t, p, q)


def _compute_rsl(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """Compute RSL path lengths."""
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(alpha)
    cb = np.cos(beta)
    c_ab = np.cos(alpha - beta)

    p_sq = -2 + d*d + 2*c_ab - 2*d*(sa + sb)
    if p_sq < 0:
        return None

    p = np.sqrt(p_sq)
    theta = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
    t = _mod2pi(alpha - theta)
    q = _mod2pi(beta - theta)

    return (t, p, q)


def _compute_rlr(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """Compute RLR path lengths."""
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(alpha)
    cb = np.cos(beta)
    c_ab = np.cos(alpha - beta)

    mode = (6.0 - d*d + 2*c_ab + 2*d*(sa - sb)) / 8.0
    if abs(mode) > 1.0:
        return None

    p = _mod2pi(2*np.pi - np.arccos(mode))
    theta = np.arctan2(ca - cb, d - sa + sb)
    t = _mod2pi(alpha - theta + _mod2pi(p/2.0))
    q = _mod2pi(alpha - beta - t + _mod2pi(p))

    return (t, p, q)


def _compute_lrl(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """Compute LRL path lengths."""
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(alpha)
    cb = np.cos(beta)
    c_ab = np.cos(alpha - beta)

    mode = (6.0 - d*d + 2*c_ab + 2*d*(sb - sa)) / 8.0
    if abs(mode) > 1.0:
        return None

    p = _mod2pi(2*np.pi - np.arccos(mode))
    theta = np.arctan2(ca - cb, d + sa - sb)
    t = _mod2pi(-alpha + theta + _mod2pi(p/2.0))
    q = _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p))

    return (t, p, q)


def compute_dubins_path(start_x: float, start_y: float, start_heading: float,
                         end_x: float, end_y: float, end_heading: float,
                         turn_radius: float) -> DubinsPath:
    """
    Compute the shortest Dubins path between two poses.

    Args:
        start_x, start_y: Start position
        start_heading: Start heading in radians
        end_x, end_y: End position
        end_heading: End heading in radians
        turn_radius: Minimum turn radius

    Returns:
        DubinsPath with the shortest path
    """
    # Normalize to unit problem
    dx = end_x - start_x
    dy = end_y - start_y
    d = np.sqrt(dx*dx + dy*dy) / turn_radius

    theta = np.arctan2(dy, dx)
    alpha = _mod2pi(start_heading - theta)
    beta = _mod2pi(end_heading - theta)

    # Try all path types
    path_functions = [
        (DubinsPathType.LSL, _compute_lsl),
        (DubinsPathType.RSR, _compute_rsr),
        (DubinsPathType.LSR, _compute_lsr),
        (DubinsPathType.RSL, _compute_rsl),
        (DubinsPathType.RLR, _compute_rlr),
        (DubinsPathType.LRL, _compute_lrl),
    ]

    best_path = None
    best_length = float('inf')

    for path_type, compute_func in path_functions:
        result = compute_func(alpha, beta, d)
        if result is not None:
            t, p, q = result
            # Convert arc lengths back to actual lengths
            if path_type in (DubinsPathType.LRL, DubinsPathType.RLR):
                # CCC paths: all segments are arcs
                length = (t + p + q) * turn_radius
            else:
                # CSC paths: t and q are arcs, p is straight
                length = (t + q) * turn_radius + p * turn_radius

            if length < best_length:
                best_length = length
                if path_type in (DubinsPathType.LRL, DubinsPathType.RLR):
                    segment_lengths = (t * turn_radius, p * turn_radius, q * turn_radius)
                else:
                    segment_lengths = (t * turn_radius, p * turn_radius, q * turn_radius)
                best_path = DubinsPath(
                    path_type=path_type,
                    turn_radius=turn_radius,
                    segment_lengths=segment_lengths,
                    total_length=length,
                    start_x=start_x,
                    start_y=start_y,
                    start_heading=start_heading,
                    end_x=end_x,
                    end_y=end_y,
                    end_heading=end_heading,
                )

    return best_path


def sample_dubins_path(path: DubinsPath, num_points: int = 100) -> List[Tuple[float, float, float]]:
    """
    Sample points along a Dubins path.

    Args:
        path: DubinsPath to sample
        num_points: Number of points to generate

    Returns:
        List of (x, y, heading) tuples
    """
    if path is None:
        return []

    points = []

    # Current pose
    x, y, heading = path.start_x, path.start_y, path.start_heading

    # Determine turn directions based on path type
    if path.path_type == DubinsPathType.LSL:
        dirs = (1, 0, 1)  # Left, Straight, Left
    elif path.path_type == DubinsPathType.RSR:
        dirs = (-1, 0, -1)  # Right, Straight, Right
    elif path.path_type == DubinsPathType.LSR:
        dirs = (1, 0, -1)  # Left, Straight, Right
    elif path.path_type == DubinsPathType.RSL:
        dirs = (-1, 0, 1)  # Right, Straight, Left
    elif path.path_type == DubinsPathType.LRL:
        dirs = (1, -1, 1)  # Left, Right, Left
    else:  # RLR
        dirs = (-1, 1, -1)  # Right, Left, Right

    r = path.turn_radius

    # Sample each segment
    for seg_idx, (seg_len, direction) in enumerate(zip(path.segment_lengths, dirs)):
        if seg_len < 1e-6:
            continue

        # Number of points for this segment (proportional to length)
        seg_points = max(2, int(num_points * seg_len / path.total_length))

        for i in range(seg_points):
            t = i / (seg_points - 1) if seg_points > 1 else 0
            s = t * seg_len

            if direction == 0:
                # Straight segment
                px = x + s * np.cos(heading)
                py = y + s * np.sin(heading)
                points.append((px, py, heading))
            else:
                # Arc segment
                # Arc angle traversed
                theta = s / r
                # Turn direction
                sign = direction

                # Center of turn circle
                cx = x - sign * r * np.sin(heading)
                cy = y + sign * r * np.cos(heading)

                # Point on arc
                phi = heading + sign * (np.pi / 2)  # Angle from center to start
                new_phi = phi + sign * theta
                px = cx + r * np.cos(new_phi)
                py = cy + r * np.sin(new_phi)
                new_heading = heading + sign * theta

                points.append((px, py, new_heading))

        # Update pose for next segment
        if direction == 0:
            x = x + seg_len * np.cos(heading)
            y = y + seg_len * np.sin(heading)
        else:
            theta = seg_len / r
            sign = direction
            cx = x - sign * r * np.sin(heading)
            cy = y + sign * r * np.cos(heading)
            phi = heading + sign * (np.pi / 2)
            new_phi = phi + sign * theta
            x = cx + r * np.cos(new_phi)
            y = cy + r * np.sin(new_phi)
            heading = heading + sign * theta

    return points


def compute_turn_radius(airspeed_fps: float, bank_angle_deg: float = 25.0) -> float:
    """
    Compute minimum turn radius for a coordinated turn.

    R = V^2 / (g * tan(bank))

    Args:
        airspeed_fps: True airspeed in ft/s
        bank_angle_deg: Bank angle in degrees

    Returns:
        Turn radius in feet
    """
    g = 32.174  # ft/s^2
    bank_rad = np.deg2rad(bank_angle_deg)
    return (airspeed_fps ** 2) / (g * np.tan(bank_rad))


if __name__ == "__main__":
    # Test Dubins path computation
    print("Dubins Path Test")
    print("=" * 50)

    # Cessna 172 at 100 kts, 25 deg bank
    airspeed = 100 * 1.68781  # kts to fps
    turn_radius = compute_turn_radius(airspeed, 25.0)
    print(f"Turn radius at 100 kts, 25° bank: {turn_radius:.0f} ft ({turn_radius/6076:.2f} nm)")

    # Test path from (0,0) heading north to (10000, 5000) heading east
    path = compute_dubins_path(
        0, 0, np.deg2rad(0),
        10000, 5000, np.deg2rad(90),
        turn_radius
    )

    print(f"\nPath type: {path.path_type.value}")
    print(f"Total length: {path.total_length:.0f} ft ({path.total_length/6076:.2f} nm)")
    print(f"Segments: {path.segment_lengths[0]:.0f}, {path.segment_lengths[1]:.0f}, {path.segment_lengths[2]:.0f} ft")

    # Sample and print a few points
    points = sample_dubins_path(path, 20)
    print(f"\nSampled {len(points)} points:")
    for i, (x, y, h) in enumerate(points[::5]):
        print(f"  {i*5}: ({x:.0f}, {y:.0f}) hdg={np.rad2deg(h):.0f}°")
