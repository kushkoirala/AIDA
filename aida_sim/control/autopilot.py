#!/usr/bin/env python3
"""
Unified Trajectory Autopilot for AIDA

A single autopilot that can follow any trajectory, replacing
the phase-specific logic in triangle_controller.py.

Architecture:
    Trajectory -> Guidance -> Autopilot -> Control Surfaces

The autopilot receives guidance commands (target heading, altitude,
speed) and outputs control surface deflections.

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from control.pid import (
    HeadingController, PitchController, AltitudeController,
    AirspeedController, GlideslopeController, CrossTrackController
)
from guidance.commands import GuidanceCommands, FlightState, LateralMode, VerticalMode
from guidance.vector_field import VectorFieldGuidance, VectorFieldConfig
from trajectory.trajectory import Trajectory, TrajectoryPoint, FlightSegment


# Unit conversion constants
FT_TO_M = 0.3048
M_TO_FT = 3.28084
KTS_TO_FPS = 1.68781
FPS_TO_KTS = 1 / 1.68781


class AutopilotMode(Enum):
    """Autopilot engagement modes."""
    MANUAL = 0
    HEADING_HOLD = 1
    ALTITUDE_HOLD = 2
    NAV = 3           # Follow trajectory
    APPROACH = 4      # ILS/GPS approach
    TAKEOFF = 5
    LANDING = 6


@dataclass
class AutopilotConfig:
    """Autopilot configuration parameters."""
    # Heading control (tuned for faster response)
    heading_kp: float = 1.5       # More aggressive turn initiation
    roll_kp: float = 2.0          # Faster roll response
    roll_kd: float = 0.6          # Good damping
    max_bank_deg: float = 25.0

    # Pitch control
    pitch_kp: float = 1.0
    pitch_kd: float = 0.7

    # Altitude control (simple P-cascade: altitude -> VS -> pitch)
    alt_kp: float = 0.08          # Outer loop: aggressive for faster response
    alt_ki: float = 0.0           # (unused, kept for API compatibility)
    vs_kp: float = 1.5            # Inner loop: responsive
    max_vs_fpm: float = 800.0     # Allow faster climb/descent

    # Speed control (tuned for faster response)
    speed_kp: float = 0.10        # More aggressive
    speed_ki: float = 0.02        # Stronger integral
    base_throttle: float = 0.5

    # Glideslope
    glideslope_deg: float = 3.0

    # Target pitch angles (degrees)
    pitch_climb: float = 5.0  # Reduced for ~600-800 fpm climb
    pitch_cruise: float = 0.0
    pitch_descent: float = -3.0
    pitch_approach: float = -2.0
    pitch_flare: float = 5.0

    # Vector Field Guidance parameters
    # See docs/guidance_algorithms.md for tuning guidelines
    vfg_k_path: float = 0.005     # Path following gain (1/ft), higher = faster convergence
    vfg_max_correction_deg: float = 30.0  # Maximum heading correction angle (reduced for smoother tracking)


class TrajectoryAutopilot:
    """
    Unified autopilot that follows trajectories.

    This replaces the phase-specific controller with a
    trajectory-following autopilot that handles all phases.
    """

    def __init__(self, config: Optional[AutopilotConfig] = None):
        """Initialize autopilot with configuration."""
        self.config = config or AutopilotConfig()

        # Mode
        self.mode = AutopilotMode.MANUAL

        # Trajectory
        self.trajectory: Optional[Trajectory] = None
        self.current_segment = FlightSegment.CRUISE

        # Controllers
        self.heading_ctrl = HeadingController(
            heading_kp=self.config.heading_kp,
            roll_kp=self.config.roll_kp,
            roll_kd=self.config.roll_kd,
            max_bank_angle=self.config.max_bank_deg
        )
        self.pitch_ctrl = PitchController(
            kp=self.config.pitch_kp,
            kd=self.config.pitch_kd
        )
        self.altitude_ctrl = AltitudeController(
            alt_kp=self.config.alt_kp,
            alt_ki=self.config.alt_ki,
            vs_kp=self.config.vs_kp,
            max_vs_fpm=self.config.max_vs_fpm
        )
        self.speed_ctrl = AirspeedController(
            kp=self.config.speed_kp,
            ki=self.config.speed_ki,
            base_throttle=self.config.base_throttle
        )
        self.glideslope_ctrl = GlideslopeController(
            glideslope_deg=self.config.glideslope_deg
        )
        self.crosstrack_ctrl = CrossTrackController()

        # Vector Field Guidance for path following
        # This provides Lyapunov-stable convergence to the trajectory
        vfg_config = VectorFieldConfig(
            k_path=self.config.vfg_k_path,
            max_course_change=self.config.vfg_max_correction_deg
        )
        self.vector_field_guidance = VectorFieldGuidance(vfg_config)

        # Guidance commands (what we're trying to achieve)
        self.commands = GuidanceCommands()

        # State
        self.distance_to_destination = float('inf')
        self.has_landed = False

        # Runway elevation for AGL calculation during approach (set from trajectory)
        self.runway_elevation_ft = 0.0

    def set_trajectory(self, trajectory: Trajectory):
        """Set the trajectory to follow."""
        self.trajectory = trajectory
        self.mode = AutopilotMode.NAV
        self.has_landed = False

        # Extract runway elevation from last waypoint (touchdown point)
        if trajectory and len(trajectory.points) > 0:
            self.runway_elevation_ft = trajectory.points[-1].z

    def set_commands(self, commands: GuidanceCommands):
        """Set guidance commands directly."""
        self.commands = commands

    def compute_action(self, state: np.ndarray, sim_time: float) -> np.ndarray:
        """
        Compute control action from state.

        Args:
            state: Aircraft state vector (from flight_dynamics)
            sim_time: Simulation time in seconds

        Returns:
            Action array [throttle, aileron, elevator, rudder, flaps, spoilers, brakes]
        """
        # Extract state (assuming StateIndex ordering)
        x_ft = state[0] * M_TO_FT
        y_ft = state[1] * M_TO_FT
        z_ft = state[2] * M_TO_FT
        altitude_ft = -z_ft  # Z is down

        u_fps = state[3] * M_TO_FT
        v_fps = state[4] * M_TO_FT
        w_fps = state[5] * M_TO_FT
        airspeed_fps = np.sqrt(u_fps**2 + v_fps**2 + w_fps**2)

        phi = state[6]    # Roll
        theta = state[7]  # Pitch
        psi = state[8]    # Heading

        # Compute inertial climb rate from body velocities and attitude
        # h_dot = u*sin(theta) - v*cos(theta)*sin(phi) - w*cos(theta)*cos(phi)
        climb_rate_fps = (u_fps * np.sin(theta)
                         - v_fps * np.cos(theta) * np.sin(phi)
                         - w_fps * np.cos(theta) * np.cos(phi))
        p = state[9]      # Roll rate
        q = state[10]     # Pitch rate
        r = state[11]     # Yaw rate

        # Update guidance from trajectory if in NAV mode
        if self.mode == AutopilotMode.NAV and self.trajectory:
            self._update_guidance_from_trajectory(x_ft, y_ft, altitude_ft, airspeed_fps)

        # Compute control based on current commands
        return self._compute_control(
            altitude_ft, airspeed_fps, climb_rate_fps,
            phi, theta, psi, p, q, r
        )

    def _update_guidance_from_trajectory(self, x_ft: float, y_ft: float,
                                          altitude_ft: float, airspeed_fps: float):
        """Update guidance commands from trajectory."""
        if not self.trajectory:
            return

        # Find closest point and lookahead
        closest, dist_to_path, cross_track = self.trajectory.find_closest_point(x_ft, y_ft)

        # Update segment - but override based on actual altitude
        trajectory_segment = closest.segment
        cruise_altitude = 5500  # TODO: get from trajectory

        # Don't transition to CRUISE until we're actually at cruise altitude
        if trajectory_segment == FlightSegment.CRUISE and altitude_ft < cruise_altitude - 500:
            self.current_segment = FlightSegment.CLIMB
        else:
            self.current_segment = trajectory_segment

        self.distance_to_destination = self.trajectory.total_distance - closest.s

        # Check for landing - use AGL altitude and require being close to runway
        altitude_agl_ft = altitude_ft - self.runway_elevation_ft
        # Landing conditions: in landing segment, low AGL, and reasonably close to destination
        # OR: very low AGL regardless of segment (safety catch)
        if ((self.current_segment == FlightSegment.LANDING and altitude_agl_ft < 50 and self.distance_to_destination < 2000)
            or (altitude_agl_ft < 10 and self.distance_to_destination < 5000)):
            self.has_landed = True
            self.mode = AutopilotMode.LANDING
            return

        # Use lookahead for altitude targeting during climb/descent
        if self.current_segment in (FlightSegment.CLIMB, FlightSegment.TAKEOFF, FlightSegment.DESCENT):
            lookahead_dist = max(2000, airspeed_fps * 10)  # 10 seconds ahead during climb
        else:
            lookahead_dist = max(500, airspeed_fps * 2)  # 2 seconds ahead in cruise
        lookahead = self.trajectory.get_lookahead_point(x_ft, y_ft, lookahead_dist)

        # Update lateral commands using Vector Field Guidance
        # This provides smooth, Lyapunov-stable convergence to the path
        # See docs/guidance_algorithms.md for details
        target_heading, cross_track, along_track = self.vector_field_guidance.compute_desired_heading(
            x_ft, y_ft, self.trajectory, airspeed_fps
        )

        self.commands.target_heading = target_heading
        self.commands.target_track = closest.heading
        self.commands.cross_track_error = cross_track
        self.commands.lateral_mode = LateralMode.TRACK

        # Update vertical commands - use lookahead altitude during climb/descent
        if self.current_segment in (FlightSegment.CLIMB, FlightSegment.TAKEOFF):
            # During climb, target lookahead altitude to drive climb
            self.commands.target_altitude = lookahead.z
        elif self.current_segment == FlightSegment.DESCENT:
            # During descent, also use lookahead
            self.commands.target_altitude = lookahead.z
        else:
            # Cruise/approach: use closest point for tighter tracking
            self.commands.target_altitude = closest.z

        self.commands.target_airspeed = closest.target_airspeed or closest.speed

        # Update flags
        self.commands.on_approach = self.current_segment == FlightSegment.APPROACH
        self.commands.on_final = self.current_segment == FlightSegment.LANDING

        # Flaps based on segment
        if self.current_segment == FlightSegment.LANDING:
            self.commands.target_flaps = 1.0
        elif self.current_segment == FlightSegment.APPROACH:
            self.commands.target_flaps = 0.5
        elif self.current_segment == FlightSegment.TAKEOFF:
            self.commands.target_flaps = 0.0
        else:
            self.commands.target_flaps = 0.0

    def _compute_control(self, altitude_ft: float, airspeed_fps: float, climb_rate_fps: float,
                          phi: float, theta: float, psi: float,
                          p: float, q: float, r: float) -> np.ndarray:
        """Compute control surfaces from guidance commands."""

        # Check for landed state
        if self.has_landed:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Lateral control
        aileron = self.heading_ctrl.compute(
            self.commands.target_heading, psi, phi, p
        )

        # Vertical control
        if self.commands.on_approach or self.commands.on_final:
            # Glideslope tracking - use AGL altitude (above runway elevation)
            distance_ft = self.distance_to_destination
            altitude_agl_ft = altitude_ft - self.runway_elevation_ft
            target_pitch, throttle, spoiler = self.glideslope_ctrl.compute(
                distance_ft, altitude_agl_ft, climb_rate_fps, airspeed_fps
            )
            elevator = self.pitch_ctrl.compute(target_pitch, theta, q)
            flaps = self.commands.target_flaps
        elif self.current_segment in (FlightSegment.CLIMB, FlightSegment.TAKEOFF):
            # During climb: use constant climb pitch like expert controller
            # Use full throttle and maintain climb pitch
            target_pitch_rad = np.deg2rad(self.config.pitch_climb)
            throttle = 1.0
            elevator = self.pitch_ctrl.compute(target_pitch_rad, theta, q)
            spoiler = 0.0
            flaps = self.commands.target_flaps
        else:
            # Cruise: altitude hold tracking trajectory
            target_pitch_rad = self.altitude_ctrl.compute(
                self.commands.target_altitude, altitude_ft, climb_rate_fps,
                np.deg2rad(self.config.pitch_cruise)
            )
            elevator = self.pitch_ctrl.compute(target_pitch_rad, theta, q)

            # Throttle for speed
            throttle = self.speed_ctrl.compute(
                self.commands.target_airspeed, airspeed_fps
            )
            spoiler = 0.0
            flaps = self.commands.target_flaps

        # Rudder - simple coordinated turn
        # Note: A proper yaw damper requires a washout filter to only dampen
        # oscillations, not steady-state turn rate. For now, keep rudder neutral.
        rudder = 0.0

        # Brakes
        brakes = 0.0

        return np.array([throttle, aileron, elevator, rudder, flaps, spoiler, brakes],
                        dtype=np.float32)

    def reset(self):
        """Reset autopilot state."""
        self.mode = AutopilotMode.MANUAL
        self.trajectory = None
        self.has_landed = False
        self.speed_ctrl.reset()


class SimpleWaypointAutopilot:
    """
    Simplified autopilot for waypoint-to-waypoint navigation.

    Does not require a full trajectory - just flies between waypoints.
    """

    def __init__(self, config: Optional[AutopilotConfig] = None):
        """Initialize autopilot."""
        self.config = config or AutopilotConfig()

        # Controllers
        self.heading_ctrl = HeadingController(
            heading_kp=self.config.heading_kp,
            roll_kp=self.config.roll_kp,
            roll_kd=self.config.roll_kd,
            max_bank_angle=self.config.max_bank_deg
        )
        self.pitch_ctrl = PitchController(
            kp=self.config.pitch_kp,
            kd=self.config.pitch_kd
        )
        self.altitude_ctrl = AltitudeController(
            alt_kp=self.config.alt_kp,
            alt_ki=self.config.alt_ki,
            vs_kp=self.config.vs_kp,
            max_vs_fpm=self.config.max_vs_fpm
        )
        self.speed_ctrl = AirspeedController(
            kp=self.config.speed_kp,
            ki=self.config.speed_ki,
            base_throttle=self.config.base_throttle
        )

        # Current targets
        self.target_heading = 0.0
        self.target_altitude_ft = 5500.0
        self.target_airspeed_fps = 185.0
        self.target_flaps = 0.0

    def set_targets(self, heading: float = None, altitude_ft: float = None,
                    airspeed_fps: float = None, flaps: float = None):
        """Set autopilot targets."""
        if heading is not None:
            self.target_heading = heading
        if altitude_ft is not None:
            self.target_altitude_ft = altitude_ft
        if airspeed_fps is not None:
            self.target_airspeed_fps = airspeed_fps
        if flaps is not None:
            self.target_flaps = flaps

    def fly_to_waypoint(self, x_ft: float, y_ft: float,
                        waypoint_x: float, waypoint_y: float,
                        track_heading: Optional[float] = None):
        """
        Set heading to fly to a waypoint.

        Args:
            x_ft, y_ft: Current position
            waypoint_x, waypoint_y: Target waypoint
            track_heading: Optional desired track (for cross-track correction)
        """
        dx = waypoint_x - x_ft
        dy = waypoint_y - y_ft

        if track_heading is not None:
            # Cross-track correction
            track_dx = np.cos(track_heading)
            track_dy = np.sin(track_heading)
            cross_track = -dx * track_dy + dy * track_dx

            intercept = np.arctan(0.0005 * cross_track)
            intercept = np.clip(intercept, np.deg2rad(-20), np.deg2rad(20))
            self.target_heading = track_heading - intercept
        else:
            # Direct to waypoint
            self.target_heading = np.arctan2(dy, dx)

    def compute_action(self, altitude_ft: float, airspeed_fps: float, climb_rate_fps: float,
                       phi: float, theta: float, psi: float,
                       p: float, q: float, r: float) -> np.ndarray:
        """
        Compute control action.

        Returns:
            Action array [throttle, aileron, elevator, rudder, flaps, spoilers, brakes]
        """
        # Lateral
        aileron = self.heading_ctrl.compute(self.target_heading, psi, phi, p)

        # Vertical
        target_pitch = self.altitude_ctrl.compute(
            self.target_altitude_ft, altitude_ft, climb_rate_fps,
            np.deg2rad(self.config.pitch_cruise)
        )
        elevator = self.pitch_ctrl.compute(target_pitch, theta, q)

        # Throttle
        throttle = self.speed_ctrl.compute(self.target_airspeed_fps, airspeed_fps)

        return np.array([throttle, aileron, elevator, 0.0, self.target_flaps, 0.0, 0.0],
                        dtype=np.float32)

    def reset(self):
        """Reset controller state."""
        self.speed_ctrl.reset()
