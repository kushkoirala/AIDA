#!/usr/bin/env python3
"""
PID Controllers for AIDA Flight Control

Extracted and generalized from triangle_controller.py for reuse
across different flight modes and trajectory following.

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0


class PIDController:
    """
    Generic PID controller with anti-windup and output limiting.
    """

    def __init__(self,
                 kp: float = 1.0,
                 ki: float = 0.0,
                 kd: float = 0.0,
                 output_min: float = -1.0,
                 output_max: float = 1.0,
                 integral_limit: float = 10.0,
                 derivative_filter: float = 0.1):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_min: Minimum output value
            output_max: Maximum output value
            integral_limit: Anti-windup integral limit
            derivative_filter: Low-pass filter for derivative (0-1, lower = more filtering)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit
        self.derivative_filter = derivative_filter

        # State
        self._integral = 0.0
        self._last_error = 0.0
        self._last_derivative = 0.0
        self._initialized = False

    def reset(self):
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_derivative = 0.0
        self._initialized = False

    def compute(self, error: float, dt: float = 0.02, derivative: Optional[float] = None) -> float:
        """
        Compute PID output.

        Args:
            error: Current error (setpoint - measurement)
            dt: Time step in seconds
            derivative: Optional pre-computed derivative (e.g., from rate gyro)

        Returns:
            Control output clipped to [output_min, output_max]
        """
        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self._integral

        # Derivative term
        if derivative is not None:
            # Use provided derivative (e.g., from rate gyro)
            d_raw = derivative
        elif self._initialized:
            # Compute derivative from error
            d_raw = (error - self._last_error) / dt
        else:
            d_raw = 0.0
            self._initialized = True

        # Low-pass filter on derivative
        self._last_derivative = (self.derivative_filter * d_raw +
                                  (1 - self.derivative_filter) * self._last_derivative)
        d_term = self.kd * self._last_derivative

        self._last_error = error

        # Compute output and clip
        output = p_term + i_term + d_term
        return np.clip(output, self.output_min, self.output_max)


class HeadingController:
    """
    Heading controller that outputs aileron command.

    Uses a two-level cascade:
    1. Heading error -> target roll angle
    2. Roll error -> aileron command
    """

    def __init__(self,
                 heading_kp: float = 0.5,
                 roll_kp: float = 1.2,
                 roll_kd: float = 0.4,
                 max_bank_angle: float = 25.0):
        """
        Initialize heading controller.

        Args:
            heading_kp: Heading-to-roll gain
            roll_kp: Roll proportional gain
            roll_kd: Roll derivative gain
            max_bank_angle: Maximum bank angle in degrees
        """
        self.heading_kp = heading_kp
        self.roll_kp = roll_kp
        self.roll_kd = roll_kd
        self.max_bank_rad = np.deg2rad(max_bank_angle)

    def compute(self,
                target_heading: float,
                current_heading: float,
                current_roll: float,
                roll_rate: float) -> float:
        """
        Compute aileron command to achieve target heading.

        Args:
            target_heading: Target heading in radians
            current_heading: Current heading in radians
            current_roll: Current roll angle in radians
            roll_rate: Current roll rate in rad/s

        Returns:
            Aileron command in [-1, 1]
        """
        # Heading error (normalized to [-pi, pi])
        heading_error = self._normalize_angle(target_heading - current_heading)

        # Target roll angle proportional to heading error
        target_roll = np.clip(heading_error * self.heading_kp,
                              -self.max_bank_rad, self.max_bank_rad)

        # Roll PD control
        roll_error = target_roll - current_roll
        aileron = self.roll_kp * roll_error - self.roll_kd * roll_rate

        return np.clip(aileron, -1.0, 1.0)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class PitchController:
    """
    Pitch controller that outputs elevator command.

    Note: Assumes negative elevator = pitch up (standard convention).
    """

    def __init__(self,
                 kp: float = 0.8,
                 kd: float = 0.6,
                 invert_output: bool = True):
        """
        Initialize pitch controller.

        Args:
            kp: Pitch proportional gain
            kd: Pitch derivative gain
            invert_output: If True, negate output (for -elev = pitch up convention)
        """
        self.kp = kp
        self.kd = kd
        self.invert = -1.0 if invert_output else 1.0

    def compute(self,
                target_pitch: float,
                current_pitch: float,
                pitch_rate: float) -> float:
        """
        Compute elevator command to achieve target pitch.

        Args:
            target_pitch: Target pitch angle in radians
            current_pitch: Current pitch angle in radians
            pitch_rate: Current pitch rate in rad/s

        Returns:
            Elevator command in [-1, 1]
        """
        error = target_pitch - current_pitch
        elevator = self.invert * (self.kp * error - self.kd * pitch_rate)
        return np.clip(elevator, -1.0, 1.0)


class AltitudeController:
    """
    Altitude controller using vertical speed feedback.

    Two-level cascade:
    1. Altitude error -> target vertical speed
    2. VS error -> pitch target

    Uses aggressive P-only control on altitude with VS damping.
    """

    def __init__(self,
                 alt_kp: float = 0.015,
                 alt_ki: float = 0.001,
                 vs_kp: float = 0.4,
                 max_vs_fpm: float = 500.0,
                 max_pitch_deg: float = 8.0):
        """
        Initialize altitude controller.

        Args:
            alt_kp: Altitude-to-VS gain (ft/s per ft error)
            alt_ki: (unused, kept for API compatibility)
            vs_kp: VS-to-pitch gain (deg per ft/s)
            max_vs_fpm: Maximum vertical speed in ft/min
            max_pitch_deg: Maximum pitch correction in degrees
        """
        self.alt_kp = alt_kp
        self.alt_ki = alt_ki  # Keep for API compatibility
        self.vs_kp = vs_kp
        self.max_vs_fps = max_vs_fpm / 60.0  # Convert to ft/s
        self.max_pitch_rad = np.deg2rad(max_pitch_deg)

    def reset(self):
        """Reset controller state."""
        pass

    def compute(self,
                target_alt_ft: float,
                current_alt_ft: float,
                current_vs_fps: float,
                base_pitch: float = 0.0,
                dt: float = 0.02) -> float:
        """
        Compute target pitch to achieve target altitude.

        Args:
            target_alt_ft: Target altitude in feet
            current_alt_ft: Current altitude in feet
            current_vs_fps: Current vertical speed in ft/s (positive = climbing)
            base_pitch: Base pitch angle in radians (e.g., for cruise)
            dt: Time step in seconds (unused)

        Returns:
            Target pitch angle in radians
        """
        # Altitude error (positive when below target)
        alt_error_ft = target_alt_ft - current_alt_ft

        # Target vertical speed proportional to altitude error
        target_vs_fps = self.alt_kp * alt_error_ft
        target_vs_fps = np.clip(target_vs_fps, -self.max_vs_fps, self.max_vs_fps)

        # VS error (positive when climbing too slow)
        vs_error_fps = target_vs_fps - current_vs_fps

        # Pitch correction proportional to VS error
        # Positive VS error -> need to pitch up -> positive pitch correction
        pitch_correction = np.deg2rad(self.vs_kp * vs_error_fps)
        pitch_correction = np.clip(pitch_correction, -self.max_pitch_rad, self.max_pitch_rad)

        return base_pitch + pitch_correction


class AirspeedController:
    """
    Airspeed controller that outputs throttle command.
    """

    def __init__(self,
                 kp: float = 0.05,
                 ki: float = 0.01,
                 base_throttle: float = 0.5,
                 min_throttle: float = 0.0,
                 max_throttle: float = 1.0):
        """
        Initialize airspeed controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            base_throttle: Base throttle setting
            min_throttle: Minimum throttle
            max_throttle: Maximum throttle
        """
        self.kp = kp
        self.ki = ki
        self.base = base_throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self._integral = 0.0

    def reset(self):
        """Reset integral state."""
        self._integral = 0.0

    def compute(self,
                target_airspeed: float,
                current_airspeed: float,
                dt: float = 0.02) -> float:
        """
        Compute throttle command to achieve target airspeed.

        Args:
            target_airspeed: Target airspeed (any unit, consistent)
            current_airspeed: Current airspeed (same unit)
            dt: Time step in seconds

        Returns:
            Throttle command in [min_throttle, max_throttle]
        """
        error = target_airspeed - current_airspeed

        # Integral with anti-windup
        self._integral += error * dt
        self._integral = np.clip(self._integral, -20.0, 20.0)

        throttle = self.base + self.kp * error + self.ki * self._integral
        return np.clip(throttle, self.min_throttle, self.max_throttle)


class GlideslopeController:
    """
    Glideslope tracking controller for approach and landing.

    Computes target pitch and throttle to stay on glideslope.
    """

    def __init__(self,
                 glideslope_deg: float = 3.0,
                 approach_speed_fps: float = 110.0):
        """
        Initialize glideslope controller.

        Args:
            glideslope_deg: Glideslope angle in degrees
            approach_speed_fps: Target approach speed in ft/s
        """
        self.glideslope_rad = np.deg2rad(glideslope_deg)
        self.approach_speed = approach_speed_fps

    def compute(self,
                distance_ft: float,
                altitude_ft: float,
                current_vs_fps: float,
                airspeed_fps: float) -> tuple:
        """
        Compute glideslope tracking commands.

        Args:
            distance_ft: Distance to touchdown point in feet
            altitude_ft: Current altitude AGL in feet
            current_vs_fps: Current vertical speed in ft/s
            airspeed_fps: Current airspeed in ft/s

        Returns:
            Tuple of (target_pitch_rad, throttle, spoiler)
        """
        # Target altitude on glideslope
        target_alt_ft = distance_ft * np.tan(self.glideslope_rad)
        alt_error_ft = altitude_ft - target_alt_ft

        # Target descent rate on glideslope
        gs_descent_fps = -airspeed_fps * np.tan(self.glideslope_rad)

        # Aggressive descent if too high
        if alt_error_ft > 500:
            target_vs_fps = -2000 / 60  # -2000 fpm
            throttle = 0.3
            spoiler = 0.5
            pitch_base = np.deg2rad(-5.0)
        elif alt_error_ft > 200:
            target_vs_fps = -1000 / 60
            throttle = 0.4
            spoiler = 0.3
            pitch_base = np.deg2rad(-3.0)
        else:
            # On glideslope
            target_vs_fps = gs_descent_fps - 0.1 * alt_error_ft
            throttle = 0.45
            spoiler = 0.0
            pitch_base = np.deg2rad(-2.0)

        # Pitch adjustment based on VS error
        vs_error = current_vs_fps - target_vs_fps
        pitch_adjust = vs_error * 0.003
        pitch = pitch_base + pitch_adjust

        return pitch, throttle, spoiler


class CrossTrackController:
    """
    Cross-track error controller for lateral path following.

    Computes heading correction to return to desired track.
    """

    def __init__(self,
                 kp: float = 0.0005,
                 max_intercept_deg: float = 20.0):
        """
        Initialize cross-track controller.

        Args:
            kp: Cross-track gain (rad per ft)
            max_intercept_deg: Maximum intercept angle in degrees
        """
        self.kp = kp
        self.max_intercept_rad = np.deg2rad(max_intercept_deg)

    def compute(self,
                cross_track_ft: float,
                track_heading: float) -> float:
        """
        Compute target heading to correct cross-track error.

        Args:
            cross_track_ft: Cross-track error in feet (positive = right of track)
            track_heading: Desired track heading in radians

        Returns:
            Target heading in radians
        """
        intercept_angle = np.clip(-self.kp * cross_track_ft,
                                  -self.max_intercept_rad, self.max_intercept_rad)
        return track_heading + intercept_angle
