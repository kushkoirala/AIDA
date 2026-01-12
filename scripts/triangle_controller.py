#!/usr/bin/env python3
"""
Triangle Intercept Controller for KHUT RWY 31 Approach
All internal calculations in IMPERIAL UNITS: ft, ft/s, nm
Converts to/from metric only at flight_dynamics interface boundary
"""
import numpy as np
import sys
from pathlib import Path
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpu-flight-dynamics' / 'python'))
from flight_dynamics import StateIndex

# Unit conversion constants
FT_TO_M = 0.3048
M_TO_FT = 3.28084
NM_TO_FT = 6076.12
FT_TO_NM = 1 / 6076.12
KTS_TO_FPS = 1.68781  # knots to ft/s
FPS_TO_KTS = 1 / 1.68781

class XCPhase(Enum):
    GROUND_ROLL = 1
    ROTATION = 2
    INITIAL_CLIMB = 3
    CLIMB = 4
    CRUISE_TO_TP = 5
    TURN_TO_INTERCEPT = 6
    INTERCEPT_LEG = 7
    FINAL_APPROACH = 8
    SHORT_FINAL = 9
    LANDING = 10
    LANDED = 11

class TriangleInterceptController:
    """
    All parameters and internal state in IMPERIAL units (ft, ft/s, nm)
    """
    def __init__(self,
                 v_rotate_kts=54.0,      # Rotation speed in knots
                 v_climb_kts=74.0,       # Climb speed in knots
                 v_cruise_kts=110.0,     # Cruise speed in knots
                 v_approach_kts=65.0,    # Approach speed in knots
                 v_touchdown_kts=50.0,   # Touchdown speed in knots
                 cruise_altitude_ft=5500.0,
                 pattern_altitude_ft=1500.0):

        # Speeds in ft/s (internal unit)
        self.v_rotate = v_rotate_kts * KTS_TO_FPS      # ~91 ft/s
        self.v_climb = v_climb_kts * KTS_TO_FPS        # ~125 ft/s
        self.v_cruise = v_cruise_kts * KTS_TO_FPS     # ~185 ft/s
        self.v_approach = v_approach_kts * KTS_TO_FPS  # ~110 ft/s
        self.v_touchdown = v_touchdown_kts * KTS_TO_FPS # ~84 ft/s

        # Altitudes in ft
        self.cruise_altitude_ft = cruise_altitude_ft
        self.pattern_altitude_ft = pattern_altitude_ft
        self.initial_climb_alt_ft = 100.0
        self.flare_altitude_ft = 50.0
        self.touchdown_altitude_ft = 3.0
        self.short_final_distance_ft = 2500.0  # ~2500ft from threshold

        # KHUT airport position (in ft from SN65 origin)
        # Original metric: khut_x=52800m, khut_y=-21300m
        self.khut_x_ft = 52800.0 * M_TO_FT   # ~173,228 ft
        self.khut_y_ft = -21300.0 * M_TO_FT  # ~-69,882 ft

        self.runway_heading = np.deg2rad(314.0)
        self.backcourse = np.deg2rad(134.0)

        # Runway 31 threshold: ~2000ft from airport center along backcourse
        threshold_offset_ft = 610.0 * M_TO_FT  # ~2001 ft
        self.threshold_x_ft = self.khut_x_ft + threshold_offset_ft * np.cos(self.backcourse)
        self.threshold_y_ft = self.khut_y_ft + threshold_offset_ft * np.sin(self.backcourse)

        # Touchdown aimpoint: AT the threshold (aim short to land on target)
        aimpoint_dist_ft = 0.0
        self.aimpoint_x_ft = self.threshold_x_ft + aimpoint_dist_ft * np.cos(self.runway_heading)
        self.aimpoint_y_ft = self.threshold_y_ft + aimpoint_dist_ft * np.sin(self.runway_heading)

        # Turning point: 10nm behind threshold on backcourse
        tp_distance_ft = 10.0 * NM_TO_FT  # ~60,761 ft
        self.tp_x_ft = self.threshold_x_ft + tp_distance_ft * np.cos(self.backcourse)
        self.tp_y_ft = self.threshold_y_ft + tp_distance_ft * np.sin(self.backcourse)

        # For compatibility - expose metric versions (converted from ft)
        self.khut_x = self.khut_x_ft * FT_TO_M
        self.khut_y = self.khut_y_ft * FT_TO_M
        self.threshold_x = self.threshold_x_ft * FT_TO_M
        self.threshold_y = self.threshold_y_ft * FT_TO_M
        self.aimpoint_x = self.aimpoint_x_ft * FT_TO_M
        self.aimpoint_y = self.aimpoint_y_ft * FT_TO_M
        self.tp_x = self.tp_x_ft * FT_TO_M
        self.tp_y = self.tp_y_ft * FT_TO_M
        self.cruise_altitude = cruise_altitude_ft * FT_TO_M
        self.pattern_altitude = pattern_altitude_ft * FT_TO_M

        self.cruise_heading = np.arctan2(self.tp_y_ft, self.tp_x_ft)
        self.departure_heading = np.deg2rad(4.0)
        self.turn_to_cruise_altitude_ft = 1000.0
        self.tp_trigger_distance_ft = 3000.0  # ~3000 ft = 0.5nm (must be close to TP to turn)

        self.phase = XCPhase.GROUND_ROLL
        self.phase_start_time = 0.0
        self.has_turned_to_cruise = False

        # Control gains
        self.kp_pitch = 0.8
        self.kd_pitch = 0.6
        self.kp_roll = 1.2
        self.kd_roll = 0.4
        self.turn_bank_angle = np.deg2rad(25.0)

        # Target pitch angles
        self.pitch_rotate = np.deg2rad(10.0)
        self.pitch_climb = np.deg2rad(8.0)
        self.pitch_cruise = np.deg2rad(0.0)  # True level flight for cruise
        self.pitch_descent = np.deg2rad(-3.0)
        self.pitch_approach = np.deg2rad(-2.0)
        self.pitch_flare = np.deg2rad(5.0)

        # Glideslope angle (degrees) - steeper for earlier touchdown
        # Standard ILS is 3.0°, using 3.5° for steeper descent
        self.glideslope_deg = 3.5

    def reset(self):
        self.phase = XCPhase.GROUND_ROLL
        self.phase_start_time = 0.0
        self.has_turned_to_cruise = False

    def _normalize_angle(self, angle):
        while angle > np.pi: angle -= 2*np.pi
        while angle < -np.pi: angle += 2*np.pi
        return angle

    def _heading_control(self, target, current, roll, roll_rate):
        error = self._normalize_angle(target - current)
        target_roll = np.clip(error * 0.5, -self.turn_bank_angle, self.turn_bank_angle)
        roll_error = target_roll - roll
        return np.clip(self.kp_roll * roll_error - self.kd_roll * roll_rate, -1.0, 1.0)

    def _pitch_control(self, target, current, pitch_rate):
        # Elevator convention: negative elevator = pitch UP, so invert output
        error = target - current
        return -np.clip(self.kp_pitch * error - self.kd_pitch * pitch_rate, -1.0, 1.0)

    def compute_action(self, state, sim_time):
        """
        Takes state in METRIC (from flight_dynamics), converts to imperial internally,
        computes control, returns action array.
        """
        # Extract state and convert to imperial (ft, ft/s)
        x_ft = state[StateIndex.X] * M_TO_FT
        y_ft = state[StateIndex.Y] * M_TO_FT
        z_ft = state[StateIndex.Z] * M_TO_FT
        altitude_ft = -z_ft  # Z is down, so altitude = -Z

        u_fps = state[StateIndex.U] * M_TO_FT  # ft/s
        v_fps = state[StateIndex.V] * M_TO_FT
        w_fps = state[StateIndex.W] * M_TO_FT
        airspeed_fps = np.sqrt(u_fps**2 + v_fps**2 + w_fps**2)
        climb_rate_fps = -w_fps  # positive = climbing

        phi = state[StateIndex.PHI]
        theta = state[StateIndex.THETA]
        psi = state[StateIndex.PSI]
        p = state[StateIndex.P]
        q = state[StateIndex.Q]
        r = state[StateIndex.R]

        # Distances in ft
        dist_to_tp_ft = np.sqrt((x_ft - self.tp_x_ft)**2 + (y_ft - self.tp_y_ft)**2)
        dist_to_threshold_ft = np.sqrt((x_ft - self.threshold_x_ft)**2 + (y_ft - self.threshold_y_ft)**2)
        dist_to_aimpoint_ft = np.sqrt((x_ft - self.aimpoint_x_ft)**2 + (y_ft - self.aimpoint_y_ft)**2)

        # Phase transitions
        if self.phase == XCPhase.GROUND_ROLL and airspeed_fps >= self.v_rotate:
            self.phase = XCPhase.ROTATION
            self.phase_start_time = sim_time
        elif self.phase == XCPhase.ROTATION and altitude_ft > self.initial_climb_alt_ft:
            self.phase = XCPhase.INITIAL_CLIMB
        elif self.phase == XCPhase.INITIAL_CLIMB and altitude_ft > self.turn_to_cruise_altitude_ft:
            self.phase = XCPhase.CLIMB
        elif self.phase == XCPhase.CLIMB and altitude_ft >= self.cruise_altitude_ft - 50:
            self.phase = XCPhase.CRUISE_TO_TP
        elif self.phase == XCPhase.CRUISE_TO_TP and dist_to_tp_ft < self.tp_trigger_distance_ft:
            self.phase = XCPhase.TURN_TO_INTERCEPT
        elif self.phase == XCPhase.TURN_TO_INTERCEPT:
            heading_error = abs(self._normalize_angle(psi - self.runway_heading))
            if heading_error < np.deg2rad(10):
                self.phase = XCPhase.INTERCEPT_LEG
        elif self.phase == XCPhase.INTERCEPT_LEG and dist_to_threshold_ft <= 3 * NM_TO_FT:  # 3nm
            self.phase = XCPhase.FINAL_APPROACH
        elif self.phase == XCPhase.FINAL_APPROACH and dist_to_threshold_ft <= self.short_final_distance_ft:
            self.phase = XCPhase.SHORT_FINAL
        elif self.phase == XCPhase.SHORT_FINAL and altitude_ft <= self.flare_altitude_ft:
            self.phase = XCPhase.LANDING
        elif self.phase == XCPhase.LANDING and altitude_ft <= self.touchdown_altitude_ft:
            self.phase = XCPhase.LANDED

        # Control logic by phase
        if self.phase == XCPhase.GROUND_ROLL:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.ROTATION:
            return np.array([1.0, 0.0, self._pitch_control(self.pitch_rotate, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.INITIAL_CLIMB:
            return np.array([1.0, self._heading_control(self.departure_heading, psi, phi, p),
                           self._pitch_control(self.pitch_climb, theta, q), 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.CLIMB:
            if altitude_ft > self.turn_to_cruise_altitude_ft:
                self.has_turned_to_cruise = True
            hdg = self.cruise_heading if self.has_turned_to_cruise else self.departure_heading
            return np.array([1.0, self._heading_control(hdg, psi, phi, p),
                           self._pitch_control(self.pitch_climb, theta, q), 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.CRUISE_TO_TP:
            bearing = np.arctan2(self.tp_y_ft - y_ft, self.tp_x_ft - x_ft)
            alt_error_ft = altitude_ft - self.cruise_altitude_ft

            # Altitude hold using vertical speed feedback (autopilot-style)
            # Step 1: Compute target vertical speed based on altitude error
            # Target VS: -2 ft/s per 100ft error (200ft high -> descend at 400 fpm)
            target_vs_fps = -0.02 * alt_error_ft
            target_vs_fps = np.clip(target_vs_fps, -500/60, 500/60)  # Limit to ±500 fpm

            # Step 2: Compute VS error and use it for pitch control
            vs_error_fps = climb_rate_fps - target_vs_fps

            # Pitch: proportional to VS error
            # If climbing too fast (vs_error > 0), pitch down
            # -0.5 deg per 1 ft/s VS error
            pitch_correction = np.deg2rad(-0.5 * vs_error_fps)
            pitch_correction = np.clip(pitch_correction, np.deg2rad(-8.0), np.deg2rad(8.0))
            pitch_target = self.pitch_cruise + pitch_correction

            # Throttle: Use lower base throttle and adjust based on altitude
            # This aircraft tends to climb, so use lower power setting
            base_throttle = 0.50  # Reduced from 0.6
            throttle_correction = -0.0005 * alt_error_ft  # Gentler throttle response
            throttle = np.clip(base_throttle + throttle_correction, 0.3, 0.8)

            return np.array([throttle, self._heading_control(bearing, psi, phi, p),
                           self._pitch_control(pitch_target, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.TURN_TO_INTERCEPT:
            # Turn to runway heading, maintain altitude and speed during turn
            # Don't start descent until established on the intercept leg
            return np.array([0.7, self._heading_control(self.runway_heading, psi, phi, p),
                           self._pitch_control(self.pitch_cruise, theta, q), 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.INTERCEPT_LEG:
            # Cross-track correction to centerline
            dx_ft = x_ft - self.aimpoint_x_ft
            dy_ft = y_ft - self.aimpoint_y_ft
            rwy_dx = np.cos(self.runway_heading)
            rwy_dy = np.sin(self.runway_heading)
            cross_track_ft = dx_ft * (-rwy_dy) + dy_ft * rwy_dx

            intercept_angle = np.clip(-cross_track_ft * 0.0005, -np.deg2rad(20), np.deg2rad(20))
            target_heading = self.runway_heading + intercept_angle

            # Glideslope target altitude based on distance to aimpoint
            target_alt_ft = dist_to_aimpoint_ft * np.tan(np.deg2rad(self.glideslope_deg))
            alt_error_ft = altitude_ft - target_alt_ft

            # Speed protection: minimum 50 kts = ~84 ft/s (stall is around 45kts)
            min_speed_fps = 50.0 * KTS_TO_FPS
            speed_low = airspeed_fps < min_speed_fps

            if speed_low:
                # Speed recovery: increase throttle, reduce descent
                target_descent_fps = -300 / 60  # -300 fpm gentle descent
                throttle = 0.9
                spoiler = 0.0
                pitch_base = np.deg2rad(0.0)  # Level to gain speed
            elif alt_error_ft > 2000:  # Way above glideslope - aggressive descent
                target_descent_fps = -3000 / 60  # -3000 fpm max descent
                throttle = 0.3  # Idle for speed control
                spoiler = 0.8
                pitch_base = np.deg2rad(-8.0)  # Steep nose down
            elif alt_error_ft > 1000:
                target_descent_fps = -2500 / 60  # -2500 fpm
                throttle = 0.35
                spoiler = 0.6
                pitch_base = np.deg2rad(-6.0)
            elif alt_error_ft > 500:
                target_descent_fps = -1500 / 60  # -1500 fpm
                throttle = 0.4
                spoiler = 0.4
                pitch_base = np.deg2rad(-4.0)
            elif alt_error_ft > 200:
                target_descent_fps = -1000 / 60  # -1000 fpm
                throttle = 0.45
                spoiler = 0.2
                pitch_base = np.deg2rad(-3.0)
            else:
                # On glideslope: 3 deg at ~90 kts = ~500 fpm
                gs_descent_fps = -airspeed_fps * np.tan(np.deg2rad(self.glideslope_deg))
                target_descent_fps = gs_descent_fps - 0.05 * alt_error_ft
                throttle = 0.5
                spoiler = 0.0
                pitch_base = self.pitch_descent

            target_descent_fps = np.clip(target_descent_fps, -3000/60, 0)
            # Pitch adjustment: error in fps, convert to pitch adjustment in radians
            pitch_adjust = (climb_rate_fps - target_descent_fps) * 0.003
            pitch = pitch_base + pitch_adjust
            return np.array([throttle, self._heading_control(target_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q), 0.0, 0.3, spoiler, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.FINAL_APPROACH:
            # Stay on centerline and glideslope
            dx_ft = x_ft - self.aimpoint_x_ft
            dy_ft = y_ft - self.aimpoint_y_ft
            rwy_dx = np.cos(self.runway_heading)
            rwy_dy = np.sin(self.runway_heading)
            cross_track_ft = dx_ft * (-rwy_dy) + dy_ft * rwy_dx

            intercept_angle = np.clip(-cross_track_ft * 0.001, -np.deg2rad(15), np.deg2rad(15))
            target_heading = self.runway_heading + intercept_angle

            target_alt_ft = dist_to_aimpoint_ft * np.tan(np.deg2rad(self.glideslope_deg))
            alt_error_ft = altitude_ft - target_alt_ft

            # Aggressive glideslope capture if still high
            if alt_error_ft > 500:
                target_descent_fps = -2000 / 60
                throttle = 0.3
                spoiler = 0.5
                pitch_base = np.deg2rad(-5.0)
            elif alt_error_ft > 200:
                target_descent_fps = -1000 / 60
                throttle = 0.4
                spoiler = 0.3
                pitch_base = np.deg2rad(-3.0)
            else:
                # On glideslope
                gs_descent_fps = -airspeed_fps * np.tan(np.deg2rad(self.glideslope_deg))
                target_descent_fps = gs_descent_fps - 0.1 * alt_error_ft
                throttle = 0.45
                spoiler = 0.0
                pitch_base = self.pitch_approach

            target_descent_fps = np.clip(target_descent_fps, -2000/60, 0)
            pitch_adjust = (climb_rate_fps - target_descent_fps) * 0.003
            pitch = pitch_base + pitch_adjust
            return np.array([throttle, self._heading_control(target_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q), 0.0, 0.5, spoiler, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.SHORT_FINAL:
            # Maintain glideslope, prepare for landing
            target_alt_ft = dist_to_aimpoint_ft * np.tan(np.deg2rad(self.glideslope_deg))
            alt_error_ft = altitude_ft - target_alt_ft

            # Still need aggressive descent if high on short final
            if alt_error_ft > 300:
                target_descent_fps = -1500 / 60
                throttle = 0.2
                spoiler = 0.4
                pitch_base = np.deg2rad(-4.0)
            elif alt_error_ft > 100:
                target_descent_fps = -800 / 60
                throttle = 0.3
                spoiler = 0.2
                pitch_base = np.deg2rad(-2.0)
            else:
                # On glideslope - standard descent
                gs_descent_fps = -airspeed_fps * np.tan(np.deg2rad(self.glideslope_deg))
                target_descent_fps = gs_descent_fps - 0.1 * alt_error_ft
                throttle = 0.35
                spoiler = 0.0
                pitch_base = self.pitch_approach

            target_descent_fps = np.clip(target_descent_fps, -1500/60, 0)
            pitch_adjust = (climb_rate_fps - target_descent_fps) * 0.003
            pitch = pitch_base + pitch_adjust
            return np.array([throttle, self._heading_control(self.runway_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q), 0.0, 0.7, spoiler, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.LANDING:
            # Flare: arrest descent gently, pitch up proportional to descent rate
            # climb_rate_fps is negative when descending, so add positive pitch to flare
            pitch_adjust = -climb_rate_fps * 0.005  # Flare based on descent rate
            pitch = self.pitch_flare + pitch_adjust
            return np.array([0.0, self._heading_control(self.runway_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q), 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.LANDED:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        return np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
