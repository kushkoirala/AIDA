#!/usr/bin/env python3
"""Classical Traffic Pattern Controller for Cessna 172

This module implements a finite state machine (FSM) based autopilot that flies
a complete traffic pattern: takeoff, climb, rectangular pattern, approach, and landing.

Architecture Overview
---------------------
The controller uses a hierarchical control structure:

1. **Mission Phase Manager (FSM)**: High-level state machine that determines
   current flight phase based on position, altitude, airspeed, and heading.

2. **Phase-Specific Controllers**: Each phase has a dedicated control law that
   generates appropriate control surface commands.

3. **Inner-Loop PD Controllers**: Low-level attitude control using proportional-
   derivative feedback for pitch, roll, and heading stabilization.

Control Outputs (Action Vector)
-------------------------------
The controller outputs a 6-element action vector:
    [throttle, aileron, elevator, rudder, flaps, spoilers]

- throttle: 0.0 (idle) to 1.0 (full power)
- aileron:  -1.0 (left roll) to +1.0 (right roll)
- elevator: -1.0 (nose down) to +1.0 (nose up)
- rudder:   -1.0 (left yaw) to +1.0 (right yaw)
- flaps:    0.0 (retracted) to 1.0 (full extension)
- spoilers: 0.0 (retracted) to 1.0 (full deployment)

Traffic Pattern Geometry
------------------------
Standard right-hand traffic pattern:

    UPWIND (N) ──────────────────► CROSSWIND TURN
         ▲                                │
         │                                ▼
      RUNWAY                         CROSSWIND (E)
         │                                │
         ▲                                ▼
    FINAL ◄────── BASE TURN ◄────── DOWNWIND TURN
         │                                │
         │                                ▼
    APPROACH                         DOWNWIND (S)
         │
         ▼
      LANDING

Coordinate System (NED)
-----------------------
- X: North (positive = north of runway)
- Y: East (positive = east of runway)
- Z: Down (negative = altitude above ground)

Key waypoints:
- Runway: Centered at origin, aligned N-S
- Upwind end: X = +4000m
- Crosswind end: Y = +2000m (east offset)
- Downwind end: X = -4000m
- Pattern altitude: 5000 ft (1524m) cruise, 1000 ft (305m) base/final

Usage for Neural Network Training
---------------------------------
This controller serves as an expert demonstrator for imitation learning:

1. **Behavior Cloning**: Record (observation, action) pairs during flight
   to train a neural network policy via supervised learning.

2. **Hybrid Architecture**: Neural network handles normal flight while
   classical controller provides safety backup:

   ```python
   if safety_monitor.is_safe(state):
       action = neural_net_policy(state)
   else:
       action = classical_controller.compute_action(state)
   ```

3. **Curriculum Learning**: Train on individual phases first (ground roll,
   climb, cruise, approach) before full traffic pattern.

Example
-------
```python
from classical_mission_controller import MissionController, MissionPhase
from flight_dynamics import FlightSimulator, StateIndex

controller = MissionController(
    cruise_altitude_ft=5000.0,
    pattern_altitude_ft=1000.0,
)

simulator = FlightSimulator(n_instances=1, dt=0.02)
state = simulator.get_states()[0]

while controller.phase != MissionPhase.LANDED:
    action = controller.compute_action(state, sim_time)
    simulator.set_controls(action.reshape(1, -1))
    simulator.step()
    state = simulator.get_states()[0]
```

Author: AIDA Project
License: MIT
"""
import numpy as np
import sys
from pathlib import Path
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))
from flight_dynamics import StateIndex

class MissionPhase(Enum):
    """Flight phases for the traffic pattern mission.

    State Transitions:
        GROUND_ROLL → ROTATION → INITIAL_CLIMB → CLIMB → CRUISE_UPWIND
        → TURN_CROSSWIND → CRUISE_CROSSWIND → TURN_DOWNWIND → CRUISE_DOWNWIND
        → TURN_BASE → DESCENT_BASE → TURN_FINAL → APPROACH → LANDING → LANDED
    """
    GROUND_ROLL = 1       # Accelerating on runway
    ROTATION = 2          # Pitching up for liftoff
    INITIAL_CLIMB = 3     # Immediate post-liftoff climb
    CLIMB = 4             # Climbing to cruise altitude
    CRUISE_UPWIND = 5     # Level flight heading north (upwind leg)
    TURN_CROSSWIND = 6    # 90° right turn to crosswind
    CRUISE_CROSSWIND = 7  # Level flight heading east (crosswind leg)
    TURN_DOWNWIND = 8     # 90° right turn to downwind
    CRUISE_DOWNWIND = 9   # Level flight heading south (downwind leg)
    TURN_BASE = 10        # 90° right turn to base, begin descent
    DESCENT_BASE = 11     # Descending on base leg heading west
    TURN_FINAL = 12       # Turn to align with runway centerline
    APPROACH = 13         # Final approach with glideslope tracking
    LANDING = 14          # Flare and touchdown
    LANDED = 15           # Mission complete, on ground


class MissionController:
    """Classical autopilot for flying a complete traffic pattern.

    This controller implements a finite state machine that manages 15 flight
    phases from ground roll through landing. Each phase uses PD feedback
    control for attitude stabilization.

    Parameters
    ----------
    v_rotate : float
        Rotation speed in m/s (default: 28.0 ≈ 54 kts)
    v_climb : float
        Target climb speed in m/s (default: 38.0 ≈ 74 kts)
    v_cruise : float
        Target cruise speed in m/s (default: 52.0 ≈ 101 kts)
    v_approach : float
        Final approach speed in m/s (default: 32.0 ≈ 62 kts)
    v_touchdown : float
        Target touchdown speed in m/s (default: 26.0 ≈ 50 kts)
    cruise_altitude_ft : float
        Target cruise altitude in feet (default: 5000)
    pattern_altitude_ft : float
        Pattern altitude for base/final in feet (default: 1000)
    runway_heading : float
        Runway heading in radians (default: 0.0 = north)
    runway_start_x : float
        X position of runway start in meters (default: -500)
    runway_length : float
        Runway length in meters (default: 1000)
    flight_area_length : float
        N-S extent of flight area in meters (default: 10000)
    flight_area_width : float
        E-W extent of flight area in meters (default: 5000)

    Attributes
    ----------
    phase : MissionPhase
        Current flight phase
    kp_pitch, kd_pitch : float
        Pitch rate PD gains
    kp_roll, kd_roll : float
        Roll rate PD gains
    kp_heading, kd_heading : float
        Heading hold PD gains
    kp_altitude, kd_altitude : float
        Altitude hold PD gains
    kp_speed : float
        Speed control proportional gain
    """

    def __init__(self, v_rotate=28.0, v_climb=38.0, v_cruise=52.0, v_approach=32.0, v_touchdown=26.0,
                 cruise_altitude_ft=5000.0, pattern_altitude_ft=1000.0, runway_heading=0.0,
                 runway_start_x=-500.0, runway_length=1000.0,
                 flight_area_length=10000.0, flight_area_width=5000.0):
        self.v_rotate = v_rotate
        self.v_climb = v_climb
        self.v_cruise = v_cruise
        self.v_approach = v_approach
        self.v_touchdown = v_touchdown
        self.cruise_altitude = cruise_altitude_ft * 0.3048
        self.pattern_altitude = pattern_altitude_ft * 0.3048
        self.initial_climb_alt = 30.0
        self.flare_altitude = 5.0
        self.touchdown_altitude = 0.5
        self.upwind_end_x = 4000.0
        self.crosswind_end_y = 2000.0
        self.downwind_end_x = -4000.0
        self.base_end_y = 0.0
        self.heading_upwind = np.deg2rad(0.0)
        self.heading_crosswind = np.deg2rad(90.0)
        self.heading_downwind = np.deg2rad(180.0)
        self.heading_base = np.deg2rad(270.0)
        self.landing_heading = np.deg2rad(0.0)
        self.phase = MissionPhase.GROUND_ROLL
        self.kp_pitch, self.kd_pitch = 1.5, 0.5
        self.kp_roll, self.kd_roll = 1.2, 0.4
        self.kp_heading, self.kd_heading = 0.8, 0.3
        self.kp_altitude, self.kd_altitude = 0.02, 0.005
        self.kp_speed = 0.05
        self.turn_bank_angle = np.deg2rad(20.0)
        self.pitch_rotate = np.deg2rad(10.0)
        self.pitch_climb = np.deg2rad(8.0)
        self.pitch_cruise = np.deg2rad(2.0)
        self.pitch_descent = np.deg2rad(-3.0)
        self.pitch_approach = np.deg2rad(-5.0)
        self.pitch_flare = np.deg2rad(5.0)
        self.descent_rate = 3.0

    def reset(self):
        """Reset controller to initial state (GROUND_ROLL)."""
        self.phase = MissionPhase.GROUND_ROLL

    def _wrap_angle(self, a):
        while a > np.pi: a -= 2*np.pi
        while a < -np.pi: a += 2*np.pi
        return a

    def _heading_near(self, psi, tgt, tol):
        return abs(self._wrap_angle(tgt - psi)) < tol

    def _heading_hold(self, psi, r, tgt):
        return np.clip(self.kp_heading * self._wrap_angle(tgt - psi) - self.kd_heading * r, -1, 1)

    def _wings_level(self, phi, p):
        return np.clip(self.kp_roll * (-phi) - self.kd_roll * p, -1, 1)

    def _update_phase(self, x, y, alt, spd, psi):
        P = self.phase
        if P == MissionPhase.GROUND_ROLL and spd >= self.v_rotate: self.phase = MissionPhase.ROTATION
        elif P == MissionPhase.ROTATION and alt > 2.0: self.phase = MissionPhase.INITIAL_CLIMB
        elif P == MissionPhase.INITIAL_CLIMB and alt > self.initial_climb_alt: self.phase = MissionPhase.CLIMB
        elif P == MissionPhase.CLIMB and (x >= 2000.0 or alt >= self.cruise_altitude * 0.95): self.phase = MissionPhase.CRUISE_UPWIND
        elif P == MissionPhase.CRUISE_UPWIND and x >= self.upwind_end_x: self.phase = MissionPhase.TURN_CROSSWIND
        elif P == MissionPhase.TURN_CROSSWIND and self._heading_near(psi, self.heading_crosswind, 0.087): self.phase = MissionPhase.CRUISE_CROSSWIND
        elif P == MissionPhase.CRUISE_CROSSWIND and y >= self.crosswind_end_y: self.phase = MissionPhase.TURN_DOWNWIND
        elif P == MissionPhase.TURN_DOWNWIND and self._heading_near(psi, self.heading_downwind, 0.087): self.phase = MissionPhase.CRUISE_DOWNWIND
        elif P == MissionPhase.CRUISE_DOWNWIND and x <= self.downwind_end_x: self.phase = MissionPhase.TURN_BASE
        elif P == MissionPhase.TURN_BASE and self._heading_near(psi, self.heading_base, 0.087): self.phase = MissionPhase.DESCENT_BASE
        elif P == MissionPhase.DESCENT_BASE and y <= self.base_end_y + 500: self.phase = MissionPhase.TURN_FINAL
        elif P == MissionPhase.TURN_FINAL and self._heading_near(psi, self.landing_heading, 0.17) and abs(y) < 200: self.phase = MissionPhase.APPROACH
        elif P == MissionPhase.APPROACH and alt <= self.flare_altitude: self.phase = MissionPhase.LANDING
        elif P == MissionPhase.LANDING and alt <= self.touchdown_altitude and spd < self.v_touchdown: self.phase = MissionPhase.LANDED

    def compute_action(self, obs, sim_time=0.0):
        """Compute control action based on current aircraft state.

        This is the main entry point called each simulation timestep.
        It updates the phase state machine and dispatches to the
        appropriate phase-specific controller.

        Parameters
        ----------
        obs : np.ndarray
            12-element state vector [x, y, z, u, v, w, phi, theta, psi, p, q, r]
            - x, y, z: Position in NED frame (m)
            - u, v, w: Body velocity (m/s)
            - phi, theta, psi: Euler angles (rad)
            - p, q, r: Angular rates (rad/s)
        sim_time : float
            Current simulation time in seconds (for logging)

        Returns
        -------
        np.ndarray
            6-element action vector [throttle, aileron, elevator, rudder, flaps, spoilers]
        """
        x, y, z = obs[StateIndex.X], obs[StateIndex.Y], obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi, theta, psi = obs[StateIndex.PHI], obs[StateIndex.THETA], obs[StateIndex.PSI]
        p, q, r = obs[StateIndex.P], obs[StateIndex.Q], obs[StateIndex.R]
        spd, alt, clmb = np.sqrt(u**2 + v**2 + w**2), -z, -w
        self._update_phase(x, y, alt, spd, psi)
        P = self.phase
        if P == MissionPhase.GROUND_ROLL: return self._ground_roll(spd, phi, psi, p, r)
        elif P == MissionPhase.ROTATION: return self._rotation(phi, theta, psi, p, q, r)
        elif P == MissionPhase.INITIAL_CLIMB: return self._initial_climb(phi, theta, psi, p, q, r)
        elif P == MissionPhase.CLIMB: return self._climb(alt, phi, theta, psi, p, q, r)
        elif P == MissionPhase.CRUISE_UPWIND: return self._cruise(alt, spd, phi, theta, psi, p, q, r, self.heading_upwind)
        elif P == MissionPhase.TURN_CROSSWIND: return self._turn(alt, spd, phi, theta, psi, p, q, r, self.heading_crosswind, False)
        elif P == MissionPhase.CRUISE_CROSSWIND: return self._cruise(alt, spd, phi, theta, psi, p, q, r, self.heading_crosswind)
        elif P == MissionPhase.TURN_DOWNWIND: return self._turn(alt, spd, phi, theta, psi, p, q, r, self.heading_downwind, False)
        elif P == MissionPhase.CRUISE_DOWNWIND: return self._cruise(alt, spd, phi, theta, psi, p, q, r, self.heading_downwind)
        elif P == MissionPhase.TURN_BASE: return self._turn(alt, spd, phi, theta, psi, p, q, r, self.heading_base, True)
        elif P == MissionPhase.DESCENT_BASE: return self._descent(alt, spd, clmb, phi, theta, psi, p, q, r, self.heading_base)
        elif P == MissionPhase.TURN_FINAL: return self._turn_final_intercept(alt, spd, phi, theta, psi, p, q, r, x, y)
        elif P == MissionPhase.APPROACH: return self._approach(alt, spd, phi, theta, psi, p, q, r, y)
        elif P == MissionPhase.LANDING: return self._landing(alt, phi, theta, psi, p, q, r, y)
        return np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)

    def _ground_roll(self, spd, phi, psi, p, r):
        e = -0.2 * min(1, max(0, (spd - 0.85*self.v_rotate) / (0.15*self.v_rotate)))
        return np.array([1, self._wings_level(phi, p), e, self._heading_hold(psi, r, self.heading_upwind), 0.1, 0], dtype=np.float32)

    def _rotation(self, phi, theta, psi, p, q, r):
        e = np.clip(-(self.kp_pitch * (self.pitch_rotate - theta) - self.kd_pitch * q), -0.8, 0.5)
        return np.array([1, self._wings_level(phi, p), e, self._heading_hold(psi, r, self.heading_upwind)*0.7, 0.1, 0], dtype=np.float32)

    def _initial_climb(self, phi, theta, psi, p, q, r):
        e = np.clip(-(self.kp_pitch * (self.pitch_climb - theta) - self.kd_pitch * q), -0.6, 0.5)
        return np.array([1, self._wings_level(phi, p), e, self._heading_hold(psi, r, self.heading_upwind)*0.5, 0, 0], dtype=np.float32)

    def _climb(self, alt, phi, theta, psi, p, q, r):
        blend = np.clip((self.cruise_altitude - alt) / 50.0, 0, 1)
        pitch = blend * self.pitch_climb + (1 - blend) * self.pitch_cruise
        e = np.clip(-(self.kp_pitch * (pitch - theta) - self.kd_pitch * q), -0.5, 0.4)
        return np.array([1, self._wings_level(phi, p), e, self._heading_hold(psi, r, self.heading_upwind)*0.3, 0, 0], dtype=np.float32)

    def _cruise(self, alt, spd, phi, theta, psi, p, q, r, tgt_hdg):
        # Speed control with protection
        if spd > 56: thr, flp, spl = 0, 0, 0.4
        elif spd > 52: thr, flp, spl = 0.3, 0, 0.2
        else: thr, flp, spl = np.clip(0.7 + self.kp_speed * (self.v_cruise - spd), 0.3, 1), 0, 0
        # Altitude hold
        pitch = np.clip(self.pitch_cruise + self.kp_altitude * (self.cruise_altitude - alt), -0.087, 0.14)
        e = np.clip(-(self.kp_pitch * (pitch - theta) - self.kd_pitch * q), -0.4, 0.4)
        # HEADING CONTROL: Use coordinated turn for heading corrections
        hdg_err = self._wrap_angle(tgt_hdg - psi)
        bank = np.clip(hdg_err * 0.5, -0.26, 0.26)  # 15 deg max bank
        a = np.clip(self.kp_roll * (bank - phi) - self.kd_roll * p, -0.4, 0.4)
        rd = 0.3 * np.sin(phi)  # Coordinated turn
        return np.array([thr, a, e, rd, flp, spl], dtype=np.float32)

    def _turn(self, alt, spd, phi, theta, psi, p, q, r, tgt_hdg, desc, y=0.0):
        tspd = self.v_approach if desc else self.v_cruise
        if spd > 56: thr, spl = 0, 0.4
        else: thr, spl = np.clip((0.5 if desc else 0.7) + self.kp_speed * (tspd - spd), 0.3, 1), 0.3 if desc else 0
        flp = 0.2 if desc else 0
        hdg_err = self._wrap_angle(tgt_hdg - psi)
        bank = self.turn_bank_angle if abs(hdg_err) > 0.087 else 0
        a = np.clip(self.kp_roll * (bank - phi) - self.kd_roll * p, -0.6, 0.6)
        talt = self.pattern_altitude if desc else self.cruise_altitude
        pitch = np.clip((self.pitch_descent if desc else self.pitch_cruise) + self.kp_altitude * (talt - alt) * 0.5, -0.14, 0.14)
        e = np.clip(-(self.kp_pitch * (pitch - theta) - self.kd_pitch * q), -0.5, 0.5)
        rd = 0.3 * np.sin(phi)
        return np.array([thr, a, e, rd, flp, spl], dtype=np.float32)

    def _turn_final_intercept(self, alt, spd, phi, theta, psi, p, q, r, x, y):
        """Turn final with runway intercept - continuous tracking toward intercept point.

        Aircraft enters from base leg (heading ~270°, Y~500m east of centerline).
        Must turn toward runway heading (0°) while intercepting centerline (Y=0).

        Strategy: Always aim for an intercept point on the centerline ahead.
        The intercept point moves as aircraft gets closer, creating a smooth curve.
        """
        tspd = self.v_approach
        if spd > 56: thr, spl = 0, 0.4
        else: thr, spl = np.clip(0.5 + self.kp_speed * (tspd - spd), 0.3, 1), 0.3
        flp = 0.3

        # Calculate intercept point on centerline
        # Distance ahead scales with lateral offset - farther from centerline, aim farther ahead
        # This creates a smooth intercept arc
        intercept_distance = max(300.0, abs(y) * 2.0)  # At least 300m ahead, more if far from centerline
        target_x = x + intercept_distance
        target_y = 0.0  # Centerline

        # Calculate heading to intercept point
        dx = target_x - x
        dy = target_y - y
        hdg_to_target = np.arctan2(dy, dx)
        hdg_err = self._wrap_angle(hdg_to_target - psi)

        # Bank angle: proportional to heading error, max 30° for aggressive intercept
        max_bank = np.deg2rad(30) if abs(y) > 200 else np.deg2rad(20)
        bank = np.clip(hdg_err * 1.2, -max_bank, max_bank)

        # Once close to centerline and aligned, reduce bank aggressiveness
        if abs(y) < 50 and abs(hdg_err) < np.deg2rad(15):
            # Fine tracking mode - localizer style
            localizer_correction = np.clip(-y * 0.008, -0.15, 0.15)
            hdg_err = self._wrap_angle(self.landing_heading + localizer_correction - psi)
            bank = np.clip(hdg_err * 0.6, -0.15, 0.15)

        a = np.clip(self.kp_roll * (bank - phi) - self.kd_roll * p, -0.7, 0.7)

        # Descend toward pattern altitude
        pitch = np.clip(self.pitch_descent + self.kp_altitude * (self.pattern_altitude - alt) * 0.3, -0.14, 0.087)
        e = np.clip(-(self.kp_pitch * (pitch - theta) - self.kd_pitch * q), -0.5, 0.5)
        rd = 0.3 * np.sin(phi)
        return np.array([thr, a, e, rd, flp, spl], dtype=np.float32)

    def _descent(self, alt, spd, clmb, phi, theta, psi, p, q, r, tgt_hdg):
        thr = np.clip(0.4 + self.kp_speed * (self.v_approach - spd), 0.2, 0.6)
        pitch = np.clip(self.pitch_descent + self.kd_altitude * (-self.descent_rate - clmb) * 10, -0.14, 0.087)
        e = np.clip(-(self.kp_pitch * (pitch - theta) - self.kd_pitch * q), -0.4, 0.5)
        hdg_err = self._wrap_angle(tgt_hdg - psi)
        bank = np.clip(hdg_err * 0.5, -0.17, 0.17)
        a = np.clip(self.kp_roll * (bank - phi) - self.kd_roll * p, -0.3, 0.3)
        rd = 0.3 * np.sin(phi)
        return np.array([thr, a, e, rd, 0.3, 0.4], dtype=np.float32)

    def _approach(self, alt, spd, phi, theta, psi, p, q, r, y=0.0):
        thr = np.clip(0.35 + self.kp_speed * (self.v_approach - spd) * 0.5, 0.15, 0.5)
        pitch = -0.14 if spd < self.v_approach * 0.9 else self.pitch_approach
        e = np.clip(-(self.kp_pitch * (pitch - theta) - self.kd_pitch * q), -0.5, 0.5)
        # Localizer tracking: steer toward runway centerline (Y=0)
        # Add lateral offset correction to heading - stronger correction for larger offsets
        localizer_correction = np.clip(-y * 0.005, -0.26, 0.26)  # ~15 deg max correction
        hdg_err = self._wrap_angle(self.landing_heading + localizer_correction - psi)
        bank = np.clip(hdg_err * 0.8, -0.26, 0.26)  # Up to 15 deg bank
        a = np.clip(self.kp_roll * (bank - phi) - self.kd_roll * p, -0.4, 0.4)
        rd = 0.3 * np.sin(phi)
        return np.array([thr, a, e, rd, 0.5, 0.2], dtype=np.float32)

    def _landing(self, alt, phi, theta, psi, p, q, r, y=0.0):
        flare = np.clip(1 - alt / self.flare_altitude, 0, 1)
        pitch = self.pitch_approach + flare * (self.pitch_flare - self.pitch_approach)
        e = np.clip(-(self.kp_pitch * (pitch - theta) - self.kd_pitch * q), -0.6, 0.3)
        # Localizer tracking during landing flare
        localizer_correction = np.clip(-y * 0.001, -0.1, 0.1)  # Gentler correction during flare
        hdg_target = self.landing_heading + localizer_correction
        hdg_err = self._wrap_angle(hdg_target - psi)
        bank = np.clip(hdg_err * 0.3, -0.1, 0.1)  # Very gentle bank in flare
        a = np.clip(self.kp_roll * (bank - phi) - self.kd_roll * p, -0.2, 0.2)
        rd = self._heading_hold(psi, r, hdg_target) * 0.6
        return np.array([0, a, e, rd, 0.5, 0.5], dtype=np.float32)

    def get_phase_name(self):
        return self.phase.name
