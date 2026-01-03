#!/usr/bin/env python3
import numpy as np
import sys
from pathlib import Path
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))
from aida_sim.env.flight_env_cessna172 import StateIndex

class TakeoffPhase(Enum):
    GROUND_ROLL = 1
    ROTATION = 2
    INITIAL_CLIMB = 3
    CLIMB = 4

class TakeoffController:
    def __init__(self, v_rotate=28.0, runway_heading=0.0):
        self.v_rotate = v_rotate
        self.runway_heading = runway_heading
        self.phase = TakeoffPhase.GROUND_ROLL
        self.kp_heading = 0.5
        self.kd_heading = 0.2
        self.kp_roll = 1.0
        self.kd_roll = 0.3
        self.kp_pitch = 1.5
        self.kd_pitch = 0.5
        self.pitch_rotate = np.deg2rad(10.0)
        self.pitch_climb = np.deg2rad(8.0)
        self.elevator_neutral = 0.0
        self.liftoff_alt = 2.0
        self.climb_alt = 30.0

    def reset(self):
        self.phase = TakeoffPhase.GROUND_ROLL

    def compute_action(self, obs):
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi = obs[StateIndex.PHI]
        theta = obs[StateIndex.THETA]
        psi = obs[StateIndex.PSI]
        p, q, r = obs[StateIndex.P], obs[StateIndex.Q], obs[StateIndex.R]
        z = obs[StateIndex.Z]
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        altitude = -z
        self._update_phase(airspeed, altitude)
        if self.phase == TakeoffPhase.GROUND_ROLL:
            return self._ground_roll_control(airspeed, phi, psi, p, r)
        elif self.phase == TakeoffPhase.ROTATION:
            return self._rotation_control(phi, theta, psi, p, q, r)
        elif self.phase == TakeoffPhase.INITIAL_CLIMB:
            return self._initial_climb_control(phi, theta, psi, p, q, r)
        else:
            return self._climb_control(phi, theta, psi, p, q, r)

    def _update_phase(self, airspeed, altitude):
        if self.phase == TakeoffPhase.GROUND_ROLL and airspeed >= self.v_rotate:
            self.phase = TakeoffPhase.ROTATION
        elif self.phase == TakeoffPhase.ROTATION and altitude > self.liftoff_alt:
            self.phase = TakeoffPhase.INITIAL_CLIMB
        elif self.phase == TakeoffPhase.INITIAL_CLIMB and altitude > self.climb_alt:
            self.phase = TakeoffPhase.CLIMB

    def _ground_roll_control(self, airspeed, phi, psi, p, r):
        throttle = 1.0
        if airspeed < 0.85 * self.v_rotate:
            elevator = self.elevator_neutral
        else:
            # Slight nose-up preparation (negative elevator = stick back)
            blend = (airspeed - 0.85 * self.v_rotate) / (0.15 * self.v_rotate)
            elevator = -blend * 0.2
        aileron = self._wings_level(phi, p)
        rudder = self._heading_hold(psi, r)
        return np.array([throttle, aileron, elevator, rudder], dtype=np.float32)

    def _rotation_control(self, phi, theta, psi, p, q, r):
        throttle = 1.0
        pitch_error = self.pitch_rotate - theta
        # Negative elevator = stick back = nose up (standard convention)
        elevator = -(self.kp_pitch * pitch_error - self.kd_pitch * q)
        elevator = np.clip(elevator, -0.8, 0.5)
        aileron = self._wings_level(phi, p)
        rudder = self._heading_hold(psi, r) * 0.7
        return np.array([throttle, aileron, elevator, rudder], dtype=np.float32)

    def _initial_climb_control(self, phi, theta, psi, p, q, r):
        throttle = 1.0
        pitch_error = self.pitch_climb - theta
        elevator = -(self.kp_pitch * pitch_error - self.kd_pitch * q)
        elevator = np.clip(elevator, -0.6, 0.5)
        aileron = self._wings_level(phi, p)
        rudder = self._heading_hold(psi, r) * 0.5
        return np.array([throttle, aileron, elevator, rudder], dtype=np.float32)

    def _climb_control(self, phi, theta, psi, p, q, r):
        throttle = 1.0
        pitch_error = self.pitch_climb - theta
        elevator = -(self.kp_pitch * pitch_error - self.kd_pitch * q)
        elevator = np.clip(elevator, -0.5, 0.4)
        aileron = self._wings_level(phi, p)
        rudder = self._heading_hold(psi, r) * 0.3
        return np.array([throttle, aileron, elevator, rudder], dtype=np.float32)

    def _wings_level(self, phi, p):
        roll_error = 0.0 - phi
        aileron = self.kp_roll * roll_error - self.kd_roll * p
        return np.clip(aileron, -1.0, 1.0)

    def _heading_hold(self, psi, r):
        heading_error = self._wrap_angle(self.runway_heading - psi)
        rudder = self.kp_heading * heading_error - self.kd_heading * r
        return np.clip(rudder, -1.0, 1.0)

    def _wrap_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def get_phase_name(self):
        return self.phase.name

