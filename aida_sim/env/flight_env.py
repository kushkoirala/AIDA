import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aida_sim.dynamics.state import VehicleState
from aida_sim.dynamics.forces import aero_forces_moments, thrust_force
from aida_sim.dynamics.integrator import integrate_step, _quat_to_rotmat
from aida_sim.safety.guards import clamp_actions, enforce_limits


# Gymnasium-compatible environment for the fixed-wing simulator.
# Placeholder physics; swap in calibrated models as they become available.
class FlightEnv(gym.Env):
    metadata = {"render_modes": ["none"], "render_fps": 50}

    def __init__(self, dt: float = 0.02):
        super().__init__()
        self.dt = dt
        self.mass = 3.4  # kg (~7.5 lb)
        self.inertia_diag = np.array([0.25, 0.35, 0.45], dtype=np.float32)  # rough placeholder
        self.wing_area = 0.5  # m^2 placeholder
        self.rho = 1.225
        self.geofence = np.array([122.0, 30.0, 30.0], dtype=np.float32)  # meters (400x100x100 ft approx)

        # Action: throttle [0,1], elevator/aileron/rudder in [-1,1] mapped to deflection limits.
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Observation: [pos(3), vel(3), quat(4), body_rates(3), surfaces(3), soc, voltage, load_factor]
        obs_dim = 3 + 3 + 4 + 3 + 3 + 1 + 1 + 1
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dim, dtype=np.float32),
            high=np.inf * np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.state = None
        self.surface_limits = np.deg2rad(np.array([20.0, 20.0, 20.0], dtype=np.float32))  # elevator, aileron, rudder

    def _obs(self, vs: VehicleState) -> np.ndarray:
        return vs.as_vector()

    def _reward(self, vs: VehicleState) -> float:
        # Simple shaping: reward staying within bounds and modest attitude.
        pos_penalty = 0.001 * np.linalg.norm(vs.position)
        attitude_penalty = 0.001 * np.linalg.norm(vs.body_rates)
        load_penalty = 0.01 * max(0.0, vs.load_factor - 1.0)
        return float(1.0 - pos_penalty - attitude_penalty - load_penalty)

    def _terminated(self, vs: VehicleState):
        out_of_bounds = np.any(np.abs(vs.position) > self.geofence)
        below_ground = vs.position[2] < 0.0
        return out_of_bounds or below_ground

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # On-ground spawn with small perturbations (world frame: X forward, Y right, Z up).
        pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        vel = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # level
        rates = np.zeros(3, dtype=np.float32)
        surfaces = np.zeros(3, dtype=np.float32)
        self.state = VehicleState(position=pos, velocity=vel, orientation=orientation, body_rates=rates,
                                  surfaces=surfaces, soc=1.0, voltage=12.0, load_factor=1.0)
        info = {"termination_reason": None}
        return self._obs(self.state), info

    def step(self, action):
        if self.state is None:
            obs, info = self.reset()
            return obs, 0.0, False, False, info

        action = np.asarray(action, dtype=np.float32)
        throttle_raw = float(action[0])
        surfaces_raw = action[1:4]
        throttle, surfaces_cmd = clamp_actions(throttle_raw, surfaces_raw, self.surface_limits)

        # Map to physical deflections
        elevator, aileron, rudder = surfaces_cmd
        surfaces = np.array([elevator, aileron, rudder], dtype=np.float32)

        # Body-frame velocity from world velocity
        R_bw = _quat_to_rotmat(self.state.orientation)
        vel_body = R_bw.T @ self.state.velocity
        airspeed = max(1e-2, float(np.linalg.norm(vel_body)))
        alpha = float(np.arctan2(-vel_body[2], vel_body[0]))  # Z up convention
        beta = float(np.arctan2(vel_body[1], max(1e-3, np.linalg.norm(vel_body))))

        q_dyn = 0.5 * self.rho * airspeed * airspeed
        aero_forces, aero_moments = aero_forces_moments(alpha, beta, self.state.body_rates, q_dyn, self.wing_area)
        thrust = thrust_force(throttle, airspeed)

        gravity_world = np.array([0.0, 0.0, -9.81 * self.mass], dtype=np.float32)
        gravity_body = R_bw.T @ gravity_world

        total_forces_body = aero_forces + thrust + gravity_body
        total_moments_body = aero_moments  # control moments not modeled yet

        next_state = integrate_step(self.state, total_forces_body, total_moments_body,
                                    self.mass, self.inertia_diag, self.dt)
        next_state.surfaces = surfaces

        # Safety checks
        over_g, over_current = enforce_limits(next_state.load_factor, 0.0)
        terminated = self._terminated(next_state) or over_g
        truncated = False
        reward = self._reward(next_state)
        info = {"termination_reason": None}
        if over_g:
            info["termination_reason"] = "over_g"
        if self._terminated(next_state):
            info["termination_reason"] = "geofence"

        self.state = next_state
        return self._obs(next_state), reward, terminated, truncated, info
