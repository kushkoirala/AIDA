import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aida_sim.dynamics.state import VehicleState
from aida_sim.dynamics.forces import aero_forces_moments, propulsion_model, AeroParams
from aida_sim.dynamics.integrator import integrate_step, _quat_to_rotmat
from aida_sim.safety.guards import clamp_actions, enforce_limits
from aida_sim.systems.battery import Battery


# Gymnasium-compatible environment for the fixed-wing simulator.
# Placeholder physics; swap in calibrated models as they become available.
class FlightEnv(gym.Env):
    metadata = {"render_modes": ["none"], "render_fps": 50}

    def __init__(self, dt: float = 0.02):
        import json
        import os
        super().__init__()
        self.dt = dt
        self.mass = 3.63  # kg (~8 lb per spec)
        self.inertia_diag = np.array([0.25, 0.35, 0.45], dtype=np.float32)  # rough placeholder
        self.wing_area = 0.432  # m^2 (≈4.65 ft^2 from aero spec)
        self.rho = 1.225
        self.geofence = np.array([304.8, 121.9, 61.0], dtype=np.float32)  # 1000ft x 400ft x 200ft
        self.aero_params = AeroParams(wing_area=self.wing_area, wing_span=1.372, mean_chord=0.315)
        self.battery = Battery(capacity_ah=2.0, nominal_voltage=16.0, r_internal=0.06)

        # Load runway/property config from shared JSON
        config_path = os.path.join(os.path.dirname(__file__), '../../runway_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        r = config["runway"]
        self.runway_length = float(r["length"])
        self.runway_width = float(r["width"])
        self.runway_start = np.array(r["start"], dtype=np.float32)
        self.runway_heading = float(r["heading"])
        # Property config available as config["property"] if needed

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
        if out_of_bounds or below_ground:
            print(f"  Termination: pos={vs.position}, bounds={self.geofence}, below_ground={below_ground}")
        return out_of_bounds or below_ground

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Start ON THE GROUND at the beginning of the default runway, aligned for takeoff
        pos = self.runway_start.copy()
        vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Stationary
        # Orientation: quaternion for heading along X+ (East), level
        # For heading = 0, quaternion = [1, 0, 0, 0] (w, x, y, z)
        # For heading != 0, rotate about Z axis
        heading = self.runway_heading
        half_angle = heading / 2.0
        orientation = np.array([
            np.cos(half_angle),  # w
            0.0,                # x
            0.0,                # y
            np.sin(half_angle)  # z
        ], dtype=np.float32)
        rates = np.zeros(3, dtype=np.float32)
        surfaces = np.zeros(3, dtype=np.float32)
        # Fresh battery every reset
        self.battery = Battery(capacity_ah=2.0, nominal_voltage=16.0, r_internal=0.06)
        self.state = VehicleState(position=pos, velocity=vel, orientation=orientation, body_rates=rates,
                                  surfaces=surfaces, soc=self.battery.soc, voltage=self.battery.voltage,
                                  load_factor=1.0)
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

        aero_forces, aero_moments = aero_forces_moments(
            vel_body,
            self.state.body_rates,
            surfaces,
            self.rho,
            self.aero_params,
        )
        thrust, electrical_power = propulsion_model(throttle, airspeed, self.rho)

        gravity_world = np.array([0.0, 0.0, -9.81 * self.mass], dtype=np.float32)
        gravity_body = R_bw.T @ gravity_world

        total_forces_body = aero_forces + thrust + gravity_body
        total_moments_body = aero_moments

        next_state = integrate_step(self.state, total_forces_body, total_moments_body,
                                    self.mass, self.inertia_diag, self.dt)
        
        # Limit pitch rate to reasonable values (max ~60 deg/s for aggressive maneuvers)
        MAX_PITCH_RATE = np.deg2rad(60)
        next_state.body_rates[1] = np.clip(next_state.body_rates[1], -MAX_PITCH_RATE, MAX_PITCH_RATE)
        
        # GROUND CONTACT HANDLING
        GROUND_HEIGHT = 0.3  # gear height in meters
        weight = self.mass * 9.81
        
        # Calculate lift force from aero (Z component)
        lift_force = aero_forces[2]  # positive = up
        
        if next_state.position[2] < GROUND_HEIGHT:
            # On the ground
            if lift_force > weight * 1.1:
                # Enough lift to fly - allow liftoff!
                pass  # Let the integration result stand
            else:
                # Stay on ground
                next_state.position[2] = GROUND_HEIGHT
                # Zero vertical velocity
                next_state.velocity[2] = max(0.0, next_state.velocity[2])
                # Prevent pitching below level (no nose-down on ground)
                # But ALLOW nose-up pitch from elevator!
                # Constraint: don't roll on ground
                next_state.body_rates[0] = 0.0  # no roll rate
        
        if next_state.position[2] < 0.0:
            # Hard ground collision 
            next_state.position[2] = 0.0
        
        # Battery and electrical
        voltage_before = max(self.battery.voltage, 1.0)
        electrical_current = electrical_power / voltage_before if electrical_power > 0.0 else 0.0
        voltage_after = self.battery.step(electrical_current, self.dt)
        next_state.soc = self.battery.soc
        next_state.voltage = voltage_after

        # Correct load factor (felt Gs)
        non_grav_forces = aero_forces + thrust
        next_state.load_factor = float(np.linalg.norm(non_grav_forces) / (self.mass * 9.81))

        next_state.surfaces = surfaces

        # Safety checks
        over_g, over_current = enforce_limits(next_state.load_factor, 0.0)
        geo_term = self._terminated(next_state)
        terminated = geo_term or over_g
        truncated = False
        reward = self._reward(next_state)
        info = {"termination_reason": None}
        if over_g:
            info["termination_reason"] = "over_g"
            print(f"  Over-G: load_factor={next_state.load_factor:.2f}")
        if geo_term:
            info["termination_reason"] = "geofence"

        self.state = next_state
        return self._obs(next_state), reward, terminated, truncated, info
