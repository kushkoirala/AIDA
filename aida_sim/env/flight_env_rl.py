"""
RL-optimized Flight Environment with improved reward shaping and stability.
Designed for PPO training with Monte Carlo rollouts.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aida_sim.dynamics.state import VehicleState
from aida_sim.dynamics.forces import aero_forces_moments, thrust_force
from aida_sim.dynamics.integrator import integrate_step, _quat_to_rotmat


class FlightEnvRL(gym.Env):
    """
    RL-friendly flight environment with:
    - Normalized observations
    - Dense reward shaping
    - Configurable task modes (hover, waypoint, landing)
    - Stable physics for training
    """
    metadata = {"render_modes": ["none"], "render_fps": 50}

    def __init__(self, 
                 dt: float = 0.02,
                 max_episode_steps: int = 1500,  # 30 seconds
                 task: str = "takeoff_and_fly",
                 target_position: np.ndarray = None):
        super().__init__()
        
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.task = task
        self.step_count = 0
        
        # Aircraft parameters
        self.mass = 3.4  # kg
        self.inertia_diag = np.array([0.25, 0.35, 0.45], dtype=np.float32)
        self.wing_area = 0.5  # m² (fixed)
        self.rho = 1.225
        
        # Flight envelope constraints (from analysis)
        self.V_STALL = 11.0  # m/s
        self.V_MAX = 25.0    # m/s (safe operating speed)
        self.MAX_G = 6.0     # More permissive for training
        self.MAX_BANK = np.deg2rad(45)
        self.MAX_PITCH = np.deg2rad(30)
        
        # Geofence - parking lot
        self.geofence = np.array([150.0, 60.0, 50.0], dtype=np.float32)  # Half-size for center origin
        
        # Target zone (10ft x 10ft = 3m x 3m)
        self.target_position = target_position if target_position is not None else np.array([50.0, 30.0, 15.0])
        self.target_radius = 5.0  # meters
        
        # Action space: [throttle, elevator, aileron, rudder] all in [-1, 1]
        # Throttle remapped from [-1,1] to [0.2, 0.8] for safe range
        self.action_space = spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        
        # Normalized observation space
        # [pos_norm(3), vel_norm(3), attitude(3), rates_norm(3), 
        #  airspeed_norm, altitude_norm, distance_to_target, heading_to_target]
        obs_dim = 16
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32) * 10,
            high=np.ones(obs_dim, dtype=np.float32) * 10,
            dtype=np.float32,
        )

        self.state = None
        self.surface_limits = np.deg2rad(np.array([15.0, 15.0, 15.0], dtype=np.float32))

    def _get_attitude(self, quat):
        """Extract roll, pitch, yaw from quaternion."""
        qw, qx, qy, qz = quat
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
        pitch = np.arcsin(sinp)
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _obs(self, vs: VehicleState) -> np.ndarray:
        """Create normalized observation vector."""
        # Position normalized by geofence
        pos_norm = vs.position / self.geofence
        
        # Velocity normalized by max speed
        vel_norm = vs.velocity / self.V_MAX
        
        # Attitude (roll, pitch, yaw)
        attitude = self._get_attitude(vs.orientation)
        
        # Body rates normalized
        rates_norm = vs.body_rates / np.deg2rad(60)
        
        # Airspeed normalized
        airspeed = np.linalg.norm(vs.velocity)
        airspeed_norm = (airspeed - self.V_STALL) / (self.V_MAX - self.V_STALL)
        
        # Altitude normalized
        altitude_norm = vs.position[2] / self.geofence[2]
        
        # Distance and heading to target
        to_target = self.target_position - vs.position
        distance = np.linalg.norm(to_target)
        distance_norm = distance / 100.0  # Normalize by ~max distance
        
        # Heading to target (relative to current heading)
        yaw = attitude[2]
        target_heading = np.arctan2(to_target[1], to_target[0])
        heading_error = target_heading - yaw
        # Wrap to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        heading_norm = heading_error / np.pi
        
        obs = np.concatenate([
            pos_norm,           # 3
            vel_norm,           # 3
            attitude,           # 3
            rates_norm,         # 3
            [airspeed_norm],    # 1
            [altitude_norm],    # 1
            [distance_norm],    # 1
            [heading_norm],     # 1
        ]).astype(np.float32)
        
        return obs

    def _reward(self, vs: VehicleState, action: np.ndarray, terminated: bool, info: dict) -> float:
        """
        Dense reward function for RL training.
        """
        reward = 0.0
        
        airspeed = np.linalg.norm(vs.velocity)
        altitude = vs.position[2]
        attitude = self._get_attitude(vs.orientation)
        roll, pitch, yaw = attitude
        
        # === SURVIVAL REWARD ===
        reward += 0.1  # Small reward for each step survived
        
        # === AIRSPEED REWARD ===
        # Encourage staying in safe speed envelope
        target_speed = 18.0  # m/s - comfortable cruise
        speed_error = abs(airspeed - target_speed)
        reward -= 0.05 * speed_error
        
        # Penalize being too slow (stall risk) or too fast (G risk)
        if airspeed < self.V_STALL:
            reward -= 0.5  # Stall penalty
        if airspeed > self.V_MAX:
            reward -= 0.3  # Overspeed penalty
        
        # === ALTITUDE REWARD ===
        target_altitude = 15.0  # meters
        alt_error = abs(altitude - target_altitude)
        reward -= 0.02 * alt_error
        
        # Big penalty for ground proximity
        if altitude < 2.0:
            reward -= 0.5
        
        # === ATTITUDE REWARD ===
        # Penalize excessive bank and pitch
        reward -= 0.1 * abs(roll)
        reward -= 0.1 * abs(pitch)
        
        # === SMOOTHNESS REWARD ===
        # Penalize large control inputs (encourage smooth flying)
        control_mag = np.linalg.norm(action[1:])  # elevator, aileron, rudder
        reward -= 0.05 * control_mag
        
        # === G-LOAD PENALTY ===
        if vs.load_factor > 3.0:
            reward -= 0.2 * (vs.load_factor - 3.0)
        
        # === TARGET APPROACH REWARD ===
        to_target = self.target_position - vs.position
        distance = np.linalg.norm(to_target)
        reward += 0.01 * (100.0 - distance)  # Closer is better
        
        # Bonus for reaching target area
        if distance < self.target_radius:
            reward += 5.0
        
        # === TERMINATION PENALTIES ===
        if terminated:
            reason = info.get("termination_reason", "")
            if reason == "over_g":
                reward -= 10.0
            elif reason == "geofence":
                reward -= 5.0
            elif reason == "crash":
                reward -= 15.0
            elif reason == "stall":
                reward -= 8.0
        
        return float(reward)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # Start airborne at a safe altitude and speed for easier learning
        # Random position within safe zone
        if self.np_random is not None:
            x = self.np_random.uniform(-50, 50)
            y = self.np_random.uniform(-30, 30)
            z = self.np_random.uniform(10, 25)
            heading = self.np_random.uniform(-np.pi, np.pi)
            speed = self.np_random.uniform(14, 20)
        else:
            x, y, z = 0, 0, 15
            heading = 0
            speed = 16
        
        pos = np.array([x, y, z], dtype=np.float32)
        
        # Velocity in heading direction
        vel = np.array([
            speed * np.cos(heading),
            speed * np.sin(heading),
            0.0
        ], dtype=np.float32)
        
        # Level orientation with given heading
        # Quaternion for yaw rotation: [cos(yaw/2), 0, 0, sin(yaw/2)]
        orientation = np.array([
            np.cos(heading/2), 
            0.0, 
            0.0, 
            np.sin(heading/2)
        ], dtype=np.float32)
        
        rates = np.zeros(3, dtype=np.float32)
        surfaces = np.zeros(3, dtype=np.float32)
        
        self.state = VehicleState(
            position=pos, velocity=vel, orientation=orientation, 
            body_rates=rates, surfaces=surfaces, 
            soc=1.0, voltage=12.0, load_factor=1.0
        )
        
        info = {"termination_reason": None}
        return self._obs(self.state), info

    def step(self, action):
        if self.state is None:
            return self.reset()
        
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)
        
        # Map actions to control inputs
        # Throttle: [-1,1] -> [0.25, 0.65] (safe range from envelope analysis)
        throttle = 0.45 + 0.2 * action[0]
        
        # Control surfaces: [-1,1] -> deflection angles
        elevator = action[1] * self.surface_limits[0]
        aileron = action[2] * self.surface_limits[1]
        rudder = action[3] * self.surface_limits[2]
        
        surfaces = np.array([elevator, aileron, rudder], dtype=np.float32)

        # Physics calculations
        R_bw = _quat_to_rotmat(self.state.orientation)
        vel_body = R_bw.T @ self.state.velocity
        airspeed = max(1e-2, float(np.linalg.norm(vel_body)))
        
        alpha = float(np.arctan2(vel_body[2], max(1e-2, vel_body[0])))
        beta = float(np.arctan2(vel_body[1], max(1e-2, np.linalg.norm(vel_body))))

        q_dyn = 0.5 * self.rho * airspeed * airspeed
        aero_forces, aero_moments = aero_forces_moments(
            alpha, beta, self.state.body_rates, q_dyn, self.wing_area
        )
        thrust = thrust_force(throttle, airspeed)

        gravity_world = np.array([0.0, 0.0, -9.81 * self.mass], dtype=np.float32)
        gravity_body = R_bw.T @ gravity_world

        total_forces_body = aero_forces + thrust + gravity_body
        
        # Control moments with moderate effectiveness
        control_effectiveness = 0.12
        control_moments = np.array([
            q_dyn * self.wing_area * control_effectiveness * aileron,
            q_dyn * self.wing_area * control_effectiveness * elevator,
            q_dyn * self.wing_area * control_effectiveness * rudder
        ], dtype=np.float32)
        total_moments_body = aero_moments + control_moments

        next_state = integrate_step(
            self.state, total_forces_body, total_moments_body,
            self.mass, self.inertia_diag, self.dt
        )
        
        # Rate limiting for stability
        MAX_RATE = np.deg2rad(90)
        next_state.body_rates = np.clip(next_state.body_rates, -MAX_RATE, MAX_RATE)
        
        # Calculate load factor
        non_grav_forces = aero_forces + thrust
        next_state.load_factor = float(np.linalg.norm(non_grav_forces) / (self.mass * 9.81))
        next_state.surfaces = surfaces

        # Check termination conditions
        terminated = False
        truncated = False
        info = {"termination_reason": None}
        
        # Ground crash
        if next_state.position[2] < 0.5:
            terminated = True
            info["termination_reason"] = "crash"
        
        # Geofence
        if np.any(np.abs(next_state.position) > self.geofence):
            terminated = True
            info["termination_reason"] = "geofence"
        
        # Over-G
        if next_state.load_factor > self.MAX_G:
            terminated = True
            info["termination_reason"] = "over_g"
        
        # Stall (too slow)
        new_airspeed = np.linalg.norm(next_state.velocity)
        if new_airspeed < self.V_STALL * 0.8:
            terminated = True
            info["termination_reason"] = "stall"
        
        # Episode timeout
        if self.step_count >= self.max_episode_steps:
            truncated = True
        
        reward = self._reward(next_state, action, terminated, info)
        self.state = next_state
        
        return self._obs(next_state), reward, terminated, truncated, info


# Register the environment
gym.register(
    id='AIDA-Flight-RL-v0',
    entry_point='aida_sim.env.flight_env_rl:FlightEnvRL',
    max_episode_steps=1500,
)
