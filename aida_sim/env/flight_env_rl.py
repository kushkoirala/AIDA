"""
RL-optimized Flight Environment with improved reward shaping and stability.
Designed for PPO training with Monte Carlo rollouts.
"""
import json
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aida_sim.dynamics.state import VehicleState
from aida_sim.dynamics.forces import aero_forces_moments, propulsion_model, AeroParams
from aida_sim.dynamics.integrator import integrate_step, _quat_to_rotmat
from aida_sim.systems.battery import Battery


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
        self.mass = 3.63  # kg (8 lb spec)
        self.inertia_diag = np.array([0.25, 0.35, 0.45], dtype=np.float32)
        self.wing_area = 0.480  # m² (5.17 ft²)
        self.rho = 1.225
        
        # Flight envelope constraints (from analysis)
        self.V_STALL = 11.0  # m/s
        self.V_MAX = 25.0    # m/s (safe operating speed)
        self.MAX_G = 6.0     # More permissive for training
        self.MAX_BANK = np.deg2rad(45)
        self.MAX_PITCH = np.deg2rad(30)
        
        # Geofence - parking lot
        self.geofence = np.array([500.0, 500.0, 150.0], dtype=np.float32)  # Half-size for center origin (≈1000x1000ft)
        self.aero_params = AeroParams(wing_area=self.wing_area, wing_span=1.372, mean_chord=0.432)
        self.battery = Battery(capacity_ah=2.0, nominal_voltage=16.0, r_internal=0.06)

        # Runway geometry for takeoff task
        config_path = os.path.join(os.path.dirname(__file__), "../../runway_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        r = config["runway"]
        self.runway_start = np.array(r["start"], dtype=np.float32)
        self.runway_heading = float(r["heading"])
        self.runway_length = float(r["length"])
        self.runway_width = float(r["width"])
        self.runway_dir = np.array(
            [np.cos(self.runway_heading), np.sin(self.runway_heading), 0.0],
            dtype=np.float32,
        )
        self.runway_right = np.array([-self.runway_dir[1], self.runway_dir[0], 0.0], dtype=np.float32)
        self.ground_height = 0.3
        self.takeoff_distance = 60.0  # meters along runway
        self.takeoff_altitude = 30.5  # ≈100 ft climb goal
        self.takeoff_target_speed = 12.0  # ≈40 ft/s
        self.rotation_pitch = np.deg2rad(10.0)
        self.roll_friction = 2.0   # N/(m/s) along runway
        self.side_friction = 8.0   # N/(m/s) lateral damping
        
        # Target zone (10ft x 10ft = 3m x 3m)
        self.target_position = target_position if target_position is not None else np.array([50.0, 30.0, 15.0])
        self.target_radius = 5.0  # meters
        
        # Action space: [throttle, trim_elev, trim_rudder, elevator_stick, aileron, rudder_stick]
        self.action_space = spaces.Box(
            low=-np.ones(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        
        # Normalized observation space
        # [pos_norm(3), vel_norm(3), attitude(3), rates_norm(3), 
        #  airspeed_norm, altitude_norm, distance_to_target, heading_to_target, trim_norm]
        obs_dim = 18
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32) * 10,
            high=np.ones(obs_dim, dtype=np.float32) * 10,
            dtype=np.float32,
        )

        self.state = None
        self.surface_limits = np.deg2rad(np.array([20.0, 20.0, 20.0], dtype=np.float32))
        self.trim_elevator = 0.0
        self.trim_rudder = 0.0
        self.trim_rudder = 0.0
        self.trim_rudder = 0.0
        self.trim_limit = np.deg2rad(10.0)
        self.trim_rate = np.deg2rad(1.0)

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
        
        trim_norm = self.trim_elevator / self.surface_limits[0]
        rudder_trim_norm = self.trim_rudder / self.surface_limits[2]
        obs = np.concatenate([
            pos_norm,           # 3
            vel_norm,           # 3
            attitude,           # 3
            rates_norm,         # 3
            [airspeed_norm],    # 1
            [altitude_norm],    # 1
            [distance_norm],    # 1
            [heading_norm],     # 1
            [trim_norm],        # 1
            [rudder_trim_norm], # 1
        ]).astype(np.float32)
        
        return obs

    def _reward(self, vs: VehicleState, action: np.ndarray, terminated: bool, info: dict) -> float:
        if self.task == "takeoff":
            return self._reward_takeoff(vs, action, terminated, info)
        return self._reward_cruise(vs, action, terminated, info)

    def _ground_reaction_forces(self, R_bw: np.ndarray):
        zero = np.zeros(3, dtype=np.float32)
        if self.task != "takeoff":
            return zero, False
        if self.state is None:
            return zero, False
        near_ground = self.state.position[2] <= self.ground_height + 0.05
        descending = self.state.velocity[2] <= 1.0
        if not (near_ground and descending):
            return zero, False

        normal_world = np.array([0.0, 0.0, self.mass * 9.81], dtype=np.float32)
        vel_world = self.state.velocity
        forward_speed = float(np.dot(vel_world, self.runway_dir))
        lateral_speed = float(np.dot(vel_world, self.runway_right))
        friction_world = (
            -self.roll_friction * forward_speed * self.runway_dir
            -self.side_friction * lateral_speed * self.runway_right
        )
        return R_bw.T @ (normal_world + friction_world), True

    def _reward_cruise(self, vs: VehicleState, action: np.ndarray, terminated: bool, info: dict) -> float:
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

    def _reward_takeoff(self, vs: VehicleState, action: np.ndarray, terminated: bool, info: dict) -> float:
        reward = 0.0
        rel_pos = vs.position - self.runway_start
        forward_progress = float(np.dot(rel_pos, self.runway_dir))
        lateral_offset = abs(np.dot(rel_pos, self.runway_right))
        forward_speed = float(np.dot(vs.velocity, self.runway_dir))
        lateral_speed = abs(np.dot(vs.velocity, self.runway_right))
        altitude = vs.position[2]
        roll, pitch, yaw = self._get_attitude(vs.orientation)
        yaw_error = np.arctan2(np.sin(yaw - self.runway_heading), np.cos(yaw - self.runway_heading))
        commanded_throttle = np.clip(0.5 * (action[0] + 1.0), 0.0, 1.0)
        takeoff_speed = 13.4
        throttle = 0.98 if (self.task == "takeoff" and forward_speed < takeoff_speed) else commanded_throttle

        # Strong incentive to stay aligned with runway centerline
        center_penalty = lateral_offset / max(1e-3, self.runway_width * 0.5)
        reward -= 0.8 * center_penalty
        reward -= 0.1 * lateral_speed
        reward -= 0.05 * abs(roll)
        reward -= 0.06 * abs(yaw_error)
        if lateral_offset < self.runway_width * 0.1 and abs(yaw_error) < np.deg2rad(3):
            reward += 0.5  # strong bonus for staying centered and aligned

        # Forward speed shaping toward 12 m/s (~40 ft/s)
        speed_ratio = np.clip(forward_speed / max(1e-3, self.takeoff_target_speed), 0.0, 1.5)
        reward += 0.6 * speed_ratio
        reward -= 0.03 * max(0.0, self.takeoff_target_speed - forward_speed) / self.takeoff_target_speed

        # Encourage throttle-up during ground roll
        reward += 0.2 * np.clip(throttle - 0.4, 0.0, 0.6)
        reward -= 0.02 * (abs(self.trim_elevator) / self.surface_limits[0] + abs(self.trim_rudder) / self.surface_limits[2])

        # Reward pitching up once acceleration is sufficient
        if forward_speed > 0.4 * self.takeoff_target_speed:
            pitch_error = pitch - self.rotation_pitch
            reward += 0.25 * np.exp(-0.5 * (pitch_error / np.deg2rad(5.0)) ** 2)

        # Altitude progress once rotation complete
        if altitude > self.ground_height + 0.5:
            reward += 0.1 * (altitude - self.ground_height)

        # Encourage meeting the 100 ft climb target near end of runway
        progress_ratio = np.clip(forward_progress / max(1e-3, self.takeoff_distance), 0.0, 1.5)
        climb_ratio = np.clip((altitude - self.ground_height) / max(1e-3, self.takeoff_altitude), 0.0, 1.5)
        reward += 0.4 * min(progress_ratio, climb_ratio)
        if forward_progress >= self.takeoff_distance * 0.9 and altitude < self.ground_height + self.takeoff_altitude:
            reward -= 4.0  # strong penalty for being too low by the fence

        # Encourage leaving the runway after 5 m of roll
        if altitude > self.ground_height + 1.0 and forward_progress > 5.0:
            reward += 0.5

        if terminated:
            reason = info.get("termination_reason", "")
            if reason == "takeoff_success":
                reward += 30.0
            elif reason == "runway_deviation":
                reward -= 8.0
            elif reason == "crash":
                reward -= 12.0
            elif reason == "stall":
                reward -= 6.0
            elif reason == "geofence":
                reward -= 5.0

        return float(reward)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        if self.task == "takeoff":
            return self._reset_takeoff()
        return self._reset_airborne()

    def _reset_airborne(self):
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
        vel = np.array(
            [
                speed * np.cos(heading),
                speed * np.sin(heading),
                0.0,
            ],
            dtype=np.float32,
        )
        orientation = np.array(
            [np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2)], dtype=np.float32
        )

        rates = np.zeros(3, dtype=np.float32)
        surfaces = np.zeros(3, dtype=np.float32)

        self.battery = Battery(capacity_ah=2.0, nominal_voltage=16.0, r_internal=0.06)
        self.state = VehicleState(
            position=pos,
            velocity=vel,
            orientation=orientation,
            body_rates=rates,
            surfaces=surfaces,
            soc=self.battery.soc,
            voltage=self.battery.voltage,
            load_factor=1.0,
        )
        self.trim_elevator = 0.0
        return self._obs(self.state), {"termination_reason": None}

    def _reset_takeoff(self):
        if self.np_random is not None:
            lateral = self.np_random.uniform(-self.runway_width * 0.1, self.runway_width * 0.1)
            heading_offset = self.np_random.uniform(-np.deg2rad(2), np.deg2rad(2))
        else:
            lateral = 0.0
            heading_offset = 0.0

        pos = self.runway_start.copy()
        pos += self.runway_right * lateral
        pos[2] = self.ground_height

        heading = self.runway_heading + heading_offset
        orientation = np.array(
            [np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2)], dtype=np.float32
        )
        vel = np.zeros(3, dtype=np.float32)
        rates = np.zeros(3, dtype=np.float32)
        surfaces = np.zeros(3, dtype=np.float32)

        self.battery = Battery(capacity_ah=2.0, nominal_voltage=16.0, r_internal=0.06)
        self.state = VehicleState(
            position=pos,
            velocity=vel,
            orientation=orientation,
            body_rates=rates,
            surfaces=surfaces,
            soc=self.battery.soc,
            voltage=self.battery.voltage,
            load_factor=1.0,
        )
        self.trim_elevator = 0.0
        return self._obs(self.state), {"termination_reason": None}

    def step(self, action):
        if self.state is None:
            return self.reset()
        
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)
        
        # Map actions to control inputs
        # Use 98% throttle until approach takeoff speed, then follow policy command
        commanded_throttle = 0.5 * (action[0] + 1.0)
        commanded_throttle = np.clip(commanded_throttle, 0.0, 1.0)
        forward_speed = float(np.dot(self.state.velocity, self.runway_dir))
        takeoff_speed = 13.4  # m/s (≈44 ft/s)
        if self.task == "takeoff" and forward_speed < takeoff_speed:
            throttle = 0.98
        else:
            throttle = commanded_throttle
        
        # Trim adjustment integrates slowly; stick deflections applied on top
        trim_delta = action[1] * self.trim_rate
        self.trim_elevator = np.clip(self.trim_elevator + trim_delta, -self.trim_limit, self.trim_limit)
        rudder_trim_delta = action[2] * self.trim_rate
        self.trim_rudder = np.clip(self.trim_rudder + rudder_trim_delta, -self.trim_limit, self.trim_limit)
        stick_elevator = action[3] * self.surface_limits[0]
        elevator = np.clip(self.trim_elevator + stick_elevator, -self.surface_limits[0], self.surface_limits[0])
        aileron = action[4] * self.surface_limits[1]
        stick_rudder = action[5] * self.surface_limits[2]
        rudder = np.clip(self.trim_rudder + stick_rudder, -self.surface_limits[2], self.surface_limits[2])
        
        surfaces = np.array([elevator, aileron, rudder], dtype=np.float32)

        # Physics calculations
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
        ground_forces_body, grounded = self._ground_reaction_forces(R_bw)

        total_forces_body = aero_forces + thrust + gravity_body + ground_forces_body
        total_moments_body = aero_moments

        next_state = integrate_step(
            self.state, total_forces_body, total_moments_body,
            self.mass, self.inertia_diag, self.dt
        )

        if self.task == "takeoff" and next_state.position[2] < self.ground_height:
            next_state.position[2] = self.ground_height
            next_state.velocity[2] = max(0.0, next_state.velocity[2])
            next_state.velocity[1] *= 0.8
        
        # Rate limiting for stability
        MAX_RATE = np.deg2rad(90)
        next_state.body_rates = np.clip(next_state.body_rates, -MAX_RATE, MAX_RATE)
        
        # Electrical / battery update
        voltage_before = max(self.battery.voltage, 1.0)
        electrical_current = electrical_power / voltage_before if electrical_power > 0.0 else 0.0
        voltage_after = self.battery.step(electrical_current, self.dt)
        next_state.soc = self.battery.soc
        next_state.voltage = voltage_after

        # Calculate load factor
        non_grav_forces = aero_forces + thrust
        next_state.load_factor = float(np.linalg.norm(non_grav_forces) / (self.mass * 9.81))
        next_state.surfaces = surfaces

        # Check termination conditions
        terminated = False
        truncated = False
        info = {"termination_reason": None}
        
        # Ground crash
        if self.task != "takeoff":
            if next_state.position[2] < 0.5:
                terminated = True
                info["termination_reason"] = "crash"
        else:
            if next_state.position[2] < 0.0:
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
        stall_threshold = self.V_STALL * 0.7
        stall_check = True
        if self.task == "takeoff":
            # Allow a long ground roll to build speed before considering stall
            forward_speed = float(np.dot(next_state.velocity, self.runway_dir))
            if forward_speed < self.takeoff_target_speed * 1.1 and next_state.position[2] <= self.ground_height + 2.0:
                stall_check = False
        if stall_check and new_airspeed < stall_threshold:
            terminated = True
            info["termination_reason"] = "stall"
        
        # Episode timeout
        if self.step_count >= self.max_episode_steps:
            truncated = True

        if self.task == "takeoff" and not terminated:
            rel_pos = next_state.position - self.runway_start
            forward_progress = float(np.dot(rel_pos, self.runway_dir))
            lateral_offset = abs(np.dot(rel_pos, self.runway_right))
            if lateral_offset > self.runway_width * 0.4:
                terminated = True
                info["termination_reason"] = "runway_deviation"
            elif forward_progress >= self.takeoff_distance and next_state.position[2] >= self.ground_height + self.takeoff_altitude:
                terminated = True
                info["termination_reason"] = "takeoff_success"
        
        reward = self._reward(next_state, action, terminated, info)
        self.state = next_state
        
        return self._obs(next_state), reward, terminated, truncated, info


# Register the environment
gym.register(
    id='AIDA-Flight-RL-v0',
    entry_point='aida_sim.env.flight_env_rl:FlightEnvRL',
    max_episode_steps=1500,
)
