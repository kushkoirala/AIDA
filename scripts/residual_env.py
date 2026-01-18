#!/usr/bin/env python3
"""
Residual RL Environment for AIDA

The NN learns small corrections to the expert controller:
    u_total = u_expert + λ * tanh(π_θ(s))

Where λ limits the NN's authority (e.g., 0.1 = ±10% correction)
"""

import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from flight_dynamics import FlightSimulator, StateIndex
from triangle_controller import TriangleInterceptController, XCPhase


class ResidualFlightEnv(gym.Env):
    """
    Residual RL Environment

    The expert controller handles the base control.
    The NN learns residual corrections within bounds [-λ, +λ].

    If the NN outputs zeros, the aircraft flies on pure expert control.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dt: float = 0.02,
        max_episode_steps: int = 30000,  # 10 minutes
        residual_scale: float = 0.15,     # λ: NN can adjust ±15%
        cruise_altitude_ft: float = 5500.0,
        use_gpu: bool = True,  # Use GPU by default
    ):
        super().__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.residual_scale = residual_scale
        self.cruise_altitude_ft = cruise_altitude_ft
        self.cruise_altitude_m = cruise_altitude_ft * 0.3048

        # Flight simulator
        self.sim = FlightSimulator(n_instances=1, dt=dt, use_gpu=use_gpu)

        # Expert controller
        self.expert = TriangleInterceptController(cruise_altitude_ft=cruise_altitude_ft)

        # Action space: residual corrections [-1, 1] scaled by λ
        # [throttle_residual, aileron_residual, elevator_residual, rudder_residual]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space: 12 state + 4 expert action + 3 target info
        # State: [x, y, z, u, v, w, phi, theta, psi, p, q, r]
        # Expert: [throttle, aileron, elevator, rudder]
        # Target: [alt_error, heading_error, dist_to_target]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )

        # Episode tracking
        self.step_count = 0
        self.sim_time = 0.0
        self.state = None
        self.prev_dist_to_khut = None

        # KHUT position (destination)
        self.khut_x = 113000.0  # meters
        self.khut_y = 1500.0

    def _get_obs(self, state: np.ndarray, expert_action: np.ndarray) -> np.ndarray:
        """Build observation: state + expert action + target info"""

        # Current state values
        x, y, z = state[0], state[1], state[2]
        psi = state[8]
        alt = -z

        # Target info
        alt_error = (self.cruise_altitude_m - alt) / 1000.0  # Normalized

        # Bearing to KHUT
        dx = self.khut_x - x
        dy = self.khut_y - y
        bearing = np.arctan2(dy, dx)
        heading_error = self._normalize_angle(bearing - psi)

        # Distance to KHUT
        dist = np.sqrt(dx**2 + dy**2) / 10000.0  # Normalized (10km scale)

        # Combine observation
        obs = np.concatenate([
            state.astype(np.float32),           # 12 dims
            expert_action[:4].astype(np.float32),  # 4 dims
            np.array([alt_error, heading_error, dist], dtype=np.float32)  # 3 dims
        ])

        return obs

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _compute_reward(self, state: np.ndarray, action: np.ndarray, expert_action: np.ndarray) -> tuple:
        """
        Reward function focusing on:
        1. Progress toward destination
        2. Altitude tracking
        3. Smooth control (small residuals preferred)
        4. Safety (within envelope)
        """
        x, y, z = state[0], state[1], state[2]
        u, v, w = state[3], state[4], state[5]
        phi, theta, psi = state[6], state[7], state[8]

        alt = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        airspeed_kts = airspeed * 1.944

        # Distance to KHUT
        dist_to_khut = np.sqrt((x - self.khut_x)**2 + (y - self.khut_y)**2)

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # 1. Progress reward (approaching destination)
        if self.prev_dist_to_khut is not None:
            progress = self.prev_dist_to_khut - dist_to_khut
            reward += progress * 0.01  # Small positive for approaching
        self.prev_dist_to_khut = dist_to_khut

        # 2. Altitude tracking (during cruise)
        if self.expert.phase in [XCPhase.CRUISE_TO_TP, XCPhase.CLIMB]:
            alt_error = abs(alt - self.cruise_altitude_m)
            if alt_error < 50:
                reward += 0.5  # Good altitude hold
            elif alt_error < 200:
                reward += 0.2
            else:
                reward -= alt_error / 1000.0  # Penalty for large error

        # 3. Smooth control (prefer small residuals)
        residual_magnitude = np.sqrt(np.sum(action**2))
        reward -= 0.1 * residual_magnitude  # Encourage minimal corrections

        # Soft penalties for approaching envelope limits (encourages staying centered)
        # Only apply after takeoff
        if self.expert.phase not in [XCPhase.GROUND_ROLL, XCPhase.ROTATION]:
            # Airspeed: prefer 60-140 kts (Vy range)
            if airspeed_kts < 55 or airspeed_kts > 140:
                reward -= 1.0  # Mild penalty near limits

            # Bank: prefer < 30°
            if abs(phi) > np.deg2rad(30):
                reward -= 0.5

            # Pitch: prefer ±15°
            if abs(theta) > np.deg2rad(15):
                reward -= 0.5

        # =================================================================
        # HARD ENVELOPE LIMITS - Terminate episode if exceeded
        # These are non-negotiable safety boundaries
        # =================================================================

        # Altitude limits
        if alt < -10:  # Underground
            terminated = True
            reward -= 100.0
            info['termination'] = 'crashed'
        elif alt > 3000:  # 10,000 ft ceiling
            terminated = True
            reward -= 50.0
            info['termination'] = 'too_high'

        # Airspeed limits
        elif airspeed_kts < 35 and alt > 50:  # Stall (after takeoff)
            terminated = True
            reward -= 100.0
            info['termination'] = 'stalled'
        elif airspeed_kts > 180:  # Vne exceeded
            terminated = True
            reward -= 50.0
            info['termination'] = 'overspeed'

        # Attitude limits
        elif abs(phi) > np.deg2rad(60):  # Max bank 60°
            terminated = True
            reward -= 50.0
            info['termination'] = 'excessive_bank'
        elif theta > np.deg2rad(45) or theta < np.deg2rad(-30):  # Pitch limits
            terminated = True
            reward -= 50.0
            info['termination'] = 'excessive_pitch'

        # Success: Arrived at destination
        elif dist_to_khut < 2000:  # Within 2km of KHUT
            terminated = True
            reward += 500.0
            info['termination'] = 'arrived'

        # Truncation
        if self.step_count >= self.max_episode_steps:
            truncated = True
            info['termination'] = 'max_steps'

        info['phase'] = self.expert.phase.name
        info['altitude'] = alt
        info['airspeed_kts'] = airspeed_kts
        info['dist_to_khut'] = dist_to_khut / 1852  # nm

        return reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Ground start position
        initial_state = np.zeros((1, 12), dtype=np.float32)
        initial_state[0, 0] = -400.0 + np.random.uniform(-50, 50)  # X
        initial_state[0, 3] = 5.0 + np.random.uniform(-1, 1)       # U (taxi speed)

        self.sim.reset(initial_state)
        self.expert.reset()

        self.step_count = 0
        self.sim_time = 0.0
        self.state = self.sim.get_states()[0]
        self.prev_dist_to_khut = None

        # Get expert action for initial observation
        expert_action = self.expert.compute_action(self.state, self.sim_time)
        obs = self._get_obs(self.state, expert_action)

        info = {'phase': self.expert.phase.name}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Apply residual action on top of expert.

        u_total = u_expert + λ * action
        """
        self.step_count += 1
        self.sim_time += self.dt

        # Get expert control
        expert_action = self.expert.compute_action(self.state, self.sim_time)

        # Combine: expert + scaled residual
        # Throttle: expert[0] + λ * action[0], clipped to [0, 1]
        # Surfaces: expert[1:4] + λ * action[1:4], clipped to [-1, 1]
        throttle = np.clip(
            expert_action[0] + self.residual_scale * action[0],
            0.0, 1.0
        )
        aileron = np.clip(
            expert_action[1] + self.residual_scale * action[1],
            -1.0, 1.0
        )
        elevator = np.clip(
            expert_action[2] + self.residual_scale * action[2],
            -1.0, 1.0
        )
        rudder = np.clip(
            expert_action[3] + self.residual_scale * action[3],
            -1.0, 1.0
        )

        # Apply combined control
        controls = np.array([[throttle, aileron, elevator, rudder]], dtype=np.float32)
        self.sim.set_controls(controls)
        self.sim.step()

        # Get new state
        self.state = self.sim.get_states()[0]

        # Compute reward
        reward, terminated, truncated, info = self._compute_reward(
            self.state, action, expert_action
        )

        # Build observation
        obs = self._get_obs(self.state, expert_action)

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


# Register with Gymnasium
gym.register(
    id='ResidualFlight-v0',
    entry_point='residual_env:ResidualFlightEnv',
    max_episode_steps=30000,
)


if __name__ == "__main__":
    # Quick test
    print("Testing ResidualFlightEnv...")
    env = ResidualFlightEnv()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial phase: {info['phase']}")

    # Run a few steps with zero residual (pure expert)
    total_reward = 0
    for i in range(500):
        action = np.zeros(4)  # Zero residual = pure expert
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 100 == 0:
            print(f"Step {i}: phase={info['phase']}, alt={info['altitude']:.0f}m, "
                  f"ias={info['airspeed_kts']:.0f}kt, reward={reward:.2f}")

        if terminated or truncated:
            print(f"Episode ended: {info.get('termination', 'unknown')}")
            break

    print(f"\nTotal reward (pure expert): {total_reward:.1f}")
    env.close()
