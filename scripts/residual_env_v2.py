#!/usr/bin/env python3
"""
Residual RL Environment v2 for AIDA - Full 7-Control Training

Changes from v1:
- 7 controls instead of 4 (adds flaps, spoilers, brakes)
- GPU-accelerated parallel instances (1000+)
- Training on ALL flight phases (ground roll to landing)
- Phase-specific reward shaping
- Fixed KHUT coordinates

The NN learns small corrections to the expert controller:
    u_total = u_expert + lambda * tanh(pi_theta(s))

Where lambda limits the NN's authority (e.g., 0.15 = +/-15% correction)
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


class ResidualFlightEnvV2(gym.Env):
    """
    Residual RL Environment V2 - Full 7-Control Training

    Key improvements:
    - 7 action dimensions: [throttle, aileron, elevator, rudder, flaps, spoilers, brakes]
    - GPU-accelerated simulation (1000+ parallel instances)
    - Training on all phases including takeoff and landing
    - Phase-aware reward shaping
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dt: float = 0.02,
        max_episode_steps: int = 45000,  # 15 minutes (longer for full mission)
        residual_scale: float = 0.15,     # lambda: NN can adjust +/-15%
        cruise_altitude_ft: float = 5500.0,
        use_gpu: bool = True,
        n_instances: int = 1,  # Can be 1000+ for GPU training
    ):
        super().__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.residual_scale = residual_scale
        self.cruise_altitude_ft = cruise_altitude_ft
        self.cruise_altitude_m = cruise_altitude_ft * 0.3048
        self.n_instances = n_instances
        self.use_gpu = use_gpu

        # Flight simulator - supports parallel instances on GPU
        self.sim = FlightSimulator(n_instances=n_instances, dt=dt, use_gpu=use_gpu)

        # Expert controller (one per instance for multi-env training)
        self.experts = [TriangleInterceptController(cruise_altitude_ft=cruise_altitude_ft)
                        for _ in range(n_instances)]

        # Different residual scales per control type
        # Lower scales during ground ops - NN should learn to output ~0
        # since aircraft is symmetric and expert gives rudder=0
        self.residual_scales = np.array([
            0.15,  # throttle
            0.10,  # aileron - small, wings should stay level
            0.15,  # elevator
            0.10,  # rudder - small, should be ~0 during ground roll
            0.10,  # flaps
            0.15,  # spoilers (important for descent)
            0.10,  # brakes
        ], dtype=np.float32)

        # Action space: residual corrections [-1, 1] for all 7 controls
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Observation space: 12 state + 7 expert action + 3 target info + 3 phase encoding = 25
        # State: [x, y, z, u, v, w, phi, theta, psi, p, q, r]
        # Expert: [throttle, aileron, elevator, rudder, flaps, spoilers, brakes]
        # Target: [alt_error, heading_error, dist_to_target]
        # Phase: [is_takeoff, is_cruise, is_approach] one-hot
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )

        # Episode tracking
        self.step_count = 0
        self.sim_time = 0.0
        self.state = None
        self.prev_dist_to_khut = None

        # CORRECT KHUT position (destination) - fixed from v1
        self.khut_x = 52800.0  # meters (was wrong: 113000)
        self.khut_y = -21300.0  # meters (was wrong: 1500)

        # Runway 31 heading
        self.runway_heading = np.deg2rad(314.0)

        # Glideslope for descent rewards
        self.glideslope_deg = 3.5

    def _get_phase_encoding(self, phase: XCPhase) -> np.ndarray:
        """Encode flight phase as one-hot vector [takeoff, cruise, approach]"""
        if phase in [XCPhase.GROUND_ROLL, XCPhase.ROTATION, XCPhase.INITIAL_CLIMB]:
            return np.array([1, 0, 0], dtype=np.float32)  # Takeoff
        elif phase in [XCPhase.CLIMB, XCPhase.CRUISE_TO_TP]:
            return np.array([0, 1, 0], dtype=np.float32)  # Cruise
        else:
            return np.array([0, 0, 1], dtype=np.float32)  # Approach

    def _get_obs(self, state: np.ndarray, expert_action: np.ndarray, phase: XCPhase) -> np.ndarray:
        """Build observation: state + expert action (7) + target info + phase encoding"""

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

        # Phase encoding
        phase_enc = self._get_phase_encoding(phase)

        # Combine observation (25 dims total)
        obs = np.concatenate([
            state.astype(np.float32),                    # 12 dims
            expert_action[:7].astype(np.float32),        # 7 dims (all controls)
            np.array([alt_error, heading_error, dist], dtype=np.float32),  # 3 dims
            phase_enc                                     # 3 dims
        ])

        return obs

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _compute_reward(self, state: np.ndarray, action: np.ndarray, expert_action: np.ndarray, phase: XCPhase) -> tuple:
        """
        Phase-aware reward function:
        1. Progress toward destination (all phases)
        2. Phase-specific objectives
        3. Smooth control (small residuals preferred)
        4. Safety (within envelope)
        """
        x, y, z = state[0], state[1], state[2]
        u, v, w = state[3], state[4], state[5]
        phi, theta, psi = state[6], state[7], state[8]

        alt = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        airspeed_kts = airspeed * 1.944
        climb_rate = -w  # Positive = climbing (m/s)
        climb_rate_fpm = climb_rate * 196.85  # ft/min

        # Distance to KHUT
        dist_to_khut = np.sqrt((x - self.khut_x)**2 + (y - self.khut_y)**2)

        # Distance to threshold (for glideslope calculation)
        threshold_x = self.khut_x + 610.0 * np.cos(np.deg2rad(134.0))
        threshold_y = self.khut_y + 610.0 * np.sin(np.deg2rad(134.0))
        dist_to_threshold = np.sqrt((x - threshold_x)**2 + (y - threshold_y)**2)

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # =================================================================
        # 1. PROGRESS REWARD (all phases)
        # =================================================================
        if self.prev_dist_to_khut is not None:
            progress = self.prev_dist_to_khut - dist_to_khut
            reward += progress * 0.01
        self.prev_dist_to_khut = dist_to_khut

        # =================================================================
        # 2. PHASE-SPECIFIC REWARDS
        # =================================================================

        if phase == XCPhase.GROUND_ROLL:
            # CRITICAL: Y must be 0. Aircraft is symmetric, expert gives rudder=0.
            # NN should learn to output ~0 residuals and keep aircraft on centerline.
            centerline_error = abs(y)
            if centerline_error < 0.3:
                reward += 5.0  # Perfect - stay exactly on centerline
            else:
                # ANY deviation is penalized - quadratic growth, no safe zone
                reward -= centerline_error ** 2 * 1.0  # 0.5m=-0.25, 1m=-1, 2m=-4, 5m=-25

            # Wings level - must be zero
            roll_error = abs(phi)
            reward -= roll_error * 10.0

            # Heading - must stay aligned (runway heading ~4 deg at SN65)
            heading_error = abs(self._normalize_angle(psi - np.deg2rad(4.0)))
            reward -= heading_error ** 2 * 5.0  # Quadratic penalty

            # Accelerating is good but secondary
            if u > 10:
                reward += 0.1

        elif phase == XCPhase.ROTATION:
            # Same - Y must stay 0 during rotation
            centerline_error = abs(y)
            if centerline_error < 0.3:
                reward += 4.0  # Perfect centerline
            else:
                # ANY deviation is penalized - quadratic growth
                reward -= centerline_error ** 2 * 0.8  # 0.5m=-0.2, 1m=-0.8, 2m=-3.2

            # Wings level during rotation
            roll_error = abs(phi)
            reward -= roll_error * 8.0

            # Heading alignment - quadratic
            heading_error = abs(self._normalize_angle(psi - np.deg2rad(4.0)))
            reward -= heading_error ** 2 * 3.0

            # Smooth pitch up to rotation attitude
            if theta > np.deg2rad(5) and theta < np.deg2rad(12):
                reward += 0.3
            elif theta > np.deg2rad(12):
                reward -= 0.5  # Don't over-rotate

        elif phase == XCPhase.INITIAL_CLIMB:
            # Positive climb rate
            if climb_rate > 0:
                reward += 0.3
            # Wings reasonably level
            if abs(phi) < np.deg2rad(15):
                reward += 0.2
            # Good airspeed (above stall)
            if airspeed_kts > 60:
                reward += 0.2

        elif phase == XCPhase.CLIMB:
            # Altitude gain
            if climb_rate > 0:
                reward += 0.3
            # On track toward turning point
            bearing_to_tp = np.arctan2(self.experts[0].tp_y - y, self.experts[0].tp_x - x)
            heading_error = abs(self._normalize_angle(psi - bearing_to_tp))
            if heading_error < np.deg2rad(20):
                reward += 0.2

        elif phase == XCPhase.CRUISE_TO_TP:
            # Altitude hold
            alt_error = abs(alt - self.cruise_altitude_m)
            if alt_error < 50:
                reward += 0.5
            elif alt_error < 200:
                reward += 0.2
            else:
                reward -= alt_error / 1000.0

        elif phase == XCPhase.TURN_TO_INTERCEPT:
            # Capture runway heading
            heading_error = abs(self._normalize_angle(psi - self.runway_heading))
            if heading_error < np.deg2rad(15):
                reward += 0.5
            # Maintain reasonable altitude during turn
            if alt > self.cruise_altitude_m - 300:
                reward += 0.2

        elif phase in [XCPhase.INTERCEPT_LEG, XCPhase.FINAL_APPROACH]:
            # Glideslope tracking
            target_alt = dist_to_threshold * np.tan(np.deg2rad(self.glideslope_deg))
            gs_error = abs(alt - target_alt)
            if gs_error < 50:
                reward += 0.5
            elif gs_error < 150:
                reward += 0.2
            else:
                reward -= gs_error / 500.0

            # Speed control (approach: 65-75 kts)
            if 60 < airspeed_kts < 80:
                reward += 0.3
            elif airspeed_kts < 50 or airspeed_kts > 90:
                reward -= 0.5

            # Descent rate (should be negative)
            if climb_rate < 0:
                reward += 0.2

        elif phase in [XCPhase.SHORT_FINAL, XCPhase.LANDING]:
            # Glideslope tracking - follow 3 degree slope to touchdown
            target_alt = dist_to_threshold * np.tan(np.deg2rad(self.glideslope_deg))
            gs_error = abs(alt - target_alt)
            if gs_error < 5:
                reward += 1.5  # Perfect glideslope
            elif gs_error < 15:
                reward += 1.0
            elif gs_error < 30:
                reward += 0.5
            else:
                reward -= gs_error * 0.05  # Stronger penalty

            # Stable descent rate (-400 to -600 fpm ideal for C172)
            if -700 < climb_rate_fpm < -300:
                reward += 0.5
            elif climb_rate_fpm > -100:
                reward -= 1.0  # Not descending enough - stronger
            elif climb_rate_fpm < -1000:
                reward -= 1.0  # Descending too fast - stronger

            # Proper flare (pitch up as altitude decreases)
            if alt < 20:
                if theta > np.deg2rad(3):  # Nose up for flare
                    reward += 1.5
                else:
                    reward -= 1.0  # Need to flare!

            # CRITICAL: Centerline tracking for landing
            rwy_dx = np.cos(self.runway_heading)
            rwy_dy = np.sin(self.runway_heading)
            dx = x - threshold_x
            dy = y - threshold_y
            cross_track = abs(dx * (-rwy_dy) + dy * rwy_dx)
            if cross_track < 3:
                reward += 2.0  # Perfect centerline
            elif cross_track < 8:
                reward += 1.0
            elif cross_track < 15:
                reward += 0.3
            else:
                reward -= cross_track * 0.1  # Strong penalty for offset

            # Speed control - slow down for landing
            if 55 < airspeed_kts < 70:
                reward += 0.5
            elif airspeed_kts > 80:
                reward -= 0.5  # Too fast

        elif phase == XCPhase.LANDED:
            # Successful landing bonus - scale by quality
            rwy_dx = np.cos(self.runway_heading)
            rwy_dy = np.sin(self.runway_heading)
            dx = x - threshold_x
            dy = y - threshold_y
            final_cross_track = abs(dx * (-rwy_dy) + dy * rwy_dx)

            # Base bonus
            reward += 50.0

            # Quality bonus - centerline accuracy
            if final_cross_track < 5:
                reward += 50.0  # Perfect landing
            elif final_cross_track < 15:
                reward += 25.0  # Good landing
            elif final_cross_track < 30:
                reward += 10.0  # Acceptable
            else:
                reward -= final_cross_track  # Penalty for bad landing

            terminated = True
            info['termination'] = 'landed'
            info['landing_cross_track_m'] = final_cross_track

        # =================================================================
        # 3. SMOOTH CONTROL (prefer small residuals)
        # =================================================================
        residual_magnitude = np.sqrt(np.sum(action**2))
        reward -= 0.05 * residual_magnitude

        # =================================================================
        # 4. ENVELOPE PROTECTION (soft penalties)
        # =================================================================
        if phase not in [XCPhase.GROUND_ROLL, XCPhase.ROTATION, XCPhase.LANDED]:
            # Airspeed
            if airspeed_kts < 50 or airspeed_kts > 140:
                reward -= 0.5
            # Bank angle
            if abs(phi) > np.deg2rad(35):
                reward -= 0.3
            # Pitch
            if abs(theta) > np.deg2rad(20):
                reward -= 0.3

        # =================================================================
        # 5. HARD LIMITS - TERMINATE
        # =================================================================

        # Crashed (underground)
        if alt < -10:
            terminated = True
            reward -= 100.0
            info['termination'] = 'crashed'

        # Too high
        elif alt > 3000:
            terminated = True
            reward -= 50.0
            info['termination'] = 'too_high'

        # Stalled (after takeoff)
        elif airspeed_kts < 30 and alt > 50:
            terminated = True
            reward -= 100.0
            info['termination'] = 'stalled'

        # Overspeed
        elif airspeed_kts > 180:
            terminated = True
            reward -= 50.0
            info['termination'] = 'overspeed'

        # Extreme attitudes
        elif abs(phi) > np.deg2rad(70):
            terminated = True
            reward -= 50.0
            info['termination'] = 'excessive_bank'
        elif theta > np.deg2rad(50) or theta < np.deg2rad(-35):
            terminated = True
            reward -= 50.0
            info['termination'] = 'excessive_pitch'

        # Success: Very close to KHUT and low altitude (landing approach)
        elif dist_to_khut < 500 and alt < 100:
            terminated = True
            reward += 500.0
            info['termination'] = 'arrived'

        # Truncation
        if self.step_count >= self.max_episode_steps:
            truncated = True
            info['termination'] = 'max_steps'

        info['phase'] = phase.name
        info['altitude'] = alt
        info['airspeed_kts'] = airspeed_kts
        info['dist_to_khut'] = dist_to_khut / 1852  # nm

        return reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Ground start position with small randomization
        if self.n_instances == 1:
            initial_state = np.zeros((1, 12), dtype=np.float32)
            initial_state[0, 0] = -400.0 + np.random.uniform(-50, 50)  # X
            initial_state[0, 3] = 5.0 + np.random.uniform(-1, 1)       # U (taxi speed)
        else:
            # Parallel instances with varied initial conditions
            initial_state = np.zeros((self.n_instances, 12), dtype=np.float32)
            initial_state[:, 0] = -400.0 + np.random.uniform(-50, 50, self.n_instances)
            initial_state[:, 3] = 5.0 + np.random.uniform(-1, 1, self.n_instances)

        self.sim.reset(initial_state)
        for expert in self.experts:
            expert.reset()

        self.step_count = 0
        self.sim_time = 0.0
        self.state = self.sim.get_states()[0] if self.n_instances == 1 else self.sim.get_states()
        self.prev_dist_to_khut = None

        # Get expert action for initial observation
        if self.n_instances == 1:
            expert_action = self.experts[0].compute_action(self.state, self.sim_time)
            phase = self.experts[0].phase
            obs = self._get_obs(self.state, expert_action, phase)
        else:
            # For parallel envs, return first instance obs
            expert_action = self.experts[0].compute_action(self.state[0], self.sim_time)
            phase = self.experts[0].phase
            obs = self._get_obs(self.state[0], expert_action, phase)

        info = {'phase': phase.name}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Apply residual action on top of expert for all 7 controls.

        u_total = u_expert + lambda * action
        """
        self.step_count += 1
        self.sim_time += self.dt

        # Get expert control (7 dimensions)
        if self.n_instances == 1:
            expert_action = self.experts[0].compute_action(self.state, self.sim_time)
            phase = self.experts[0].phase
        else:
            expert_action = self.experts[0].compute_action(self.state[0], self.sim_time)
            phase = self.experts[0].phase

        # Combine: expert + scaled residual for ALL 7 controls
        throttle = np.clip(
            expert_action[0] + self.residual_scales[0] * action[0],
            0.0, 1.0
        )
        aileron = np.clip(
            expert_action[1] + self.residual_scales[1] * action[1],
            -1.0, 1.0
        )
        elevator = np.clip(
            expert_action[2] + self.residual_scales[2] * action[2],
            -1.0, 1.0
        )
        rudder = np.clip(
            expert_action[3] + self.residual_scales[3] * action[3],
            -1.0, 1.0
        )
        flaps = np.clip(
            expert_action[4] + self.residual_scales[4] * action[4],
            0.0, 1.0
        )
        spoilers = np.clip(
            expert_action[5] + self.residual_scales[5] * action[5],
            0.0, 1.0
        )
        brakes = np.clip(
            expert_action[6] + self.residual_scales[6] * action[6],
            0.0, 1.0
        )

        # Apply ALL 7 controls
        if self.n_instances == 1:
            controls = np.array([[throttle, aileron, elevator, rudder, flaps, spoilers, brakes]], dtype=np.float32)
        else:
            # Broadcast to all instances (simplified - in reality each would have own action)
            controls = np.tile(
                np.array([throttle, aileron, elevator, rudder, flaps, spoilers, brakes], dtype=np.float32),
                (self.n_instances, 1)
            )

        self.sim.set_controls(controls)
        self.sim.step()

        # Get new state
        self.state = self.sim.get_states()[0] if self.n_instances == 1 else self.sim.get_states()

        # Compute reward
        state_for_reward = self.state if self.n_instances == 1 else self.state[0]
        reward, terminated, truncated, info = self._compute_reward(
            state_for_reward, action, expert_action, phase
        )

        # Build observation
        obs = self._get_obs(state_for_reward, expert_action, phase)

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


# Register with Gymnasium
gym.register(
    id='ResidualFlight-v2',
    entry_point='residual_env_v2:ResidualFlightEnvV2',
    max_episode_steps=45000,
)


if __name__ == "__main__":
    # Quick test
    print("Testing ResidualFlightEnvV2 (7 controls, all phases)...")
    print("=" * 60)

    # Test single instance
    env = ResidualFlightEnvV2(use_gpu=True, n_instances=1)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial phase: {info['phase']}")

    # Run with zero residual (pure expert)
    total_reward = 0
    for i in range(1000):
        action = np.zeros(7)  # Zero residual = pure expert
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 200 == 0:
            print(f"Step {i}: phase={info['phase']:>18s}, alt={info['altitude']:5.0f}m, "
                  f"ias={info['airspeed_kts']:5.0f}kt, dist={info['dist_to_khut']:.1f}nm, r={reward:.2f}")

        if terminated or truncated:
            print(f"\nEpisode ended: {info.get('termination', 'unknown')}")
            break

    print(f"\nTotal reward (pure expert, 1000 steps): {total_reward:.1f}")
    env.close()
