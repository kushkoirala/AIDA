"""
GPU Flight Dynamics to AIDA Environment Adapter

Converts between GPU physics format and AIDA flight_env_rl format for BC training.

GPU Format (12-dim state, 4-dim control):
  State: [x, y, z, u, v, w, phi, theta, psi, p, q, r]
  Control: [throttle, aileron, elevator, rudder]

AIDA Format (19-dim obs, 6-dim action):
  Obs: [pos_norm(3), vel_norm(3), attitude(3), rates_norm(3),
        airspeed_norm, altitude_norm, distance_to_target, heading_to_target, trim_norm, mission_phase]
  Action: [throttle, trim_elev, trim_rudder, elevator_stick, aileron, rudder_stick]

Author: Kushal Koirala
Date: December 2024
"""

import numpy as np


class GPUToAIDAAdapter:
    """Adapter to convert GPU physics data to AIDA format"""

    def __init__(self, target_altitude=30.0, target_heading=0.0):
        """
        Args:
            target_altitude: Target cruise altitude in meters
            target_heading: Target heading in radians
        """
        self.target_altitude = target_altitude
        self.target_heading = target_heading

        # Normalization scales (approximate, matching AIDA env)
        self.pos_scale = 100.0    # meters
        self.vel_scale = 25.0     # m/s
        self.angle_scale = np.pi / 4  # radians (45 deg)
        self.rate_scale = np.deg2rad(30)  # rad/s
        self.airspeed_scale = 10.0  # m/s around 20 m/s cruise
        self.altitude_scale = 20.0  # meters around 30m

    def gpu_state_to_aida_obs(self, gpu_state):
        """
        Convert GPU state (12-dim) to AIDA observation (19-dim)

        Args:
            gpu_state: (N, 12) or (12,) array
                [x, y, z, u, v, w, phi, theta, psi, p, q, r]

        Returns:
            aida_obs: (N, 19) or (19,) array
                [pos_norm(3), vel_norm(3), attitude(3), rates_norm(3),
                 airspeed_norm, altitude_norm, distance_to_target,
                 heading_to_target, trim_norm, mission_phase]
        """
        single_input = gpu_state.ndim == 1
        if single_input:
            gpu_state = gpu_state[np.newaxis, :]

        N = len(gpu_state)
        aida_obs = np.zeros((N, 19), dtype=np.float32)

        # Extract GPU state components
        x, y, z = gpu_state[:, 0], gpu_state[:, 1], gpu_state[:, 2]
        u, v, w = gpu_state[:, 3], gpu_state[:, 4], gpu_state[:, 5]
        phi, theta, psi = gpu_state[:, 6], gpu_state[:, 7], gpu_state[:, 8]
        p, q, r = gpu_state[:, 9], gpu_state[:, 10], gpu_state[:, 11]

        # 0-2: Position normalized
        aida_obs[:, 0] = x / self.pos_scale
        aida_obs[:, 1] = y / self.pos_scale
        aida_obs[:, 2] = z / self.pos_scale

        # 3-5: Velocity normalized (body frame velocities)
        aida_obs[:, 3] = u / self.vel_scale
        aida_obs[:, 4] = v / self.vel_scale
        aida_obs[:, 5] = w / self.vel_scale

        # 6-8: Attitude (Euler angles, already in radians)
        aida_obs[:, 6] = phi / self.angle_scale
        aida_obs[:, 7] = theta / self.angle_scale
        aida_obs[:, 8] = psi / np.pi  # Yaw normalized to [-1, 1]

        # 9-11: Body rates normalized
        aida_obs[:, 9] = p / self.rate_scale
        aida_obs[:, 10] = q / self.rate_scale
        aida_obs[:, 11] = r / self.rate_scale

        # 12: Airspeed normalized
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        aida_obs[:, 12] = (airspeed - 20.0) / self.airspeed_scale

        # 13: Altitude normalized (NED convention: z is down, so altitude = -z)
        altitude = -z
        aida_obs[:, 13] = (altitude - self.target_altitude) / self.altitude_scale

        # 14: Distance to target (horizontal distance to origin)
        distance_to_target = np.sqrt(x**2 + y**2)
        aida_obs[:, 14] = np.clip(distance_to_target / 100.0, -1, 1)

        # 15: Heading to target
        heading_to_target = np.arctan2(y, x + 1e-6)  # Avoid division by zero
        heading_error = self._wrap_angle(heading_to_target - self.target_heading)
        aida_obs[:, 15] = heading_error / np.pi

        # 16: Trim normalized (set to 0 for now, BC will learn this)
        aida_obs[:, 16] = 0.0

        # 17-18: Mission phase (cruise = 1.0, others = 0.0)
        aida_obs[:, 17] = 1.0  # Cruise phase
        aida_obs[:, 18] = 0.0  # Reserved

        if single_input:
            return aida_obs[0]
        return aida_obs

    def aida_action_to_gpu_control(self, aida_action):
        """
        Convert AIDA action (6-dim) to GPU control (4-dim)

        Args:
            aida_action: (N, 6) or (6,) array
                [throttle, trim_elev, trim_rudder, elevator_stick, aileron, rudder_stick]

        Returns:
            gpu_control: (N, 4) or (4,) array
                [throttle, aileron, elevator, rudder]
        """
        single_input = aida_action.ndim == 1
        if single_input:
            aida_action = aida_action[np.newaxis, :]

        N = len(aida_action)
        gpu_control = np.zeros((N, 4), dtype=np.float32)

        # Extract AIDA action components
        throttle = aida_action[:, 0]
        trim_elev = aida_action[:, 1]
        trim_rudder = aida_action[:, 2]
        elevator_stick = aida_action[:, 3]
        aileron = aida_action[:, 4]
        rudder_stick = aida_action[:, 5]

        # 0: Throttle (map from [-1, 1] to [0, 1] if needed)
        # AIDA uses normalized throttle, GPU expects [0, 1]
        gpu_control[:, 0] = np.clip((throttle + 1.0) / 2.0, 0.0, 1.0)

        # 1: Aileron (direct mapping, both use [-1, 1])
        gpu_control[:, 1] = aileron

        # 2: Elevator (combine trim and stick input)
        gpu_control[:, 2] = np.clip(trim_elev + elevator_stick, -1.0, 1.0)

        # 3: Rudder (combine trim and stick input)
        gpu_control[:, 3] = np.clip(trim_rudder + rudder_stick, -1.0, 1.0)

        if single_input:
            return gpu_control[0]
        return gpu_control

    def gpu_control_to_aida_action(self, gpu_control):
        """
        Convert GPU control (4-dim) to AIDA action (6-dim)

        Args:
            gpu_control: (N, 4) or (4,) array
                [throttle, aileron, elevator, rudder]

        Returns:
            aida_action: (N, 6) or (6,) array
                [throttle, trim_elev, trim_rudder, elevator_stick, aileron, rudder_stick]
        """
        single_input = gpu_control.ndim == 1
        if single_input:
            gpu_control = gpu_control[np.newaxis, :]

        N = len(gpu_control)
        aida_action = np.zeros((N, 6), dtype=np.float32)

        # Extract GPU control components
        throttle = gpu_control[:, 0]
        aileron = gpu_control[:, 1]
        elevator = gpu_control[:, 2]
        rudder = gpu_control[:, 3]

        # 0: Throttle (map from [0, 1] to [-1, 1] for AIDA)
        aida_action[:, 0] = throttle * 2.0 - 1.0

        # 1-2: Trim (set to 0, let stick do all the work)
        aida_action[:, 1] = 0.0  # trim_elev
        aida_action[:, 2] = 0.0  # trim_rudder

        # 3: Elevator stick
        aida_action[:, 3] = elevator

        # 4: Aileron
        aida_action[:, 4] = aileron

        # 5: Rudder stick
        aida_action[:, 5] = rudder

        if single_input:
            return aida_action[0]
        return aida_action

    @staticmethod
    def _wrap_angle(angle):
        """Wrap angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))


def convert_gpu_dataset_to_aida(gpu_dataset_path, output_path, adapter=None):
    """
    Convert GPU-generated BC dataset to AIDA format

    Args:
        gpu_dataset_path: Path to GPU dataset .npz file
        output_path: Path to save converted AIDA dataset
        adapter: GPUToAIDAAdapter instance (creates default if None)
    """
    if adapter is None:
        adapter = GPUToAIDAAdapter(target_altitude=30.0, target_heading=0.0)

    # Load GPU dataset
    print(f"Loading GPU dataset from {gpu_dataset_path}...")
    gpu_data = np.load(gpu_dataset_path)

    gpu_observations = gpu_data['observations']  # (N, 12)
    gpu_actions = gpu_data['actions']            # (N, 4)
    gpu_rewards = gpu_data.get('rewards', np.zeros(len(gpu_observations)))

    print(f"GPU dataset:")
    print(f"  Observations: {gpu_observations.shape}")
    print(f"  Actions: {gpu_actions.shape}")
    print(f"  Samples: {len(gpu_observations):,}")

    # Convert to AIDA format
    print("Converting to AIDA format...")
    aida_observations = adapter.gpu_state_to_aida_obs(gpu_observations)
    aida_actions = adapter.gpu_control_to_aida_action(gpu_actions)

    print(f"AIDA dataset:")
    print(f"  Observations: {aida_observations.shape}")
    print(f"  Actions: {aida_actions.shape}")

    # Save converted dataset
    np.savez_compressed(
        output_path,
        observations=aida_observations,
        actions=aida_actions,
        rewards=gpu_rewards,
        n_episodes=gpu_data.get('n_episodes', len(gpu_observations)),
    )

    print(f"Saved AIDA dataset to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Convert GPU physics BC dataset to AIDA format")
    parser.add_argument("--input", type=str, required=True, help="Input GPU dataset .npz file")
    parser.add_argument("--output", type=str, required=True, help="Output AIDA dataset .npz file")
    parser.add_argument("--target-altitude", type=float, default=30.0, help="Target altitude (m)")
    parser.add_argument("--target-heading", type=float, default=0.0, help="Target heading (rad)")

    args = parser.parse_args()

    adapter = GPUToAIDAAdapter(
        target_altitude=args.target_altitude,
        target_heading=args.target_heading
    )

    convert_gpu_dataset_to_aida(args.input, args.output, adapter)

    print("\nNext steps:")
    print(f"  1. Train BC policy:")
    print(f"     python scripts/train_bc_policy.py --dataset {args.output}")
    print(f"  2. Warm-start PPO:")
    print(f"     python scripts/train_ppo_flight.py --policy-init checkpoints/bc_policy.pt --task cruise")
