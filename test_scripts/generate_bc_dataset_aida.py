"""
Behavior Cloning Dataset Generation using AIDA Flight Environment

Generates expert flight trajectories for PPO warm-start using:
- AIDA flight_env_rl environment (correct observation/action dimensions)
- Classical PID flight controller for stable cruise flight
- Diverse initial conditions for robust learning

Author: Kushal Koirala
Date: December 2024

Usage:
    python generate_bc_dataset_aida.py --num-episodes 1000 --output checkpoints/bc_dataset_aida.npz
"""

import numpy as np
import argparse
import sys
import os
from pathlib import Path
import time
from tqdm import tqdm

# Add AIDA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aida_sim.env.flight_env_rl import FlightEnvRL


class ClassicalFlightController:
    """
    PID-based flight controller for generating expert trajectories.
    Maps to AIDA action space: [throttle, trim_elev, trim_rudder, elevator_stick, aileron, rudder_stick]
    """

    def __init__(
        self,
        target_altitude=30.0,  # meters
        target_airspeed=20.0,  # m/s
        target_heading=0.0,    # radians
    ):
        self.target_altitude = target_altitude
        self.target_airspeed = target_airspeed
        self.target_heading = target_heading

        # PID gains (tuned for cruise flight)
        self.speed_Kp = 0.08
        self.altitude_Kp = 0.3
        self.altitude_Kd = 0.6
        self.heading_Kp = 1.2
        self.roll_Kp = 2.5
        self.roll_Kd = 0.5

        # Control limits
        self.max_elevator = np.deg2rad(15)
        self.max_aileron = np.deg2rad(20)
        self.max_rudder = np.deg2rad(15)

    def compute_controls(self, obs):
        """
        Compute control actions from observation.

        AIDA observation (19-dim):
        [pos_norm(3), vel_norm(3), attitude(3), rates_norm(3),
         airspeed_norm, altitude_norm, distance_to_target, heading_to_target, trim_norm, mission_phase]

        Returns:
        [throttle, trim_elev, trim_rudder, elevator_stick, aileron, rudder_stick]
        """
        # Extract relevant state from normalized observation
        # Note: These are normalized, so we need to denormalize for control

        # Attitude (indices 6-8): [roll, pitch, yaw]
        phi = obs[6]    # roll (normalized ~[-1, 1] for reasonable angles)
        theta = obs[7]  # pitch
        psi = obs[8]    # yaw

        # Angular rates (indices 9-11, normalized)
        p = obs[9]   # roll rate
        q = obs[10]  # pitch rate
        r = obs[11]  # yaw rate

        # Airspeed and altitude (normalized)
        airspeed_norm = obs[12]
        altitude_norm = obs[13]

        # Heading to target
        heading_error = obs[14]

        # Denormalize for control (approximate)
        # Airspeed: normalized around 20 m/s ± 10
        airspeed = 20.0 + airspeed_norm * 10.0
        # Altitude: normalized around 30m ± 20
        altitude = 30.0 + altitude_norm * 20.0

        # Angles are already in reasonable ranges after normalization
        roll = phi * np.deg2rad(45)  # Assume ±45° normalization
        pitch = theta * np.deg2rad(20)

        # Rates (assume ±30 deg/s normalization)
        p_rad = p * np.deg2rad(30)
        q_rad = q * np.deg2rad(30)
        r_rad = r * np.deg2rad(30)

        # === Throttle control (airspeed hold) ===
        speed_error = self.target_airspeed - airspeed
        throttle = 0.45 + self.speed_Kp * speed_error
        throttle = np.clip(throttle, 0.25, 0.75)  # AIDA flight envelope

        # === Elevator control (altitude hold via pitch) ===
        altitude_error = self.target_altitude - altitude
        desired_climb_rate = self.altitude_Kp * altitude_error
        desired_climb_rate = np.clip(desired_climb_rate, -2.0, 2.0)

        # Use pitch rate as proxy for climb rate
        elevator_stick = self.altitude_Kd * desired_climb_rate - 0.3 * q_rad
        elevator_stick = np.clip(elevator_stick, -1.0, 1.0)

        # === Aileron control (wings level + heading hold) ===
        desired_roll = self.heading_Kp * heading_error
        desired_roll = np.clip(desired_roll, -np.deg2rad(30), np.deg2rad(30))

        roll_error = desired_roll - roll
        aileron = self.roll_Kp * roll_error - self.roll_Kd * p_rad
        aileron = np.clip(aileron, -1.0, 1.0)

        # === Rudder control (coordinated turn + yaw damping) ===
        rudder_stick = -0.2 * r_rad
        rudder_stick = np.clip(rudder_stick, -1.0, 1.0)

        # === Trim controls (slowly learned, keep near neutral for cruise) ===
        trim_elev = 0.0   # Let PPO learn trim
        trim_rudder = 0.0

        # AIDA action: [throttle, trim_elev, trim_rudder, elevator_stick, aileron, rudder_stick]
        controls = np.array([
            throttle,
            trim_elev,
            trim_rudder,
            elevator_stick,
            aileron,
            rudder_stick
        ], dtype=np.float32)

        return controls


def collect_trajectories(n_episodes, episode_length=500, task='cruise'):
    """
    Collect expert trajectories using AIDA environment.

    Args:
        n_episodes: Total number of episodes to collect
        episode_length: Maximum steps per episode
        task: 'cruise' or 'takeoff'

    Returns:
        dataset: Dictionary with 'observations', 'actions', 'rewards'
    """
    print(f"Collecting {n_episodes} expert trajectories...")
    print(f"  Task: {task}")
    print(f"  Max episode length: {episode_length} steps")

    env = FlightEnvRL(task=task)
    controller = ClassicalFlightController(
        target_altitude=30.0,
        target_airspeed=20.0,
        target_heading=0.0
    )

    # Storage
    all_observations = []
    all_actions = []
    all_rewards = []

    start_time = time.time()
    successful_episodes = 0

    for ep in tqdm(range(n_episodes), desc="Collecting episodes"):
        obs, _ = env.reset()
        episode_obs = []
        episode_actions = []
        episode_rewards = []

        for step in range(episode_length):
            # Compute expert action
            action = controller.compute_controls(obs)

            # Store
            episode_obs.append(obs.copy())
            episode_actions.append(action.copy())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)

            if terminated or truncated:
                if terminated and reward > 0:  # Successful completion
                    successful_episodes += 1
                break

        # Only keep reasonable episodes (not immediate failures)
        if len(episode_obs) >= 10:
            all_observations.extend(episode_obs)
            all_actions.extend(episode_actions)
            all_rewards.extend(episode_rewards)

    elapsed_time = time.time() - start_time
    total_steps = len(all_observations)

    print(f"\nData collection complete!")
    print(f"  Total episodes: {n_episodes}")
    print(f"  Successful episodes: {successful_episodes} ({100*successful_episodes/n_episodes:.1f}%)")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Elapsed time: {elapsed_time:.2f} s")
    print(f"  Steps/second: {total_steps / elapsed_time:.0f}")

    dataset = {
        'observations': np.array(all_observations, dtype=np.float32),
        'actions': np.array(all_actions, dtype=np.float32),
        'rewards': np.array(all_rewards, dtype=np.float32),
        'n_episodes': n_episodes,
        'successful_episodes': successful_episodes,
    }

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate BC dataset using AIDA flight environment")
    parser.add_argument("--num-episodes", type=int, default=500, help="Number of expert episodes")
    parser.add_argument("--episode-length", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--task", type=str, default="cruise", choices=["cruise", "takeoff"], help="Flight task")
    parser.add_argument("--output", type=str, default="checkpoints/bc_dataset_aida.npz", help="Output file")

    args = parser.parse_args()

    # Collect dataset
    dataset = collect_trajectories(
        n_episodes=args.num_episodes,
        episode_length=args.episode_length,
        task=args.task
    )

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total timesteps: {len(dataset['observations']):,}")
    print(f"  Observation shape: {dataset['observations'].shape}")
    print(f"  Action shape: {dataset['actions'].shape}")
    print(f"  Mean reward: {dataset['rewards'].mean():.3f} ± {dataset['rewards'].std():.3f}")

    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        **dataset
    )

    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print("\n" + "="*60)
    print("  Next Steps:")
    print("="*60)
    print("  1. Train BC policy:")
    print(f"     python scripts/train_bc_policy.py --dataset {output_path}")
    print("  2. Warm-start PPO with BC policy:")
    print(f"     python scripts/train_ppo_flight.py --policy-init checkpoints/bc_policy.pt --task {args.task}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
