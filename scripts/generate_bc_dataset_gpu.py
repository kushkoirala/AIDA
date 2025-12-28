"""
Behavior Cloning Dataset Generation using GPU-Accelerated Udaan Simulation

Generates expert flight trajectories for PPO warm-start using:
- GPU-accelerated parallel simulation (gpu-flight-dynamics)
- Classical flight controller for stable flight demonstration
- Diverse flight conditions for robust learning

The generated dataset can be used to:
1. Pre-train the PPO policy via behavior cloning
2. Reduce exploration time in early RL training
3. Provide baseline performance metrics

Author: Kushal Koirala
Date: December 2024

Usage:
    python generate_bc_dataset_gpu.py --num-episodes 1000 --output bc_dataset.npz
"""

import numpy as np
import argparse
import sys
import os
from pathlib import Path
import time
from tqdm import tqdm

# Add gpu-flight-dynamics to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))

from flight_dynamics import FlightSimulator, StateIndex, ControlIndex, STATE_DIM, CONTROL_DIM
from aircraft_database import get_aircraft


def config_to_params(config):
    """Convert AircraftConfig to FlightSimulator parameters"""
    from flight_dynamics import (
        AircraftParams, MassProperties, Geometry,
        LongitudinalDerivatives, LateralDerivatives, PropulsionParams
    )

    return AircraftParams(
        mass=MassProperties(
            mass=config.mass.mass,
            Ixx=config.mass.Ixx,
            Iyy=config.mass.Iyy,
            Izz=config.mass.Izz,
            Ixz=config.mass.Ixz,
        ),
        geom=Geometry(
            S=config.geom.S,
            b=config.geom.b,
            c=config.geom.c,
            e=config.geom.e,
        ),
        longi=LongitudinalDerivatives(
            CL0=config.longi.CL0,
            CLa=config.longi.CLa,
            CLq=config.longi.CLq,
            CLde=config.longi.CLde,
            CLmax=config.longi.CLmax,
            CLmin=config.longi.CLmin,
            CD0=config.longi.CD0,
            K=config.longi.K,
            CDa=config.longi.CDa,
            Cm0=config.longi.Cm0,
            Cma=config.longi.Cma,
            Cmq=config.longi.Cmq,
            Cmde=config.longi.Cmde,
        ),
        latdi=LateralDerivatives(
            CYb=config.latdi.CYb,
            CYp=config.latdi.CYp,
            CYr=config.latdi.CYr,
            CYda=config.latdi.CYda,
            CYdr=config.latdi.CYdr,
            Clb=config.latdi.Clb,
            Clp=config.latdi.Clp,
            Clr=config.latdi.Clr,
            Clda=config.latdi.Clda,
            Cldr=config.latdi.Cldr,
            Cnb=config.latdi.Cnb,
            Cnp=config.latdi.Cnp,
            Cnr=config.latdi.Cnr,
            Cnda=config.latdi.Cnda,
            Cndr=config.latdi.Cndr,
        ),
        prop=PropulsionParams(
            thrust_max=config.prop.thrust_max,
            thrust_min=config.prop.thrust_min,
            tau=config.prop.tau,
        ),
    )


class ClassicalFlightController:
    """
    Simple PID-based flight controller for generating expert trajectories.

    Maintains:
    - Altitude hold
    - Airspeed hold
    - Wings-level flight
    - Heading hold
    """

    def __init__(self, target_altitude=30.0, target_airspeed=22.4, target_heading=0.0):
        """
        Args:
            target_altitude: Target altitude in meters
            target_airspeed: Target airspeed in m/s
            target_heading: Target heading in radians
        """
        self.target_altitude = target_altitude
        self.target_airspeed = target_airspeed
        self.target_heading = target_heading

        # PID gains (tuned for Udaan)
        self.altitude_Kp = 0.056  # Tuned for CLα=3.82
        self.altitude_Kd = 0.104  # Tuned for CLα=3.82
        self.speed_Kp = 0.028  # Tuned for CLα=3.82
        self.roll_Kp = 1.391  # Tuned for CLα=3.82
        self.roll_Kd = 0.348  # Tuned for CLα=3.82
        self.heading_Kp = 0.209  # Tuned for CLα=3.82

        # Control limits
        self.max_elevator = np.deg2rad(15.0)
        self.max_aileron = np.deg2rad(20.0)
        self.max_rudder = np.deg2rad(15.0)

    def compute_controls(self, state):
        """
        Compute control inputs based on current state.

        Args:
            state: State vector [x, y, z, u, v, w, phi, theta, psi, p, q, r]

        Returns:
            controls: Control vector [throttle, aileron, elevator, rudder]
        """
        # Extract state variables
        z = state[StateIndex.Z]
        u = state[StateIndex.U]
        phi = state[StateIndex.PHI]
        theta = state[StateIndex.THETA]
        psi = state[StateIndex.PSI]
        p = state[StateIndex.P]
        q = state[StateIndex.Q]
        r = state[StateIndex.R]

        altitude = -z  # NED to AGL
        climb_rate = -state[StateIndex.W]  # NED w-component

        # Throttle control (airspeed hold)
        speed_error = self.target_airspeed - u
        throttle = 0.45 + self.speed_Kp * speed_error
        throttle = np.clip(throttle, 0.0, 1.0)

        # Elevator control (altitude hold via pitch)
        altitude_error = self.target_altitude - altitude
        desired_climb_rate = self.altitude_Kp * altitude_error
        desired_climb_rate = np.clip(desired_climb_rate, -3.0, 3.0)

        climb_error = desired_climb_rate - climb_rate
        elevator_cmd = self.altitude_Kd * climb_error - 0.3 * q

        # CRITICAL FIX: Negate elevator for Udaan's sign convention
        # In this aircraft model, positive elevator produces nose-DOWN moment
        # (see Dynamics PDF Table 2: Cm,δe = -1.13)
        # Therefore, to pitch UP (positive climb error), we need NEGATIVE elevator
        elevator = -elevator_cmd
        elevator = np.clip(elevator, -self.max_elevator, self.max_elevator)

        # Aileron control (wings level + heading hold)
        heading_error = self.target_heading - psi
        # Wrap heading error to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        desired_roll = self.heading_Kp * heading_error
        desired_roll = np.clip(desired_roll, -np.deg2rad(30), np.deg2rad(30))

        roll_error = desired_roll - phi
        aileron = self.roll_Kp * roll_error - self.roll_Kd * p
        aileron = np.clip(aileron, -self.max_aileron, self.max_aileron)

        # Rudder control (coordinated turn + yaw damping)
        rudder = -0.2 * r  # Simple yaw damper
        rudder = np.clip(rudder, -self.max_rudder, self.max_rudder)

        # Convert to simulator format
        controls = np.array([throttle, aileron, elevator, rudder], dtype=np.float32)
        return controls


def generate_random_initial_conditions(n_instances, seed=None):
    """
    Generate diverse initial conditions for parallel episodes.

    Args:
        n_instances: Number of parallel instances
        seed: Random seed for reproducibility

    Returns:
        initial_states: (n_instances, STATE_DIM) array
    """
    if seed is not None:
        np.random.seed(seed)

    initial_states = np.zeros((n_instances, STATE_DIM), dtype=np.float32)

    # Random positions (within reasonable flight box)
    initial_states[:, StateIndex.X] = np.random.uniform(-100, 100, n_instances)
    initial_states[:, StateIndex.Y] = np.random.uniform(-100, 100, n_instances)
    initial_states[:, StateIndex.Z] = -np.random.uniform(15, 50, n_instances)  # 15-50m AGL

    # Random cruise speeds (±20% variation)
    cruise_speed = 22.4  # m/s
    initial_states[:, StateIndex.U] = np.random.uniform(cruise_speed * 0.85, cruise_speed * 1.15, n_instances)

    # Small attitude variations
    initial_states[:, StateIndex.PHI] = np.random.uniform(-5, 5, n_instances) * np.pi / 180
    initial_states[:, StateIndex.THETA] = np.random.uniform(-3, 5, n_instances) * np.pi / 180
    initial_states[:, StateIndex.PSI] = np.random.uniform(-180, 180, n_instances) * np.pi / 180

    return initial_states


def collect_trajectories(n_episodes, episode_length, batch_size=100, use_gpu=True):
    """
    Collect expert trajectories using parallel GPU simulation.

    Args:
        n_episodes: Total number of episodes to collect
        episode_length: Steps per episode
        batch_size: Number of parallel episodes per batch
        use_gpu: Use GPU acceleration

    Returns:
        dataset: Dictionary with keys 'observations', 'actions', 'rewards'
    """
    print(f"Collecting {n_episodes} expert trajectories...")
    print(f"  Episode length: {episode_length} steps")
    print(f"  Batch size: {batch_size} parallel instances")
    print(f"  Device: {'GPU (CuPy)' if use_gpu else 'CPU (NumPy)'}")

    # Initialize Udaan simulator
    udaan_config = get_aircraft("udaan")
    udaan_params = config_to_params(udaan_config)
    sim = FlightSimulator(n_instances=batch_size, params=udaan_params, dt=0.02, use_gpu=use_gpu)

    # Storage for dataset
    all_observations = []
    all_actions = []
    all_rewards = []

    # Classical controller
    controller = ClassicalFlightController(
        target_altitude=30.0,  # ~100 ft
        target_airspeed=22.4,  # ~73.5 ft/s
        target_heading=0.0
    )

    n_batches = int(np.ceil(n_episodes / batch_size))

    start_time = time.time()

    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        # Generate random initial conditions for this batch
        initial_states = generate_random_initial_conditions(batch_size, seed=batch_idx)
        sim.reset(initial_state=initial_states)

        batch_observations = []
        batch_actions = []
        batch_rewards = []

        for step in range(episode_length):
            # Get current states
            states = sim.get_states()  # (batch_size, STATE_DIM)

            # Compute expert actions for each instance
            actions = np.zeros((batch_size, CONTROL_DIM), dtype=np.float32)
            for i in range(batch_size):
                actions[i] = controller.compute_controls(states[i])

            # Store observations and actions
            batch_observations.append(states.copy())
            batch_actions.append(actions.copy())

            # Simple reward: penalize deviation from targets
            altitude = -states[:, StateIndex.Z]
            airspeed = states[:, StateIndex.U]
            roll = states[:, StateIndex.PHI]

            altitude_error = np.abs(altitude - controller.target_altitude)
            speed_error = np.abs(airspeed - controller.target_airspeed)
            roll_error = np.abs(roll)

            rewards = 1.0 - 0.1 * (altitude_error + speed_error) - 0.2 * roll_error
            batch_rewards.append(rewards)

            # Step simulation
            sim.set_controls(actions)
            sim.step()

        # Append batch data
        all_observations.append(np.array(batch_observations))  # (episode_length, batch_size, STATE_DIM)
        all_actions.append(np.array(batch_actions))            # (episode_length, batch_size, CONTROL_DIM)
        all_rewards.append(np.array(batch_rewards))            # (episode_length, batch_size)

    elapsed_time = time.time() - start_time
    total_steps = n_episodes * episode_length

    print(f"\nData collection complete!")
    print(f"  Total episodes: {n_episodes}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Elapsed time: {elapsed_time:.2f} s")
    print(f"  Steps/second: {total_steps / elapsed_time:,.0f}")

    # Concatenate all batches
    all_observations = np.concatenate(all_observations, axis=1)  # (episode_length, n_episodes, STATE_DIM)
    all_actions = np.concatenate(all_actions, axis=1)            # (episode_length, n_episodes, CONTROL_DIM)
    all_rewards = np.concatenate(all_rewards, axis=1)            # (episode_length, n_episodes)

    # Reshape to (total_timesteps, dim)
    observations = all_observations.reshape(-1, STATE_DIM)
    actions = all_actions.reshape(-1, CONTROL_DIM)
    rewards = all_rewards.reshape(-1)

    dataset = {
        'observations': observations[:n_episodes * episode_length],  # Trim to exact count
        'actions': actions[:n_episodes * episode_length],
        'rewards': rewards[:n_episodes * episode_length],
        'episode_length': episode_length,
        'n_episodes': n_episodes,
    }

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate BC dataset using GPU-accelerated Udaan simulation")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of expert episodes")
    parser.add_argument("--episode-length", type=int, default=500, help="Steps per episode (500 = 10s @ 50Hz)")
    parser.add_argument("--batch-size", type=int, default=100, help="Parallel instances per batch")
    parser.add_argument("--output", type=str, default="checkpoints/bc_dataset_udaan.npz", help="Output file")
    parser.add_argument("--use-cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  BEHAVIOR CLONING DATASET GENERATION")
    print("  Udaan UAV - GPU-Accelerated Simulation")
    print("="*60)

    # Set random seed
    np.random.seed(args.seed)

    # Collect trajectories
    dataset = collect_trajectories(
        n_episodes=args.num_episodes,
        episode_length=args.episode_length,
        batch_size=args.batch_size,
        use_gpu=not args.use_cpu
    )

    # Print dataset statistics
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
        observations=dataset['observations'],
        actions=dataset['actions'],
        rewards=dataset['rewards'],
        episode_length=dataset['episode_length'],
        n_episodes=dataset['n_episodes'],
    )

    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print("\n" + "="*60)
    print("  Next Steps:")
    print("="*60)
    print("  1. Train BC policy:")
    print(f"     python scripts/train_bc_policy.py --dataset {output_path}")
    print("  2. Warm-start PPO with BC policy:")
    print(f"     python scripts/train_ppo_flight.py --policy-init checkpoints/bc_policy.pt")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
