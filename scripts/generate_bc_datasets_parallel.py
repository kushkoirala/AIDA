#!/usr/bin/env python3
"""
Generate Multiple BC Datasets with Different Initial Conditions in Parallel

Tests various initial condition configurations to find the most stable
for BC training with updated CATIA inertia values.

Author: Kushal Koirala (with Claude Code)
Date: December 27, 2024
"""

import numpy as np
import sys
import os
from pathlib import Path
from multiprocessing import Process, Queue
import time

# Add gpu-flight-dynamics to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))

from flight_dynamics import FlightSimulator, StateIndex, ControlIndex, STATE_DIM, CONTROL_DIM
from aircraft_database import get_aircraft


def config_to_params(config):
    """Convert AircraftConfig to params dict for FlightSimulator"""
    return {
        'mass': config.mass.mass,
        'Ixx': config.mass.Ixx,
        'Iyy': config.mass.Iyy,
        'Izz': config.mass.Izz,
        'Ixz': config.mass.Ixz,
        'S': config.geom.S,
        'b': config.geom.b,
        'c': config.geom.c,
        'CL0': config.longi.CL0,
        'CLa': config.longi.CLa,
        'CLq': config.longi.CLq,
        'CLde': config.longi.CLde,
        'CD0': config.longi.CD0,
        'K': config.longi.K,
        'Cm0': config.longi.Cm0,
        'Cma': config.longi.Cma,
        'Cmq': config.longi.Cmq,
        'Cmde': config.longi.Cmde,
        'CYb': config.latdi.CYb,
        'Clb': config.latdi.Clb,
        'Clp': config.latdi.Clp,
        'Clr': config.latdi.Clr,
        'Clda': config.latdi.Clda,
        'Cnb': config.latdi.Cnb,
        'Cnp': config.latdi.Cnp,
        'Cnr': config.latdi.Cnr,
        'thrust_max': config.prop.thrust_max,
    }


class ClassicalFlightController:
    """Classical PID controller for altitude/heading hold"""
    def __init__(self, altitude_Kp=0.15, altitude_Kd=0.5, max_elevator=np.deg2rad(15)):
        self.altitude_Kp = altitude_Kp
        self.altitude_Kd = altitude_Kd
        self.max_elevator = max_elevator

    def get_action(self, state, target_altitude=30.0):
        """Compute control action for altitude hold"""
        # Extract state
        z = state[StateIndex.Z]
        w = state[StateIndex.W]
        theta = state[StateIndex.THETA]
        q = state[StateIndex.Q]

        altitude = -z  # NED to AGL
        climb_rate = -w

        # Altitude hold with climb rate
        altitude_error = target_altitude - altitude
        desired_climb_rate = self.altitude_Kp * altitude_error
        desired_climb_rate = np.clip(desired_climb_rate, -5.0, 5.0)

        climb_error = desired_climb_rate - climb_rate
        elevator_cmd = self.altitude_Kd * climb_error - 0.3 * q

        # CRITICAL FIX: Negate elevator for Udaan's sign convention
        elevator = -elevator_cmd
        elevator = np.clip(elevator, -self.max_elevator, self.max_elevator)

        # Throttle to maintain speed
        throttle = 0.6

        return np.array([throttle, 0.0, elevator, 0.0], dtype=np.float32)


def generate_dataset_config(config_id, init_params, n_episodes=100, episode_length=500, result_queue=None):
    """
    Generate BC dataset with specific initial condition parameters.
    Runs in separate process for parallelization.
    """
    try:
        # Don't set CUDA_VISIBLE_DEVICES - let all processes share the GPU
        # CuPy can handle multiple processes accessing the same GPU

        # Get aircraft config
        aircraft_config = get_aircraft('udaan')
        params = config_to_params(aircraft_config)

        # Create simulator
        batch_size = 20  # Smaller batches for parallel execution
        n_batches = n_episodes // batch_size
        dt = 0.02

        print(f"\nConfig {config_id}: Starting generation...")
        print(f"  Init params: {init_params['name']}")
        print(f"  Episodes: {n_episodes}, Batch size: {batch_size}")

        sim = FlightSimulator(n_instances=batch_size, params=params, dt=dt)
        controller = ClassicalFlightController(
            altitude_Kp=init_params.get('altitude_Kp', 0.15),
            altitude_Kd=init_params.get('altitude_Kd', 0.5)
        )

        # Storage
        all_obs = []
        all_actions = []
        all_rewards = []

        for batch_idx in range(n_batches):
            # Initialize states with config-specific parameters
            n = batch_size
            altitude_init = np.random.uniform(
                init_params['alt_min'],
                init_params['alt_max'],
                n
            )
            airspeed_init = np.random.uniform(
                init_params['speed_min'],
                init_params['speed_max'],
                n
            )
            pitch_init = np.random.uniform(
                init_params['pitch_min'],
                init_params['pitch_max'],
                n
            )

            # Set initial states - convert to CuPy if needed
            try:
                import cupy as cp
                sim.states[:, StateIndex.Z] = cp.asarray(-altitude_init)
                sim.states[:, StateIndex.U] = cp.asarray(airspeed_init)
                sim.states[:, StateIndex.W] = 0.0
                sim.states[:, StateIndex.THETA] = cp.deg2rad(cp.asarray(pitch_init))
                sim.states[:, StateIndex.PHI] = 0.0
                sim.states[:, StateIndex.PSI] = 0.0
            except:
                # Fallback to numpy if CuPy not available
                sim.states[:, StateIndex.Z] = -altitude_init
                sim.states[:, StateIndex.U] = airspeed_init
                sim.states[:, StateIndex.W] = 0.0
                sim.states[:, StateIndex.THETA] = np.deg2rad(pitch_init)
                sim.states[:, StateIndex.PHI] = 0.0
                sim.states[:, StateIndex.PSI] = 0.0

            # Rollout episodes
            batch_obs = []
            batch_actions = []
            batch_rewards = []

            for step in range(episode_length):
                # Get observations (convert from CuPy if needed)
                obs = sim.get_states()  # Returns (batch_size, STATE_DIM)
                try:
                    obs_np = obs.get() if hasattr(obs, 'get') else obs
                except:
                    obs_np = np.array(obs)

                batch_obs.append(obs_np)

                # Compute actions for each instance
                actions = np.zeros((n, CONTROL_DIM), dtype=np.float32)
                for i in range(n):
                    actions[i] = controller.get_action(obs_np[i])

                batch_actions.append(actions)

                # Step simulation
                sim.set_controls(actions)
                sim.step()

                # Compute reward (simple altitude tracking)
                altitudes = -obs_np[:, StateIndex.Z]
                altitude_errors = np.abs(altitudes - 30.0)
                rewards = -altitude_errors / 10.0  # Normalize
                batch_rewards.append(rewards)

            # Convert to numpy arrays
            batch_obs = np.array(batch_obs).transpose(1, 0, 2).reshape(-1, STATE_DIM)
            batch_actions = np.array(batch_actions).transpose(1, 0, 2).reshape(-1, CONTROL_DIM)
            batch_rewards = np.array(batch_rewards).transpose(1, 0).reshape(-1)

            all_obs.append(batch_obs)
            all_actions.append(batch_actions)
            all_rewards.append(batch_rewards)

            if (batch_idx + 1) % 2 == 0:
                print(f"Config {config_id}: Batch {batch_idx+1}/{n_batches} complete")

        # Combine all batches
        observations = np.vstack(all_obs)
        actions = np.vstack(all_actions)
        rewards = np.concatenate(all_rewards)

        # Compute statistics
        altitude_data = -observations[:, StateIndex.Z]
        pitch_data = np.rad2deg(observations[:, StateIndex.THETA])
        airspeed_data = observations[:, StateIndex.U]

        stats = {
            'config_id': config_id,
            'name': init_params['name'],
            'n_timesteps': len(observations),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'altitude_mean': float(np.mean(altitude_data)),
            'altitude_min': float(np.min(altitude_data)),
            'altitude_max': float(np.max(altitude_data)),
            'altitude_negative_pct': float(np.mean(altitude_data < 0) * 100),
            'pitch_mean': float(np.mean(pitch_data)),
            'pitch_std': float(np.std(pitch_data)),
            'pitch_flips': int(np.sum(np.abs(pitch_data) > 170)),
            'airspeed_mean': float(np.mean(airspeed_data)),
            'airspeed_min': float(np.min(airspeed_data)),
        }

        # Save dataset
        output_file = f"checkpoints/bc_dataset_config{config_id}.npz"
        np.savez_compressed(
            output_file,
            observations=observations,
            actions=actions,
            rewards=rewards,
            init_params=init_params
        )

        stats['file'] = output_file
        stats['file_size_mb'] = os.path.getsize(output_file) / (1024 * 1024)

        print(f"\nConfig {config_id}: COMPLETE")
        print(f"  Mean reward: {stats['mean_reward']:.3f}")
        print(f"  Altitude: {stats['altitude_mean']:.1f}m (range: {stats['altitude_min']:.1f} to {stats['altitude_max']:.1f}m)")
        print(f"  Underground: {stats['altitude_negative_pct']:.1f}%")
        print(f"  Pitch flips: {stats['pitch_flips']}")
        print(f"  File: {output_file} ({stats['file_size_mb']:.1f} MB)")

        if result_queue:
            result_queue.put(stats)

        return stats

    except Exception as e:
        error_stats = {
            'config_id': config_id,
            'name': init_params['name'],
            'error': str(e)
        }
        if result_queue:
            result_queue.put(error_stats)
        print(f"\nConfig {config_id}: ERROR - {e}")
        return error_stats


def main():
    print("="*70)
    print("  PARALLEL BC DATASET GENERATION")
    print("  Testing Multiple Initial Condition Sets")
    print("="*70)

    # Define initial condition configurations to test
    configs = [
        {
            'name': 'Conservative (Safe)',
            'alt_min': 30.0, 'alt_max': 60.0,
            'speed_min': 18.0, 'speed_max': 28.0,
            'pitch_min': -5.0, 'pitch_max': 5.0,
            'altitude_Kp': 0.15, 'altitude_Kd': 0.5,
        },
        {
            'name': 'Moderate',
            'alt_min': 20.0, 'alt_max': 50.0,
            'speed_min': 15.0, 'speed_max': 30.0,
            'pitch_min': -10.0, 'pitch_max': 10.0,
            'altitude_Kp': 0.15, 'altitude_Kd': 0.5,
        },
        {
            'name': 'Original (Aggressive)',
            'alt_min': 5.0, 'alt_max': 60.0,
            'speed_min': 11.2, 'speed_max': 30.0,
            'pitch_min': -15.0, 'pitch_max': 15.0,
            'altitude_Kp': 0.15, 'altitude_Kd': 0.5,
        },
        {
            'name': 'High Altitude',
            'alt_min': 40.0, 'alt_max': 80.0,
            'speed_min': 20.0, 'speed_max': 30.0,
            'pitch_min': -3.0, 'pitch_max': 3.0,
            'altitude_Kp': 0.15, 'altitude_Kd': 0.5,
        },
        {
            'name': 'Lower Gains',
            'alt_min': 30.0, 'alt_max': 60.0,
            'speed_min': 18.0, 'speed_max': 28.0,
            'pitch_min': -5.0, 'pitch_max': 5.0,
            'altitude_Kp': 0.10, 'altitude_Kd': 0.3,
        },
    ]

    print(f"\nTesting {len(configs)} configurations:")
    for i, cfg in enumerate(configs):
        print(f"\n  Config {i}: {cfg['name']}")
        print(f"    Altitude: {cfg['alt_min']}-{cfg['alt_max']} m")
        print(f"    Speed: {cfg['speed_min']}-{cfg['speed_max']} m/s")
        print(f"    Pitch: {cfg['pitch_min']:.1f} to {cfg['pitch_max']:.1f}°")
        print(f"    Gains: Kp={cfg['altitude_Kp']}, Kd={cfg['altitude_Kd']}")

    # Create result queue for inter-process communication
    result_queue = Queue()

    # Launch parallel processes
    print(f"\n{'='*70}")
    print("  LAUNCHING PARALLEL GENERATION")
    print("="*70)

    n_episodes_per_config = 100  # Smaller for faster testing
    episode_length = 500

    processes = []
    for i, cfg in enumerate(configs):
        p = Process(
            target=generate_dataset_config,
            args=(i, cfg, n_episodes_per_config, episode_length, result_queue)
        )
        p.start()
        processes.append(p)
        time.sleep(1)  # Stagger starts slightly

    # Wait for all to complete
    for p in processes:
        p.join()

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Sort by config_id
    results.sort(key=lambda x: x['config_id'])

    # Print comparison table
    print(f"\n{'='*70}")
    print("  RESULTS COMPARISON")
    print("="*70)

    print(f"\n{'ID':<4} {'Name':<20} {'Reward':<12} {'Alt':<10} {'Underground':<12} {'Flips':<8}")
    print("-"*70)

    for r in results:
        if 'error' in r:
            print(f"{r['config_id']:<4} {r['name']:<20} ERROR: {r['error']}")
        else:
            print(f"{r['config_id']:<4} {r['name']:<20} "
                  f"{r['mean_reward']:>6.3f} ± {r['std_reward']:>4.2f}  "
                  f"{r['altitude_mean']:>6.1f}m  "
                  f"{r['altitude_negative_pct']:>6.1f}%     "
                  f"{r['pitch_flips']:>6d}")

    # Find best configuration
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        # Best = highest reward, lowest underground %, fewest flips
        best = max(valid_results, key=lambda x: (
            x['mean_reward'],
            -x['altitude_negative_pct'],
            -x['pitch_flips']
        ))

        print(f"\n{'='*70}")
        print("  BEST CONFIGURATION")
        print("="*70)
        print(f"\nConfig {best['config_id']}: {best['name']}")
        print(f"  Mean reward: {best['mean_reward']:.3f} ± {best['std_reward']:.3f}")
        print(f"  Altitude: {best['altitude_mean']:.1f}m (min: {best['altitude_min']:.1f}m)")
        print(f"  Underground: {best['altitude_negative_pct']:.1f}%")
        print(f"  Pitch flips: {best['pitch_flips']}")
        print(f"  File: {best['file']}")

        # Copy best to standard location
        import shutil
        shutil.copy(best['file'], 'checkpoints/bc_dataset_udaan_best.npz')
        print(f"\n  ✓ Copied to: checkpoints/bc_dataset_udaan_best.npz")

        print(f"\n{'='*70}")
        print("  NEXT STEPS")
        print("="*70)
        print(f"\nTrain PPO with best dataset:")
        print(f"  python scripts/train_ppo_flight.py --device cuda")
        print(f"\nOr visualize best dataset:")
        print(f"  python scripts/visualize_bc_dataset.py --dataset {best['file']}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
