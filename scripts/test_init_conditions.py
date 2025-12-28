#!/usr/bin/env python3
"""
Test Multiple Initial Condition Sets Sequentially

Faster, simpler version that tests configs one at a time.
No multiprocessing complications.

Author: Kushal Koirala (with Claude Code)
Date: December 27, 2024
"""

import numpy as np
import sys
from pathlib import Path

# Add gpu-flight-dynamics to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))

# Import original BC dataset generation script
from generate_bc_dataset_gpu import (
    main as generate_bc_main,
    config_to_params,
    ClassicalFlightController,
    FlightSimulator,
    generate_random_initial_conditions,
    StateIndex,
    ControlIndex,
    STATE_DIM,
    CONTROL_DIM
)
from aircraft_database import get_aircraft

def test_init_config(config_name, alt_range, speed_range, pitch_range, n_episodes=100):
    """Test a single initial condition configuration"""
    print(f"\n{'='*70}")
    print(f"  Testing: {config_name}")
    print("="*70)
    print(f"  Altitude: {alt_range[0]:.1f}-{alt_range[1]:.1f} m")
    print(f"  Speed: {speed_range[0]:.1f}-{speed_range[1]:.1f} m/s")
    print(f"  Pitch: {pitch_range[0]:.1f} to {pitch_range[1]:.1f}°")

    # Get aircraft
    aircraft = get_aircraft('udaan')
    params = config_to_params(aircraft)

    # Create simulator
    batch_size = 50
    n_batches = n_episodes // batch_size
    episode_length = 500
    dt = 0.02

    sim = FlightSimulator(n_instances=batch_size, params=params, dt=dt)
    controller = ClassicalFlightController()

    # Storage
    all_altitudes = []
    all_pitches = []
    all_rewards = []

    for batch_idx in range(n_batches):
        # Custom initial states
        altitude_init = np.random.uniform(alt_range[0], alt_range[1], batch_size)
        airspeed_init = np.random.uniform(speed_range[0], speed_range[1], batch_size)
        pitch_init = np.deg2rad(np.random.uniform(pitch_range[0], pitch_range[1], batch_size))

        # Build initial state array
        initial_state = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
        initial_state[:, StateIndex.Z] = -altitude_init  # NED frame
        initial_state[:, StateIndex.U] = airspeed_init
        initial_state[:, StateIndex.THETA] = pitch_init

        sim.reset(initial_state=initial_state)

        # Rollout
        for step in range(episode_length):
            states = sim.get_states()

            # Convert from CuPy if needed
            try:
                states_np = states.get() if hasattr(states, 'get') else states
            except:
                states_np = np.array(states)

            # Compute actions
            actions = np.zeros((batch_size, CONTROL_DIM), dtype=np.float32)
            for i in range(batch_size):
                actions[i] = controller.compute_controls(states_np[i])

            # Record stats
            altitude = -states_np[:, StateIndex.Z]
            pitch = np.rad2deg(states_np[:, StateIndex.THETA])

            all_altitudes.extend(altitude)
            all_pitches.extend(pitch)

            # Simple reward
            altitude_error = np.abs(altitude - 30.0)
            rewards = 1.0 - 0.1 * altitude_error
            all_rewards.extend(rewards)

            # Step
            sim.set_controls(actions)
            sim.step()

    # Compute statistics
    all_altitudes = np.array(all_altitudes)
    all_pitches = np.array(all_pitches)
    all_rewards = np.array(all_rewards)

    stats = {
        'name': config_name,
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'altitude_mean': float(np.mean(all_altitudes)),
        'altitude_min': float(np.min(all_altitudes)),
        'altitude_max': float(np.max(all_altitudes)),
        'underground_pct': float(np.mean(all_altitudes < 0) * 100),
        'pitch_mean': float(np.mean(all_pitches)),
        'pitch_std': float(np.std(all_pitches)),
        'pitch_flips': int(np.sum(np.abs(all_pitches) > 170)),
    }

    print(f"\n  Results:")
    print(f"    Mean reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"    Altitude: {stats['altitude_mean']:.1f}m (range: {stats['altitude_min']:.1f} to {stats['altitude_max']:.1f}m)")
    print(f"    Underground: {stats['underground_pct']:.1f}%")
    print(f"    Pitch flips: {stats['pitch_flips']}")

    return stats


def main():
    print("="*70)
    print("  BC DATASET INITIAL CONDITION TESTING")
    print("  Sequential Testing with Updated Inertias")
    print("="*70)

    # Define test configurations
    configs = [
        ("Conservative (Safe)", (30.0, 60.0), (18.0, 28.0), (-5.0, 5.0)),
        ("Moderate", (20.0, 50.0), (15.0, 30.0), (-10.0, 10.0)),
        ("Original (Aggressive)", (5.0, 60.0), (11.2, 30.0), (-15.0, 15.0)),
        ("High Altitude", (40.0, 80.0), (20.0, 30.0), (-3.0, 3.0)),
        ("Very Conservative", (40.0, 60.0), (20.0, 25.0), (-3.0, 3.0)),
    ]

    results = []
    for config_name, alt_range, speed_range, pitch_range in configs:
        stats = test_init_config(config_name, alt_range, speed_range, pitch_range, n_episodes=100)
        results.append(stats)

    # Print comparison table
    print(f"\n{'='*70}")
    print("  RESULTS COMPARISON")
    print("="*70)

    print(f"\n{'Config':<25} {'Reward':<12} {'Alt':<10} {'Underground':<12} {'Flips':<8}")
    print("-"*70)

    for r in results:
        print(f"{r['name']:<25} "
              f"{r['mean_reward']:>6.3f} ± {r['std_reward']:>4.2f}  "
              f"{r['altitude_mean']:>6.1f}m  "
              f"{r['underground_pct']:>6.1f}%     "
              f"{r['pitch_flips']:>6d}")

    # Find best
    best = max(results, key=lambda x: (
        x['mean_reward'],
        -x['underground_pct'],
        -x['pitch_flips']
    ))

    print(f"\n{'='*70}")
    print("  BEST CONFIGURATION")
    print("="*70)
    print(f"\n  {best['name']}")
    print(f"    Mean reward: {best['mean_reward']:.3f}")
    print(f"    Underground: {best['underground_pct']:.1f}%")
    print(f"    Pitch flips: {best['pitch_flips']}")

    print(f"\n{'='*70}")
    print("  RECOMMENDATION")
    print("="*70)

    if best['underground_pct'] < 5 and best['pitch_flips'] < 100:
        print(f"\n  ✓ Good configuration found: {best['name']}")
        print(f"\n  Regenerate full BC dataset with these params and train PPO:")
        print(f"    python scripts/train_ppo_flight.py --device cuda")
    else:
        print(f"\n  ⚠ Best config still has issues:")
        print(f"    - Underground: {best['underground_pct']:.1f}%")
        print(f"    - Pitch flips: {best['pitch_flips']}")
        print(f"\n  Recommendation: Skip BC, train PPO from scratch")
        print(f"    python scripts/train_ppo_flight.py --device cuda")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
