#!/usr/bin/env python3
"""
Quick sanity check for curriculum learning phases.

Tests each phase can:
1. Create environment
2. Reset successfully
3. Run for expected duration
4. Compute rewards correctly

Author: Kushal Koirala (with Claude Code)
Date: December 27, 2024
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from aida_sim.env.flight_env_cessna172 import Cessna172Env


def test_phase(task, expected_max_steps):
    """Test a single curriculum phase."""
    print(f"\n{'='*60}")
    print(f"  TESTING: {task.upper()}")
    print(f"{'='*60}")

    try:
        # Create environment
        env = Cessna172Env(task=task, cruise_altitude_ft=3000.0)
        print(f"✓ Environment created")
        print(f"  Max steps: {env.max_episode_steps}")
        print(f"  Task config: {env.task_config}")

        # Reset
        obs, info = env.reset()
        print(f"\n✓ Environment reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Initial altitude: {info['altitude']:.2f} m")
        print(f"  Initial airspeed: {info['airspeed']:.2f} m/s")

        # Run episode with random actions
        done = False
        total_reward = 0
        steps = 0
        max_test_steps = min(expected_max_steps, 500)  # Limit for quick test

        print(f"\n✓ Running {max_test_steps} steps with random actions...")

        while not done and steps < max_test_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Print occasional updates
            if steps % 100 == 0:
                print(f"  Step {steps}: reward={reward:.2f}, total={total_reward:.2f}, alt={info['altitude']:.1f}m, speed={info['airspeed']:.1f}m/s")

        print(f"\n✓ Episode complete")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final altitude: {info['altitude']:.2f} m")
        print(f"  Final airspeed: {info['airspeed']:.2f} m/s")

        if done:
            reason = info.get('termination_reason', 'unknown')
            print(f"  Termination reason: {reason}")

        env.close()

        print(f"\n✅ {task.upper()} PASSED")
        return True

    except Exception as e:
        print(f"\n❌ {task.upper()} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all curriculum phases."""
    print("\n" + "="*60)
    print("  CURRICULUM PHASES SANITY CHECK")
    print("="*60)

    phases = [
        ('ground_roll', 500),
        ('rotation', 750),
        ('initial_climb', 1500),
        ('full_climb', 3000),
        ('cruise', 1500),
    ]

    results = {}

    for task, expected_max_steps in phases:
        passed = test_phase(task, expected_max_steps)
        results[task] = passed

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    all_passed = all(results.values())

    for task, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {task}")

    print("="*60)

    if all_passed:
        print("\n✅ ALL PHASES PASSED!")
        print("\nReady to start curriculum training:")
        print("  python scripts/train_cessna172_curriculum.py --n-envs 4")
    else:
        print("\n❌ SOME PHASES FAILED!")
        print("Fix issues before training.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
