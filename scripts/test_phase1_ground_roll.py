#!/usr/bin/env python3
"""
Test Phase 1 Ground Roll Model

Runs the trained Phase 1 model and shows detailed performance metrics
to verify the agent can successfully accelerate down the runway.

Author: Kushal Koirala (with Claude Code)
Date: December 27, 2024
"""

import sys
import os
from pathlib import Path
import numpy as np
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from aida_sim.env.flight_env_cessna172 import Cessna172Env


def test_ground_roll(model_path, n_episodes=10, render_details=True):
    """
    Test the ground roll phase model.

    Args:
        model_path: Path to trained model
        n_episodes: Number of test episodes
        render_details: Print detailed step-by-step info
    """
    print("\n" + "="*60)
    print("  PHASE 1: GROUND ROLL SIMULATION")
    print("="*60)
    print(f"\nModel: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("="*60 + "\n")

    # Create environment
    env = Cessna172Env(task='ground_roll', cruise_altitude_ft=3000.0)

    # Load model
    print(f"Loading model...")
    model = PPO.load(model_path)
    print(f"✓ Model loaded\n")

    # Statistics
    episode_rewards = []
    episode_lengths = []
    max_airspeeds = []
    max_altitudes = []
    lateral_deviations = []
    heading_errors = []
    termination_reasons = {}

    for ep in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"  EPISODE {ep+1}/{n_episodes}")
        print(f"{'='*60}\n")

        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        max_airspeed = 0
        max_altitude = 0
        max_lateral_dev = 0
        max_heading_err = 0

        while not done and steps < 500:
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            # Track metrics
            airspeed = info['airspeed']
            altitude = info['altitude']
            max_airspeed = max(max_airspeed, airspeed)
            max_altitude = max(max_altitude, altitude)

            # Calculate position metrics
            x, y = obs[0], obs[1]
            psi = obs[8]  # Heading

            # Runway centerline deviation (runway is along Y axis)
            lateral_dev = abs(x)
            max_lateral_dev = max(max_lateral_dev, lateral_dev)

            # Heading error (runway heading is 0)
            heading_err = abs(psi)
            max_heading_err = max(max_heading_err, heading_err)

            # Print periodic updates
            if render_details and (steps % 50 == 0 or done):
                print(f"Step {steps:3d}: "
                      f"Speed={airspeed:5.1f} m/s ({airspeed*1.944:5.1f} KIAS), "
                      f"Alt={altitude:5.2f}m, "
                      f"LatDev={lateral_dev:5.1f}m, "
                      f"Heading={np.rad2deg(psi):5.1f}°, "
                      f"Reward={reward:6.1f}")

        # Record episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        max_airspeeds.append(max_airspeed)
        max_altitudes.append(max_altitude)
        lateral_deviations.append(max_lateral_dev)
        heading_errors.append(np.rad2deg(max_heading_err))

        reason = info.get('termination_reason', 'max_steps')
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

        # Episode summary
        print(f"\n{'-'*60}")
        print(f"Episode {ep+1} Summary:")
        print(f"  Steps:           {steps}")
        print(f"  Total Reward:    {total_reward:.1f}")
        print(f"  Max Airspeed:    {max_airspeed:.1f} m/s ({max_airspeed*1.944:.1f} KIAS)")
        print(f"  Max Altitude:    {max_altitude:.2f} m")
        print(f"  Max Lateral Dev: {max_lateral_dev:.1f} m")
        print(f"  Max Heading Err: {np.rad2deg(max_heading_err):.1f}°")
        print(f"  Termination:     {reason}")

        # Success check
        rotation_speed = 28.0  # 55 KIAS
        if max_airspeed >= rotation_speed:
            print(f"  ✅ SUCCESS: Reached rotation speed ({max_airspeed:.1f} >= {rotation_speed} m/s)")
        else:
            print(f"  ❌ FAILED: Below rotation speed ({max_airspeed:.1f} < {rotation_speed} m/s)")

        print(f"{'-'*60}")

    env.close()

    # Overall statistics
    print(f"\n{'='*60}")
    print(f"  OVERALL STATISTICS ({n_episodes} episodes)")
    print(f"{'='*60}\n")

    print(f"Episode Length:")
    print(f"  Mean:   {np.mean(episode_lengths):.1f} steps")
    print(f"  Std:    {np.std(episode_lengths):.1f} steps")
    print(f"  Range:  [{np.min(episode_lengths)}, {np.max(episode_lengths)}]")

    print(f"\nEpisode Reward:")
    print(f"  Mean:   {np.mean(episode_rewards):.1f}")
    print(f"  Std:    {np.std(episode_rewards):.1f}")
    print(f"  Range:  [{np.min(episode_rewards):.1f}, {np.max(episode_rewards):.1f}]")

    print(f"\nMax Airspeed:")
    print(f"  Mean:   {np.mean(max_airspeeds):.1f} m/s ({np.mean(max_airspeeds)*1.944:.1f} KIAS)")
    print(f"  Std:    {np.std(max_airspeeds):.1f} m/s")
    print(f"  Range:  [{np.min(max_airspeeds):.1f}, {np.max(max_airspeeds):.1f}] m/s")

    print(f"\nRunway Tracking:")
    print(f"  Lateral Dev (mean): {np.mean(lateral_deviations):.1f} m")
    print(f"  Heading Err (mean): {np.mean(heading_errors):.1f}°")

    print(f"\nTermination Reasons:")
    for reason, count in sorted(termination_reasons.items(), key=lambda x: -x[1]):
        pct = 100 * count / n_episodes
        print(f"  {reason:20s}: {count:2d} ({pct:5.1f}%)")

    # Success rate
    rotation_speed = 28.0  # 55 KIAS
    successes = sum(1 for speed in max_airspeeds if speed >= rotation_speed)
    success_rate = successes / n_episodes

    print(f"\n{'='*60}")
    print(f"  SUCCESS RATE")
    print(f"{'='*60}")
    print(f"\nReached Rotation Speed (≥{rotation_speed} m/s / 55 KIAS):")
    print(f"  {successes}/{n_episodes} episodes ({success_rate*100:.1f}%)")

    if success_rate >= 0.9:
        print(f"\n✅ EXCELLENT: Success rate {success_rate*100:.1f}% ≥ 90%")
        print(f"   Ready for Phase 2 (Rotation)!")
    elif success_rate >= 0.7:
        print(f"\n⚠️  GOOD: Success rate {success_rate*100:.1f}% ≥ 70%")
        print(f"   Could proceed to Phase 2, but may want more training")
    else:
        print(f"\n❌ INSUFFICIENT: Success rate {success_rate*100:.1f}% < 70%")
        print(f"   Need more training before Phase 2")

    print(f"\n{'='*60}\n")

    return {
        'success_rate': success_rate,
        'mean_reward': np.mean(episode_rewards),
        'mean_airspeed': np.mean(max_airspeeds),
        'mean_length': np.mean(episode_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description='Test Phase 1 Ground Roll Model')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model (default: auto-find latest)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes (default: 10)')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable step-by-step output')
    args = parser.parse_args()

    # Auto-find model if not specified
    if args.model is None:
        # Try to find the best model
        best_model = "checkpoints/cessna172_curriculum/phase1_ground_roll/best_model.zip"
        if os.path.exists(best_model):
            model_path = best_model
            print(f"Using best model: {model_path}")
        else:
            # Try final model
            final_model = "checkpoints/cessna172_curriculum/phase1_ground_roll/phase1_ground_roll_ppo_final.zip"
            if os.path.exists(final_model):
                model_path = final_model
                print(f"Using final model: {model_path}")
            else:
                print(f"❌ No trained model found!")
                print(f"   Expected: {best_model}")
                print(f"   Or: {final_model}")
                print(f"\nPlease wait for Phase 1 training to complete.")
                return 1
    else:
        model_path = args.model

    # Test the model
    results = test_ground_roll(
        model_path=model_path,
        n_episodes=args.episodes,
        render_details=not args.quiet
    )

    return 0 if results['success_rate'] >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
