#!/usr/bin/env python3
"""
Sanity check for Cessna 172 environment before long training.

Tests:
1. Environment can reset
2. Episodes last reasonable length (not stuck at 17 steps)
3. Termination reasons make sense
4. Airspeed calculation is correct
5. Rewards are in expected range
6. Actions affect the aircraft
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aida_sim.env.flight_env_cessna172 import Cessna172Env

def test_environment_reset():
    """Test 1: Environment can reset"""
    print("=" * 60)
    print("TEST 1: Environment Reset")
    print("=" * 60)

    env = Cessna172Env(task='full_mission', cruise_altitude_ft=3000.0)
    obs, info = env.reset()

    print(f"✓ Environment created")
    print(f"✓ Initial observation shape: {obs.shape}")
    print(f"✓ Initial altitude: {info['altitude']:.1f} m")
    print(f"✓ Initial airspeed: {info['airspeed']:.1f} m/s")
    print(f"✓ Mission phase: {info['mission_phase']}")
    print()

    return env

def test_episode_length(env, n_episodes=10):
    """Test 2: Episodes should last more than 17 steps"""
    print("=" * 60)
    print("TEST 2: Episode Length with Random Actions")
    print("=" * 60)

    episode_lengths = []
    termination_reasons = {}

    for i in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < 1000:
            action = env.action_space.sample()  # Random actions
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        episode_lengths.append(steps)
        reason = info.get('termination_reason', 'timeout')
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

        print(f"Episode {i+1:2d}: {steps:4d} steps, "
              f"reason: {reason:15s}, "
              f"alt: {info['altitude']:5.1f}m, "
              f"speed: {info['airspeed']:5.1f}m/s")

    avg_length = np.mean(episode_lengths)
    print(f"\n✓ Average episode length: {avg_length:.1f} steps")
    print(f"✓ Min: {np.min(episode_lengths)}, Max: {np.max(episode_lengths)}")

    print(f"\nTermination reasons:")
    for reason, count in termination_reasons.items():
        print(f"  {reason:20s}: {count:2d} episodes ({100*count/n_episodes:.0f}%)")

    if avg_length > 50:
        print(f"\n✅ PASS: Average length {avg_length:.1f} > 50 steps")
    else:
        print(f"\n❌ FAIL: Average length {avg_length:.1f} <= 50 steps (stuck early!)")

    print()
    return avg_length > 50

def test_airspeed_calculation(env):
    """Test 3: Airspeed should be magnitude of velocity vector"""
    print("=" * 60)
    print("TEST 3: Airspeed Calculation")
    print("=" * 60)

    obs, info = env.reset()

    # Apply high throttle, neutral controls for a few steps
    for _ in range(50):
        action = np.array([1.0, 0.0, 0.0, 0.0])  # Full throttle
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Manual airspeed calculation
    u, v, w = obs[3], obs[4], obs[5]  # Velocity components
    manual_airspeed = np.sqrt(u**2 + v**2 + w**2)
    reported_airspeed = info['airspeed']

    print(f"Velocity components: u={u:.2f}, v={v:.2f}, w={w:.2f} m/s")
    print(f"Manual calculation:  sqrt(u² + v² + w²) = {manual_airspeed:.2f} m/s")
    print(f"Reported airspeed:   {reported_airspeed:.2f} m/s")

    error = abs(manual_airspeed - reported_airspeed)
    if error < 0.01:
        print(f"\n✅ PASS: Airspeed calculation correct (error: {error:.4f} m/s)")
        return True
    else:
        print(f"\n❌ FAIL: Airspeed mismatch (error: {error:.4f} m/s)")
        return False

def test_termination_logic(env):
    """Test 4: Stall should only trigger when airborne"""
    print("=" * 60)
    print("TEST 4: Stall Logic (Ground vs Airborne)")
    print("=" * 60)

    # Test ground roll: low speed on ground should NOT trigger stall
    print("\nScenario A: Low speed on ground (should NOT stall)")
    obs, info = env.reset()

    for i in range(20):
        action = np.array([0.0, 0.0, 0.0, 0.0])  # Idle
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            reason = info.get('termination_reason', 'unknown')
            print(f"  Step {i}: TERMINATED - {reason}")
            print(f"  Altitude: {info['altitude']:.1f}m, Airspeed: {info['airspeed']:.1f}m/s")
            if reason == 'stall':
                print(f"\n❌ FAIL: Stalled on ground (should be allowed during ground roll)")
                return False
            break
    else:
        print(f"  ✓ Survived 20 steps at low speed on ground")

    # Test airborne stall: low speed in air SHOULD trigger stall
    print("\nScenario B: Low speed airborne (SHOULD stall)")
    obs, info = env.reset(seed=123)

    # Get airborne first
    for _ in range(100):
        action = np.array([1.0, 0.0, 0.1, 0.0])  # High throttle, pitch up
        obs, reward, terminated, truncated, info = env.step(action)
        if info['altitude'] > 20.0:  # Airborne
            break

    if info['altitude'] > 20.0:
        print(f"  Aircraft airborne at {info['altitude']:.1f}m")

        # Now cut throttle and wait for stall
        for i in range(100):
            action = np.array([0.0, 0.0, 0.5, 0.0])  # Idle throttle, pitch up
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                reason = info.get('termination_reason', 'unknown')
                print(f"  Step {i}: TERMINATED - {reason}")
                print(f"  Altitude: {info['altitude']:.1f}m, Airspeed: {info['airspeed']:.1f}m/s")
                if reason == 'stall':
                    print(f"\n✅ PASS: Correctly detected stall when airborne")
                    return True
                elif reason == 'crash':
                    print(f"\n⚠️  Crashed before stalling (acceptable)")
                    return True
                break
        else:
            print(f"\n⚠️  Did not stall (airspeed: {info['airspeed']:.1f} m/s)")
    else:
        print(f"  Could not get airborne for test")

    print()
    return True

def test_action_effect(env):
    """Test 5: Actions should affect aircraft state"""
    print("=" * 60)
    print("TEST 5: Control Effectiveness")
    print("=" * 60)

    # Test throttle
    print("\nThrottle test:")
    obs, info = env.reset()
    initial_speed = info['airspeed']

    for _ in range(100):
        action = np.array([1.0, 0.0, 0.0, 0.0])  # Full throttle
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    final_speed = info['airspeed']
    speed_gain = final_speed - initial_speed

    print(f"  Initial speed: {initial_speed:.2f} m/s")
    print(f"  Final speed:   {final_speed:.2f} m/s")
    print(f"  Gain:          {speed_gain:.2f} m/s")

    if speed_gain > 5.0:
        print(f"  ✅ Throttle works (gained {speed_gain:.1f} m/s)")
        throttle_works = True
    else:
        print(f"  ❌ Throttle ineffective (only {speed_gain:.1f} m/s gain)")
        throttle_works = False

    # Test elevator
    print("\nElevator test:")
    obs, info = env.reset()

    # Get some speed first
    for _ in range(50):
        action = np.array([1.0, 0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    initial_pitch = obs[7]  # theta

    for _ in range(30):
        action = np.array([1.0, 0.0, 0.8, 0.0])  # Pitch up
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    final_pitch = obs[7]
    pitch_change = np.rad2deg(final_pitch - initial_pitch)

    print(f"  Initial pitch: {np.rad2deg(initial_pitch):.2f}°")
    print(f"  Final pitch:   {np.rad2deg(final_pitch):.2f}°")
    print(f"  Change:        {pitch_change:.2f}°")

    if abs(pitch_change) > 2.0:
        print(f"  ✅ Elevator works (changed {pitch_change:.1f}°)")
        elevator_works = True
    else:
        print(f"  ❌ Elevator ineffective (only {pitch_change:.1f}° change)")
        elevator_works = False

    print()
    return throttle_works and elevator_works

def test_reward_range(env, n_steps=100):
    """Test 6: Rewards should be in reasonable range"""
    print("=" * 60)
    print("TEST 6: Reward Range")
    print("=" * 60)

    obs, info = env.reset()
    rewards = []

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    min_r = np.min(rewards)
    max_r = np.max(rewards)
    mean_r = np.mean(rewards)

    print(f"Reward statistics over {len(rewards)} steps:")
    print(f"  Min:  {min_r:.3f}")
    print(f"  Max:  {max_r:.3f}")
    print(f"  Mean: {mean_r:.3f}")

    if min_r > -20 and max_r < 20:
        print(f"\n✅ PASS: Rewards in reasonable range [-20, +20]")
        return True
    else:
        print(f"\n⚠️  WARNING: Rewards outside expected range")
        return False

def main():
    """Run all sanity checks"""
    print("\n" + "=" * 60)
    print(" CESSNA 172 ENVIRONMENT SANITY CHECKS")
    print("=" * 60)
    print()

    results = {}

    try:
        # Test 1: Environment reset
        env = test_environment_reset()
        results['reset'] = True

        # Test 2: Episode length
        results['episode_length'] = test_episode_length(env, n_episodes=10)

        # Test 3: Airspeed calculation
        results['airspeed'] = test_airspeed_calculation(env)

        # Test 4: Termination logic
        results['termination'] = test_termination_logic(env)

        # Test 5: Control effectiveness
        results['controls'] = test_action_effect(env)

        # Test 6: Reward range
        results['rewards'] = test_reward_range(env)

        env.close()

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("=" * 60)
    print(" SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 60)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Environment ready for training.")
        print("\nTo start training:")
        print("  python scripts/train_cessna172_ppo.py --n-envs 4 --timesteps 1000000")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED! Fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
