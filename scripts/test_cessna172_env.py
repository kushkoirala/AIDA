#!/usr/bin/env python3
"""Test Cessna 172 environment initialization"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from aida_sim.env.flight_env_cessna172 import Cessna172Env

print("="*70)
print("  CESSNA 172 ENVIRONMENT TEST")
print("="*70)

print("\nCreating Cessna 172 environment...")
env = Cessna172Env(task="full_mission", cruise_altitude_ft=3000.0)
print("✓ Environment created successfully")

print("\nEnvironment details:")
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: {env.action_space.shape}")
print(f"  Max episode steps: {env.max_episode_steps}")
print(f"  Task: {env.task}")
print(f"  Cruise altitude: {env.cruise_altitude:.1f} m ({env.cruise_altitude / 0.3048:.0f} ft)")
print(f"  Runway: {env.runway_length:.0f}m x {env.runway_width:.0f}m")
print(f"  Flight bounds: {env.flight_bounds[0]:.0f}m x {env.flight_bounds[1]:.0f}m x {env.flight_bounds[2]:.0f}m")

print("\nResetting environment...")
obs, info = env.reset(seed=42)
print("✓ Reset successful")
print(f"  Initial altitude: {info['altitude']:.1f} m")
print(f"  Initial airspeed: {info['airspeed']:.1f} m/s")
print(f"  Mission phase: {info['mission_phase']}")

print("\nTesting 10 steps with high throttle...")
for i in range(10):
    action = np.array([0.8, 0.0, 0.0, 0.0])  # High throttle, no control inputs
    obs, reward, terminated, truncated, info = env.step(action)

    alt = info['altitude']
    spd = info['airspeed']
    phase = info['mission_phase']

    print(f"  Step {i+1:2d}: reward={reward:6.3f}, alt={alt:6.1f}m, spd={spd:5.1f}m/s, phase={phase}, term={terminated}")

    if terminated or truncated:
        print(f"    Episode ended: {info.get('termination_reason', 'max_steps')}")
        break

print("\n✓ All environment tests passed!")
print("\nNext steps:")
print("  1. Train Cessna 172 PPO:")
print("     python scripts/train_cessna172_ppo.py --device cuda --timesteps 1000000")
print("  2. Test different tasks:")
print("     python scripts/test_cessna172_env.py  (this script with different tasks)")
print()
