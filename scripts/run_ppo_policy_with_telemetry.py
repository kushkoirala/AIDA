#!/usr/bin/env python3
"""
Run Neural Network (PPO) Policy with 3D Telemetry Viewer

NEURAL NETWORK CONTROLLER - Separate from classical controller
=============================================================
This script runs a trained PPO neural network policy for autonomous flight.
The NN learns from expert demonstrations (BC) and reinforcement learning (PPO).

For the classical controller (rule-based, PID), use:
    python scripts/triangle_controller.py
    python scripts/run_xc_sn65_khut.py

This NN controller is experimental and under development.
The classical controller is the reliable baseline.

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import asyncio
import threading
import time
import sys
import math
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from aida_sim.env.flight_env_cessna172 import Cessna172Env, StateIndex
from aida_sim.io.telemetry import telemetry_server

TELEMETRY_UNITS = "metric"

shared_state = {
    "position": [0, 0, 0],
    "quaternion": [1, 0, 0, 0],
    "velocity": [0, 0, 0],
    "rates": [0, 0, 0],
    "surfaces": [0, 0, 0],
    "throttle": 0.0,
    "soc": 1.0,
    "voltage": 12.0,
    "load_factor": 1.0,
    "units": TELEMETRY_UNITS,
    "model": "cessna172",
    "phase": "NN_POLICY",
}


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]


def update_telemetry(obs, action, phase_name):
    x, y, z = obs[StateIndex.X], obs[StateIndex.Y], obs[StateIndex.Z]
    u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
    phi, theta, psi = obs[StateIndex.PHI], obs[StateIndex.THETA], obs[StateIndex.PSI]
    p, q, r = obs[StateIndex.P], obs[StateIndex.Q], obs[StateIndex.R]

    viewer_x = x + 500.0
    viewer_y = y
    viewer_z = -z

    shared_state["position"] = [float(viewer_x), float(viewer_y), float(viewer_z)]

    quat = euler_to_quaternion(phi, theta, psi)
    shared_state["quaternion"] = [float(q) for q in quat]
    shared_state["velocity"] = [float(u), float(v), float(w)]
    shared_state["rates"] = [float(p), float(q), float(r)]
    shared_state["phase"] = phase_name

    throttle_norm = (action[0] + 1.0) / 2.0
    shared_state["throttle"] = float(np.clip(throttle_norm, 0.0, 1.0))
    shared_state["surfaces"] = [
        float(np.clip(action[1], -1.0, 1.0)),
        float(np.clip(action[2], -1.0, 1.0)),
        float(np.clip(action[3], -1.0, 1.0))
    ]


def run_simulation(model_path, task='full_mission', dt=0.02, max_steps=3000, deterministic=True):
    print("\n" + "="*70)
    print("  NEURAL NETWORK (PPO) CONTROLLER - TELEMETRY VIEWER")
    print("="*70)
    print("  NOTE: This is the NN controller (experimental)")
    print("        For classical controller: python scripts/run_xc_sn65_khut.py")
    print("="*70)
    print(f"\nModel: {model_path}")
    print(f"Task: {task}")
    print("\nOpen browser to http://localhost:8000")
    print("Press Ctrl+C to stop\n")

    # Load PPO policy
    print("Loading PPO policy...")
    model = PPO.load(model_path)
    print(f"✓ Policy loaded successfully\n")

    # Create environment
    env = Cessna172Env(task=task, dt=dt, cruise_altitude_ft=3000.0)
    print(f"Environment created:")
    print(f"  Task: {task}")
    print(f"  V_rotate: {env.V_ROTATE:.1f} m/s ({env.V_ROTATE*1.944:.1f} KIAS)")
    print(f"  V_climb: {env.V_CLIMB:.1f} m/s ({env.V_CLIMB*1.944:.1f} KIAS)")
    print(f"  Cruise altitude: {env.cruise_altitude:.0f} m ({env.cruise_altitude/0.3048:.0f} ft)")

    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n{'='*60}")
            print(f"  Episode {episode}")
            print(f"{'='*60}")

            obs, info = env.reset()
            steps = 0
            total_reward = 0.0
            max_altitude = 0.0
            max_airspeed = 0.0

            while steps < max_steps:
                # Get NN policy action
                action, _states = model.predict(obs, deterministic=deterministic)

                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # Update telemetry for viewer
                phase_name = info.get('mission_phase', task.upper())
                update_telemetry(obs, action, phase_name)

                # Extract metrics
                airspeed = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)
                alt = -obs[2]
                pitch = np.rad2deg(obs[7])
                roll = np.rad2deg(obs[6])

                max_altitude = max(max_altitude, alt)
                max_airspeed = max(max_airspeed, airspeed)

                # Print status every 5 seconds
                if steps % int(5.0 / dt) == 0:
                    print(f"  {steps*dt:5.1f}s: {airspeed*1.944:5.1f} KIAS | Alt={alt:6.1f}m ({alt/0.3048:5.0f}ft) | "
                          f"Pitch={pitch:+5.1f}° | Roll={roll:+5.1f}° | Phase={phase_name}")

                steps += 1
                time.sleep(dt)

                if terminated or truncated:
                    reason = info.get('termination_reason', 'max_steps')
                    print(f"\n  Episode ended: {reason}")
                    break

            print(f"\nEpisode {episode} Summary:")
            print(f"  Duration: {steps*dt:.1f}s ({steps} steps)")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Max Altitude: {max_altitude:.1f}m ({max_altitude/0.3048:.0f}ft)")
            print(f"  Max Airspeed: {max_airspeed*1.944:.1f} KIAS")
            print(f"  Final Phase: {info.get('mission_phase', 'unknown')}")

            # Pause before next episode
            time.sleep(5.0)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")

    env.close()


async def main_async(model_path, task, dt, max_steps, deterministic):
    server_task = asyncio.create_task(
        telemetry_server(lambda: shared_state, host="0.0.0.0", port=8765)
    )

    await asyncio.sleep(1)

    sim_thread = threading.Thread(
        target=run_simulation,
        args=(model_path, task, dt, max_steps, deterministic),
        daemon=True
    )
    sim_thread.start()

    try:
        await server_task
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(description='Run PPO policy with telemetry viewer')
    parser.add_argument('--model', type=str,
                        default='checkpoints/cessna172_curriculum/phase5_cruise/best_model.zip',
                        help='Path to PPO model checkpoint')
    parser.add_argument('--task', type=str, default='full_mission',
                        choices=['ground_roll', 'rotation', 'initial_climb', 'full_climb', 'cruise', 'full_mission'],
                        help='Flight task')
    parser.add_argument('--dt', type=float, default=0.02, help='Simulation timestep')
    parser.add_argument('--max-steps', type=int, default=3000, help='Max steps per episode')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic policy (default: deterministic)')

    args = parser.parse_args()

    asyncio.run(main_async(args.model, args.task, args.dt, args.max_steps, not args.stochastic))
    return 0


if __name__ == "__main__":
    sys.exit(main())
