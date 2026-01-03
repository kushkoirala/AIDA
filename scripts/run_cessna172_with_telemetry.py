#!/usr/bin/env python3
"""
Run Cessna 172 Ground Roll with 3D Telemetry Viewer

Loads a trained Phase 1 model and visualizes it in the 3D viewer
with the Cessna 172 model, runway, and environment.

Author: Kushal Koirala (with Claude Code)
Date: December 27, 2024
"""

import argparse
import asyncio
import threading
import time
import sys
from pathlib import Path
import numpy as np
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from aida_sim.env.flight_env_cessna172 import Cessna172Env
from aida_sim.io.telemetry import telemetry_server

# Shared state for telemetry
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
    "model": "cessna172",  # Tell viewer which aircraft model to use
}


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion [w, x, y, z]."""
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


def update_telemetry_from_obs(obs, action, info):
    """Update shared telemetry state from Cessna172 observation."""
    # State indices (from flight_env_cessna172.py)
    # [x, y, z, u, v, w, phi, theta, psi, p, q, r]
    x, y, z = obs[0], obs[1], obs[2]
    u, v, w = obs[3], obs[4], obs[5]
    phi, theta, psi = obs[6], obs[7], obs[8]
    p, q, r = obs[9], obs[10], obs[11]

    # Position (NED coordinates)
    # Transform position from simulation to viewer coordinates
    # Simulation: runway_start = [-500, 0, 0], viewer expects [0, 0, 0]
    viewer_x = x + 500.0  # Shift so runway starts at x=0
    viewer_y = y          # Y stays the same (lateral)
    viewer_z = -z         # NED Z is down, viewer Z is up
    shared_state["position"] = [float(viewer_x), float(viewer_y), float(viewer_z)]

    # Orientation (convert Euler to quaternion)
    quat = euler_to_quaternion(phi, theta, psi)
    shared_state["quaternion"] = [float(q) for q in quat]

    # Velocity (body frame)
    shared_state["velocity"] = [float(u), float(v), float(w)]

    # Angular rates (body frame)
    shared_state["rates"] = [float(p), float(q), float(r)]

    # Control surfaces (from action)
    # Action: [throttle, aileron, elevator, rudder] (normalized -1 to 1)
    throttle_norm = (action[0] + 1.0) / 2.0  # Map to 0-1
    aileron = action[1]   # -1 to 1
    elevator = action[2]  # -1 to 1
    rudder = action[3]    # -1 to 1

    shared_state["throttle"] = float(np.clip(throttle_norm, 0.0, 1.0))
    shared_state["surfaces"] = [
        float(np.clip(aileron, -1.0, 1.0)),
        float(np.clip(elevator, -1.0, 1.0)),
        float(np.clip(rudder, -1.0, 1.0))
    ]

    # Battery/power (fake for now)
    shared_state["soc"] = 1.0
    shared_state["voltage"] = 12.0

    # Load factor (approximate from vertical acceleration)
    # For ground roll, ~1.0
    shared_state["load_factor"] = 1.0

    # Euler angles in degrees for attitude indicator
    shared_state["roll_deg"] = float(np.rad2deg(phi))
    shared_state["pitch_deg"] = float(np.rad2deg(theta))
    shared_state["heading_deg"] = float(np.rad2deg(psi)) % 360.0

    # Angle of attack (alpha) - calculated from body velocities
    # AoA = atan2(w, u) where w=down velocity, u=forward velocity
    if abs(u) > 0.1:  # Avoid division issues at low speed
        alpha_rad = math.atan2(w, u)
        shared_state["alpha_deg"] = float(np.rad2deg(alpha_rad))
    else:
        shared_state["alpha_deg"] = 0.0

    # Airspeed in knots
    airspeed_mps = math.sqrt(u**2 + v**2 + w**2)
    shared_state["airspeed_kts"] = float(airspeed_mps * 1.94384)

    # Altitude in feet
    altitude_m = -z  # NED: Z is down, altitude is up
    shared_state["altitude_ft"] = float(altitude_m * 3.28084)

    # Vertical speed in fpm
    # Transform body velocity to NED frame to get true vertical speed
    # The NED z-velocity (z_dot) can be computed from body velocities:
    # z_dot = -u*sin(theta) + v*cos(theta)*sin(phi) + w*cos(theta)*cos(phi)
    # Negative z_dot = climb rate (positive when climbing)
    z_dot = (-u * math.sin(theta) +
             v * math.cos(theta) * math.sin(phi) +
             w * math.cos(theta) * math.cos(phi))
    vertical_speed_mps = -z_dot  # Positive when climbing (NED z is down)
    shared_state["vertical_speed_fpm"] = float(vertical_speed_mps * 196.85)


def run_simulation(model_path, task='ground_roll', dt=0.02, max_steps=500):
    """
    Run Cessna 172 simulation with telemetry.

    Args:
        model_path: Path to trained PPO model
        task: Task type (ground_roll, rotation, etc.)
        dt: Timestep
        max_steps: Maximum steps per episode
    """
    print(f"\n{'='*60}")
    print(f"  CESSNA 172 TELEMETRY VIEWER")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Task: {task}")
    print(f"Timestep: {dt}s")
    print(f"{'='*60}\n")

    # Create environment
    print("Creating environment...")
    env = Cessna172Env(task=task, cruise_altitude_ft=3000.0, dt=dt)
    print("✓ Environment created\n")

    # Load model
    print("Loading PPO model...")
    model = PPO.load(model_path)
    print("✓ Model loaded\n")

    print("Starting simulation...")
    print("Open browser to http://localhost:8000 to view telemetry")
    print("Press Ctrl+C to stop\n")

    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n--- Episode {episode} ---")

            obs, info = env.reset()
            done = False
            steps = 0
            total_reward = 0

            while not done and steps < max_steps:
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Update telemetry
                update_telemetry_from_obs(obs, action, info)

                # Print status every 2 seconds
                if steps % int(2.0 / dt) == 0:
                    airspeed_kias = info['airspeed'] * 1.944
                    altitude = info['altitude']
                    print(f"  Step {steps:3d}: Speed={airspeed_kias:5.1f} KIAS, "
                          f"Alt={altitude:5.2f}m, Reward={reward:6.1f}")

                steps += 1

                # Limit update rate for smooth visualization
                time.sleep(dt)

            # Episode summary
            reason = info.get('termination_reason', 'max_steps')
            max_airspeed_kias = info['airspeed'] * 1.944

            print(f"\nEpisode {episode} complete:")
            print(f"  Steps: {steps}")
            print(f"  Total Reward: {total_reward:.1f}")
            print(f"  Final Airspeed: {max_airspeed_kias:.1f} KIAS")
            print(f"  Termination: {reason}")

            # Wait a bit before next episode
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")

    env.close()


async def main_async(model_path, task, dt, max_steps):
    """Run telemetry server and simulation concurrently."""
    # Start telemetry server in background (WebSocket on port 8765)
    server_task = asyncio.create_task(
        telemetry_server(lambda: shared_state, host="0.0.0.0", port=8765)
    )

    # Give server time to start
    await asyncio.sleep(1)

    # Run simulation in thread
    sim_thread = threading.Thread(
        target=run_simulation,
        args=(model_path, task, dt, max_steps),
        daemon=True
    )
    sim_thread.start()

    # Wait for server
    try:
        await server_task
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(description='Run Cessna 172 with 3D Telemetry Viewer')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to PPO model (default: auto-find best)')
    parser.add_argument('--task', type=str, default='ground_roll',
                       choices=['ground_roll', 'rotation', 'initial_climb', 'full_climb', 'cruise'],
                       help='Task type (default: ground_roll)')
    parser.add_argument('--dt', type=float, default=0.02,
                       help='Timestep in seconds (default: 0.02)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Max steps per episode (default: 500)')
    args = parser.parse_args()

    # Auto-find model if not specified
    if args.model is None:
        import os
        # Try to find the best model for the task
        best_model = f"checkpoints/cessna172_curriculum/phase1_{args.task}/best_model.zip"
        if os.path.exists(best_model):
            model_path = best_model
        else:
            # Fall back to Phase 1 ground roll
            best_model = "checkpoints/cessna172_curriculum/phase1_ground_roll/best_model.zip"
            if os.path.exists(best_model):
                model_path = best_model
                print(f"Using Phase 1 ground roll model: {model_path}")
            else:
                print(f"❌ No trained model found!")
                print(f"   Expected: {best_model}")
                print(f"\nPlease wait for training to complete.")
                return 1
    else:
        model_path = args.model

    # Run async main
    asyncio.run(main_async(model_path, args.task, args.dt, args.max_steps))

    return 0


if __name__ == "__main__":
    sys.exit(main())
