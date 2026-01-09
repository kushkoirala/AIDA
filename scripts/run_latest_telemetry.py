#!/usr/bin/env python3
"""
Auto-Launch Telemetry Viewer with Latest Best Checkpoint

Automatically finds and loads the most recent best_model.zip from
the checkpoints directory and launches the telemetry viewer.

Author: Kushal Koirala (with Claude Code)
Date: December 29, 2024
"""

import argparse
import asyncio
import threading
import time
import sys
from pathlib import Path
import numpy as np
import math
import os
from datetime import datetime

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
    "model": "cessna172",
    "sim_time": 0.0,
}


def find_latest_best_model(checkpoint_dir="checkpoints"):
    """
    Find the most recently modified best_model.zip in checkpoints directory.
    
    Returns:
        tuple: (path, phase_name, modification_time) or (None, None, None)
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None, None, None
    
    best_models = list(checkpoint_path.glob("**/best_model.zip"))
    
    if not best_models:
        return None, None, None
    
    # Sort by modification time (most recent first)
    best_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    latest = best_models[0]
    phase_name = latest.parent.name
    mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
    
    return str(latest), phase_name, mod_time


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


def update_telemetry_from_obs(obs, action, info, sim_time):
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
    viewer_z = z         # NED Z is down, viewer Z is up
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
    shared_state["load_factor"] = 1.0
    
    # Sim time
    shared_state["sim_time"] = float(sim_time)


def run_simulation(model_path, task='ground_roll', dt=0.02, max_steps=800):
    """
    Run Cessna 172 simulation with telemetry.

    Args:
        model_path: Path to trained PPO model
        task: Task type (ground_roll, rotation, etc.)
        dt: Timestep
        max_steps: Maximum steps per episode
    """
    print(f"Creating environment...")
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
            sim_start = time.time()

            while not done and steps < max_steps:
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Update telemetry
                sim_time = time.time() - sim_start
                update_telemetry_from_obs(obs, action, info, sim_time)

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
    parser = argparse.ArgumentParser(
        description='Auto-Launch Telemetry Viewer with Latest Best Checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-find latest checkpoint
  python run_latest_telemetry.py
  
  # Specify task
  python run_latest_telemetry.py --task rotation
  
  # Use specific checkpoint
  python run_latest_telemetry.py --model checkpoints/my_model.zip
        """
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Path to PPO model (default: auto-find latest best)')
    parser.add_argument('--task', type=str, default='ground_roll',
                       choices=['ground_roll', 'rotation', 'initial_climb', 'full_climb', 'cruise'],
                       help='Task type (default: ground_roll)')
    parser.add_argument('--dt', type=float, default=0.02,
                       help='Timestep in seconds (default: 0.02)')
    parser.add_argument('--max-steps', type=int, default=800,
                       help='Max steps per episode (default: 800)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoints directory (default: checkpoints)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CESSNA 172 TELEMETRY VIEWER")
    print(f"  Auto-Loading Latest Best Checkpoint")
    print(f"{'='*60}\n")

    # Auto-find model if not specified
    if args.model is None:
        print("🔍 Searching for latest best checkpoint...")
        model_path, phase_name, mod_time = find_latest_best_model(args.checkpoint_dir)
        
        if model_path is None:
            print(f"❌ No trained models found in {args.checkpoint_dir}/")
            print(f"\nPlease wait for training to complete or specify --model")
            return 1
        
        print(f"✓ Found: {model_path}")
        print(f"  Phase: {phase_name}")
        print(f"  Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Auto-detect task from phase name
        if 'ground_roll' in phase_name:
            auto_task = 'ground_roll'
        elif 'rotation' in phase_name:
            auto_task = 'rotation'
        elif 'initial_climb' in phase_name:
            auto_task = 'initial_climb'
        elif 'full_climb' in phase_name:
            auto_task = 'full_climb'
        elif 'cruise' in phase_name:
            auto_task = 'cruise'
        else:
            auto_task = args.task
        
        # Override task if not explicitly set
        if args.task == 'ground_roll':  # Default value
            args.task = auto_task
            print(f"  Task: {auto_task} (auto-detected)")
    else:
        model_path = args.model
        print(f"Model: {model_path}")
    
    print(f"Task: {args.task}")
    print(f"Timestep: {args.dt}s")
    print(f"Max steps: {args.max_steps}")
    print(f"{'='*60}\n")

    # Run async main
    asyncio.run(main_async(model_path, args.task, args.dt, args.max_steps))

    return 0


if __name__ == "__main__":
    sys.exit(main())
