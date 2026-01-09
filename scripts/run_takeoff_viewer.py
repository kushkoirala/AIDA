#!/usr/bin/env python3
"""
Run Full Takeoff Sequence with 3D Telemetry Viewer

Shows the complete takeoff sequence:
- Phase 1: Ground Roll (accelerate to rotation speed)
- Phase 2: Rotation (pitch up and lift off)

Uses the best available trained model and loops continuously.

Author: Kushal Koirala (with Claude Code)
Date: December 31, 2024
"""

# CuPy cache directory (use existing cache from training)
import os
os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy-cache'

import argparse
import asyncio
import threading
import time
import sys
from pathlib import Path
import numpy as np
import math
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from aida_sim.env.flight_env_cessna172 import Cessna172Env
from aida_sim.io.telemetry import telemetry_server

# Shared state for telemetry
TELEMETRY_UNITS = "metric"

# Cessna172Env runway starts at (0, -500, 0) in NED coordinates
# Viewer expects runway at (0, 0, 0), so we offset positions
RUNWAY_START_OFFSET = np.array([0.0, -500.0, 0.0])

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
    x, y, z = obs[0], obs[1], obs[2]
    u, v, w = obs[3], obs[4], obs[5]
    phi, theta, psi = obs[6], obs[7], obs[8]
    p, q, r = obs[9], obs[10], obs[11]

    # Transform position to viewer coordinates (offset runway start to origin)
    pos = np.array([x, y, z]) - RUNWAY_START_OFFSET
    shared_state["position"] = [float(pos[0]), float(pos[1]), float(pos[2])]

    quat = euler_to_quaternion(phi, theta, psi)
    shared_state["quaternion"] = [float(q) for q in quat]
    shared_state["velocity"] = [float(u), float(v), float(w)]
    shared_state["rates"] = [float(p), float(q), float(r)]

    throttle_norm = (action[0] + 1.0) / 2.0
    aileron = action[1]
    elevator = action[2]
    rudder = action[3]

    shared_state["throttle"] = float(np.clip(throttle_norm, 0.0, 1.0))
    shared_state["surfaces"] = [
        float(np.clip(aileron, -1.0, 1.0)),
        float(np.clip(elevator, -1.0, 1.0)),
        float(np.clip(rudder, -1.0, 1.0))
    ]
    shared_state["soc"] = 1.0
    shared_state["voltage"] = 12.0
    shared_state["load_factor"] = 1.0


def load_phase_models():
    """Load all available phase models for end-to-end takeoff.

    Returns dict with models for each phase that exists.
    """
    checkpoint_base = "checkpoints/cessna172_curriculum"

    models = {}
    phase_paths = {
        'ground_roll': f"{checkpoint_base}/phase1_ground_roll/best_model.zip",
        'rotation': f"{checkpoint_base}/phase2_rotation/best_model.zip",
        'initial_climb': f"{checkpoint_base}/phase3_initial_climb/best_model.zip",
    }

    for phase, path in phase_paths.items():
        if os.path.exists(path):
            print(f"Loading {phase} model: {path}")
            models[phase] = PPO.load(path)

    return models


# Phase transition thresholds (matching Cessna 172 speeds)
V_ROTATE = 28.0  # m/s (~55 KIAS) - switch to rotation model
V_LIFTOFF = 31.0  # m/s (~60 KIAS) - rotation complete, switch to climb
LIFTOFF_ALT = 1.0  # meters - consider airborne above this


def get_current_phase(airspeed, altitude):
    """Determine which phase we're in based on airspeed and altitude."""
    if altitude > LIFTOFF_ALT:
        return 'initial_climb'
    elif airspeed >= V_ROTATE:
        return 'rotation'
    else:
        return 'ground_roll'


def run_simulation(models, dt=0.02, max_steps=2000, host="0.0.0.0"):
    """
    Run end-to-end takeoff simulation with model chaining.

    Switches between phase models as the aircraft progresses:
    - ground_roll model: 0 to V_ROTATE
    - rotation model: V_ROTATE to liftoff
    - initial_climb model: after liftoff
    """
    print(f"\n{'='*60}")
    print(f"  CESSNA 172 END-TO-END TAKEOFF VIEWER")
    print(f"{'='*60}")
    print(f"Models loaded: {list(models.keys())}")
    print(f"Phase transitions:")
    print(f"  Ground Roll → Rotation at {V_ROTATE * 1.944:.0f} KIAS")
    print(f"  Rotation → Climb at {LIFTOFF_ALT * 3.281:.0f} ft AGL")
    print(f"Timestep: {dt}s")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}\n")

    # Create environment - use 'full_mission' task which starts from rest and allows full flight
    print("Creating environment...")
    env = Cessna172Env(task='full_mission', cruise_altitude_ft=3000.0, dt=dt)
    print("Environment created\n")

    print("Starting simulation loop...")
    print(f"Telemetry: ws://{host}:8765")
    print(f"Viewer: http://{host}:8000")
    print("Press Ctrl+C to stop\n")

    # Default to first available model if a phase is missing
    fallback_model = list(models.values())[0] if models else None

    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n--- Takeoff {episode} ---")

            obs, info = env.reset()
            done = False
            steps = 0
            total_reward = 0
            max_altitude = 0
            max_airspeed = 0
            last_phase = None

            while not done and steps < max_steps:
                # Determine current phase and select model
                airspeed = info.get('airspeed', 0)
                altitude = info.get('altitude', 0)
                current_phase = get_current_phase(airspeed, altitude)

                # Select appropriate model for current phase
                model = models.get(current_phase, fallback_model)

                # Log phase transitions
                if current_phase != last_phase:
                    print(f"  [Phase: {current_phase}] at {airspeed * 1.944:.1f} KIAS, {altitude * 3.281:.1f} ft")
                    last_phase = current_phase

                # Get action from selected model
                action, _states = model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Track max values
                altitude = info['altitude']
                airspeed = info['airspeed']
                max_altitude = max(max_altitude, altitude)
                max_airspeed = max(max_airspeed, airspeed)

                # Update telemetry
                update_telemetry_from_obs(obs, action, info)

                # Print status every 2 seconds
                if steps % int(2.0 / dt) == 0:
                    airspeed_kias = airspeed * 1.944
                    altitude_ft = altitude * 3.281
                    print(f"  t={steps*dt:5.1f}s: Speed={airspeed_kias:5.1f} KIAS, "
                          f"Alt={altitude_ft:6.1f} ft, Phase={current_phase}")

                steps += 1
                time.sleep(dt)

            # Episode summary
            reason = info.get('termination_reason', 'max_steps')
            print(f"\nTakeoff {episode} complete:")
            print(f"  Duration: {steps * dt:.1f}s ({steps} steps)")
            print(f"  Max Altitude: {max_altitude * 3.281:.1f} ft")
            print(f"  Max Airspeed: {max_airspeed * 1.944:.1f} KIAS")
            print(f"  Total Reward: {total_reward:.1f}")
            print(f"  Result: {reason}")

            # Brief pause before next takeoff
            time.sleep(3.0)

    except KeyboardInterrupt:
        print("\n\nViewer stopped")

    env.close()


async def main_async(models, dt, max_steps, host):
    """Run telemetry server and simulation concurrently."""
    # Start telemetry server
    server_task = asyncio.create_task(
        telemetry_server(lambda: shared_state, host=host, port=8765)
    )

    await asyncio.sleep(1)

    # Run simulation in thread
    sim_thread = threading.Thread(
        target=run_simulation,
        args=(models, dt, max_steps, host),
        daemon=True
    )
    sim_thread.start()

    try:
        await server_task
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(description='Run Cessna 172 End-to-End Takeoff Viewer')
    parser.add_argument('--dt', type=float, default=0.02,
                       help='Timestep in seconds (default: 0.02)')
    parser.add_argument('--max-steps', type=int, default=2000,
                       help='Max steps per episode (default: 2000 = 40 sec)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind telemetry server (default: 0.0.0.0)')
    args = parser.parse_args()

    # Load all available phase models
    print("Loading phase models for end-to-end takeoff...")
    models = load_phase_models()

    if not models:
        print("No trained models found!")
        print("\nLooking for models in: checkpoints/cessna172_curriculum/")
        print("Please wait for training to complete.")
        return 1

    print(f"\nLoaded {len(models)} phase model(s): {list(models.keys())}")

    asyncio.run(main_async(models, args.dt, args.max_steps, args.host))
    return 0


if __name__ == "__main__":
    sys.exit(main())
