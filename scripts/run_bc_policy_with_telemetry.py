#!/usr/bin/env python3
"""
Run BC-trained Policy with 3D Telemetry Viewer

Visualizes the behavioral cloning policy performing takeoff.

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import asyncio
import threading
import time
import sys
import math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from aida_sim.env.flight_env_cessna172 import Cessna172Env, StateIndex
from aida_sim.io.telemetry import telemetry_server
from train_ppo_flight import ActorCritic

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
    "phase": "BC_POLICY",
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


def update_telemetry(obs, action):
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

    throttle_norm = (action[0] + 1.0) / 2.0
    shared_state["throttle"] = float(np.clip(throttle_norm, 0.0, 1.0))
    shared_state["surfaces"] = [
        float(np.clip(action[1], -1.0, 1.0)),
        float(np.clip(action[2], -1.0, 1.0)),
        float(np.clip(action[3], -1.0, 1.0))
    ]


def run_simulation(dt=0.02, max_steps=1500):
    print("\n" + "="*60)
    print("  BC POLICY TAKEOFF - TELEMETRY VIEWER")
    print("="*60)
    print("Open browser to http://localhost:8000")
    print("Press Ctrl+C to stop\n")

    # Load BC policy
    print("Loading BC policy...")
    checkpoint = torch.load('checkpoints/bc_takeoff/bc_policy_v1.pt', map_location='cpu', weights_only=False)
    policy = ActorCritic(checkpoint['obs_dim'], checkpoint['action_dim'], checkpoint['hidden_dim'])
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    print(f"Policy loaded (val_loss: {checkpoint['val_loss']:.6f})\n")

    env = Cessna172Env(task='ground_roll', dt=dt)
    print(f"V_rotate: {env.V_ROTATE:.1f} m/s ({env.V_ROTATE*1.944:.1f} KIAS)")

    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n{'='*60}")
            print(f"  Episode {episode}")
            print(f"{'='*60}")

            obs, info = env.reset()
            steps = 0

            while steps < max_steps:
                # Get BC policy action
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = policy.get_action(obs_tensor, deterministic=True)
                action_np = action.squeeze().numpy()

                obs, reward, terminated, truncated, info = env.step(action_np)

                update_telemetry(obs, action_np)

                airspeed = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)
                alt = -obs[2]
                pitch = np.rad2deg(obs[7])

                if steps % int(5.0 / dt) == 0:
                    print(f"  {steps*dt:5.1f}s: {airspeed*1.944:5.1f} KIAS | Alt={alt:6.1f}m | Pitch={pitch:+5.1f}deg")

                steps += 1
                time.sleep(dt)

                if alt > 50:
                    print(f"\n  SUCCESS: Reached {alt:.1f}m altitude!")
                    break

            print(f"\nEpisode {episode} complete:")
            print(f"  Duration: {steps*dt:.1f}s")
            print(f"  Final Airspeed: {airspeed*1.944:.1f} KIAS")
            print(f"  Final Altitude: {alt:.1f}m")

            time.sleep(3.0)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")

    env.close()


async def main_async(dt, max_steps):
    server_task = asyncio.create_task(
        telemetry_server(lambda: shared_state, host="0.0.0.0", port=8765)
    )

    await asyncio.sleep(1)

    sim_thread = threading.Thread(
        target=run_simulation,
        args=(dt, max_steps),
        daemon=True
    )
    sim_thread.start()

    try:
        await server_task
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    asyncio.run(main_async(0.02, 1500))
    return 0


if __name__ == "__main__":
    sys.exit(main())
