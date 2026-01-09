#!/usr/bin/env python3
"""
Run Classical Takeoff Controller with 3D Telemetry Viewer

Full takeoff sequence using classical GNC:
1. Ground Roll
2. Rotation  
3. Initial Climb
4. Climb

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from aida_sim.env.flight_env_cessna172 import Cessna172Env, StateIndex
from aida_sim.io.telemetry import telemetry_server
from classical_takeoff_controller import TakeoffController

TELEMETRY_UNITS = "metric"

MAX_ELEVATOR_DEG = 28.0
MAX_AILERON_DEG = 20.0
MAX_RUDDER_DEG = 16.0

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
    "phase": "ground_roll",
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
    viewer_z = max(0.0, -z)

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

    shared_state["aileron_deg"] = float(action[1]) * MAX_AILERON_DEG
    shared_state["elevator_deg"] = float(action[2]) * MAX_ELEVATOR_DEG
    shared_state["rudder_deg"] = float(action[3]) * MAX_RUDDER_DEG

    shared_state["roll_deg"] = float(np.rad2deg(phi))
    shared_state["pitch_deg"] = float(np.rad2deg(theta))
    shared_state["heading_deg"] = float(np.rad2deg(psi)) % 360.0

    if abs(u) > 0.1:
        alpha_rad = math.atan2(w, u)
        shared_state["alpha_deg"] = float(np.rad2deg(alpha_rad))
    else:
        shared_state["alpha_deg"] = 0.0

    airspeed_mps = math.sqrt(u**2 + v**2 + w**2)
    shared_state["airspeed_kts"] = float(airspeed_mps * 1.94384)

    altitude_m = max(0.0, -z)
    shared_state["altitude_ft"] = float(altitude_m * 3.28084)

    z_dot = (-u * math.sin(theta) +
             v * math.cos(theta) * math.sin(phi) +
             w * math.cos(theta) * math.cos(phi))
    vertical_speed_mps = -z_dot
    shared_state["vertical_speed_fpm"] = float(vertical_speed_mps * 196.85)

    shared_state["phase"] = phase_name


def run_simulation(dt=0.02, max_steps=3000):
    print("\n" + "="*60)
    print("  CLASSICAL TAKEOFF CONTROLLER - TELEMETRY VIEWER")
    print("="*60)
    print("Phases: Ground Roll -> Rotation -> Initial Climb -> Climb")
    print("Open browser to http://localhost:8000")
    print("Press Ctrl+C to stop\n")

    env = Cessna172Env(task='ground_roll', dt=dt, max_episode_steps=max_steps)
    controller = TakeoffController(v_rotate=env.V_ROTATE, runway_heading=env.runway_heading)

    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n--- TAKEOFF {episode} ---")

            obs, info = env.reset()
            controller.reset()
            done = False
            steps = 0
            total_reward = 0
            last_phase = None

            while not done and steps < max_steps:
                action = controller.compute_action(obs)
                
                action_4 = action[:4]
                
                obs, reward, terminated, truncated, info = env.step(action_4)
                done = terminated or truncated
                total_reward += reward

                phase_name = controller.get_phase_name().lower()
                
                if phase_name != last_phase:
                    airspeed = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)
                    altitude = -obs[2]
                    print(f">>> PHASE: {phase_name.upper()} (Speed={airspeed*1.944:.1f} KIAS, Alt={altitude*3.281:.1f} ft)")
                    last_phase = phase_name

                update_telemetry(obs, action, phase_name)

                if steps % int(2.0 / dt) == 0:
                    airspeed = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)
                    altitude = -obs[2]
                    print(f"  [{phase_name:15s}] Step {steps:4d}: Speed={airspeed*1.944:5.1f} KIAS, Alt={altitude*3.281:6.1f} ft")

                steps += 1
                time.sleep(dt)

            reason = info.get('termination_reason', 'max_steps')
            airspeed = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)
            altitude = -obs[2]

            print(f"\nTakeoff {episode} complete:")
            print(f"  Steps: {steps}")
            print(f"  Total Reward: {total_reward:.1f}")
            print(f"  Final Airspeed: {airspeed*1.944:.1f} KIAS")
            print(f"  Final Altitude: {altitude*3.281:.1f} ft")
            print(f"  Termination: {reason}")

            time.sleep(3.0)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")

    env.close()


async def main_async(dt, max_steps, ws_port=8765):
    server_task = asyncio.create_task(
        telemetry_server(lambda: shared_state, host="0.0.0.0", port=ws_port)
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
    asyncio.run(main_async(0.02, 3000))
    return 0


if __name__ == "__main__":
    sys.exit(main())
