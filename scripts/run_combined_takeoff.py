#!/usr/bin/env python3
"""
Combined Takeoff Viewer - Shows Phase 1 (ground roll) transitioning to Phase 2 (rotation)
Uses Cessna172Env (12-dim obs) which matches the curriculum training.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import asyncio
import websockets
import json
import threading
import time
from stable_baselines3 import PPO

from aida_sim.env.flight_env_cessna172 import Cessna172Env


# Telemetry server
connected_clients = set()
ws_loop = None

async def telemetry_handler(websocket, path=None):
    connected_clients.add(websocket)
    print(f"[telemetry] Client connected")
    try:
        async for message in websocket:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[telemetry] Client disconnected")

async def broadcast_telemetry(data):
    if connected_clients:
        message = json.dumps(data)
        await asyncio.gather(*[client.send(message) for client in connected_clients], return_exceptions=True)

def start_websocket_server():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)

    async def serve():
        async with websockets.serve(telemetry_handler, "0.0.0.0", 8765):
            print("[telemetry] WebSocket server on ws://0.0.0.0:8765")
            await asyncio.Future()  # run forever

    ws_loop.run_until_complete(serve())


def run_combined_takeoff():
    global ws_loop

    # Start WebSocket server
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()
    time.sleep(2)

    print("\n" + "="*60)
    print("  COMBINED TAKEOFF VIEWER")
    print("  Phase 1: Ground Roll -> Phase 2: Rotation -> Phase 3: Climb")
    print("="*60 + "\n")

    # Load models for each phase
    models = {}
    phase_paths = {
        1: "checkpoints/cessna172_curriculum/phase1_ground_roll/best_model.zip",
        2: "checkpoints/cessna172_curriculum/phase2_rotation/best_model.zip",
        3: "checkpoints/cessna172_curriculum/phase3_initial_climb/best_model.zip",
    }

    for phase, path in phase_paths.items():
        if os.path.exists(path):
            print(f"Loading Phase {phase} model: {path}")
            models[phase] = PPO.load(path)
        else:
            print(f"Phase {phase} model not found: {path}")

    if not models:
        print("No models found!")
        return

    print(f"Loaded {len(models)} phase models!\n")

    # Transition thresholds
    ROTATION_SPEED_MS = 28.0   # 55 KIAS - switch from Phase 1 to Phase 2
    LIFTOFF_ALT_M = 3.0        # ~10 ft AGL - switch from Phase 2 to Phase 3

    episode = 0
    while True:
        episode += 1
        print(f"\n{'='*50}")
        print(f"--- Episode {episode} ---")
        print(f"{'='*50}")

        # Start with ground_roll environment (Phase 1 initial conditions)
        env = Cessna172Env(task="ground_roll", cruise_altitude_ft=3000.0, dt=0.02)
        obs, _ = env.reset()

        current_phase = 1
        current_model = models.get(1, list(models.values())[0])
        step = 0
        total_reward = 0
        phase_switched = {1: False, 2: False, 3: False}
        phase_switched[1] = True  # We start in phase 1

        while True:
            # Get action from current model
            action, _ = current_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Get state for telemetry
            state = env.sim.get_states()[0]
            airspeed_ms = float(state[3])  # u (forward velocity)
            altitude_m = -float(state[2])  # -z in NED

            # Check for phase transitions
            old_phase = current_phase

            if current_phase == 1 and airspeed_ms >= ROTATION_SPEED_MS and 2 in models:
                current_phase = 2
                current_model = models[2]
                if not phase_switched[2]:
                    print(f"\n  >>> PHASE 2: ROTATION! Speed={airspeed_ms*1.94384:.1f} KIAS <<<")
                    phase_switched[2] = True

            elif current_phase == 2 and altitude_m >= LIFTOFF_ALT_M and 3 in models:
                current_phase = 3
                current_model = models[3]
                if not phase_switched[3]:
                    print(f"\n  >>> PHASE 3: CLIMB! Alt={altitude_m:.1f}m ({altitude_m*3.28084:.0f} ft) <<<")
                    phase_switched[3] = True

            # Send telemetry
            # State: [x, y, z, u, v, w, phi, theta, psi, p, q, r]
            # Offset X by 500m so aircraft starts at runway origin (sim starts at -500)
            pos = [float(state[0]) + 500.0, float(state[1]), float(state[2])]

            # Convert Euler angles to quaternion for viewer
            phi, theta, psi = float(state[6]), float(state[7]), float(state[8])
            cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
            cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
            cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            quat = [qw, qx, qy, qz]

            phase_names = {1: "Ground Roll", 2: "Rotation", 3: "Initial Climb"}
            phase_name = phase_names.get(current_phase, "Unknown")

            telemetry = {
                "position": pos,
                "quaternion": quat,
                "velocity": [float(state[3]), float(state[4]), float(state[5])],
                "throttle": float(action[0]) if len(action) > 0 else 0.0,
                "surfaces": [float(action[1]), float(action[2]), float(action[3])] if len(action) >= 4 else [0,0,0],
                "sim_time": step * 0.02,
                "mode": f"Phase {current_phase}: {phase_name}",
                "heartbeat": step,
                "units": "metric"
            }

            if ws_loop:
                asyncio.run_coroutine_threadsafe(broadcast_telemetry(telemetry), ws_loop)

            # Print progress
            if step % 100 == 0:
                airspeed_kts = airspeed_ms * 1.94384
                alt_ft = altitude_m * 3.28084
                print(f"  Step {step:4d}: Phase {current_phase} ({phase_name:12s}), "
                      f"Speed={airspeed_kts:5.1f} KIAS, Alt={alt_ft:6.1f} ft")

            step += 1

            # End conditions
            max_steps = 3000 if 3 in models else 2000
            if terminated or truncated or step > max_steps:
                airspeed_kts = airspeed_ms * 1.94384
                alt_ft = altitude_m * 3.28084
                print(f"\nEpisode {episode} complete:")
                print(f"  Steps: {step}")
                print(f"  Total Reward: {total_reward:.1f}")
                print(f"  Final Airspeed: {airspeed_kts:.1f} KIAS")
                print(f"  Final Altitude: {alt_ft:.1f} ft")
                print(f"  Phases completed: {[p for p, v in phase_switched.items() if v]}")
                break

            time.sleep(0.02)

        env.close()
        time.sleep(3)


if __name__ == "__main__":
    run_combined_takeoff()
