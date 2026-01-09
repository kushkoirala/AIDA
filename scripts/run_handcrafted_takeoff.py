#!/usr/bin/env python3
"""
Handcrafted Takeoff Controller - Fixed version with proper rotation
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

from aida_sim.env.flight_env_cessna172 import Cessna172Env

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
            await asyncio.Future()

    ws_loop.run_until_complete(serve())


def handcrafted_controller(state, phase):
    """
    Improved handcrafted takeoff controller.
    
    Args:
        state: [x, y, z, u, v, w, phi, theta, psi, p, q, r]
        phase: Current flight phase string
    
    Returns:
        action: [throttle, elevator, aileron, rudder] in [-1, 1]
    """
    x, y, z = state[0], state[1], state[2]
    u, v, w = state[3], state[4], state[5]
    phi, theta, psi = state[6], state[7], state[8]
    
    altitude_m = -z
    altitude_ft = altitude_m * 3.28084
    airspeed_ms = u
    airspeed_kts = airspeed_ms * 1.94384
    pitch_deg = np.rad2deg(theta)
    roll_deg = np.rad2deg(phi)
    climb_rate = -w
    
    V_ROTATE = 55  # KIAS
    V_CLIMB = 70   # KIAS - best rate of climb
    
    # Always full throttle during takeoff
    throttle = 1.0
    elevator = 0.0
    aileron = 0.0
    rudder = 0.0
    
    if altitude_ft < 5:
        # GROUND PHASE
        if airspeed_kts < V_ROTATE - 5:
            # Accelerating - neutral elevator
            elevator = 0.0
        else:
            # At/near rotation speed - apply pitch-up
            target_pitch = 10.0
            pitch_error = target_pitch - pitch_deg
            elevator = -0.3 * np.clip(pitch_error / 10.0, -1, 1)
    else:
        # AIRBORNE PHASE - maintain climb attitude
        target_pitch = 8.0
        
        # Speed protection
        if airspeed_kts < V_CLIMB - 15:
            target_pitch = 3.0  # Lower nose if too slow
        elif airspeed_kts > V_CLIMB + 10:
            target_pitch = 12.0  # Raise nose if too fast
        
        pitch_error = target_pitch - pitch_deg
        elevator = -0.15 * pitch_error
        elevator = np.clip(elevator, -0.4, 0.3)
    
    # Wings level
    if abs(roll_deg) > 2:
        aileron = -0.2 * roll_deg / 10.0
        aileron = np.clip(aileron, -0.3, 0.3)
    
    # Heading hold
    heading_error = np.rad2deg(psi)
    if abs(heading_error) > 2:
        rudder = -0.1 * heading_error / 10.0
        rudder = np.clip(rudder, -0.2, 0.2)
    
    return np.array([throttle, elevator, aileron, rudder])


def run_handcrafted_takeoff():
    global ws_loop

    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()
    time.sleep(2)

    print("\n" + "="*60)
    print("  HANDCRAFTED TAKEOFF CONTROLLER (Fixed)")
    print("="*60 + "\n")

    episode = 0
    while True:
        episode += 1
        print(f"\n{'='*50}")
        print(f"--- Episode {episode} ---")
        print(f"{'='*50}")

        env = Cessna172Env(task="initial_climb", cruise_altitude_ft=500.0, dt=0.02)
        obs, _ = env.reset()

        step = 0
        max_altitude = 0
        phase = "Ground Roll"

        while True:
            state = env.sim.get_states()[0]
            action = handcrafted_controller(state, phase)
            obs, reward, terminated, truncated, info = env.step(action)

            state = env.sim.get_states()[0]
            airspeed_kts = float(state[3]) * 1.94384
            altitude_ft = -float(state[2]) * 3.28084
            pitch_deg = np.rad2deg(float(state[7]))
            
            max_altitude = max(max_altitude, altitude_ft)
            
            # Phase detection
            if altitude_ft < 5:
                if airspeed_kts < 50:
                    phase = "Ground Roll"
                else:
                    phase = "Rotation"
            elif altitude_ft < 100:
                phase = "Liftoff"
            else:
                phase = "Climb"

            # Telemetry
            pos = [float(state[0]) + 500.0, float(state[1]), float(state[2])]
            phi, theta, psi = float(state[6]), float(state[7]), float(state[8])
            cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
            cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
            cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy

            telemetry = {
                "position": pos,
                "quaternion": [qw, qx, qy, qz],
                "velocity": [float(state[3]), float(state[4]), float(state[5])],
                "throttle": float(action[0]),
                "surfaces": [float(action[1]), float(action[2]), float(action[3])],
                "sim_time": step * 0.02,
                "mode": f"Handcrafted: {phase}",
                "heartbeat": step,
                "units": "metric"
            }

            if ws_loop:
                asyncio.run_coroutine_threadsafe(broadcast_telemetry(telemetry), ws_loop)

            if step % 100 == 0:
                print(f"  Step {step:4d}: {phase:12s} | Speed={airspeed_kts:5.1f} KIAS | "
                      f"Alt={altitude_ft:6.1f} ft | Pitch={pitch_deg:5.1f}deg | "
                      f"Elev={action[1]:.2f}")

            step += 1

            if terminated or truncated or step > 3000 or altitude_ft > 500:
                print(f"\nEpisode {episode} complete:")
                print(f"  Steps: {step}")
                print(f"  Max Altitude: {max_altitude:.1f} ft")
                print(f"  Final Speed: {airspeed_kts:.1f} KIAS")
                print(f"  Final Phase: {phase}")
                if altitude_ft > 100:
                    print(f"  *** SUCCESSFUL TAKEOFF! ***")
                break

            time.sleep(0.02)

        env.close()
        time.sleep(3)


if __name__ == "__main__":
    run_handcrafted_takeoff()
