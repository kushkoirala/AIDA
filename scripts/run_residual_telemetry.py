#!/usr/bin/env python3
"""
Run Residual PPO with Telemetry Viewer

Shows the expert controller + NN residual corrections flying the mission.
"""

import asyncio
import threading
import time
import sys
import math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from flight_dynamics import FlightSimulator, StateIndex
from triangle_controller import TriangleInterceptController, XCPhase
from aida_sim.io.telemetry import telemetry_server

TELEMETRY_UNITS = "metric"

shared_state = {
    "position": [0, 0, 0],
    "quaternion": [1, 0, 0, 0],
    "velocity": [0, 0, 0],
    "rates": [0, 0, 0],
    "surfaces": [0, 0, 0],
    "throttle": 0.0,
    "flaps": 0.0,
    "spoilers": 0.0,
    "soc": 1.0,
    "voltage": 12.0,
    "load_factor": 1.0,
    "units": TELEMETRY_UNITS,
    "model": "cessna172",
    "phase": "RESIDUAL_RL",
    "aoa_deg": 0.0,
    "alpha_deg": 0.0,
    "beta_deg": 0.0,
    "pilot_mode": "EXPERT",
    "sim_speed": 10.0,
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


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def build_obs(state, expert_action, expert, khut_x=52800, khut_y=-21300, cruise_alt_m=1676):
    """Build observation for the residual policy."""
    x, y, z = state[0], state[1], state[2]
    psi = state[8]
    alt = -z

    alt_error = (cruise_alt_m - alt) / 1000.0
    dx = khut_x - x
    dy = khut_y - y
    bearing = np.arctan2(dy, dx)
    heading_error = normalize_angle(bearing - psi)
    dist = np.sqrt(dx**2 + dy**2) / 10000.0

    obs = np.concatenate([
        state.astype(np.float32),
        expert_action[:4].astype(np.float32),
        np.array([alt_error, heading_error, dist], dtype=np.float32)
    ])
    return obs


def update_telemetry(state, throttle, aileron, elevator, rudder, flaps, spoilers, phase_name, distance_to_khut):
    x, y, z = state[0], state[1], state[2]
    phi, theta, psi = state[6], state[7], state[8]
    u, v, w_vel = state[3], state[4], state[5]
    p, q, r = state[9], state[10], state[11]

    # Convert NED coordinates from meters to feet (matching run_xc_sn65_khut.py)
    M_TO_FT = 3.28084
    shared_state["position"] = [float(x * M_TO_FT), float(y * M_TO_FT), float(z * M_TO_FT)]
    shared_state["quaternion"] = [float(qv) for qv in euler_to_quaternion(phi, theta, psi)]
    shared_state["velocity"] = [float(u * M_TO_FT), float(v * M_TO_FT), float(w_vel * M_TO_FT)]
    shared_state["rates"] = [float(p), float(q), float(r)]
    shared_state["throttle"] = float(throttle)
    shared_state["surfaces"] = [float(aileron), float(elevator), float(rudder)]
    shared_state["flaps"] = float(flaps)
    shared_state["spoilers"] = float(spoilers)
    shared_state["phase"] = phase_name

    # Additional telemetry matching expert viewer
    altitude_m = -z
    airspeed = np.sqrt(u**2 + v**2 + w_vel**2)
    shared_state["altitude_ft"] = float(altitude_m * M_TO_FT)
    shared_state["airspeed_kts"] = float(airspeed * 1.944)
    shared_state["heading_deg"] = float(np.rad2deg(psi) % 360)
    shared_state["distance_nm"] = float(distance_to_khut / 1852.0)
    shared_state["roll_deg"] = float(np.rad2deg(phi))
    shared_state["pitch_deg"] = float(np.rad2deg(theta))
    shared_state["euler"] = [float(np.rad2deg(phi)), float(np.rad2deg(theta)), float(np.rad2deg(psi))]

    # Angle of Attack (alpha) and sideslip (beta)
    if airspeed > 1.0:
        alpha_rad = np.arctan2(w_vel, u)
        beta_rad = np.arcsin(np.clip(v / airspeed, -1.0, 1.0))
    else:
        alpha_rad = 0.0
        beta_rad = 0.0
    shared_state["alpha_deg"] = float(np.rad2deg(alpha_rad))
    shared_state["beta_deg"] = float(np.rad2deg(beta_rad))
    shared_state["aoa_deg"] = float(np.rad2deg(alpha_rad))


def run_simulation(model_path=None, residual_scale=0.15, dt=0.02, sim_speed=10.0):
    print("\n" + "="*70)
    print("  RESIDUAL RL FLIGHT - EXPERT + NN CORRECTIONS")
    print("="*70)
    print(f"  Residual scale: {residual_scale} (NN can adjust ±{residual_scale*100:.0f}%)")
    if model_path:
        print(f"  Model: {model_path}")
    else:
        print("  Mode: PURE EXPERT (no NN corrections)")
    print("="*70)
    print("\nOpen browser: http://localhost:8000")
    print("Press Ctrl+C to stop\n")

    # Load model if provided
    model = None
    if model_path and Path(model_path).exists():
        print(f"Loading residual policy: {model_path}")
        model = PPO.load(model_path)
        print("Model loaded successfully")
    else:
        print("Running PURE EXPERT (no residual NN)")

    # Set pilot mode and sim speed
    if model is None:
        shared_state["pilot_mode"] = "EXPERT"
    else:
        shared_state["pilot_mode"] = "HYBRID"  # V1 model: NN on some phases
    shared_state["sim_speed"] = sim_speed

    # Setup
    sim = FlightSimulator(n_instances=1, dt=dt, use_gpu=False)
    expert = TriangleInterceptController(cruise_altitude_ft=5500.0)

    khut_x, khut_y = 52800.0, -21300.0  # KHUT airport in meters from SN65
    cruise_alt_m = 5500 * 0.3048

    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n{'='*60}")
            print(f"  Flight {episode}")
            print(f"{'='*60}")

            # Reset
            initial_state = np.zeros((1, 12), dtype=np.float32)
            initial_state[0, 0] = -400.0
            initial_state[0, 3] = 5.0
            sim.reset(initial_state)
            expert.reset()

            sim_time = 0.0
            step = 0

            while step < 90000:  # 30 min max
                state = sim.get_states()[0]

                # Expert action
                expert_action = expert.compute_action(state, sim_time)

                # Residual correction
                if model is not None:
                    obs = build_obs(state, expert_action, expert, khut_x, khut_y, cruise_alt_m)
                    residual, _ = model.predict(obs, deterministic=True)
                else:
                    residual = np.zeros(4)

                # Combine: expert + scaled residual
                # Disable residual during critical phases where expert knows best
                if expert.phase in [XCPhase.GROUND_ROLL, XCPhase.ROTATION]:
                    # No lateral correction during takeoff roll - stay on centerline
                    throttle = np.clip(expert_action[0] + residual_scale * residual[0], 0, 1)
                    aileron = 0.0
                    elevator = np.clip(expert_action[2] + residual_scale * residual[2], -1, 1)
                    rudder = 0.0
                elif expert.phase in [XCPhase.TURN_TO_INTERCEPT, XCPhase.INTERCEPT_LEG, XCPhase.FINAL_APPROACH, XCPhase.SHORT_FINAL, XCPhase.LANDING, XCPhase.LANDED]:
                    # Pure expert during approach/landing - NN not trained for descent
                    throttle = float(expert_action[0])
                    aileron = float(expert_action[1])
                    elevator = float(expert_action[2])
                    rudder = float(expert_action[3])
                else:
                    # Normal residual during climb/cruise
                    throttle = np.clip(expert_action[0] + residual_scale * residual[0], 0, 1)
                    aileron = np.clip(expert_action[1] + residual_scale * residual[1], -1, 1)
                    elevator = np.clip(expert_action[2] + residual_scale * residual[2], -1, 1)
                    rudder = np.clip(expert_action[3] + residual_scale * residual[3], -1, 1)

                # Get flaps/spoilers/brakes from expert action
                flaps = float(expert_action[4]) if len(expert_action) > 4 else 0.0
                spoilers = float(expert_action[5]) if len(expert_action) > 5 else 0.0
                brakes = float(expert_action[6]) if len(expert_action) > 6 else 0.0

                # Apply ALL 7 controls (including flaps/spoilers for descent)
                controls = np.array([[throttle, aileron, elevator, rudder, flaps, spoilers, brakes]])
                sim.set_controls(controls)
                sim.step()

                # Stats
                alt = -state[2]
                ias = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2) * 1.944
                hdg = np.degrees(state[8]) % 360
                dist_m = np.sqrt((state[0] - khut_x)**2 + (state[1] - khut_y)**2)
                dist = dist_m / 1852

                # Telemetry
                phase_name = expert.phase.name
                update_telemetry(state, throttle, aileron, elevator, rudder, flaps, spoilers, phase_name, dist_m)

                if step % 1500 == 0:
                    res_str = f"[{residual[0]:+.2f},{residual[1]:+.2f},{residual[2]:+.2f},{residual[3]:+.2f}]" if model else "[pure expert]"
                    print(f"  {sim_time:6.1f}s | {phase_name:>18s} | {alt:5.0f}m | {ias:5.1f}kt | {hdg:3.0f}° | {dist:5.1f}nm | {res_str}")

                # Check termination
                if expert.phase == XCPhase.LANDED or dist < 1.0:
                    print("\n  ARRIVED!")
                    break
                if alt < -10 or (alt > 100 and ias < 40):
                    print(f"\n  CRASH at {sim_time:.1f}s")
                    break

                step += 1
                sim_time += dt
                time.sleep(dt / sim_speed)  # 3x speed

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\nStopped by user")


async def main_async(model_path, residual_scale):
    server_task = asyncio.create_task(
        telemetry_server(lambda: shared_state, host="0.0.0.0", port=8765)
    )
    await asyncio.sleep(1)

    sim_thread = threading.Thread(
        target=run_simulation,
        args=(model_path, residual_scale),
        daemon=True
    )
    sim_thread.start()

    try:
        await server_task
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/residual_ppo/best_model.zip')
    parser.add_argument('--residual-scale', type=float, default=0.15)
    parser.add_argument('--pure-expert', action='store_true', help='Run pure expert without NN')
    args = parser.parse_args()

    model_path = None if args.pure_expert else args.model
    asyncio.run(main_async(model_path, args.residual_scale))


if __name__ == "__main__":
    main()
