#!/usr/bin/env python3
"""
Run Residual PPO V2 with Telemetry Viewer

Supports both v1 (4 control) and v2 (7 control) models.
The NN now learns corrections for ALL 7 controls during ALL flight phases.
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
    "phase": "RESIDUAL_RL_V2",
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


def get_phase_encoding(phase):
    """Encode flight phase as one-hot vector [takeoff, cruise, approach]"""
    if phase in [XCPhase.GROUND_ROLL, XCPhase.ROTATION, XCPhase.INITIAL_CLIMB]:
        return np.array([1, 0, 0], dtype=np.float32)  # Takeoff
    elif phase in [XCPhase.CLIMB, XCPhase.CRUISE_TO_TP]:
        return np.array([0, 1, 0], dtype=np.float32)  # Cruise
    else:
        return np.array([0, 0, 1], dtype=np.float32)  # Approach


def build_obs_v2(state, expert_action, phase, khut_x=52800, khut_y=-21300, cruise_alt_m=1676):
    """Build observation for v2 policy (25 dims)."""
    x, y, z = state[0], state[1], state[2]
    psi = state[8]
    alt = -z

    alt_error = (cruise_alt_m - alt) / 1000.0
    dx = khut_x - x
    dy = khut_y - y
    bearing = np.arctan2(dy, dx)
    heading_error = normalize_angle(bearing - psi)
    dist = np.sqrt(dx**2 + dy**2) / 10000.0

    phase_enc = get_phase_encoding(phase)

    obs = np.concatenate([
        state.astype(np.float32),              # 12 dims
        expert_action[:7].astype(np.float32),  # 7 dims (all controls)
        np.array([alt_error, heading_error, dist], dtype=np.float32),  # 3 dims
        phase_enc                               # 3 dims
    ])
    return obs


def build_obs_v1(state, expert_action, khut_x=52800, khut_y=-21300, cruise_alt_m=1676):
    """Build observation for v1 policy (19 dims)."""
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

    M_TO_FT = 3.28084
    shared_state["position"] = [float(x * M_TO_FT), float(y * M_TO_FT), float(z * M_TO_FT)]
    shared_state["quaternion"] = [float(qv) for qv in euler_to_quaternion(phi, theta, psi)]
    shared_state["velocity"] = [float(u * M_TO_FT), float(v * M_TO_FT), float(w_vel * M_TO_FT)]  # w positive = climbing
    shared_state["rates"] = [float(p), float(q), float(r)]
    shared_state["throttle"] = float(throttle)
    shared_state["surfaces"] = [float(aileron), float(elevator), float(rudder)]
    shared_state["flaps"] = float(flaps)
    shared_state["spoilers"] = float(spoilers)
    shared_state["phase"] = phase_name

    altitude_m = -z
    airspeed = np.sqrt(u**2 + v**2 + w_vel**2)
    shared_state["altitude_ft"] = float(altitude_m * M_TO_FT)
    shared_state["airspeed_kts"] = float(airspeed * 1.944)
    shared_state["heading_deg"] = float(np.rad2deg(psi) % 360)
    shared_state["distance_nm"] = float(distance_to_khut / 1852.0)
    shared_state["roll_deg"] = float(np.rad2deg(phi))
    shared_state["pitch_deg"] = float(np.rad2deg(theta))
    shared_state["euler"] = [float(np.rad2deg(phi)), float(np.rad2deg(theta)), float(np.rad2deg(psi))]

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
    print("  RESIDUAL RL V2 FLIGHT - EXPERT + NN CORRECTIONS (7 CONTROLS)")
    print("="*70)
    print(f"  Residual scale: {residual_scale} (NN can adjust +/-{residual_scale*100:.0f}%)")
    if model_path:
        print(f"  Model: {model_path}")
    else:
        print("  Mode: PURE EXPERT (no NN corrections)")
    print("="*70)
    print("\nOpen browser: http://localhost:8000")
    print("Press Ctrl+C to stop\n")

    # Load model if provided
    model = None
    is_v2_model = False  # v2 has 7 action dims, v1 has 4

    if model_path and Path(model_path).exists():
        print(f"Loading residual policy: {model_path}")
        model = PPO.load(model_path)

        # Detect model version from action space
        action_dim = model.action_space.shape[0]
        if action_dim == 7:
            is_v2_model = True
            print(f"Model loaded: v2 (7 controls)")
        else:
            print(f"Model loaded: v1 (4 controls)")
    else:
        print("Running PURE EXPERT (no residual NN)")

    # Set pilot mode and sim speed in shared state
    if model is None:
        shared_state["pilot_mode"] = "EXPERT"
    elif is_v2_model:
        shared_state["pilot_mode"] = "AI+EXPERT"
    else:
        shared_state["pilot_mode"] = "HYBRID"
    shared_state["sim_speed"] = sim_speed

    # Residual scales per control (v2 only)
    # Lower scales - NN should output ~0 during ground roll (aircraft is symmetric)
    residual_scales = np.array([
        0.15,  # throttle
        0.10,  # aileron - small, wings should stay level
        0.15,  # elevator
        0.10,  # rudder - small, should be ~0 during ground roll
        0.10,  # flaps
        0.15,  # spoilers
        0.10,  # brakes
    ], dtype=np.float32)

    # Setup
    sim = FlightSimulator(n_instances=1, dt=dt, use_gpu=True)
    expert = TriangleInterceptController(cruise_altitude_ft=5500.0)

    khut_x, khut_y = 52800.0, -21300.0
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

                # Expert action (7 dims)
                expert_action = expert.compute_action(state, sim_time)
                phase = expert.phase

                # Residual correction
                if model is not None:
                    if is_v2_model:
                        # V2 model: 7 control outputs - NN active on ALL phases
                        obs = build_obs_v2(state, expert_action, phase, khut_x, khut_y, cruise_alt_m)
                        residual, _ = model.predict(obs, deterministic=True)

                        # Apply residual to all 7 controls (all phases)
                        throttle = np.clip(expert_action[0] + residual_scales[0] * residual[0], 0, 1)
                        aileron = np.clip(expert_action[1] + residual_scales[1] * residual[1], -1, 1)
                        elevator = np.clip(expert_action[2] + residual_scales[2] * residual[2], -1, 1)
                        rudder = np.clip(expert_action[3] + residual_scales[3] * residual[3], -1, 1)
                        flaps = np.clip(expert_action[4] + residual_scales[4] * residual[4], 0, 1)
                        spoilers = np.clip(expert_action[5] + residual_scales[5] * residual[5], 0, 1)
                        brakes = np.clip(expert_action[6] + residual_scales[6] * residual[6], 0, 1)

                    else:
                        # V1 model: 4 control outputs (legacy)
                        obs = build_obs_v1(state, expert_action, khut_x, khut_y, cruise_alt_m)
                        residual, _ = model.predict(obs, deterministic=True)

                        # V1: Only 4 controls, fallback to expert for phases not trained
                        if phase in [XCPhase.GROUND_ROLL, XCPhase.ROTATION]:
                            throttle = np.clip(expert_action[0] + residual_scale * residual[0], 0, 1)
                            aileron = 0.0
                            elevator = np.clip(expert_action[2] + residual_scale * residual[2], -1, 1)
                            rudder = 0.0
                        elif phase in [XCPhase.TURN_TO_INTERCEPT, XCPhase.INTERCEPT_LEG, XCPhase.FINAL_APPROACH, XCPhase.SHORT_FINAL, XCPhase.LANDING, XCPhase.LANDED]:
                            throttle = float(expert_action[0])
                            aileron = float(expert_action[1])
                            elevator = float(expert_action[2])
                            rudder = float(expert_action[3])
                        else:
                            throttle = np.clip(expert_action[0] + residual_scale * residual[0], 0, 1)
                            aileron = np.clip(expert_action[1] + residual_scale * residual[1], -1, 1)
                            elevator = np.clip(expert_action[2] + residual_scale * residual[2], -1, 1)
                            rudder = np.clip(expert_action[3] + residual_scale * residual[3], -1, 1)

                        flaps = float(expert_action[4]) if len(expert_action) > 4 else 0.0
                        spoilers = float(expert_action[5]) if len(expert_action) > 5 else 0.0
                        brakes = float(expert_action[6]) if len(expert_action) > 6 else 0.0

                else:
                    # Pure expert
                    residual = np.zeros(7)
                    throttle = float(expert_action[0])
                    aileron = float(expert_action[1])
                    elevator = float(expert_action[2])
                    rudder = float(expert_action[3])
                    flaps = float(expert_action[4]) if len(expert_action) > 4 else 0.0
                    spoilers = float(expert_action[5]) if len(expert_action) > 5 else 0.0
                    brakes = float(expert_action[6]) if len(expert_action) > 6 else 0.0

                # Apply ALL 7 controls
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
                phase_name = phase.name
                update_telemetry(state, throttle, aileron, elevator, rudder, flaps, spoilers, phase_name, dist_m)

                if step % 1500 == 0:
                    if model and is_v2_model:
                        res_str = f"[{residual[0]:+.2f},{residual[1]:+.2f},{residual[2]:+.2f},{residual[3]:+.2f},{residual[4]:+.2f},{residual[5]:+.2f},{residual[6]:+.2f}]"
                    elif model:
                        res_str = f"[{residual[0]:+.2f},{residual[1]:+.2f},{residual[2]:+.2f},{residual[3]:+.2f}]"
                    else:
                        res_str = "[pure expert]"
                    print(f"  {sim_time:6.1f}s | {phase_name:>18s} | {alt:5.0f}m | {ias:5.1f}kt | {hdg:3.0f}° | {dist:5.1f}nm | {res_str[:40]}")

                # Check termination - must be in LANDED phase AND on ground (alt < 10m)
                if phase == XCPhase.LANDED and alt < 10:
                    print(f"\n  LANDED SUCCESSFULLY! (alt={alt:.1f}m, dist={dist:.2f}nm)")
                    break
                if alt < -10 or (alt > 100 and ias < 40):
                    print(f"\n  CRASH at {sim_time:.1f}s")
                    break

                step += 1
                sim_time += dt
                time.sleep(dt / sim_speed)

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
    parser.add_argument('--model', type=str, default='checkpoints/residual_ppo_v2/best_model.zip')
    parser.add_argument('--residual-scale', type=float, default=0.15)
    parser.add_argument('--pure-expert', action='store_true', help='Run pure expert without NN')
    args = parser.parse_args()

    model_path = None if args.pure_expert else args.model
    asyncio.run(main_async(model_path, args.residual_scale))


if __name__ == "__main__":
    main()
