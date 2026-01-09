#!/usr/bin/env python3
"""Cross-Country Flight: SN65 -> KHUT with 3D Telemetry Viewer

Coordinate Systems:
- Simulation (NED): X=North, Y=East, Z=Down (negative Z is altitude)
- Viewer (Three.js): X=East, Y=Up, Z=South

Transform: viewer_x = sim_y, viewer_y = -sim_z (altitude), viewer_z = -sim_x
"""

import asyncio
import threading
import time
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))

from triangle_controller import TriangleInterceptController as CrossCountryController, XCPhase
from flight_dynamics import FlightSimulator, StateIndex
from aida_sim.io.telemetry import telemetry_server, get_command

TELEMETRY_UNITS = "metric"
start_phase = 'ground_roll'  # Default starting phase

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
    "phase": "GROUND_ROLL",
    "altitude_ft": 0.0,
    "airspeed_kts": 0.0,
    "heading_deg": 0.0,
    "distance_nm": 0.0,
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


def update_telemetry(state, action, phase_name, distance_to_khut):
    x, y, z = state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z]
    u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]
    phi, theta, psi = state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI]
    p, q, r = state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]

    altitude_m = -z
    altitude_ft = altitude_m * 3.28084
    airspeed = np.sqrt(u**2 + v**2 + w**2)
    # Decouple display heading during takeoff phases
    # During GROUND_ROLL, ROTATION, INITIAL_CLIMB: show runway heading (4 deg)
    # After INITIAL_CLIMB (CLIMB onwards): show actual aircraft heading
    takeoff_phases = ["GROUND_ROLL", "ROTATION", "INITIAL_CLIMB"]
    if phase_name in takeoff_phases:
        heading = 4.0  # True runway heading for display
    else:
        heading = np.rad2deg(psi) % 360

    # Convert NED coordinates from meters to feet for viewer
    # Raw simulation coords: X=North, Y=East, Z=Down (negative Z is altitude)
    M_TO_FT = 3.28084
    shared_state["position"] = [float(x * M_TO_FT), float(y * M_TO_FT), float(z * M_TO_FT)]
    
    quat = euler_to_quaternion(phi, theta, psi)
    shared_state["quaternion"] = [float(q) for q in quat]
    # Convert velocity from m/s to ft/s
    shared_state["velocity"] = [float(u * M_TO_FT), float(v * M_TO_FT), float(w * M_TO_FT)]
    shared_state["rates"] = [float(p), float(q), float(r)]

    shared_state["throttle"] = float(np.clip(action[0], 0.0, 1.0))
    shared_state["surfaces"] = [
        float(np.clip(action[1], -1.0, 1.0)),
        float(np.clip(action[2], -1.0, 1.0)),
        float(np.clip(action[3], -1.0, 1.0))
    ]
    shared_state["flaps"] = float(np.clip(action[4], 0.0, 1.0))
    shared_state["spoilers"] = float(np.clip(action[5], 0.0, 1.0))
    shared_state["brakes"] = float(np.clip(action[6], 0.0, 1.0)) if len(action) > 6 else 0.0

    shared_state["phase"] = phase_name
    shared_state["altitude_ft"] = float(altitude_ft)
    shared_state["airspeed_kts"] = float(airspeed * 1.944)
    shared_state["heading_deg"] = float(heading)
    shared_state["distance_nm"] = float(distance_to_khut / 1852.0)

    # Euler angles for attitude indicator (in degrees)
    shared_state["roll_deg"] = float(np.rad2deg(phi))
    shared_state["pitch_deg"] = float(np.rad2deg(theta))
    shared_state["euler"] = [float(np.rad2deg(phi)), float(np.rad2deg(theta)), float(np.rad2deg(psi))]
    
    # Angle of Attack (alpha) and sideslip (beta) in degrees
    if airspeed > 1.0:
        alpha_rad = np.arctan2(w, u)
        beta_rad = np.arcsin(np.clip(v / airspeed, -1.0, 1.0))
    else:
        alpha_rad = 0.0
        beta_rad = 0.0
    shared_state["alpha_deg"] = float(np.rad2deg(alpha_rad))
    shared_state["beta_deg"] = float(np.rad2deg(beta_rad))
    shared_state["aoa_deg"] = float(np.rad2deg(alpha_rad))


# Phase initial conditions for phase jumping
PHASE_INITIAL_CONDITIONS = {
    'ground_roll': {
        'x': -400.0, 'y': 0.0, 'z': 0.0,          # On runway threshold
        'u': 5.0, 'v': 0.0, 'w': 0.0,             # Taxi speed
        'psi': 0.0, 'theta': 0.0, 'phi': 0.0,    # Level, heading north
        'phase': XCPhase.GROUND_ROLL
    },
    'climb': {
        'x': 500.0, 'y': -200.0, 'z': -150.0,    # 150m altitude, turning west
        'u': 45.0, 'v': 0.0, 'w': 3.0,           # Climb speed
        'psi': np.deg2rad(338.0), 'theta': np.deg2rad(8.0), 'phi': 0.0,
        'phase': XCPhase.CLIMB
    },
    'cruise': {
        'x': 15065.0, 'y': -6060.0, 'z': -1676.0,  # 5500ft (1676m), en route
        'u': 55.0, 'v': 0.0, 'w': 0.0,             # Cruise speed
        'psi': np.deg2rad(338.0), 'theta': 0.0, 'phi': 0.0,
        'phase': XCPhase.CRUISE_TO_TP
    },
    'descent': {
        'x': 40000.0, 'y': -20000.0, 'z': -1200.0,  # 4000ft, approaching
        'u': 50.0, 'v': 0.0, 'w': -2.0,             # Descent
        'psi': np.deg2rad(338.0), 'theta': np.deg2rad(-3.0), 'phi': 0.0,
        'phase': XCPhase.TURN_TO_INTERCEPT
    },
    'approach': {
        # 3nm from touchdown zone, altitude calibrated for actual descent rate
        'x': 48594.0, 'y': -16943.0, 'z': -350.0,  # 1148ft to reach TDZ
        'u': 40.0, 'v': 0.0, 'w': -1.0,
        'psi': np.deg2rad(314.0), 'theta': np.deg2rad(-3.0), 'phi': 0.0,
        'phase': XCPhase.FINAL_APPROACH
    },
}


def run_xc_flight(dt=0.02, save_dataset=True, start_phase="ground_roll", sim_speed=1.0):
    print()
    print("="*70)
    print("  CROSS-COUNTRY FLIGHT: SN65 -> KHUT")
    print("="*70)
    print("Departure: Lake Waltanna (SN65) RWY 35 | Hdg 004")
    print("Arrival:   Hutchinson Regional (KHUT) RWY 31 | Hdg 314")
    print("Distance:  31 NM (57 km) | Course: 338 deg true")
    print("Cruise:    5500 ft MSL")
    if start_phase != 'ground_roll':
        print(f"Starting at phase: {start_phase.upper()}")
    print("="*70)

    dynamics = FlightSimulator(n_instances=1, dt=dt, use_gpu=False)
    print(f"FlightSimulator: 1 instances on CPU (NumPy)")

    # Get initial conditions for the selected phase
    ic = PHASE_INITIAL_CONDITIONS.get(start_phase, PHASE_INITIAL_CONDITIONS['ground_roll'])
    
    initial_states = np.zeros((1, 12), dtype=np.float32)
    initial_states[0, 0] = ic['x']      # X position
    initial_states[0, 1] = ic['y']      # Y position
    initial_states[0, 2] = ic['z']      # Z position (negative = altitude)
    initial_states[0, 3] = ic['u']      # Forward velocity
    initial_states[0, 4] = ic['v']      # Sideways velocity
    initial_states[0, 5] = ic['w']      # Vertical velocity
    initial_states[0, 6] = ic['phi']    # Roll
    initial_states[0, 7] = ic['theta']  # Pitch
    initial_states[0, 8] = ic['psi']    # Heading
    dynamics.reset(initial_states)

    controller = CrossCountryController(
        cruise_altitude_ft=5500.0,
        pattern_altitude_ft=1500.0,
    )
    
    # Set controller to the starting phase
    controller.phase = ic['phase']
    if ic['phase'].value >= XCPhase.CLIMB.value:
        controller.has_turned_to_cruise = True

    all_observations = []
    all_actions = []
    all_phases = []
    all_distances = []

    last_phase = None
    sim_time = 0.0
    max_time = 1200.0
    print_interval = 10.0
    last_print_time = 0.0
    debug_interval = 10.0
    last_debug_time = 0.0

    try:
        while sim_time < max_time:
            state = dynamics.get_states()[0]
            action = controller.compute_action(state, sim_time)
            dynamics.set_controls(action.reshape(1, -1))
            dynamics.step()

            # Use distance to touchdown zone during approach/landing
            if controller.phase.value >= XCPhase.FINAL_APPROACH.value:
                distance_to_target = np.sqrt((controller.threshold_x - state[StateIndex.X])**2 + (controller.threshold_y - state[StateIndex.Y])**2)
            else:
                distance_to_target = np.sqrt((controller.threshold_x - state[StateIndex.X])**2 + (controller.threshold_y - state[StateIndex.Y])**2)
            update_telemetry(state, action, controller.phase.name, distance_to_target)

            all_observations.append(state.copy())
            all_actions.append(action.copy())
            all_phases.append(controller.phase.value)
            all_distances.append(distance_to_target)

            if controller.phase != last_phase:
                alt = -state[StateIndex.Z]
                spd = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2)
                hdg = np.rad2deg(state[StateIndex.PSI]) % 360
                dist_nm = distance_to_target / 1852.0
                print(f"[{sim_time:6.1f}s] {controller.phase.name:18s} | "
                      f"Alt:{alt:6.0f}m ({alt*3.28084:5.0f}ft) | "
                      f"Spd:{spd:5.1f}m/s ({spd*1.944:5.1f}kts) | "
                      f"Hdg:{hdg:5.1f} | Dist:{dist_nm:5.1f}nm")
                last_phase = controller.phase
                last_print_time = sim_time

            elif sim_time - last_print_time >= print_interval:
                alt = -state[StateIndex.Z]
                spd = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2)
                hdg = np.rad2deg(state[StateIndex.PSI]) % 360
                dist_nm = distance_to_target / 1852.0
                x, y = state[StateIndex.X], state[StateIndex.Y]
                print(f"  [{sim_time:6.1f}s] X={x/1000:.1f}km Y={y/1000:.1f}km | "
                      f"Alt:{alt:5.0f}m | Hdg:{hdg:5.1f} | To KHUT:{dist_nm:5.1f}nm")
                last_print_time = sim_time

            if sim_time - last_debug_time >= debug_interval:
                x, y = state[StateIndex.X], state[StateIndex.Y]
                alt = int(-state[StateIndex.Z])
                print(f"  [DEBUG] t={int(sim_time)}s X={int(x)} Y={int(y)} Alt={alt}m Phase={controller.phase.name}")
                last_debug_time = sim_time

            if controller.phase == XCPhase.LANDED:
                # Continue until aircraft has slowed to taxi speed
                airspeed = state[StateIndex.U]
                airspeed_kts = airspeed * 1.944
                if airspeed_kts >= 10.0:  # Still rolling, continue braking
                    continue
                print()
                print("="*70)
                print("  MISSION COMPLETE - LANDED AT KHUT!")
                print(f"  Final speed: {airspeed_kts:.1f} kts")
                print("="*70)
                break

            if abs(state[0]) > 80000 or abs(state[1]) > 50000:
                print(f"OUT OF BOUNDS (X={state[0]/1000:.1f}km, Y={state[1]/1000:.1f}km)")
                break

            sim_time += dt
            time.sleep(dt / sim_speed)  # Real-time with speed multiplier

        if save_dataset and len(all_observations) > 100:
            observations = np.array(all_observations, dtype=np.float32)
            actions = np.array(all_actions, dtype=np.float32)
            phases = np.array(all_phases, dtype=np.int32)
            distances = np.array(all_distances, dtype=np.float32)

            output_dir = Path("/home/AIDA/data/xc_demos")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"xc_SN65_KHUT_{timestamp}.npz"

            np.savez_compressed(
                filename,
                observations=observations,
                actions=actions,
                phases=phases,
                distances=distances,
                scenario_type="cross_country",
                departure="SN65_35",
                arrival="KHUT_31",
                cruise_altitude_ft=5500.0,
                num_transitions=len(observations),
                dt=dt,
                total_time=sim_time,
            )
            print(f"Dataset saved: {filename}")
            print(f"  Observations: {observations.shape}")
            print(f"  Total time: {sim_time:.1f}s")

            print("Phase distribution:")
            for phase in XCPhase:
                count = np.sum(phases == phase.value)
                if count > 0:
                    pct = count / len(phases) * 100
                    print(f"  {phase.name:18s}: {count:6d} ({pct:5.1f}%)")

    except KeyboardInterrupt:
        print("Flight stopped by user")


async def main_async(dt):
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "localhost"

    print(f"[telemetry_server] Starting on 0.0.0.0:8765...")

    server_task = asyncio.create_task(
        telemetry_server(lambda: shared_state, host="0.0.0.0", port=8765)
    )

    await asyncio.sleep(1)

    print()
    print("="*70)
    print("  CESSNA 172 CROSS-COUNTRY: SN65 -> KHUT")
    print("="*70)
    print(f"Open browser to http://{local_ip}:8000")
    print("="*70)

    sim_thread = threading.Thread(target=run_xc_flight, args=(dt, True, start_phase, 3.0), daemon=True)  # 3x speed
    sim_thread.start()

    try:
        await server_task
    except KeyboardInterrupt:
        print("Shutting down...")


def main():
    parser = argparse.ArgumentParser(description='Cross-Country Flight: SN65 -> KHUT')
    parser.add_argument('--phase', '-p', type=str, default='ground_roll',
                        choices=['ground_roll', 'climb', 'cruise', 'descent', 'approach'],
                        help='Starting phase (default: ground_roll)')
    args = parser.parse_args()
    
    # Store phase globally so main_async can access it
    global start_phase
    start_phase = args.phase
    
    asyncio.run(main_async(0.02))
    return 0


if __name__ == "__main__":
    sys.exit(main())
