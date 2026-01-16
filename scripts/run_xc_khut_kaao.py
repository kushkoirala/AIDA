#!/usr/bin/env python3
"""Cross-Country Flight: KHUT -> KAAO with 3D Telemetry Viewer

Uses the GeneralizedXCController for flexible airport-to-airport routing.

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

from generalized_xc_controller import (
    GeneralizedXCController, XCPhase, AirportConfig,
    KANSAS_AIRPORTS, create_controller
)
from flight_dynamics import FlightSimulator, StateIndex
from aida_sim.io.telemetry import telemetry_server, get_command
from earth_model import EarthModel, curvature_altitude_correction, M_TO_NM

# Unit conversions
M_TO_FT = 3.28084
FT_TO_M = 0.3048
NM_TO_FT = 6076.12
KTS_TO_FPS = 1.68781

TELEMETRY_UNITS = "metric"

# Flight configuration
ORIGIN = "KHUT"
DESTINATION = "KAAO"
CRUISE_ALTITUDE_FT = 5500.0  # Standard VFR cruise altitude

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
    "origin": ORIGIN,
    "destination": DESTINATION,
    # Airport configuration for viewer
    "origin_config": None,  # Will be set in run_xc_flight
    "dest_config": None,
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


def update_telemetry(state, action, phase_name, distance_to_dest, earth_model=None):
    x, y, z = state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z]
    u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]
    phi, theta, psi = state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI]
    p, q, r = state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]

    # Simulator altitude (flat Earth model)
    altitude_sim_m = -z

    # Apply Earth curvature correction
    # As aircraft travels away from origin, ground "drops" due to curvature
    # This means true MSL altitude increases relative to flat model
    if earth_model is not None:
        curvature_correction_m = earth_model.get_curvature_correction(x, y)
    else:
        curvature_correction_m = 0.0

    # AGL altitude (above local ground, accounting for curvature)
    altitude_agl_m = altitude_sim_m
    # MSL altitude (simulator altitude + curvature correction + origin elevation)
    altitude_msl_m = altitude_sim_m + curvature_correction_m

    altitude_ft = altitude_agl_m * M_TO_FT  # Display AGL for consistency
    airspeed = np.sqrt(u**2 + v**2 + w**2)

    # Decouple display heading during takeoff phases
    takeoff_phases = ["GROUND_ROLL", "ROTATION", "INITIAL_CLIMB"]
    if phase_name in takeoff_phases:
        heading = KANSAS_AIRPORTS[ORIGIN].runway_heading_deg
    else:
        heading = np.rad2deg(psi) % 360

    # Convert NED coordinates from meters to feet for viewer
    shared_state["position"] = [float(x * M_TO_FT), float(y * M_TO_FT), float(z * M_TO_FT)]

    quat = euler_to_quaternion(phi, theta, psi)
    shared_state["quaternion"] = [float(q) for q in quat]
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
    shared_state["altitude_msl_ft"] = float(altitude_msl_m * M_TO_FT)
    shared_state["curvature_correction_ft"] = float(curvature_correction_m * M_TO_FT)
    shared_state["airspeed_kts"] = float(airspeed * 1.944)
    shared_state["heading_deg"] = float(heading)
    shared_state["distance_nm"] = float(distance_to_dest / 1852.0)

    # Euler angles for attitude indicator
    shared_state["roll_deg"] = float(np.rad2deg(phi))
    shared_state["pitch_deg"] = float(np.rad2deg(theta))
    shared_state["euler"] = [float(np.rad2deg(phi)), float(np.rad2deg(theta)), float(np.rad2deg(psi))]

    # Angle of Attack and sideslip
    if airspeed > 1.0:
        alpha_rad = np.arctan2(w, u)
        beta_rad = np.arcsin(np.clip(v / airspeed, -1.0, 1.0))
    else:
        alpha_rad = 0.0
        beta_rad = 0.0
    shared_state["alpha_deg"] = float(np.rad2deg(alpha_rad))
    shared_state["beta_deg"] = float(np.rad2deg(beta_rad))
    shared_state["aoa_deg"] = float(np.rad2deg(alpha_rad))

    # Sim time for viewer sync
    shared_state["sim_time"] = shared_state.get("sim_time", 0.0)


def run_xc_flight(dt=0.02, save_dataset=True, sim_speed=2.0):
    """Run the cross-country flight simulation."""

    # Get airport configs
    origin = KANSAS_AIRPORTS[ORIGIN]
    destination = KANSAS_AIRPORTS[DESTINATION]

    # Initialize Earth model for curvature corrections
    # Origin airport is the reference point for the local NED frame
    earth = EarthModel(
        origin_lat_deg=origin.lat,
        origin_lon_deg=origin.lon,
        origin_alt_m=origin.elevation_ft * FT_TO_M
    )

    # Calculate great circle distance and curvature at destination
    gc_distance_m = earth.great_circle_distance(
        origin.lat, origin.lon,
        destination.lat, destination.lon
    )
    dest_north, dest_east, _ = earth.lla_to_ned(
        destination.lat, destination.lon, destination.elevation_ft * FT_TO_M
    )
    max_curvature_correction_m = earth.get_curvature_correction(dest_north, dest_east)

    print()
    print("="*70)
    print(f"  CROSS-COUNTRY FLIGHT: {ORIGIN} -> {DESTINATION}")
    print("="*70)
    print(f"Departure: {origin.name} ({origin.icao}) RWY {int(origin.runway_heading_deg/10):02d}")
    print(f"Arrival:   {destination.name} ({destination.icao}) RWY {int(destination.runway_heading_deg/10):02d}")

    # Calculate distance
    dx = destination.x_ft - origin.x_ft
    dy = destination.y_ft - origin.y_ft
    distance_ft = np.sqrt(dx**2 + dy**2)
    distance_nm = distance_ft / NM_TO_FT
    course_deg = np.rad2deg(np.arctan2(dy, dx)) % 360

    print(f"Distance:  {distance_nm:.1f} NM | Course: {course_deg:.0f}° true")
    print(f"Cruise:    {CRUISE_ALTITUDE_FT:.0f} ft MSL")
    print(f"Earth Curvature: {max_curvature_correction_m:.1f}m ({max_curvature_correction_m*M_TO_FT:.0f}ft) at destination")
    print("="*70)

    # Initialize flight dynamics
    dynamics = FlightSimulator(n_instances=1, dt=dt, use_gpu=False)
    print(f"FlightSimulator: 1 instance on CPU (NumPy)")

    # Initial position at origin airport runway
    # Aircraft starts on runway, heading runway direction
    runway_hdg_rad = np.deg2rad(origin.runway_heading_deg)

    # Starting position: 400m behind runway threshold (for takeoff roll)
    start_offset = -400.0  # meters behind threshold
    start_x = origin.x_ft * FT_TO_M + start_offset * np.cos(runway_hdg_rad)
    start_y = origin.y_ft * FT_TO_M + start_offset * np.sin(runway_hdg_rad)
    # Z is down in NED, so negative Z = altitude above ground
    # Simulator ground is always at Z=0 (no terrain model)
    # Aircraft starts on ground
    start_z = 0.0

    initial_states = np.zeros((1, 12), dtype=np.float32)
    initial_states[0, StateIndex.X] = start_x
    initial_states[0, StateIndex.Y] = start_y
    initial_states[0, StateIndex.Z] = start_z
    initial_states[0, StateIndex.U] = 5.0  # Taxi speed m/s
    initial_states[0, StateIndex.V] = 0.0
    initial_states[0, StateIndex.W] = 0.0
    initial_states[0, StateIndex.PHI] = 0.0
    initial_states[0, StateIndex.THETA] = 0.0
    initial_states[0, StateIndex.PSI] = runway_hdg_rad
    dynamics.reset(initial_states)

    # Create controller with origin at KHUT (need to adjust coordinates)
    # The generalized controller expects origin at (0,0), so we need to adjust
    # Create custom airport configs relative to KHUT as origin
    khut = KANSAS_AIRPORTS["KHUT"]
    kaao = KANSAS_AIRPORTS["KAAO"]

    # Recalculate KAAO position relative to KHUT
    kaao_x_from_khut = kaao.x_ft - khut.x_ft
    kaao_y_from_khut = kaao.y_ft - khut.y_ft

    origin_config = AirportConfig(
        icao=khut.icao,
        name=khut.name,
        x_ft=0.0,  # Origin is at (0,0) for controller
        y_ft=0.0,
        elevation_ft=khut.elevation_ft,
        runway_heading_deg=khut.runway_heading_deg,
        lat=khut.lat,
        lon=khut.lon
    )

    dest_config = AirportConfig(
        icao=kaao.icao,
        name=kaao.name,
        x_ft=kaao_x_from_khut,
        y_ft=kaao_y_from_khut,
        elevation_ft=kaao.elevation_ft,
        runway_heading_deg=kaao.runway_heading_deg,
        lat=kaao.lat,
        lon=kaao.lon
    )

    controller = GeneralizedXCController(
        origin=origin_config,
        destination=dest_config,
        cruise_altitude_ft=CRUISE_ALTITUDE_FT,
        pattern_altitude_ft=1500.0,
    )

    # Send airport config to viewer via telemetry
    # Use absolute positions (relative to SN65 origin) for viewer rendering
    shared_state["origin_config"] = {
        "icao": origin.icao,
        "name": origin.name,
        "x_ft": origin.x_ft,
        "y_ft": origin.y_ft,
        "elevation_ft": origin.elevation_ft,
        "runway_heading_deg": origin.runway_heading_deg,
    }
    shared_state["dest_config"] = {
        "icao": destination.icao,
        "name": destination.name,
        "x_ft": destination.x_ft,
        "y_ft": destination.y_ft,
        "elevation_ft": destination.elevation_ft,
        "runway_heading_deg": destination.runway_heading_deg,
    }

    # Data collection
    all_observations = []
    all_actions = []
    all_phases = []
    all_distances = []

    last_phase = None
    sim_time = 0.0
    max_time = 2400.0  # 40 minutes max (longer for approach phases)
    print_interval = 10.0
    last_print_time = 0.0

    try:
        while sim_time < max_time:
            # Check for override commands
            cmd = get_command()
            if cmd:
                print(f"[OVERRIDE] Received command: {cmd}")
                if cmd.get('type') == 'override':
                    action_type = cmd.get('action')
                    value = cmd.get('value')
                    if action_type == 'set_altitude' and value is not None:
                        controller.set_altitude_override(float(value))
                    elif action_type == 'set_heading' and value is not None:
                        controller.set_heading_override(float(value))
                    elif action_type == 'land':
                        controller.set_land_override()

            state = dynamics.get_states()[0]

            # Transform state to controller's coordinate system (relative to KHUT)
            # The controller expects positions relative to origin airport
            state_for_controller = state.copy()
            state_for_controller[StateIndex.X] = state[StateIndex.X] - khut.x_ft * FT_TO_M
            state_for_controller[StateIndex.Y] = state[StateIndex.Y] - khut.y_ft * FT_TO_M

            action = controller.compute_action(state_for_controller, sim_time)
            dynamics.set_controls(action.reshape(1, -1))
            dynamics.step()

            # Calculate distance to destination
            dest_x_m = kaao.x_ft * FT_TO_M
            dest_y_m = kaao.y_ft * FT_TO_M
            distance_to_dest = np.sqrt(
                (state[StateIndex.X] - dest_x_m)**2 +
                (state[StateIndex.Y] - dest_y_m)**2
            )

            update_telemetry(state, action, controller.phase.name, distance_to_dest, earth)
            shared_state["sim_time"] = sim_time

            all_observations.append(state.copy())
            all_actions.append(action.copy())
            all_phases.append(controller.phase.value)
            all_distances.append(distance_to_dest)

            # Phase change logging - altitude with Earth curvature correction
            if controller.phase != last_phase:
                alt_agl = -state[StateIndex.Z]  # Simulator ground is Z=0
                curv_corr = earth.get_curvature_correction(state[StateIndex.X], state[StateIndex.Y])
                alt_msl = alt_agl + curv_corr + origin.elevation_ft * FT_TO_M
                spd = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2)
                hdg = np.rad2deg(state[StateIndex.PSI]) % 360
                dist_nm = distance_to_dest / 1852.0
                print(f"[{sim_time:6.1f}s] {controller.phase.name:18s} | "
                      f"Alt:{alt_agl*M_TO_FT:5.0f}ft AGL ({alt_msl*M_TO_FT:5.0f}ft MSL) | "
                      f"Spd:{spd:5.1f}m/s ({spd*1.944:5.1f}kts) | "
                      f"Hdg:{hdg:5.1f}° | Dist:{dist_nm:5.1f}nm")
                last_phase = controller.phase
                last_print_time = sim_time

            elif sim_time - last_print_time >= print_interval:
                alt_agl = -state[StateIndex.Z]  # Simulator ground is Z=0
                curv_corr = earth.get_curvature_correction(state[StateIndex.X], state[StateIndex.Y])
                alt_msl = alt_agl + curv_corr + origin.elevation_ft * FT_TO_M
                hdg = np.rad2deg(state[StateIndex.PSI]) % 360
                dist_nm = distance_to_dest / 1852.0
                print(f"  [{sim_time:6.1f}s] Alt:{alt_agl*M_TO_FT:5.0f}ft AGL ({alt_msl*M_TO_FT:5.0f}ft MSL, curv:{curv_corr*M_TO_FT:4.0f}ft) | Hdg:{hdg:5.1f}° | To {DESTINATION}:{dist_nm:5.1f}nm")
                last_print_time = sim_time

            # Check for landing completion
            if controller.phase == XCPhase.LANDED:
                airspeed = state[StateIndex.U]
                airspeed_kts = airspeed * 1.944
                if airspeed_kts < 10.0:
                    print()
                    print("="*70)
                    print(f"  MISSION COMPLETE - LANDED AT {DESTINATION}!")
                    print(f"  Final speed: {airspeed_kts:.1f} kts")
                    print("="*70)
                    break

            # Bounds check
            if abs(state[0]) > 100000 or abs(state[1]) > 200000:
                print(f"OUT OF BOUNDS (X={state[0]/1000:.1f}km, Y={state[1]/1000:.1f}km)")
                break

            sim_time += dt
            time.sleep(dt / sim_speed)

        # Save dataset
        if save_dataset and len(all_observations) > 100:
            observations = np.array(all_observations, dtype=np.float32)
            actions = np.array(all_actions, dtype=np.float32)
            phases = np.array(all_phases, dtype=np.int32)
            distances = np.array(all_distances, dtype=np.float32)

            output_dir = Path("/home/AIDA/data/xc_demos")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"xc_{ORIGIN}_{DESTINATION}_{timestamp}.npz"

            np.savez_compressed(
                filename,
                observations=observations,
                actions=actions,
                phases=phases,
                distances=distances,
                scenario_type="cross_country",
                departure=f"{ORIGIN}_{int(origin.runway_heading_deg/10):02d}",
                arrival=f"{DESTINATION}_{int(destination.runway_heading_deg/10):02d}",
                cruise_altitude_ft=CRUISE_ALTITUDE_FT,
                num_transitions=len(observations),
                dt=dt,
                total_time=sim_time,
            )
            print(f"Dataset saved: {filename}")

    except KeyboardInterrupt:
        print("Flight stopped by user")


async def main_async(dt=0.02, sim_speed=2.0):
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
    print(f"  CESSNA 172 CROSS-COUNTRY: {ORIGIN} -> {DESTINATION}")
    print("="*70)
    print(f"Open browser to http://{local_ip}:8000")
    print("="*70)

    sim_thread = threading.Thread(
        target=run_xc_flight,
        args=(dt, True, sim_speed),
        daemon=True
    )
    sim_thread.start()

    try:
        await server_task
    except KeyboardInterrupt:
        print("Shutting down...")


def main():
    parser = argparse.ArgumentParser(description=f'Cross-Country Flight: {ORIGIN} -> {DESTINATION}')
    parser.add_argument('--speed', '-s', type=float, default=2.0,
                        help='Simulation speed multiplier (default: 2.0)')
    args = parser.parse_args()

    asyncio.run(main_async(0.02, args.speed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
