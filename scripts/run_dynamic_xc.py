#!/usr/bin/env python3
"""Dynamic Cross-Country Flight Server with 3D Telemetry Viewer

Supports dynamic flight creation from the viewer UI - can restart flights
between any configured Kansas airports.

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
CRUISE_ALTITUDE_FT = 5500.0  # Default cruise altitude

# Shared state for telemetry
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
    "phase": "IDLE",
    "altitude_ft": 0.0,
    "airspeed_kts": 0.0,
    "heading_deg": 0.0,
    "distance_nm": 0.0,
    "origin": None,
    "destination": None,
    "origin_config": None,
    "dest_config": None,
    "sim_time": 0.0,
    # Flight plan parameters (for LLM context)
    "cruise_altitude_ft": CRUISE_ALTITUDE_FT,
    "target_altitude_ft": 0.0,  # Current altitude the controller is commanding
    "target_heading_deg": 0.0,  # Current heading the controller is commanding
}

# Flight state management
flight_state = {
    "running": False,
    "restart_requested": False,
    "new_origin": None,
    "new_destination": None,
    "new_origin_config": None,
    "new_dest_config": None,
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


def body_to_ned_velocity(u, v, w, phi, theta, psi):
    """
    Transform body-frame velocity (u, v, w) to NED world-frame velocity.

    Body frame: u=forward, v=right, w=down
    NED frame: X=North, Y=East, Z=Down

    Uses rotation matrix from body to NED.
    """
    # Precompute trig functions
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)

    # Rotation matrix from body to NED (standard aerospace convention)
    # First row: X_ned = ...
    vx_ned = (ctheta * cpsi) * u + (sphi * stheta * cpsi - cphi * spsi) * v + (cphi * stheta * cpsi + sphi * spsi) * w
    # Second row: Y_ned = ...
    vy_ned = (ctheta * spsi) * u + (sphi * stheta * spsi + cphi * cpsi) * v + (cphi * stheta * spsi - sphi * cpsi) * w
    # Third row: Z_ned = ...
    vz_ned = (-stheta) * u + (sphi * ctheta) * v + (cphi * ctheta) * w

    return vx_ned, vy_ned, vz_ned


def update_telemetry(state, action, phase_name, distance_to_dest, origin, earth_model=None):
    x, y, z = state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z]
    u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]
    phi, theta, psi = state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI]
    p, q, r = state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]

    # Simulator altitude (flat Earth model)
    altitude_sim_m = -z

    # Apply Earth curvature correction
    if earth_model is not None:
        curvature_correction_m = earth_model.get_curvature_correction(x, y)
    else:
        curvature_correction_m = 0.0

    altitude_agl_m = altitude_sim_m
    altitude_msl_m = altitude_sim_m + curvature_correction_m
    altitude_ft = altitude_agl_m * M_TO_FT
    airspeed = np.sqrt(u**2 + v**2 + w**2)

    # Decouple display heading during takeoff phases
    takeoff_phases = ["GROUND_ROLL", "ROTATION", "INITIAL_CLIMB"]
    if phase_name in takeoff_phases and origin:
        heading = origin.runway_heading_deg
    else:
        heading = np.rad2deg(psi) % 360

    shared_state["position"] = [float(x * M_TO_FT), float(y * M_TO_FT), float(z * M_TO_FT)]

    quat = euler_to_quaternion(phi, theta, psi)
    shared_state["quaternion"] = [float(q) for q in quat]

    # Transform body-frame velocity to NED world-frame velocity
    vx_ned, vy_ned, vz_ned = body_to_ned_velocity(u, v, w, phi, theta, psi)
    # Display as [North, East, Up] (ft/s) — negate vz_ned so positive = climb
    shared_state["velocity"] = [float(vx_ned * M_TO_FT), float(vy_ned * M_TO_FT), float(-vz_ned * M_TO_FT)]
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

    # Vertical speed: vz_ned is positive-down (NED), so negate for positive-up (climb)
    vertical_speed_mps = -vz_ned
    shared_state["vertical_speed_fpm"] = float(vertical_speed_mps * 196.85)

    shared_state["phase"] = phase_name
    shared_state["altitude_ft"] = float(altitude_ft)
    shared_state["altitude_msl_ft"] = float(altitude_msl_m * M_TO_FT)
    shared_state["curvature_correction_ft"] = float(curvature_correction_m * M_TO_FT)
    shared_state["airspeed_kts"] = float(airspeed * 1.944)
    shared_state["heading_deg"] = float(heading)
    shared_state["distance_nm"] = float(distance_to_dest / 1852.0)

    shared_state["roll_deg"] = float(np.rad2deg(phi))
    shared_state["pitch_deg"] = float(np.rad2deg(theta))
    shared_state["euler"] = [float(np.rad2deg(phi)), float(np.rad2deg(theta)), float(np.rad2deg(psi))]

    if airspeed > 1.0:
        alpha_rad = np.arctan2(w, u)
        beta_rad = np.arcsin(np.clip(v / airspeed, -1.0, 1.0))
    else:
        alpha_rad = 0.0
        beta_rad = 0.0
    shared_state["alpha_deg"] = float(np.rad2deg(alpha_rad))
    shared_state["beta_deg"] = float(np.rad2deg(beta_rad))
    shared_state["aoa_deg"] = float(np.rad2deg(alpha_rad))


def setup_flight(origin_icao, dest_icao):
    """Setup a new flight between two airports. Returns (origin, dest, controller, dynamics, earth)."""

    origin = KANSAS_AIRPORTS[origin_icao]
    destination = KANSAS_AIRPORTS[dest_icao]

    # Initialize Earth model
    earth = EarthModel(
        origin_lat_deg=origin.lat,
        origin_lon_deg=origin.lon,
        origin_alt_m=origin.elevation_ft * FT_TO_M
    )

    # Initialize flight dynamics
    dynamics = FlightSimulator(n_instances=1, dt=0.02, use_gpu=False)

    # Initial position at origin airport runway centerline
    # Key insight: The viewer's runway visual is centered at the airport reference point
    # and the centerline passes through this center. Aircraft must start ON the centerline.
    #
    # IMPORTANT: Use viewer_runway_heading_deg for positioning, NOT runway_heading_deg.
    # The viewer may render runways at simplified headings (e.g., SN65 at 0° instead of 4°).
    # Using the viewer heading ensures aircraft starts on the visual centerline.

    # Get the viewer's runway heading for positioning (may differ from real heading)
    viewer_hdg_deg = origin.get_viewer_heading_deg()
    viewer_hdg_rad = np.deg2rad(viewer_hdg_deg)
    start_offset_m = 400.0  # meters behind threshold (along runway direction)

    # Airport center in world coordinates (meters)
    airport_x_m = origin.x_ft * FT_TO_M
    airport_y_m = origin.y_ft * FT_TO_M

    # The backcourse direction for positioning behind threshold
    backcourse_rad = viewer_hdg_rad + np.pi

    # Calculate start position: offset behind threshold along runway direction
    start_x = airport_x_m + start_offset_m * np.cos(backcourse_rad)
    start_y = airport_y_m + start_offset_m * np.sin(backcourse_rad)
    start_z = 0.0

    # Aircraft heading matches viewer runway heading for visual alignment
    aircraft_heading_rad = viewer_hdg_rad

    # Debug output to verify aircraft position
    print(f"\n[INIT] Origin: {origin_icao} ({origin.name})")
    print(f"[INIT] Airport center: ({airport_x_m:.1f}m N, {airport_y_m:.1f}m E)")
    print(f"[INIT]                 ({origin.x_ft:.0f}ft N, {origin.y_ft:.0f}ft E)")
    print(f"[INIT] Display heading: {origin.runway_heading_deg:.1f}° | Viewer heading: {viewer_hdg_deg:.1f}°")
    print(f"[INIT] Start position: ({start_x:.1f}m N, {start_y:.1f}m E)")
    print(f"[INIT] Start in feet:  ({start_x*M_TO_FT:.0f}ft N, {start_y*M_TO_FT:.0f}ft E)")
    print(f"[INIT] Viewer coords:  X={start_y*M_TO_FT:.0f}ft (E), Z={-start_x*M_TO_FT:.0f}ft (-N)")
    print(f"[INIT] Offset from center: ({(start_x-airport_x_m):.1f}m N, {(start_y-airport_y_m):.1f}m E)")

    initial_states = np.zeros((1, 12), dtype=np.float32)
    initial_states[0, StateIndex.X] = start_x
    initial_states[0, StateIndex.Y] = start_y
    initial_states[0, StateIndex.Z] = start_z
    initial_states[0, StateIndex.U] = 5.0  # Taxi speed
    initial_states[0, StateIndex.PSI] = aircraft_heading_rad
    dynamics.reset(initial_states)

    # Create controller with origin at origin airport
    dest_x_from_origin = destination.x_ft - origin.x_ft
    dest_y_from_origin = destination.y_ft - origin.y_ft

    origin_config = AirportConfig(
        icao=origin.icao,
        name=origin.name,
        x_ft=0.0,
        y_ft=0.0,
        elevation_ft=origin.elevation_ft,
        runway_heading_deg=origin.runway_heading_deg,
        lat=origin.lat,
        lon=origin.lon
    )

    dest_config = AirportConfig(
        icao=destination.icao,
        name=destination.name,
        x_ft=dest_x_from_origin,
        y_ft=dest_y_from_origin,
        elevation_ft=destination.elevation_ft,
        runway_heading_deg=destination.runway_heading_deg,
        lat=destination.lat,
        lon=destination.lon
    )

    controller = GeneralizedXCController(
        origin=origin_config,
        destination=dest_config,
        cruise_altitude_ft=CRUISE_ALTITUDE_FT,
        pattern_altitude_ft=1500.0,
    )

    # Update shared state
    shared_state["origin"] = origin_icao
    shared_state["destination"] = dest_icao
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

    return origin, destination, controller, dynamics, earth


def run_flight_loop(dt=0.02, sim_speed=2.0):
    """Main flight loop that supports dynamic restarts."""
    global flight_state

    # Start in hangar mode - wait for user to select a flight from the viewer
    origin_icao = None
    dest_icao = None

    while True:
        # Check if a new flight was requested or if we're starting fresh (hangar mode)
        if origin_icao is None or dest_icao is None or flight_state["restart_requested"]:
            if flight_state["restart_requested"]:
                new_origin = flight_state["new_origin"]
                new_dest = flight_state["new_destination"]
                flight_state["restart_requested"] = False
            else:
                new_origin = None
                new_dest = None

            # If no flight specified, wait for selection from viewer (hangar mode)
            if new_origin is None or new_dest is None:
                print("\n[HANGAR] Waiting for new flight selection from viewer...")
                # Set shared state to hangar mode so viewer knows to show hangar
                shared_state["phase"] = "HANGAR"
                shared_state["origin"] = None
                shared_state["destination"] = None
                while True:
                    cmd = get_command()
                    if cmd:
                        cmd_type = cmd.get('type')
                        if cmd_type == 'start_flight':
                            origin_icao = cmd.get('origin')
                            dest_icao = cmd.get('destination')
                            if origin_icao and dest_icao:
                                print(f"[HANGAR] New flight selected: {origin_icao} -> {dest_icao}")
                                break
                    time.sleep(0.1)
            else:
                origin_icao = new_origin
                dest_icao = new_dest
                print(f"\n[RESTART] Starting new flight: {origin_icao} -> {dest_icao}")

        # Setup the flight
        try:
            origin, destination, controller, dynamics, earth = setup_flight(origin_icao, dest_icao)
        except KeyError as e:
            print(f"[ERROR] Unknown airport: {e}")
            time.sleep(1)
            continue

        # Calculate distance
        dx = destination.x_ft - origin.x_ft
        dy = destination.y_ft - origin.y_ft
        distance_ft = np.sqrt(dx**2 + dy**2)
        distance_nm = distance_ft / NM_TO_FT
        course_deg = np.rad2deg(np.arctan2(dy, dx)) % 360

        print()
        print("="*70)
        print(f"  CROSS-COUNTRY FLIGHT: {origin_icao} -> {dest_icao}")
        print("="*70)
        print(f"Departure: {origin.name} ({origin.icao}) RWY {int(origin.runway_heading_deg/10):02d}")
        print(f"Arrival:   {destination.name} ({destination.icao}) RWY {int(destination.runway_heading_deg/10):02d}")
        print(f"Distance:  {distance_nm:.1f} NM | Course: {course_deg:.0f}° true")
        print(f"Cruise:    {CRUISE_ALTITUDE_FT:.0f} ft MSL")
        print("="*70)

        flight_state["running"] = True
        last_phase = None
        sim_time = 0.0
        max_time = 2400.0
        print_interval = 10.0
        last_print_time = 0.0

        try:
            while sim_time < max_time and not flight_state["restart_requested"]:
                # Check for commands
                cmd = get_command()
                if cmd:
                    cmd_type = cmd.get('type')
                    print(f"[CMD] Received: {cmd_type}")

                    if cmd_type == 'start_flight':
                        # Extract origin and destination from command
                        new_origin = cmd.get('origin')
                        new_dest = cmd.get('destination')

                        if new_origin and new_dest:
                            flight_state["new_origin"] = new_origin
                            flight_state["new_destination"] = new_dest
                            flight_state["restart_requested"] = True
                            print(f"[CMD] Restart requested: {new_origin} -> {new_dest}")
                            break  # Exit inner loop to restart

                    elif cmd_type == 'create_flight':
                        # Just acknowledge - actual restart happens on start_flight
                        print(f"[CMD] Flight plan created: {cmd.get('origin')} -> {cmd.get('destination')}")

                    elif cmd_type == 'stop_flight':
                        # User requested return to hangar - stop current flight and wait
                        print(f"[CMD] Stop flight requested - returning to hangar mode")
                        flight_state["restart_requested"] = True
                        flight_state["new_origin"] = None
                        flight_state["new_destination"] = None
                        break  # Exit inner loop

                    elif cmd_type == 'override':
                        action_type = cmd.get('action')
                        value = cmd.get('value')
                        if action_type == 'set_altitude' and value is not None:
                            controller.set_altitude_override(float(value))
                        elif action_type == 'set_heading' and value is not None:
                            controller.set_heading_override(float(value))
                        elif action_type == 'land':
                            controller.set_land_override()

                state = dynamics.get_states()[0]

                # Transform state to controller's coordinate system
                state_for_controller = state.copy()
                state_for_controller[StateIndex.X] = state[StateIndex.X] - origin.x_ft * FT_TO_M
                state_for_controller[StateIndex.Y] = state[StateIndex.Y] - origin.y_ft * FT_TO_M

                action = controller.compute_action(state_for_controller, sim_time)
                dynamics.set_controls(action.reshape(1, -1))
                dynamics.step()

                # Calculate distance to destination
                dest_x_m = destination.x_ft * FT_TO_M
                dest_y_m = destination.y_ft * FT_TO_M
                distance_to_dest = np.sqrt(
                    (state[StateIndex.X] - dest_x_m)**2 +
                    (state[StateIndex.Y] - dest_y_m)**2
                )

                update_telemetry(state, action, controller.phase.name, distance_to_dest, origin, earth)
                shared_state["sim_time"] = sim_time

                # Update flight plan parameters for LLM context
                shared_state["cruise_altitude_ft"] = controller.cruise_altitude_ft
                shared_state["target_altitude_ft"] = controller.get_target_altitude_ft()
                shared_state["target_heading_deg"] = controller.get_target_heading_deg()

                # Phase change logging
                if controller.phase != last_phase:
                    alt_agl = -state[StateIndex.Z]
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
                    alt_agl = -state[StateIndex.Z]
                    curv_corr = earth.get_curvature_correction(state[StateIndex.X], state[StateIndex.Y])
                    alt_msl = alt_agl + curv_corr + origin.elevation_ft * FT_TO_M
                    hdg = np.rad2deg(state[StateIndex.PSI]) % 360
                    dist_nm = distance_to_dest / 1852.0
                    print(f"  [{sim_time:6.1f}s] Alt:{alt_agl*M_TO_FT:5.0f}ft AGL | Hdg:{hdg:5.1f}° | To {dest_icao}:{dist_nm:5.1f}nm")
                    last_print_time = sim_time

                # Check for landing completion
                if controller.phase == XCPhase.LANDED:
                    airspeed = state[StateIndex.U]
                    airspeed_kts = airspeed * 1.944
                    if airspeed_kts < 10.0:
                        print()
                        print("="*70)
                        print(f"  MISSION COMPLETE - LANDED AT {dest_icao}!")
                        print(f"  Final speed: {airspeed_kts:.1f} kts")
                        print("="*70)
                        print("\n  Waiting for new flight request from viewer...")

                        # Wait for restart command
                        while not flight_state["restart_requested"]:
                            cmd = get_command()
                            if cmd:
                                cmd_type = cmd.get('type')
                                if cmd_type == 'start_flight':
                                    new_origin = cmd.get('origin')
                                    new_dest = cmd.get('destination')
                                    if new_origin and new_dest:
                                        flight_state["new_origin"] = new_origin
                                        flight_state["new_destination"] = new_dest
                                        flight_state["restart_requested"] = True
                                        print(f"[CMD] New flight: {new_origin} -> {new_dest}")
                            time.sleep(0.1)
                        break

                # Bounds check
                if abs(state[0]) > 200000 or abs(state[1]) > 200000:
                    print(f"OUT OF BOUNDS (X={state[0]/1000:.1f}km, Y={state[1]/1000:.1f}km)")
                    break

                sim_time += dt
                time.sleep(dt / sim_speed)

        except KeyboardInterrupt:
            print("\nFlight stopped by user")
            break

        flight_state["running"] = False


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
    print("  DYNAMIC CROSS-COUNTRY FLIGHT SERVER")
    print("="*70)
    print(f"Open browser to http://{local_ip}:8000")
    print("Select airports in the Flight Setup panel to start a flight")
    print("="*70)

    sim_thread = threading.Thread(
        target=run_flight_loop,
        args=(dt, sim_speed),
        daemon=True
    )
    sim_thread.start()

    try:
        await server_task
    except KeyboardInterrupt:
        print("Shutting down...")


def main():
    parser = argparse.ArgumentParser(description='Dynamic Cross-Country Flight Server')
    parser.add_argument('--speed', '-s', type=float, default=2.0,
                        help='Simulation speed multiplier (default: 2.0)')
    args = parser.parse_args()

    asyncio.run(main_async(0.02, args.speed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
