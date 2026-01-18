#!/usr/bin/env python3
"""
Trajectory-Based Cross-Country Flight Demo

Demonstrates the new trajectory architecture by flying from
SN65 (Lake Waltanna) to KHUT (Hutchinson Regional).

This script uses:
- TrajectoryPlanner: Generates smooth 4D trajectory
- TrajectoryAutopilot: Follows trajectory with unified controller
- VectorFieldGuidance: For path following

Coordinate Systems:
- Simulation (NED): X=North, Y=East, Z=Down (negative Z is altitude)
- Viewer (Three.js): X=East, Y=Up, Z=South

Transform: viewer_x = sim_y, viewer_y = -sim_z (altitude), viewer_z = -sim_x

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import sys
import time
import asyncio
import math
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpu-flight-dynamics' / 'python'))

from flight_dynamics import FlightSimulator as FlightDynamics, StateIndex
from aida_sim.trajectory.planner import TrajectoryPlanner, Waypoint, AircraftPerformance
from aida_sim.trajectory.trajectory import FlightSegment
from aida_sim.control.autopilot import TrajectoryAutopilot, AutopilotConfig
from aida_sim.guidance.vector_field import VectorFieldGuidance

# Telemetry
from aida_sim.io.telemetry import telemetry_server

# Unit conversion
M_TO_FT = 3.28084
FT_TO_M = 0.3048
NM_TO_FT = 6076.12
KTS_TO_FPS = 1.68781
FPS_TO_KTS = 1 / KTS_TO_FPS

# Shared telemetry state (same format as run_xc_sn65_khut.py)
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
    "units": "metric",
    "model": "cessna172",
    "phase": "TAKEOFF",
    "altitude_ft": 0.0,
    "airspeed_kts": 0.0,
    "heading_deg": 0.0,
    "distance_nm": 0.0,
    "roll_deg": 0.0,
    "pitch_deg": 0.0,
    "euler": [0, 0, 0],
    "alpha_deg": 0.0,
    "beta_deg": 0.0,
    "aoa_deg": 0.0,
    "trajectory": None,  # Will be populated with trajectory points for viewer
}


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (rad) to quaternion [w, x, y, z]."""
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


def update_telemetry(state, action, phase_name, distance_nm, sim_time=0.0):
    """Update shared_state with current flight data (same format as working script)."""
    x, y, z = state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z]
    u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]
    phi, theta, psi = state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI]
    p, q, r = state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]

    altitude_m = -z
    altitude_ft = altitude_m * M_TO_FT
    airspeed = np.sqrt(u**2 + v**2 + w**2)
    heading = np.rad2deg(psi) % 360

    # Position in FEET for viewer (critical!)
    shared_state["position"] = [float(x * M_TO_FT), float(y * M_TO_FT), float(z * M_TO_FT)]

    quat = euler_to_quaternion(phi, theta, psi)
    shared_state["quaternion"] = [float(q) for q in quat]

    # Velocity in ft/s
    shared_state["velocity"] = [float(u * M_TO_FT), float(v * M_TO_FT), float(w * M_TO_FT)]
    shared_state["rates"] = [float(p), float(q), float(r)]

    # Control surfaces
    # Action order from autopilot: [throttle, aileron, elevator, rudder, flaps, spoilers, brakes]
    shared_state["throttle"] = float(np.clip(action[0], 0.0, 1.0))
    if len(action) >= 4:
        shared_state["surfaces"] = [
            float(np.clip(action[1], -1.0, 1.0)),  # aileron (index 1)
            float(np.clip(action[2], -1.0, 1.0)),  # elevator (index 2)
            float(np.clip(action[3], -1.0, 1.0)),  # rudder (index 3)
        ]
    if len(action) >= 5:
        shared_state["flaps"] = float(np.clip(action[4], 0.0, 1.0))
    if len(action) >= 6:
        shared_state["spoilers"] = float(np.clip(action[5], 0.0, 1.0))

    shared_state["phase"] = phase_name
    shared_state["altitude_ft"] = float(altitude_ft)
    shared_state["airspeed_kts"] = float(airspeed * 1.944)
    shared_state["heading_deg"] = float(heading)
    shared_state["distance_nm"] = float(distance_nm)

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

    # Heartbeat and sim_time for viewer connection
    shared_state["sim_time"] = float(sim_time)
    shared_state["heartbeat"] = int(sim_time * 10)


def create_sn65_khut_trajectory():
    """
    Create a flyable trajectory from SN65 to KHUT using A* graph-based routing.

    Uses the navigation graph to find optimal waypoints and generates
    a smooth trajectory through them.
    """
    from aida_sim.trajectory.route_graph import (
        create_kansas_navgraph, FlightPlanGenerator
    )

    # Create navigation graph and flight plan generator
    graph = create_kansas_navgraph()
    fp_generator = FlightPlanGenerator(graph)

    # Generate flight plan using A* routing
    flight_plan = fp_generator.generate_flight_plan(
        origin_id="SN65",
        destination_id="KHUT",
        cruise_altitude_ft=5500.0
    )

    if not flight_plan:
        raise RuntimeError("Failed to generate flight plan from SN65 to KHUT")

    print(f"\nA* Flight Plan: {len(flight_plan)} waypoints")
    for i, wp in enumerate(flight_plan):
        print(f"  {i+1:2d}. {wp['name']:12s} {wp['segment']:10s} "
              f"alt={wp['altitude_ft']:5.0f}ft hdg={wp['heading_deg']:3.0f}°")

    # Reference point for coordinate conversion (SN65)
    ref_lat = 37.594167
    ref_lon = -97.615833

    # Convert lat/lon waypoints to local XY coordinates (feet)
    def latlon_to_xy(lat, lon):
        """Convert lat/lon to local XY in feet."""
        lat_to_ft = 364567.0  # ~60nm per degree * 6076 ft/nm
        lon_to_ft = 364567.0 * np.cos(np.deg2rad(ref_lat))
        x = (lat - ref_lat) * lat_to_ft  # North
        y = (lon - ref_lon) * lon_to_ft  # East
        return x, y

    # Build waypoint list for trajectory planner
    waypoints = []
    for wp in flight_plan:
        x, y = latlon_to_xy(wp['lat'], wp['lon'])

        # Map flight plan segment to FlightSegment enum
        seg_name = wp['segment'].upper()
        if seg_name == 'TAKEOFF':
            segment = FlightSegment.TAKEOFF
        elif seg_name == 'CLIMB':
            segment = FlightSegment.CLIMB
        elif seg_name == 'CRUISE':
            segment = FlightSegment.CRUISE
        elif seg_name == 'DESCENT':
            segment = FlightSegment.DESCENT
        elif seg_name == 'APPROACH':
            segment = FlightSegment.APPROACH
        elif seg_name == 'LANDING':
            segment = FlightSegment.LANDING
        else:
            segment = FlightSegment.CRUISE

        waypoints.append(Waypoint(
            x=x,
            y=y,
            altitude=wp['altitude_ft'],
            heading=np.deg2rad(wp['heading_deg']),
            name=wp['name'],
            segment=segment,
            speed=wp['speed_kts'] * KTS_TO_FPS if wp['speed_kts'] > 0 else 0.0
        ))

    # Generate smooth trajectory through waypoints
    planner = TrajectoryPlanner()
    start_heading = np.deg2rad(flight_plan[0]['heading_deg'])
    trajectory = planner.plan_trajectory(waypoints, start_heading=start_heading)

    print(f"\nTrajectory: {trajectory.name}")
    print(f"Distance: {trajectory.total_distance / NM_TO_FT:.1f} nm")
    print(f"Expected time: {trajectory.total_time / 60:.1f} min")
    print(f"Points: {len(trajectory)}")

    return trajectory


async def run_flight_async(sim_speed: float = 2.0):
    """Run the full SN65->KHUT flight with telemetry."""
    print("\n" + "=" * 60)
    print("AIDA Trajectory Flight Demo: SN65 -> KHUT")
    print("=" * 60)

    # Create trajectory
    trajectory = create_sn65_khut_trajectory()

    # Initialize flight dynamics
    sim = FlightDynamics(n_instances=1, dt=0.02, use_gpu=False)
    state = sim.get_states()[0]

    # Set initial position on runway
    state[StateIndex.X] = 0.0
    state[StateIndex.Y] = 0.0
    state[StateIndex.Z] = -1448.0 * FT_TO_M
    state[StateIndex.U] = 50.0
    state[StateIndex.V] = 0.0
    state[StateIndex.W] = 0.0
    state[StateIndex.PSI] = np.deg2rad(4.0)
    state[StateIndex.PHI] = 0.0
    state[StateIndex.THETA] = 0.0
    sim.states[0] = state

    # Initialize autopilot
    autopilot = TrajectoryAutopilot()
    autopilot.set_trajectory(trajectory)

    # Convert trajectory to viewer format: list of [x_north, y_east, z_alt] in feet
    # Sample every ~10 points to reduce data size (trajectory has ~2000 points)
    trajectory_for_viewer = []
    step = max(1, len(trajectory.points) // 200)  # ~200 points for viewer
    for i in range(0, len(trajectory.points), step):
        pt = trajectory.points[i]
        trajectory_for_viewer.append([float(pt.x), float(pt.y), float(pt.z)])
    # Always include last point
    if len(trajectory.points) > 0:
        pt = trajectory.points[-1]
        trajectory_for_viewer.append([float(pt.x), float(pt.y), float(pt.z)])
    shared_state["trajectory"] = trajectory_for_viewer
    print(f"Trajectory prepared for viewer: {len(trajectory_for_viewer)} points")

    # Start telemetry server (using shared_state dict like run_xc_sn65_khut.py)
    print("\n[telemetry_server] Starting on 0.0.0.0:8765...")
    print("Open viewer at http://localhost:8080")
    print("-" * 60, flush=True)

    server_task = asyncio.create_task(
        telemetry_server(lambda: shared_state, host="0.0.0.0", port=8765, interval=0.05)
    )

    await asyncio.sleep(1.0)

    # Flight loop
    dt = 0.02
    sim_time = 0.0
    last_print_time = -60.0
    real_start = time.time()

    print("\nStarting flight from SN65...", flush=True)
    print("-" * 60, flush=True)

    max_time = 90 * 60  # 90 minutes max

    # Batch physics steps to reduce async overhead
    # At dt=0.02s, we do 50 steps/sec of sim time
    # Run physics in batches, only yield every 0.1s of sim time (5 steps)
    steps_per_batch = max(1, int(0.1 / dt))  # 5 steps per batch
    last_yield_time = 0.0

    try:
        while sim_time < max_time:
            # Run a batch of physics steps
            for _ in range(steps_per_batch):
                state = sim.get_states()[0]
                action = autopilot.compute_action(state, sim_time)
                sim.set_controls(action.reshape(1, -1))
                sim.step()
                sim_time += dt

                if autopilot.has_landed:
                    break

            # Get updated state after batch
            state = sim.get_states()[0]

            # Calculate distance for telemetry
            dist = autopilot.distance_to_destination / NM_TO_FT

            # Update telemetry (same format as working script)
            update_telemetry(state, action, autopilot.current_segment.name, dist, sim_time)

            # Extract for console printing
            alt = -state[StateIndex.Z] * M_TO_FT
            hdg = np.rad2deg(state[StateIndex.PSI]) % 360
            u = state[StateIndex.U]
            v = state[StateIndex.V]
            ias = np.sqrt(u**2 + v**2) * M_TO_FT * FPS_TO_KTS
            theta = state[StateIndex.THETA]
            phi = state[StateIndex.PHI]
            w = state[StateIndex.W]
            vs_fps = (u * np.sin(theta) - v * np.cos(theta) * np.sin(phi)
                      - w * np.cos(theta) * np.cos(phi))
            vs_fpm = vs_fps * M_TO_FT * 60

            # Print every 30 seconds
            if sim_time - last_print_time >= 30.0:
                print(f"T+{sim_time/60:5.1f}min | Alt: {alt:5.0f}ft | IAS: {ias:3.0f}kts | "
                      f"VS: {vs_fpm:+5.0f}fpm | Hdg: {hdg:03.0f}° | Dist: {dist:5.1f}nm | "
                      f"Seg: {autopilot.current_segment.name}", flush=True)
                last_print_time = sim_time

            if autopilot.has_landed:
                print("-" * 60)
                print("*** LANDED SUCCESSFULLY AT KHUT! ***", flush=True)
                break

            if alt < -100 or alt > 15000:
                print(f"\n*** ERROR: Altitude out of bounds: {alt:.0f}ft ***")
                break

            # Real-time pacing - yield every 0.1s sim time for telemetry
            target_real_time = real_start + sim_time / sim_speed
            sleep_time = target_real_time - time.time()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            elif sim_time - last_yield_time >= 0.5:
                # Yield at least every 0.5s sim time to keep telemetry alive
                await asyncio.sleep(0)
                last_yield_time = sim_time

    except KeyboardInterrupt:
        print("\n\nFlight interrupted")

    # Final status
    state = sim.get_states()[0]
    final_alt = -state[StateIndex.Z] * M_TO_FT
    final_dist = autopilot.distance_to_destination / NM_TO_FT

    print("-" * 60)
    print(f"Final: T+{sim_time/60:.1f}min, Alt: {final_alt:.0f}ft, "
          f"Dist: {final_dist:.1f}nm, Phase: {autopilot.current_segment.name}", flush=True)

    # Keep server running briefly
    print("\nKeeping telemetry active for 10 more seconds...")
    await asyncio.sleep(10)
    server_task.cancel()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trajectory-based flight demo")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Simulation speed multiplier (default: 2.0)")

    args = parser.parse_args()

    asyncio.run(run_flight_async(sim_speed=args.speed))
