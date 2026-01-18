#!/usr/bin/env python3
"""
Test approach and landing phases by starting aircraft near KHUT.

This test starts the aircraft already at cruise altitude, 15nm from the destination,
to verify the approach and landing phases work correctly without waiting for
the full climb and cruise phases.

Supports telemetry streaming to the AIDA viewer.
"""

import sys
import asyncio
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpu-flight-dynamics' / 'python'))

from aida_sim.trajectory.planner import TrajectoryPlanner, Waypoint, AircraftPerformance
from aida_sim.trajectory.trajectory import FlightSegment
from flight_dynamics import FlightSimulator as FlightDynamics, StateIndex
from aida_sim.control.autopilot import TrajectoryAutopilot
from aida_sim.io.telemetry import telemetry_server

M_TO_FT = 3.28084
FT_TO_M = 0.3048
NM_TO_FT = 6076.12
KTS_TO_FPS = 1.68781


def create_approach_trajectory():
    """Create a short trajectory for approach/landing testing."""
    planner = TrajectoryPlanner()

    # KHUT airport data
    khut_x = 52800.0 * M_TO_FT
    khut_y = -21300.0 * M_TO_FT
    khut_elev = 1542.0
    khut_rwy_hdg = np.deg2rad(314.0)  # Runway 31

    # Threshold position
    threshold_offset = 610.0 * M_TO_FT
    backcourse = np.deg2rad(134.0)
    threshold_x = khut_x + threshold_offset * np.cos(backcourse)
    threshold_y = khut_y + threshold_offset * np.sin(backcourse)

    # Start point: 10nm out on final approach course
    start_dist = 10.0 * NM_TO_FT
    start_x = threshold_x + start_dist * np.cos(backcourse)
    start_y = threshold_y + start_dist * np.sin(backcourse)
    start_alt = 5500.0  # Cruise altitude

    # Intermediate point: 5nm out, start descent
    int_dist = 5.0 * NM_TO_FT
    int_x = threshold_x + int_dist * np.cos(backcourse)
    int_y = threshold_y + int_dist * np.sin(backcourse)
    int_alt = 3500.0  # Descending

    # Final approach fix: 3nm from threshold
    faf_dist = 3.0 * NM_TO_FT
    faf_x = threshold_x + faf_dist * np.cos(backcourse)
    faf_y = threshold_y + faf_dist * np.sin(backcourse)
    faf_alt = khut_elev + faf_dist * np.tan(np.deg2rad(3.0))  # 3 degree glideslope

    waypoints = [
        # Starting point (cruise)
        Waypoint(x=start_x, y=start_y, altitude=start_alt,
                 heading=khut_rwy_hdg, name="START",
                 segment=FlightSegment.CRUISE),

        # Begin descent
        Waypoint(x=int_x, y=int_y, altitude=int_alt,
                 heading=khut_rwy_hdg, name="TOD",
                 segment=FlightSegment.DESCENT),

        # Final approach fix
        Waypoint(x=faf_x, y=faf_y, altitude=faf_alt,
                 heading=khut_rwy_hdg, name="FAF",
                 segment=FlightSegment.APPROACH),

        # Threshold
        Waypoint(x=threshold_x, y=threshold_y, altitude=khut_elev + 50,
                 heading=khut_rwy_hdg, name="THR",
                 segment=FlightSegment.LANDING),

        # Touchdown
        Waypoint(x=threshold_x + 500 * np.cos(khut_rwy_hdg),
                 y=threshold_y + 500 * np.sin(khut_rwy_hdg),
                 altitude=khut_elev,
                 heading=khut_rwy_hdg, name="TD",
                 segment=FlightSegment.LANDING),
    ]

    trajectory = planner.plan_trajectory(waypoints, start_heading=khut_rwy_hdg)

    print(f"\nApproach Trajectory: {trajectory.name}")
    print(f"Distance: {trajectory.total_distance / NM_TO_FT:.1f} nm")
    print(f"Points: {len(trajectory)}")

    return trajectory, start_x, start_y, start_alt, khut_rwy_hdg


class FlightState:
    """Shared state for telemetry."""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.phi = 0.0
        self.theta = 0.0
        self.psi = 0.0
        self.airspeed = 0.0
        self.altitude = 0.0
        self.vs = 0.0
        self.heading = 0.0
        self.throttle = 0.0
        self.phase = "INIT"
        self.distance = 0.0
        self.sim_time = 0.0
        self.landed = False

    def to_dict(self):
        import math
        # Convert Euler angles to quaternion (ZYX convention)
        # q = qz * qy * qx
        cy = math.cos(self.psi * 0.5)
        sy = math.sin(self.psi * 0.5)
        cp = math.cos(self.theta * 0.5)
        sp = math.sin(self.theta * 0.5)
        cr = math.cos(self.phi * 0.5)
        sr = math.sin(self.phi * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return {
            # Position in NED meters (what viewer expects)
            "position": [self.x, self.y, self.z],
            # Quaternion [w, x, y, z] for orientation
            "quaternion": [qw, qx, qy, qz],
            # Euler angles in degrees
            "euler": [math.degrees(self.phi), math.degrees(self.theta), math.degrees(self.psi)],
            "roll_deg": math.degrees(self.phi),
            "pitch_deg": math.degrees(self.theta),
            "heading_deg": self.heading,
            # Flight data
            "altitude_ft": self.altitude,
            "airspeed_kts": self.airspeed,
            "vertical_speed_fpm": self.vs,
            # Velocities (approximate body frame)
            "velocity": [self.airspeed * 1.68781, 0.0, 0.0],  # u, v, w in ft/s
            # Controls
            "throttle": self.throttle,
            # Phase and navigation
            "phase": self.phase,
            "distance_nm": self.distance,
            "sim_time": self.sim_time,
            # Heartbeat for connection monitoring
            "heartbeat": int(self.sim_time * 10),
        }


async def run_flight_with_telemetry(sim_speed: float = 1.0):
    """Run the approach/landing flight with telemetry streaming."""
    import time

    # Create approach trajectory
    trajectory, start_x, start_y, start_alt, start_hdg = create_approach_trajectory()

    # Initialize simulation
    sim = FlightDynamics(n_instances=1, dt=0.02, use_gpu=False)
    state = sim.get_states()[0]
    state[StateIndex.X] = start_x * FT_TO_M
    state[StateIndex.Y] = start_y * FT_TO_M
    state[StateIndex.Z] = -start_alt * FT_TO_M
    state[StateIndex.U] = 110 * KTS_TO_FPS * FT_TO_M
    state[StateIndex.V] = 0.0
    state[StateIndex.W] = 0.0
    state[StateIndex.PSI] = start_hdg
    state[StateIndex.PHI] = 0.0
    state[StateIndex.THETA] = 0.0
    sim.states[0] = state

    # Initialize autopilot
    autopilot = TrajectoryAutopilot()
    autopilot.set_trajectory(trajectory)

    # Shared flight state for telemetry
    flight_state = FlightState()

    def get_state():
        return flight_state.to_dict()

    # Start telemetry server
    print("\nStarting telemetry server on port 8765...", flush=True)
    print("Open the viewer and connect to ws://localhost:8765", flush=True)
    print("-" * 70, flush=True)

    server_task = asyncio.create_task(
        telemetry_server(get_state, host="0.0.0.0", port=8765, interval=0.05)
    )

    # Give server time to start
    await asyncio.sleep(1.0)

    # Run simulation with real-time pacing
    dt = 0.02
    sim_time = 0.0
    last_phase = None
    last_print = -60.0
    real_start = time.time()

    print("\nRunning approach/landing simulation...", flush=True)
    print("-" * 70, flush=True)

    max_time = 20 * 60  # 20 minutes max
    try:
        while sim_time < max_time:
            state = sim.get_states()[0]
            action = autopilot.compute_action(state, sim_time)
            sim.set_controls(action.reshape(1, -1))
            sim.step()
            sim_time += dt

            # Get current state
            x = state[StateIndex.X]
            y = state[StateIndex.Y]
            z = state[StateIndex.Z]
            phi = state[StateIndex.PHI]
            theta = state[StateIndex.THETA]
            psi = state[StateIndex.PSI]
            u = state[StateIndex.U]
            v = state[StateIndex.V]
            w = state[StateIndex.W]

            alt = -z * M_TO_FT
            hdg = np.rad2deg(psi) % 360
            ias = np.sqrt(u**2 + v**2) * M_TO_FT / KTS_TO_FPS
            dist = autopilot.distance_to_destination / NM_TO_FT

            # Compute vertical speed
            vs_fps = (u * np.sin(theta) - v * np.cos(theta) * np.sin(phi)
                      - w * np.cos(theta) * np.cos(phi))
            vs_fpm = vs_fps * M_TO_FT * 60

            # Update shared state for telemetry
            flight_state.x = float(x)
            flight_state.y = float(y)
            flight_state.z = float(z)
            flight_state.phi = float(phi)
            flight_state.theta = float(theta)
            flight_state.psi = float(psi)
            flight_state.airspeed = float(ias)
            flight_state.altitude = float(alt)
            flight_state.vs = float(vs_fpm)
            flight_state.heading = float(hdg)
            flight_state.throttle = float(action[0])
            flight_state.phase = autopilot.current_segment.name
            flight_state.distance = float(dist)
            flight_state.sim_time = float(sim_time)

            # Print on segment change or every 30 seconds
            phase_changed = autopilot.current_segment != last_phase
            time_to_print = sim_time - last_print >= 30.0

            if phase_changed or time_to_print:
                print(f"T+{sim_time/60:5.1f}min | Alt: {alt:5.0f}ft | IAS: {ias:3.0f}kts | "
                      f"VS: {vs_fpm:+5.0f}fpm | Hdg: {hdg:03.0f}° | Dist: {dist:5.1f}nm | "
                      f"Seg: {autopilot.current_segment.name}", flush=True)
                last_print = sim_time
                last_phase = autopilot.current_segment

            if autopilot.has_landed:
                flight_state.landed = True
                print("-" * 70)
                print("*** LANDED SUCCESSFULLY! ***", flush=True)
                break

            # Emergency check
            if alt < -100 or alt > 10000:
                print(f"\n*** SIMULATION ERROR: Altitude out of bounds: {alt:.0f}ft ***")
                break

            # Real-time pacing
            target_real_time = real_start + sim_time / sim_speed
            sleep_time = target_real_time - time.time()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nFlight interrupted by user")

    # Final status
    state = sim.get_states()[0]
    final_alt = -state[StateIndex.Z] * M_TO_FT
    final_dist = autopilot.distance_to_destination / NM_TO_FT

    print("-" * 70)
    print(f"Final: T+{sim_time/60:.1f}min, Alt: {final_alt:.0f}ft, "
          f"Dist: {final_dist:.1f}nm, Phase: {autopilot.current_segment.name}", flush=True)

    # Keep server running for a bit after landing
    print("\nKeeping telemetry server running for 10 more seconds...")
    await asyncio.sleep(10)

    server_task.cancel()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Approach/landing test with viewer support")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Simulation speed multiplier (default: 1.0 for real-time)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Run without telemetry/viewer support (fast mode)")
    args = parser.parse_args()

    if args.no_viewer:
        # Run fast without telemetry (original behavior)
        run_fast()
    else:
        # Run with telemetry for viewer
        asyncio.run(run_flight_with_telemetry(sim_speed=args.speed))


def run_fast():
    """Run simulation as fast as possible without telemetry."""
    # Create approach trajectory
    trajectory, start_x, start_y, start_alt, start_hdg = create_approach_trajectory()

    # Initialize simulation
    sim = FlightDynamics(n_instances=1, dt=0.02, use_gpu=False)
    state = sim.get_states()[0]
    state[StateIndex.X] = start_x * FT_TO_M
    state[StateIndex.Y] = start_y * FT_TO_M
    state[StateIndex.Z] = -start_alt * FT_TO_M
    state[StateIndex.U] = 110 * KTS_TO_FPS * FT_TO_M
    state[StateIndex.V] = 0.0
    state[StateIndex.W] = 0.0
    state[StateIndex.PSI] = start_hdg
    state[StateIndex.PHI] = 0.0
    state[StateIndex.THETA] = 0.0
    sim.states[0] = state

    # Initialize autopilot
    autopilot = TrajectoryAutopilot()
    autopilot.set_trajectory(trajectory)

    dt = 0.02
    sim_time = 0.0
    last_phase = None
    last_print = -60.0

    print("\nRunning approach/landing simulation (fast mode)...", flush=True)
    print("-" * 70, flush=True)

    max_time = 20 * 60
    while sim_time < max_time:
        state = sim.get_states()[0]
        action = autopilot.compute_action(state, sim_time)
        sim.set_controls(action.reshape(1, -1))
        sim.step()
        sim_time += dt

        alt = -state[StateIndex.Z] * M_TO_FT
        hdg = np.rad2deg(state[StateIndex.PSI]) % 360
        ias = np.sqrt(state[StateIndex.U]**2 + state[StateIndex.V]**2) * M_TO_FT / KTS_TO_FPS
        dist = autopilot.distance_to_destination / NM_TO_FT

        u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]
        theta, phi = state[StateIndex.THETA], state[StateIndex.PHI]
        vs_fps = (u * np.sin(theta) - v * np.cos(theta) * np.sin(phi)
                  - w * np.cos(theta) * np.cos(phi))
        vs_fpm = vs_fps * M_TO_FT * 60

        phase_changed = autopilot.current_segment != last_phase
        time_to_print = sim_time - last_print >= 30.0

        if phase_changed or time_to_print:
            print(f"T+{sim_time/60:5.1f}min | Alt: {alt:5.0f}ft | IAS: {ias:3.0f}kts | "
                  f"VS: {vs_fpm:+5.0f}fpm | Hdg: {hdg:03.0f}° | Dist: {dist:5.1f}nm | "
                  f"Seg: {autopilot.current_segment.name}", flush=True)
            last_print = sim_time
            last_phase = autopilot.current_segment

        if autopilot.has_landed:
            print("-" * 70)
            print("*** LANDED SUCCESSFULLY! ***", flush=True)
            break

        if alt < -100 or alt > 10000:
            print(f"\n*** SIMULATION ERROR: Altitude out of bounds: {alt:.0f}ft ***")
            break

    state = sim.get_states()[0]
    final_alt = -state[StateIndex.Z] * M_TO_FT
    final_dist = autopilot.distance_to_destination / NM_TO_FT

    print("-" * 70)
    print(f"Final: T+{sim_time/60:.1f}min, Alt: {final_alt:.0f}ft, "
          f"Dist: {final_dist:.1f}nm, Phase: {autopilot.current_segment.name}", flush=True)


if __name__ == "__main__":
    main()
