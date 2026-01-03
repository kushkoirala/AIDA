#!/usr/bin/env python3
"""Run Traffic Pattern Mission with 3D Telemetry Viewer

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
from pathlib import Path
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))

from classical_mission_controller import MissionController, MissionPhase
from flight_dynamics import FlightSimulator, StateIndex
from aida_sim.io.telemetry import telemetry_server, get_command

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
    "phase": "GROUND_ROLL",
    "altitude_ft": 0.0,
    "airspeed_kts": 0.0,
    "heading_deg": 0.0,
    "sim_time": 0.0,
    "vertical_speed_fpm": 0.0,
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


def update_telemetry(state, action, phase_name, sim_time):
    """Update shared telemetry state from simulation state.

    Simulation NED: X=North, Y=East, Z=Down
    Viewer: Uses SIM_TO_VIEWER_QUAT to transform

    We send raw NED coordinates - the viewer handles the transform.
    """
    x, y, z = state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z]
    u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]
    phi, theta, psi = state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI]
    p, q, r = state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]

    altitude = -z  # NED: negative Z is up
    airspeed = np.sqrt(u**2 + v**2 + w**2)
    heading = np.rad2deg(psi) % 360

    # Vertical speed: Transform body velocity to NED frame to get true z_dot
    # z_dot = -u*sin(theta) + v*cos(theta)*sin(phi) + w*cos(theta)*cos(phi)
    # Negative z_dot = climb rate (positive when climbing since NED z is down)
    z_dot = (-u * np.sin(theta) +
             v * np.cos(theta) * np.sin(phi) +
             w * np.cos(theta) * np.cos(phi))
    vertical_speed_mps = -z_dot  # Positive when climbing
    vertical_speed_fpm = vertical_speed_mps * 196.85  # m/s to ft/min

    # Send raw NED coordinates - viewer handles the transform
    # Runway is centered at origin (0,0) in the viewer
    shared_state["position"] = [float(x), float(y), float(z)]

    quat = euler_to_quaternion(phi, theta, psi)
    shared_state["quaternion"] = [float(q) for q in quat]
    shared_state["velocity"] = [float(u), float(v), float(w)]
    shared_state["rates"] = [float(np.rad2deg(p)), float(np.rad2deg(q)), float(np.rad2deg(r))]

    shared_state["throttle"] = float(np.clip(action[0], 0, 1))
    shared_state["surfaces"] = [
        float(np.clip(action[1], -1, 1)),
        float(np.clip(action[2], -1, 1)),
        float(np.clip(action[3], -1, 1))
    ]
    shared_state["flaps"] = float(np.clip(action[4], 0, 1))
    shared_state["spoilers"] = float(np.clip(action[5], 0, 1))

    # Flight computer data
    shared_state["phase"] = phase_name
    shared_state["altitude_ft"] = float(altitude * 3.28084)
    shared_state["airspeed_kts"] = float(airspeed * 1.944)
    shared_state["heading_deg"] = float(heading)
    shared_state["sim_time"] = float(sim_time)
    shared_state["vertical_speed_fpm"] = float(vertical_speed_fpm)

    # Euler angles for attitude indicator
    shared_state["roll_deg"] = float(np.rad2deg(phi))
    shared_state["pitch_deg"] = float(np.rad2deg(theta))

    # Angle of attack (alpha) - calculated from body velocities
    # AoA = atan2(w, u) where w=down velocity, u=forward velocity
    if abs(u) > 0.1:  # Avoid division issues at low speed
        alpha_rad = np.arctan2(w, u)
        shared_state["alpha_deg"] = float(np.rad2deg(alpha_rad))
    else:
        shared_state["alpha_deg"] = 0.0


def get_phase_preset(phase_name):
    """Get initial state for a specific mission phase.

    Returns (state_array, phase_enum) for the requested phase.
    """
    state = np.zeros((1, 12), dtype=np.float32)

    if phase_name == "GROUND_ROLL":
        # Start at south end of runway
        state[0, StateIndex.X] = -250.0
        state[0, StateIndex.Y] = 0.0
        state[0, StateIndex.Z] = 0.0
        state[0, StateIndex.U] = 5.0
        state[0, StateIndex.PSI] = 0.0
        return state, MissionPhase.GROUND_ROLL

    elif phase_name == "CLIMB":
        # Climbing northbound at 500ft
        state[0, StateIndex.X] = 500.0
        state[0, StateIndex.Y] = 0.0
        state[0, StateIndex.Z] = -150.0  # 150m altitude
        state[0, StateIndex.U] = 45.0    # ~90 kts
        state[0, StateIndex.PSI] = 0.0   # North
        state[0, StateIndex.THETA] = np.deg2rad(5)  # Slight climb pitch
        return state, MissionPhase.CLIMB

    elif phase_name == "CRUISE_UPWIND":
        # Cruising north at 2000ft
        state[0, StateIndex.X] = 2000.0
        state[0, StateIndex.Y] = 0.0
        state[0, StateIndex.Z] = -600.0  # ~2000ft
        state[0, StateIndex.U] = 50.0    # ~100 kts
        state[0, StateIndex.PSI] = 0.0   # North
        return state, MissionPhase.CRUISE_UPWIND

    elif phase_name == "TURN_CROSSWIND":
        # About to turn right to crosswind
        state[0, StateIndex.X] = 4000.0
        state[0, StateIndex.Y] = 0.0
        state[0, StateIndex.Z] = -500.0  # ~1600ft
        state[0, StateIndex.U] = 50.0
        state[0, StateIndex.PSI] = 0.0
        return state, MissionPhase.TURN_CROSSWIND

    elif phase_name == "CRUISE_DOWNWIND":
        # Flying south on downwind leg at pattern altitude
        state[0, StateIndex.X] = 2000.0
        state[0, StateIndex.Y] = 3000.0  # East of runway
        state[0, StateIndex.Z] = -1524.0  # 5000ft
        state[0, StateIndex.U] = 50.0
        state[0, StateIndex.PSI] = np.pi  # South (180°)
        return state, MissionPhase.CRUISE_DOWNWIND

    elif phase_name == "TURN_BASE":
        # About to turn to base leg
        state[0, StateIndex.X] = -4000.0  # South of runway
        state[0, StateIndex.Y] = 3000.0
        state[0, StateIndex.Z] = -1524.0
        state[0, StateIndex.U] = 50.0
        state[0, StateIndex.PSI] = np.pi
        return state, MissionPhase.TURN_BASE

    elif phase_name == "APPROACH":
        # On final approach
        state[0, StateIndex.X] = 1500.0  # North of runway
        state[0, StateIndex.Y] = 0.0
        state[0, StateIndex.Z] = -100.0  # ~300ft
        state[0, StateIndex.U] = 35.0    # ~70 kts approach speed
        state[0, StateIndex.PSI] = 0.0   # Heading north... wait, should be south
        state[0, StateIndex.PSI] = np.pi  # South toward runway
        state[0, StateIndex.X] = -1500.0  # South of runway, approaching north
        state[0, StateIndex.PSI] = 0.0   # Heading north
        return state, MissionPhase.APPROACH

    # Default: ground roll
    state[0, StateIndex.X] = -250.0
    state[0, StateIndex.U] = 5.0
    return state, MissionPhase.GROUND_ROLL


def run_simulation(dt=0.02):
    print("\n" + "="*70)
    print("  CESSNA 172 TRAFFIC PATTERN - TELEMETRY VIEWER")
    print("="*70)
    print("Flight Area: 10km (N-S) x 5km (E-W)")
    print("Cruise Altitude: 5000 ft (1524m)")
    print("Runway: Centered at origin, aligned North-South")
    print("="*70)
    print("Open browser to http://100.79.9.48:8000")
    print("Press Ctrl+C to stop\n")
    print("Available commands from viewer:")
    print("  - restart: Reset to ground roll")
    print("  - set_phase: Jump to a specific phase")
    print("="*70 + "\n")

    dynamics = FlightSimulator(n_instances=1, dt=dt, use_gpu=False)

    # Start at south end of centered runway
    # Runway is 600m long centered at origin, so south end is at X=-300
    initial_states = np.zeros((1, 12), dtype=np.float32)
    initial_states[0, StateIndex.X] = -250.0  # Start near south end of runway
    initial_states[0, StateIndex.Y] = 0.0     # Centered on runway (East=0)
    initial_states[0, StateIndex.Z] = 0.0     # On ground (Z=0 in NED)
    initial_states[0, StateIndex.U] = 5.0     # Small initial forward velocity
    initial_states[0, StateIndex.PSI] = 0.0   # Heading North (psi=0)
    dynamics.reset(initial_states)

    controller = MissionController(
        cruise_altitude_ft=5000.0,
        pattern_altitude_ft=1000.0,
    )

    last_phase = None
    sim_time = 0.0
    max_time = 800.0
    is_paused = False

    try:
        while sim_time < max_time:
            # Check for commands from viewer
            cmd = get_command()
            if cmd:
                cmd_type = cmd.get("type", cmd.get("command", ""))
                print(f"[CMD] Received: {cmd}")

                if cmd_type == "restart":
                    print("[CMD] Restarting simulation...")
                    dynamics.reset(initial_states)
                    controller.reset()
                    last_phase = None
                    sim_time = 0.0
                    is_paused = False
                    continue

                elif cmd_type == "pause":
                    print("[CMD] Simulation paused")
                    is_paused = True
                    continue

                elif cmd_type == "resume":
                    print("[CMD] Simulation resumed")
                    is_paused = False
                    continue

                elif cmd_type == "set_phase":
                    phase_name = cmd.get("phase", "GROUND_ROLL")
                    print(f"[CMD] Setting phase to: {phase_name}")
                    preset_state, preset_phase = get_phase_preset(phase_name)
                    dynamics.reset(preset_state)
                    controller.reset()
                    controller.phase = preset_phase
                    last_phase = None
                    sim_time = 0.0
                    continue

            # When paused, still update telemetry but don't advance physics
            state = dynamics.get_states()[0]
            if is_paused:
                # Update telemetry with current frozen state
                action = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
                update_telemetry(state, action, controller.phase.name + " (PAUSED)", sim_time)
                time.sleep(dt)
                continue

            action = controller.compute_action(state, sim_time)
            dynamics.set_controls(action.reshape(1, -1))
            dynamics.step()

            update_telemetry(state, action, controller.phase.name, sim_time)

            if controller.phase != last_phase:
                alt = -state[StateIndex.Z]
                spd = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2)
                hdg = np.rad2deg(state[StateIndex.PSI]) % 360
                print(f"[{sim_time:6.1f}s] {controller.phase.name:20s} | "
                      f"Alt:{alt:6.0f}m ({alt*3.28084:5.0f}ft) | "
                      f"Spd:{spd:5.1f}m/s ({spd*1.944:5.1f}kts) | "
                      f"Hdg:{hdg:5.1f}")
                last_phase = controller.phase

            if controller.phase == MissionPhase.LANDED:
                print("\nMISSION COMPLETE!")
                print("Restarting in 5 seconds...")
                time.sleep(5.0)
                dynamics.reset(initial_states)
                controller.reset()
                last_phase = None
                sim_time = 0.0
                continue

            # Out of bounds check (10km x 5km from center - expanded for full pattern)
            if abs(state[StateIndex.X]) > 10000 or abs(state[StateIndex.Y]) > 5000:
                print(f"\nOUT OF BOUNDS at X={state[StateIndex.X]:.0f}, Y={state[StateIndex.Y]:.0f} - Restarting...")
                time.sleep(2.0)
                dynamics.reset(initial_states)
                controller.reset()
                last_phase = None
                sim_time = 0.0
                continue

            # Print position periodically for debugging
            if int(sim_time * 10) % 100 == 0:  # Every 10 seconds
                print(f"  [DEBUG] t={sim_time:.0f}s X={state[StateIndex.X]:.0f} Y={state[StateIndex.Y]:.0f} Alt={-state[StateIndex.Z]:.0f}m Phase={controller.phase.name}")

            sim_time += dt
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")


async def main_async(dt):
    server_task = asyncio.create_task(
        telemetry_server(lambda: shared_state, host="0.0.0.0", port=8765)
    )
    await asyncio.sleep(1)

    sim_thread = threading.Thread(
        target=run_simulation,
        args=(dt,),
        daemon=True
    )
    sim_thread.start()

    try:
        await server_task
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    asyncio.run(main_async(0.02))
    return 0


if __name__ == "__main__":
    sys.exit(main())
