#!/usr/bin/env python3
"""Test Waypoint-Following NN Pilot with 3D Telemetry Viewer

Runs the trained BC model to fly SN65 -> KHUT using waypoint commands.
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
import torch

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))

from triangle_controller import TriangleInterceptController, XCPhase
from generate_waypoint_demos import WaypointDemoGenerator
from flight_dynamics import FlightSimulator, StateIndex
from aida_sim.io.telemetry import telemetry_server, get_command
from train_waypoint_bc import WaypointPilotNN
import torch.nn as nn


class WaypointPilotActor(nn.Module):
    """PPO Actor network - matches BC architecture for loading PPO checkpoints."""
    def __init__(self, input_dim=18, action_dim=4, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        # Match BC architecture exactly: Linear->LayerNorm->ReLU pattern
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # actor.0
            nn.LayerNorm(hidden_dim),              # actor.1
            nn.ReLU(),                             # actor.2
            nn.Linear(hidden_dim, hidden_dim),     # actor.3
            nn.LayerNorm(hidden_dim),              # actor.4
            nn.ReLU(),                             # actor.5
            nn.Linear(hidden_dim, hidden_dim // 2),  # actor.6 (256->128)
            nn.LayerNorm(hidden_dim // 2),         # actor.7
            nn.ReLU(),                             # actor.8
            nn.Linear(hidden_dim // 2, action_dim),  # actor.9
            nn.Tanh(),                             # actor.10
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, x):
        """Return deterministic action (mean)."""
        return self.actor(x)

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
    "distance_nm": 0.0,
    "controller": "NN_PILOT",  # Indicate we're using NN
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


def update_telemetry(state, action, phase_name, distance_to_khut, waypoint_info=""):
    x, y, z = state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z]
    u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]
    phi, theta, psi = state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI]
    p, q, r = state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]

    altitude_m = -z
    altitude_ft = altitude_m * 3.28084
    airspeed = np.sqrt(u**2 + v**2 + w**2)

    takeoff_phases = ["GROUND_ROLL", "ROTATION", "INITIAL_CLIMB"]
    if phase_name in takeoff_phases:
        heading = 4.0
    else:
        heading = np.rad2deg(psi) % 360

    M_TO_FT = 3.28084
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
    shared_state["flaps"] = float(np.clip(action[4], 0.0, 1.0)) if len(action) > 4 else 0.0
    shared_state["spoilers"] = float(np.clip(action[5], 0.0, 1.0)) if len(action) > 5 else 0.0
    shared_state["brakes"] = float(np.clip(action[6], 0.0, 1.0)) if len(action) > 6 else 0.0

    shared_state["phase"] = phase_name
    shared_state["altitude_ft"] = float(altitude_ft)
    shared_state["airspeed_kts"] = float(airspeed * 1.944)
    shared_state["heading_deg"] = float(heading)
    shared_state["distance_nm"] = float(distance_to_khut / 1852.0)
    shared_state["waypoint_info"] = waypoint_info

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


def load_nn_pilot(checkpoint_path):
    """Load trained waypoint pilot NN (supports both BC and PPO checkpoints)"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Detect checkpoint type
    is_ppo = 'actor_state_dict' in checkpoint

    if is_ppo:
        print("  [PPO Model Detected]")
        model = WaypointPilotActor(
            input_dim=checkpoint.get('input_dim', 18),
            action_dim=checkpoint.get('action_dim', 4),
            hidden_dim=checkpoint.get('hidden_dim', 256)
        )
        model.load_state_dict(checkpoint['actor_state_dict'])
    else:
        print("  [BC Model Detected]")
        model = WaypointPilotNN(
            input_dim=checkpoint.get('input_dim', 18),
            action_dim=checkpoint.get('action_dim', 4),
            hidden_dim=checkpoint.get('hidden_dim', 256)
        )
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # Extract normalization stats from checkpoint
    norm_stats = {
        'obs_mean': checkpoint['obs_mean'],
        'obs_std': checkpoint['obs_std'],
        'wp_x_mean': checkpoint['wp_xy_mean'][0],
        'wp_x_std': checkpoint['wp_xy_std'][0],
        'wp_y_mean': checkpoint['wp_xy_mean'][1],
        'wp_y_std': checkpoint['wp_xy_std'][1],
        'wp_z_mean': checkpoint['wp_xy_mean'][2],
        'wp_z_std': checkpoint['wp_xy_std'][2],
        'wp_spd_mean': checkpoint['wp_spd_mean'],
        'wp_spd_std': checkpoint['wp_spd_std'],
        'wp_dist_mean': checkpoint['wp_dist_mean'],
        'wp_dist_std': checkpoint['wp_dist_std'],
    }
    return model, norm_stats


def normalize_observation(state, waypoint, norm_stats):
    """Normalize state and waypoint for NN input"""
    # State normalization (12D)
    state_norm = (state - norm_stats['obs_mean']) / norm_stats['obs_std']

    # Waypoint normalization (6D: rel_x, rel_y, rel_z, target_speed, wp_type, distance)
    wp_norm = np.zeros(6, dtype=np.float32)
    wp_norm[0] = (waypoint[0] - norm_stats['wp_x_mean']) / norm_stats['wp_x_std']
    wp_norm[1] = (waypoint[1] - norm_stats['wp_y_mean']) / norm_stats['wp_y_std']
    wp_norm[2] = (waypoint[2] - norm_stats['wp_z_mean']) / norm_stats['wp_z_std']
    wp_norm[3] = (waypoint[3] - norm_stats['wp_spd_mean']) / norm_stats['wp_spd_std']
    wp_norm[4] = waypoint[4]  # wp_type already normalized (0-4)
    wp_norm[5] = (waypoint[5] - norm_stats['wp_dist_mean']) / norm_stats['wp_dist_std']

    # Concatenate
    obs = np.concatenate([state_norm, wp_norm])
    return obs


def run_nn_flight(dt=0.02, checkpoint_path=None, sim_speed=1.0):
    print()
    print("="*70)
    print("  NN WAYPOINT PILOT: SN65 -> KHUT")
    print("="*70)
    print("Departure: Lake Waltanna (SN65) RWY 35 | Hdg 004")
    print("Arrival:   Hutchinson Regional (KHUT) RWY 31 | Hdg 314")
    print("Distance:  31 NM (57 km) | Course: 338 deg true")
    print("="*70)

    # Load NN model - default to PPO checkpoint if available
    if checkpoint_path is None:
        ppo_path = Path(__file__).parent.parent / "checkpoints" / "waypoint_pilot_ppo" / "waypoint_pilot_ppo_best.pt"
        bc_path = Path(__file__).parent.parent / "checkpoints" / "waypoint_pilot" / "waypoint_pilot_best.pt"
        checkpoint_path = ppo_path if ppo_path.exists() else bc_path

    print(f"Loading model: {checkpoint_path}")
    model, norm_stats = load_nn_pilot(checkpoint_path)
    print(f"Model loaded successfully!")
    print()

    # Setup simulator
    dynamics = FlightSimulator(n_instances=1, dt=dt, use_gpu=False)
    print(f"FlightSimulator: 1 instance on CPU (NumPy)")

    # Initialize at runway
    initial_states = np.zeros((1, 12), dtype=np.float32)
    initial_states[0, 0] = -400.0  # X position (runway)
    initial_states[0, 3] = 5.0     # U (taxi speed)
    dynamics.reset(initial_states)

    # Setup classical controller for waypoint generation (not for control)
    classical_controller = TriangleInterceptController(
        cruise_altitude_ft=5500.0,
        pattern_altitude_ft=1500.0,
    )
    wp_generator = WaypointDemoGenerator(classical_controller)

    last_phase = None
    sim_time = 0.0
    max_time = 1200.0

    try:
        while sim_time < max_time:
            state = dynamics.get_states()[0]

            # Get waypoint from classical controller (for guidance only)
            classical_action = classical_controller.compute_action(state, sim_time)
            waypoint = wp_generator.get_current_waypoint(classical_controller.phase, state)

            # Normalize and get NN prediction
            obs = normalize_observation(state, waypoint, norm_stats)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                nn_action = model(obs_tensor).numpy()[0]

            # NN outputs 4 controls: throttle, aileron, elevator, rudder
            # Map to full action array (7 elements)
            action = np.zeros(7, dtype=np.float32)
            action[0] = np.clip(nn_action[0], 0.0, 1.0)  # throttle [0,1]
            action[1] = np.clip(nn_action[1], -1.0, 1.0)  # aileron
            action[2] = np.clip(nn_action[2], -1.0, 1.0)  # elevator
            action[3] = np.clip(nn_action[3], -1.0, 1.0)  # rudder
            # Flaps, spoilers, brakes from classical controller for now
            action[4] = classical_action[4]  # flaps
            action[5] = classical_action[5]  # spoilers
            action[6] = classical_action[6]  # brakes

            dynamics.set_controls(action.reshape(1, -1))
            dynamics.step()

            # Distance to KHUT
            distance_to_target = np.sqrt(
                (classical_controller.threshold_x - state[StateIndex.X])**2 +
                (classical_controller.threshold_y - state[StateIndex.Y])**2
            )

            # Waypoint info string
            wp_type_names = ["FLYOVER", "FLYBY", "HOLD", "LAND", "TAKEOFF"]
            wp_info = f"WP: {wp_type_names[int(waypoint[4])]} dist={waypoint[5]:.0f}m"

            update_telemetry(state, action, classical_controller.phase.name, distance_to_target, wp_info)

            # Periodic status logging (every 10 seconds)
            alt = -state[StateIndex.Z]
            spd = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2)
            hdg = np.rad2deg(state[StateIndex.PSI]) % 360
            dist_nm = distance_to_target / 1852.0

            if classical_controller.phase != last_phase or int(sim_time) % 10 == 0 and int(sim_time) != int(sim_time - dt):
                print(f"[{sim_time:6.1f}s] {classical_controller.phase.name:18s} | "
                      f"Alt:{alt:6.0f}m ({alt*3.28084:5.0f}ft) | "
                      f"Spd:{spd:5.1f}m/s ({spd*1.944:5.1f}kts) | "
                      f"Hdg:{hdg:5.1f} | Dist:{dist_nm:5.1f}nm | "
                      f"Thr:{action[0]:.2f} Elv:{action[2]:.2f}")
                last_phase = classical_controller.phase

            # Check termination
            if classical_controller.phase == XCPhase.LANDED:
                print(f"\n[{sim_time:.1f}s] LANDED SUCCESSFULLY!")
                break

            # Check for crash
            altitude = -state[StateIndex.Z]
            if altitude < -10 and sim_time > 10:
                print(f"\n[{sim_time:.1f}s] CRASH DETECTED - altitude below ground")
                break

            sim_time += dt
            time.sleep(dt / sim_speed)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    print("="*70)
    print("  NN PILOT FLIGHT COMPLETE")
    print("="*70)


async def main_async(args):
    # Start telemetry server (must pass callable that returns state dict)
    server_task = asyncio.create_task(telemetry_server(lambda: shared_state, host="0.0.0.0", port=8765))
    await asyncio.sleep(0.5)

    # Run simulation in thread
    sim_thread = threading.Thread(
        target=run_nn_flight,
        kwargs={
            'dt': args.dt,
            'checkpoint_path': args.checkpoint,
            'sim_speed': args.speed,
        }
    )
    sim_thread.start()

    print("\n" + "="*70)
    print("  TELEMETRY SERVER RUNNING")
    print("="*70)
    print("Open viewer at: http://localhost:8080")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")

    try:
        while sim_thread.is_alive():
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")

    sim_thread.join(timeout=2.0)


def main():
    parser = argparse.ArgumentParser(description='NN Waypoint Pilot Flight Test')
    parser.add_argument('--dt', type=float, default=0.02, help='Simulation timestep')
    parser.add_argument('--speed', type=float, default=1.0, help='Simulation speed multiplier')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
