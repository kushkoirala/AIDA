#!/usr/bin/env python3
"""
Run Waypoint BC Policy with 3D Telemetry Viewer

Visualizes the behavioral cloning policy flying the cross-country mission.
"""

import asyncio
import threading
import time
import sys
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from flight_dynamics import FlightSimulator, StateIndex
from triangle_controller import TriangleInterceptController, XCPhase
from generate_waypoint_demos import WaypointDemoGenerator
from aida_sim.io.telemetry import telemetry_server

TELEMETRY_UNITS = "metric"

shared_state = {
    "position": [0, 0, 0],
    "quaternion": [1, 0, 0, 0],
    "velocity": [0, 0, 0],
    "rates": [0, 0, 0],
    "surfaces": [0, 0, 0],
    "throttle": 0.0,
    "soc": 1.0,
    "voltage": 12.0,
    "load_factor": 1.0,
    "units": TELEMETRY_UNITS,
    "model": "cessna172",
    "phase": "BC_WAYPOINT",
}


class WaypointPilotNN(nn.Module):
    def __init__(self, input_dim=19, action_dim=4, hidden_dim=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.actor(x)


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


def normalize_obs(state, waypoint, ns):
    state_norm = (state - ns['obs_mean']) / ns['obs_std']
    wp_dim = ns.get('wp_dim', 7)
    wp_norm = np.zeros(wp_dim, dtype=np.float32)
    wp_norm[0] = (waypoint[0] - float(ns['wp_xy_mean'][0])) / float(ns['wp_xy_std'][0])
    wp_norm[1] = (waypoint[1] - float(ns['wp_xy_mean'][1])) / float(ns['wp_xy_std'][1])
    wp_norm[2] = (waypoint[2] - float(ns['wp_xy_mean'][2])) / float(ns['wp_xy_std'][2])
    wp_norm[3] = (waypoint[3] - float(ns['wp_spd_mean'])) / float(ns['wp_spd_std'])
    wp_norm[4] = waypoint[4]
    wp_norm[5] = (waypoint[5] - float(ns['wp_dist_mean'])) / float(ns['wp_dist_std'])
    if wp_dim >= 7 and 'wp_bearing_mean' in ns:
        wp_norm[6] = (waypoint[6] - float(ns['wp_bearing_mean'])) / float(ns['wp_bearing_std'])
    return np.concatenate([state_norm, wp_norm])


def update_telemetry(state, action, phase_name):
    x = state[StateIndex.X]
    y = state[StateIndex.Y]
    z = state[StateIndex.Z]
    u = state[StateIndex.U]
    v = state[StateIndex.V]
    w = state[StateIndex.W]
    phi = state[StateIndex.PHI]
    theta = state[StateIndex.THETA]
    psi = state[StateIndex.PSI]
    p = state[StateIndex.P]
    q = state[StateIndex.Q]
    r = state[StateIndex.R]

    # Viewer coordinates (offset for runway position)
    viewer_x = x + 500.0
    viewer_y = y
    viewer_z = -z  # Z is down in sim, up in viewer

    shared_state["position"] = [float(viewer_x), float(viewer_y), float(viewer_z)]
    shared_state["quaternion"] = [float(v) for v in euler_to_quaternion(phi, theta, psi)]
    shared_state["velocity"] = [float(u), float(v), float(w)]
    shared_state["rates"] = [float(p), float(q), float(r)]
    shared_state["throttle"] = float(np.clip(action[0], 0.0, 1.0))
    shared_state["surfaces"] = [
        float(np.clip(action[1], -1.0, 1.0)),
        float(np.clip(action[2], -1.0, 1.0)),
        float(np.clip(action[3], -1.0, 1.0))
    ]
    shared_state["phase"] = phase_name


def run_simulation(dt=0.02):
    print("\n" + "="*70)
    print("  WAYPOINT BC POLICY - CROSS COUNTRY FLIGHT")
    print("="*70)
    print("Open browser to view: file:///home/AIDA/viewer/public/index.html")
    print("Press Ctrl+C to stop\n")

    # Load BC model
    checkpoint_path = "checkpoints/waypoint_pilot/waypoint_pilot_best.pt"
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    bc_model = WaypointPilotNN(
        input_dim=checkpoint['input_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    bc_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    bc_model.eval()

    norm_stats = {
        'obs_mean': checkpoint['obs_mean'],
        'obs_std': checkpoint['obs_std'],
        'wp_xy_mean': checkpoint['wp_xy_mean'],
        'wp_xy_std': checkpoint['wp_xy_std'],
        'wp_spd_mean': checkpoint['wp_spd_mean'],
        'wp_spd_std': checkpoint['wp_spd_std'],
        'wp_dist_mean': checkpoint['wp_dist_mean'],
        'wp_dist_std': checkpoint['wp_dist_std'],
    }
    if 'wp_bearing_mean' in checkpoint:
        norm_stats['wp_bearing_mean'] = checkpoint['wp_bearing_mean']
        norm_stats['wp_bearing_std'] = checkpoint['wp_bearing_std']
        norm_stats['wp_dim'] = 7
    else:
        norm_stats['wp_dim'] = 6

    print(f"Model val_loss: {checkpoint['val_loss']:.6f}")

    # Setup simulation
    sim = FlightSimulator(n_instances=1, dt=dt, use_gpu=False)
    ctrl = TriangleInterceptController()
    wp_gen = WaypointDemoGenerator(ctrl)

    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n{'='*70}")
            print(f"  Flight {episode}: SN65 -> KHUT")
            print(f"{'='*70}")

            # Ground start
            initial_state = np.zeros((1, 12), dtype=np.float32)
            initial_state[0, 0] = -400.0
            initial_state[0, 3] = 5.0
            sim.reset(initial_state)
            ctrl.reset()

            sim_time = 0.0
            step = 0
            max_steps = 150000  # 50 minutes

            while step < max_steps:
                state = sim.get_states()[0]

                # Get waypoint for current phase (using expert phase detection)
                expert_act = ctrl.compute_action(state, sim_time)
                waypoint = wp_gen.get_current_waypoint(ctrl.phase, state)

                # BC prediction
                obs_norm = normalize_obs(state, waypoint, norm_stats)
                with torch.no_grad():
                    bc_output = bc_model(torch.FloatTensor(obs_norm).unsqueeze(0)).numpy()[0]

                # Convert BC output [-1,1] to control format
                bc_throttle = (bc_output[0] + 1.0) / 2.0
                bc_aileron = bc_output[1]
                bc_elevator = bc_output[2]
                bc_rudder = bc_output[3]

                bc_controls = np.array([[bc_throttle, bc_aileron, bc_elevator, bc_rudder]])

                # Execute
                sim.set_controls(bc_controls)
                sim.step()

                # Update telemetry
                phase_name = str(ctrl.phase).replace("XCPhase.", "")
                update_telemetry(state, [bc_throttle, bc_aileron, bc_elevator, bc_rudder], phase_name)

                # Stats
                alt = -state[2]
                ias = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2) * 1.944
                hdg = np.degrees(state[8]) % 360

                # Distance to KHUT
                x, y = state[0], state[1]
                dist_to_khut = np.sqrt((x - 113000)**2 + (y - 1500)**2) / 1852

                # Print status every 30 seconds
                if step % 1500 == 0:
                    print(f"  {sim_time:6.1f}s | {phase_name:>18s} | {alt:6.0f}m ({alt*3.281:5.0f}ft) | {ias:5.1f}kt | {hdg:3.0f}° | {dist_to_khut:5.1f}nm")

                # Check mission status
                if ctrl.phase == XCPhase.LANDED or dist_to_khut < 2.0:
                    print(f"\n  ARRIVED AT KHUT!")
                    break

                if alt < -10 or (alt > 100 and ias < 40):
                    print(f"\n  CRASH/STALL at t={sim_time:.1f}s")
                    break

                step += 1
                sim_time += dt
                time.sleep(dt)  # Real-time

            print(f"\nFlight {episode} complete: {sim_time/60:.1f} min, {dist_to_khut:.1f}nm to KHUT")
            time.sleep(5.0)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")


async def main_async():
    server_task = asyncio.create_task(
        telemetry_server(lambda: shared_state, host="0.0.0.0", port=8765)
    )

    await asyncio.sleep(1)

    sim_thread = threading.Thread(
        target=run_simulation,
        daemon=True
    )
    sim_thread.start()

    try:
        await server_task
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    asyncio.run(main_async())
    return 0


if __name__ == "__main__":
    sys.exit(main())
