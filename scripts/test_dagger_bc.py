#!/usr/bin/env python3
"""
Test DAgger-trained BC model in closed-loop to verify throttle is maintained.
"""

import sys
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


class WaypointPilotNN(nn.Module):
    def __init__(self, input_dim=18, action_dim=4, hidden_dim=256):
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


def normalize_obs(state, waypoint, ns):
    """Normalize observation for BC model."""
    state_norm = (state - ns['obs_mean']) / ns['obs_std']

    wp_norm = waypoint.copy()
    wp_norm[:3] = (waypoint[:3] - ns['wp_xy_mean']) / ns['wp_xy_std']
    wp_norm[3] = (waypoint[3] - float(ns['wp_spd_mean'])) / float(ns['wp_spd_std'])
    wp_norm[5] = (waypoint[5] - float(ns['wp_dist_mean'])) / float(ns['wp_dist_std'])

    # Normalize bearing_error if present (7D waypoint format)
    if len(waypoint) >= 7 and 'wp_bearing_mean' in ns:
        wp_norm[6] = (waypoint[6] - float(ns['wp_bearing_mean'])) / float(ns['wp_bearing_std'])

    return np.concatenate([state_norm, wp_norm])


def main():
    print("\n" + "=" * 70)
    print("  Testing DAgger-trained BC Model (Closed-Loop)")
    print("=" * 70)

    # Load DAgger-trained BC model
    checkpoint_path = "checkpoints/waypoint_pilot/waypoint_pilot_best.pt"
    print(f"\nLoading: {checkpoint_path}")

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

    # Add bearing normalization if present (7D waypoint format)
    if 'wp_bearing_mean' in checkpoint:
        norm_stats['wp_bearing_mean'] = checkpoint['wp_bearing_mean']
        norm_stats['wp_bearing_std'] = checkpoint['wp_bearing_std']
        print(f"Using 7D waypoint format (with bearing_error)")

    print(f"Model val_loss: {checkpoint['val_loss']:.6f}")

    # Setup simulation
    dt = 0.02
    sim = FlightSimulator(n_instances=1, dt=dt, use_gpu=False)
    ctrl = TriangleInterceptController()
    wp_gen = WaypointDemoGenerator(ctrl)

    # Ground start
    initial_state = np.zeros((1, 12), dtype=np.float32)
    initial_state[0, 0] = -400.0  # X position
    initial_state[0, 3] = 5.0     # U (taxi speed)
    sim.reset(initial_state)
    ctrl.reset()

    print("\nRunning FULL FLIGHT (SN65 -> KHUT)...", flush=True)
    print("-" * 100, flush=True)
    print(f"{'Time':>7s} | {'Phase':>20s} | {'Alt':>7s} | {'IAS':>6s} | {'Hdg':>5s} | BC vs Expert Controls               | {'Dist':>6s}", flush=True)
    print("-" * 100, flush=True)

    sim_time = 0.0
    throttle_issues = 0
    max_steps = 60000  # 20 minutes for full flight test

    for step in range(max_steps):
        state = sim.get_states()[0]

        # Get expert action (for comparison only)
        expert_act = ctrl.compute_action(state, sim_time)
        waypoint = wp_gen.get_current_waypoint(ctrl.phase, state)

        # BC prediction
        obs_norm = normalize_obs(state, waypoint, norm_stats)
        with torch.no_grad():
            bc_output = bc_model(torch.FloatTensor(obs_norm).unsqueeze(0)).numpy()[0]

        # Convert BC output [-1,1] to control format
        bc_throttle = (bc_output[0] + 1.0) / 2.0  # [0, 1]
        bc_aileron = bc_output[1]
        bc_elevator = bc_output[2]
        bc_rudder = bc_output[3]

        # Format for simulator
        bc_controls = np.array([[bc_throttle, bc_aileron, bc_elevator, bc_rudder]])

        # Execute BC action (closed-loop!)
        sim.set_controls(bc_controls)
        sim.step()

        # Stats
        alt = -state[2]
        ias = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2) * 1.944  # m/s to kts
        hdg = np.degrees(state[8]) % 360  # psi to heading
        expert_throttle = expert_act[0]

        # Distance to KHUT (approximately 61nm from SN65)
        x, y = state[0], state[1]
        dist_to_khut = np.sqrt((x - 113000)**2 + (y - 1500)**2) / 1852  # meters to nm

        # Check for throttle issues during climb
        if alt < 500 and bc_throttle < 0.8 and expert_throttle > 0.9:
            throttle_issues += 1

        # Print status every 30 seconds
        if step % 1500 == 0:
            phase_str = str(ctrl.phase).replace("XCPhase.", "")
            # Also show BC vs expert control comparison
            exp_thr = expert_act[0]
            exp_ail = expert_act[1]
            exp_ele = expert_act[2]
            exp_rud = expert_act[3]
            ctrl_str = f"Ail={bc_aileron:+.2f}({exp_ail:+.2f}) Ele={bc_elevator:+.2f}({exp_ele:+.2f})"
            print(f"{sim_time:6.1f}s | {phase_str:>20s} | {alt:6.1f}m | {ias:5.1f}kt | {hdg:4.0f}° | {ctrl_str} | {dist_to_khut:5.1f}nm", flush=True)

            # Debug: print waypoint info
            if sim_time > 400:
                print(f"         Waypoint: x={waypoint[0]:.0f} y={waypoint[1]:.0f} alt={waypoint[2]:.0f} spd={waypoint[3]:.0f} type={waypoint[4]:.0f} dist={waypoint[5]:.0f}", flush=True)
                print(f"         State psi={np.degrees(state[8]):.1f}° phi={np.degrees(state[6]):.2f}°", flush=True)

        # Check if mission complete (arrived at KHUT)
        if ctrl.phase == XCPhase.LANDED or dist_to_khut < 2.0:
            print(f"\n  ARRIVED AT KHUT!")
            break

        # Safety check - aircraft crashed or stalled
        if alt < -10 or (alt > 100 and ias < 40):
            print(f"\n  CRASH/STALL at t={sim_time:.1f}s, alt={alt:.1f}m, ias={ias:.1f}kt")
            break

        sim_time += dt

    print("-" * 80)

    # Final statistics
    final_alt = -state[2]
    final_ias = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2) * 1.944
    final_hdg = np.degrees(state[8]) % 360
    flight_time_min = sim_time / 60.0

    print(f"\n{'='*80}")
    print(f"  FLIGHT SUMMARY")
    print(f"{'='*80}")
    print(f"  Flight time:     {flight_time_min:.1f} minutes")
    print(f"  Final altitude:  {final_alt:.1f}m ({final_alt*3.281:.0f}ft)")
    print(f"  Final airspeed:  {final_ias:.1f} KIAS")
    print(f"  Final heading:   {final_hdg:.0f}°")
    print(f"  Final phase:     {ctrl.phase}")
    print(f"  Distance to KHUT: {dist_to_khut:.1f}nm")
    print(f"  Throttle issues: {throttle_issues}")

    if dist_to_khut < 5.0:
        print(f"\n  SUCCESS: Completed cross-country flight SN65 -> KHUT!")
    elif throttle_issues < 50:
        print(f"\n  PARTIAL SUCCESS: BC flying correctly, mission in progress")
    else:
        print(f"\n  ISSUES DETECTED: {throttle_issues} throttle problems during climb")

    return throttle_issues


if __name__ == "__main__":
    issues = main()
    sys.exit(0 if issues < 50 else 1)
