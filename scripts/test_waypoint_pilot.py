#!/usr/bin/env python3
"""
Test the trained Waypoint Pilot NN.

This loads the trained model and runs inference on sample inputs
to verify it's working correctly.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_waypoint_bc import WaypointPilotNN


def load_model(checkpoint_path):
    """Load a trained waypoint pilot model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model = WaypointPilotNN(
        input_dim=checkpoint['input_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract normalization stats
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

    return model, norm_stats, checkpoint


def normalize_input(observation, waypoint, norm_stats):
    """Normalize inputs using saved statistics."""
    # Normalize observation
    obs_norm = (observation - norm_stats['obs_mean']) / norm_stats['obs_std']

    # Normalize waypoint
    wp_norm = waypoint.copy()
    wp_norm[:3] = (waypoint[:3] - norm_stats['wp_xy_mean']) / norm_stats['wp_xy_std']
    wp_norm[3] = (waypoint[3] - norm_stats['wp_spd_mean']) / norm_stats['wp_spd_std']
    # wp_norm[4] stays as-is (categorical waypoint type)
    wp_norm[5] = (waypoint[5] - norm_stats['wp_dist_mean']) / norm_stats['wp_dist_std']

    # Combine
    combined = np.concatenate([obs_norm, wp_norm])
    return torch.FloatTensor(combined).unsqueeze(0)


def test_model(model, norm_stats):
    """Test the model with various inputs."""
    print("\n" + "=" * 60)
    print("  TESTING WAYPOINT PILOT NN")
    print("=" * 60)

    # Test case 1: Aircraft on ground, takeoff waypoint ahead
    print("\n--- Test 1: Ground roll phase ---")
    obs1 = np.array([0, 0, 0,  # x, y, z
                     20, 0, 0,  # u, v, w (20 m/s forward)
                     0, 0, 0,   # phi, theta, psi
                     0, 0, 0])  # p, q, r
    wp1 = np.array([500, 0, 0,  # target ahead on runway
                    50,  # target speed 50 m/s
                    4,   # TAKEOFF type
                    500])  # 500m away

    inp1 = normalize_input(obs1, wp1, norm_stats)
    with torch.no_grad():
        action1 = model(inp1).numpy()[0]
    print(f"  State: ground roll at 20 m/s")
    print(f"  Waypoint: takeoff point 500m ahead")
    print(f"  Output: throttle={action1[0]:.3f}, aileron={action1[1]:.3f}, elevator={action1[2]:.3f}, rudder={action1[3]:.3f}")

    # Test case 2: In air, climbing
    print("\n--- Test 2: Climb phase ---")
    obs2 = np.array([500, 0, 100,  # 100m altitude
                     45, 0, 5,     # climbing
                     0, 0.1, 0,    # slight pitch up
                     0, 0, 0])
    wp2 = np.array([2000, 0, 500,  # waypoint at 500m alt
                    55,    # target speed
                    1,     # FLYBY type
                    1500]) # 1500m away

    inp2 = normalize_input(obs2, wp2, norm_stats)
    with torch.no_grad():
        action2 = model(inp2).numpy()[0]
    print(f"  State: climbing at 45 m/s, 100m alt")
    print(f"  Waypoint: 1500m ahead at 500m alt")
    print(f"  Output: throttle={action2[0]:.3f}, aileron={action2[1]:.3f}, elevator={action2[2]:.3f}, rudder={action2[3]:.3f}")

    # Test case 3: Level flight, waypoint to the right
    print("\n--- Test 3: Turn to waypoint ---")
    obs3 = np.array([0, 0, 300,    # 300m altitude
                     50, 0, 0,     # level flight
                     0, 0, 0,      # wings level, heading north
                     0, 0, 0])
    wp3 = np.array([0, 500, 300,   # waypoint to the east
                    50,    # maintain speed
                    1,     # FLYBY type
                    500])  # 500m away

    inp3 = normalize_input(obs3, wp3, norm_stats)
    with torch.no_grad():
        action3 = model(inp3).numpy()[0]
    print(f"  State: level at 300m, heading north")
    print(f"  Waypoint: 500m to the east")
    print(f"  Output: throttle={action3[0]:.3f}, aileron={action3[1]:.3f}, elevator={action3[2]:.3f}, rudder={action3[3]:.3f}")

    # Test case 4: Approaching landing
    print("\n--- Test 4: Landing approach ---")
    obs4 = np.array([300, 0, 50,   # 50m altitude, 300m from threshold
                     40, 0, -3,    # descending
                     0, -0.05, 0,  # slight pitch down
                     0, 0, 0])
    wp4 = np.array([0, 0, 0,       # runway threshold
                    35,    # approach speed
                    3,     # LAND type
                    300])  # 300m away

    inp4 = normalize_input(obs4, wp4, norm_stats)
    with torch.no_grad():
        action4 = model(inp4).numpy()[0]
    print(f"  State: on approach, 50m alt, 300m from threshold")
    print(f"  Waypoint: runway threshold (LAND type)")
    print(f"  Output: throttle={action4[0]:.3f}, aileron={action4[1]:.3f}, elevator={action4[2]:.3f}, rudder={action4[3]:.3f}")

    print("\n" + "=" * 60)
    print("  Model outputs are in range [-1, 1] (Tanh activation)")
    print("  Note: Model was trained on early flight phases only")
    print("       (ground roll, rotation, climb) due to demo limitations")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/waypoint_pilot/waypoint_pilot_best.pt')
    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    model, norm_stats, checkpoint = load_model(args.checkpoint)

    print(f"\nModel info:")
    print(f"  Input dim: {checkpoint['input_dim']}")
    print(f"  Action dim: {checkpoint['action_dim']}")
    print(f"  Hidden dim: {checkpoint['hidden_dim']}")
    print(f"  Best val loss: {checkpoint['val_loss']:.6f}")
    print(f"  Trained for: {checkpoint['epoch']} epochs")

    test_model(model, norm_stats)


if __name__ == "__main__":
    main()
