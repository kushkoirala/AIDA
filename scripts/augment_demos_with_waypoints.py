#!/usr/bin/env python3
"""
Augment Existing Demos with Waypoint Labels

This script takes existing BC demos (which have observations, actions, phases)
and adds waypoint labels based on the flight phase and aircraft position.

This is much faster than re-generating demos from scratch.

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import sys
from pathlib import Path
import numpy as np
import argparse

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpu-flight-dynamics' / 'python'))

from flight_dynamics import StateIndex


def compute_waypoint_from_state_and_phase(state, phase, flight_config):
    """
    Compute the waypoint command based on aircraft state and flight phase.

    This reverse-engineers what the classical controller was targeting
    based on the flight phase and aircraft position.

    Returns: [target_x, target_y, target_alt, target_speed, wp_type, dist_to_wp]
    """
    x, y, z = state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z]
    altitude = -z  # Convert from NED

    # Flight config (SN65 -> KHUT triangle approach)
    tp_x = flight_config['tp_x']  # Turning point x
    tp_y = flight_config['tp_y']  # Turning point y
    runway_x = flight_config['runway_x']  # KHUT runway threshold
    runway_y = flight_config['runway_y']
    cruise_alt = flight_config['cruise_alt']
    pattern_alt = flight_config['pattern_alt']

    # Speeds (m/s)
    v_climb = 27.0
    v_cruise = 40.0
    v_approach = 28.0
    v_touchdown = 25.0

    # Waypoint types
    FLYOVER = 0
    FLYBY = 1
    LAND = 3
    TAKEOFF = 4

    # Determine waypoint based on phase
    if phase == 0:  # GROUND_ROLL
        target = (500.0, 0.0, 0.0, 22.0, TAKEOFF)
    elif phase == 1:  # ROTATION
        target = (1000.0, 0.0, 30.0, v_climb, FLYOVER)
    elif phase == 2:  # INITIAL_CLIMB
        target = (2000.0, 0.0, 300.0, v_climb, FLYOVER)
    elif phase == 3:  # CLIMB
        target = (10000.0, 0.0, cruise_alt, v_climb, FLYOVER)
    elif phase == 4:  # CRUISE_TO_TP
        target = (tp_x, tp_y, cruise_alt, v_cruise, FLYBY)
    elif phase == 5:  # TURN_TO_INTERCEPT
        target = (runway_x, runway_y, cruise_alt, v_cruise, FLYOVER)
    elif phase == 6:  # INTERCEPT_LEG
        target = (runway_x, runway_y, pattern_alt, v_approach, FLYOVER)
    elif phase == 7:  # FINAL_APPROACH
        target = (runway_x, runway_y, 30.0, v_approach, FLYOVER)
    elif phase == 8:  # SHORT_FINAL
        target = (runway_x + 100, runway_y, 5.0, v_touchdown, LAND)
    elif phase == 9:  # LANDING
        target = (runway_x + 200, runway_y, 0.0, v_touchdown * 0.8, LAND)
    elif phase == 10:  # LANDED
        target = (runway_x + 300, runway_y, 0.0, 0.0, LAND)
    else:
        # Default to cruise
        target = (tp_x, tp_y, cruise_alt, v_cruise, FLYOVER)

    # Calculate distance to waypoint
    dist = np.sqrt((target[0] - x)**2 + (target[1] - y)**2)

    return np.array([
        target[0],  # target_x
        target[1],  # target_y
        target[2],  # target_alt
        target[3],  # target_speed
        float(target[4]),  # wp_type
        dist,  # dist_to_wp
    ], dtype=np.float32)


def augment_demos(input_path, output_path):
    """
    Add waypoint labels to existing demos.
    """
    print("=" * 70)
    print("  AUGMENTING DEMOS WITH WAYPOINT LABELS")
    print("=" * 70)

    # Load existing demos
    print(f"\nLoading: {input_path}")
    data = np.load(input_path)

    observations = data['observations']
    actions = data['actions']
    phases = data['phases']

    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Phases: {phases.shape}")

    # Flight configuration (SN65 -> KHUT)
    flight_config = {
        'tp_x': 80000.0,  # Turning point east of KHUT
        'tp_y': -32000.0,  # South
        'runway_x': 52800.0,  # KHUT RWY 31 threshold
        'runway_y': -21300.0,
        'cruise_alt': 1676.0,  # 5500 ft
        'pattern_alt': 457.0,  # 1500 ft AGL
    }

    # Generate waypoint labels
    print("\nComputing waypoint labels...")
    n_samples = len(observations)
    waypoints = np.zeros((n_samples, 6), dtype=np.float32)

    # Process in batches for progress reporting
    batch_size = 10000
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        for j in range(i, end_idx):
            waypoints[j] = compute_waypoint_from_state_and_phase(
                observations[j], phases[j], flight_config
            )
        print(f"  Processed {end_idx:,}/{n_samples:,} samples")

    # Save augmented dataset
    print(f"\nSaving to: {output_path}")
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        waypoints=waypoints,
        phases=phases,
        num_episodes=data.get('num_episodes', 0),
        successful_episodes=data.get('successful_episodes', 0),
        dt=data.get('dt', 0.02),
        waypoint_format=['target_x', 'target_y', 'target_alt', 'target_speed', 'wp_type', 'dist_to_wp'],
    )

    # Print statistics
    print("\nWaypoint type distribution:")
    unique, counts = np.unique(waypoints[:, 4], return_counts=True)
    wp_types = ['FLYOVER', 'FLYBY', 'HOLD', 'LAND', 'TAKEOFF']
    for t, c in zip(unique, counts):
        print(f"  {wp_types[int(t)]}: {c:,} ({100*c/len(waypoints):.1f}%)")

    print("\nPhase distribution:")
    unique, counts = np.unique(phases, return_counts=True)
    phase_names = ['GROUND_ROLL', 'ROTATION', 'INITIAL_CLIMB', 'CLIMB',
                   'CRUISE_TO_TP', 'TURN_TO_INTERCEPT', 'INTERCEPT_LEG',
                   'FINAL_APPROACH', 'SHORT_FINAL', 'LANDING', 'LANDED']
    for p, c in zip(unique, counts):
        name = phase_names[p] if p < len(phase_names) else f"PHASE_{p}"
        print(f"  {name}: {c:,} ({100*c/len(phases):.1f}%)")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Augment demos with waypoint labels')
    parser.add_argument('--input', type=str, required=True, help='Input .npz file')
    parser.add_argument('--output', type=str, default=None, help='Output .npz file')

    args = parser.parse_args()

    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_waypoints.npz")

    augment_demos(args.input, args.output)


if __name__ == "__main__":
    main()
