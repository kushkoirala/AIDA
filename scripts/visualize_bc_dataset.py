#!/usr/bin/env python3
"""
Visualize Behavior Cloning Dataset

Shows flight dynamics from the generated BC dataset to verify:
1. Elevator control is working correctly (positive pitch error → nose up)
2. Stable flight behavior
3. Proper altitude and speed control
4. No oscillations or divergence

Author: Kushal Koirala
Date: December 26, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path

# State indices (from flight_dynamics.py)
class StateIndex:
    X = 0
    Y = 1
    Z = 2
    U = 3
    V = 4
    W = 5
    PHI = 6
    THETA = 7
    PSI = 8
    P = 9
    Q = 10
    R = 11

# Control indices
class ControlIndex:
    THROTTLE = 0
    AILERON = 1
    ELEVATOR = 2
    RUDDER = 3

def main():
    parser = argparse.ArgumentParser(description='Visualize BC dataset')
    parser.add_argument('--dataset', type=str, default='bc_data/bc_dataset_corrected.npz',
                       help='Path to BC dataset')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to plot')
    parser.add_argument('--output', type=str, default='bc_dataset_visualization_CORRECTED.png',
                       help='Output plot filename')
    args = parser.parse_args()

    print("=" * 70)
    print("  BC DATASET VISUALIZATION - CORRECTED ELEVATOR")
    print("=" * 70)

    # Load dataset
    data = np.load(args.dataset)
    observations = data['observations']  # (N, 12)
    actions = data['actions']           # (N, 4)
    rewards = data['rewards']           # (N,)

    print(f"\nDataset: {args.dataset}")
    print(f"  Total timesteps: {len(observations):,}")
    print(f"  Observation shape: {observations.shape}")
    print(f"  Action shape: {actions.shape}")
    print(f"  Mean reward: {rewards.mean():.3f} ± {rewards.std():.3f}")

    # Assume episodes are stored sequentially (500 steps each)
    episode_length = 500
    n_episodes = len(observations) // episode_length

    print(f"\nEpisode structure:")
    print(f"  Episode length: {episode_length}")
    print(f"  Number of episodes: {n_episodes}")
    print(f"\nPlotting first {args.episodes} episodes...")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Time vector for one episode
    dt = 0.02  # 50 Hz
    time = np.arange(episode_length) * dt

    # Plot first few episodes
    n_plot = min(args.episodes, n_episodes)
    colors = plt.cm.viridis(np.linspace(0, 1, n_plot))

    # Extract data for each episode
    for ep in range(n_plot):
        start_idx = ep * episode_length
        end_idx = start_idx + episode_length

        obs = observations[start_idx:end_idx]
        act = actions[start_idx:end_idx]
        rew = rewards[start_idx:end_idx]

        # Extract states
        altitude = -obs[:, StateIndex.Z]  # NED to AGL
        u = obs[:, StateIndex.U]
        w = obs[:, StateIndex.W]
        pitch = np.rad2deg(obs[:, StateIndex.THETA])
        roll = np.rad2deg(obs[:, StateIndex.PHI])
        p = np.rad2deg(obs[:, StateIndex.P])
        q = np.rad2deg(obs[:, StateIndex.Q])
        r = np.rad2deg(obs[:, StateIndex.R])

        # Derived quantities
        airspeed = u  # Approximate
        climb_rate = -w

        # Extract actions
        throttle = act[:, ControlIndex.THROTTLE]
        aileron = np.rad2deg(act[:, ControlIndex.AILERON])
        elevator = np.rad2deg(act[:, ControlIndex.ELEVATOR])
        rudder = np.rad2deg(act[:, ControlIndex.RUDDER])

        label = f'Episode {ep+1}'
        alpha = 0.7 if ep == 0 else 0.4
        lw = 2 if ep == 0 else 1

        # Row 1: Altitude, Airspeed, Pitch
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time, altitude, color=colors[ep], alpha=alpha, linewidth=lw, label=label)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Altitude (m)')
        ax1.set_title('Altitude AGL', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        if ep == 0:
            ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time, airspeed, color=colors[ep], alpha=alpha, linewidth=lw)
        ax2.axhline(y=11.2, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Airspeed (m/s)')
        ax2.set_title('Airspeed', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(time, pitch, color=colors[ep], alpha=alpha, linewidth=lw)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Pitch (deg)')
        ax3.set_title('Pitch Attitude', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Row 2: Roll, Pitch Rate, Climb Rate
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(time, roll, color=colors[ep], alpha=alpha, linewidth=lw)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Roll (deg)')
        ax4.set_title('Roll Attitude', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(time, q, color=colors[ep], alpha=alpha, linewidth=lw)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Pitch Rate (deg/s)')
        ax5.set_title('Pitch Rate', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(time, climb_rate, color=colors[ep], alpha=alpha, linewidth=lw)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Climb Rate (m/s)')
        ax6.set_title('Vertical Speed', fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # Row 3: Controls - Throttle, Elevator, Aileron
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(time, throttle, color=colors[ep], alpha=alpha, linewidth=lw)
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Throttle')
        ax7.set_title('Throttle Setting', fontweight='bold')
        ax7.set_ylim([0, 1.1])
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(time, elevator, color=colors[ep], alpha=alpha, linewidth=lw)
        ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax8.axhline(y=15, color='r', linestyle='--', alpha=0.2, linewidth=0.5)
        ax8.axhline(y=-15, color='r', linestyle='--', alpha=0.2, linewidth=0.5)
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Elevator (deg)')
        ax8.set_title('Elevator Deflection (CORRECTED)', fontweight='bold')
        ax8.grid(True, alpha=0.3)

        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(time, aileron, color=colors[ep], alpha=alpha, linewidth=lw)
        ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Aileron (deg)')
        ax9.set_title('Aileron Deflection', fontweight='bold')
        ax9.grid(True, alpha=0.3)

        # Row 4: Rewards and Phase Diagram
        ax10 = fig.add_subplot(gs[3, 0])
        ax10.plot(time, rew, color=colors[ep], alpha=alpha, linewidth=lw)
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('Reward')
        ax10.set_title('Instantaneous Reward', fontweight='bold')
        ax10.grid(True, alpha=0.3)

        # Phase diagram: pitch vs pitch rate
        ax11 = fig.add_subplot(gs[3, 1])
        ax11.scatter(pitch, q, c=time, cmap='viridis', s=1, alpha=0.5)
        ax11.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax11.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax11.set_xlabel('Pitch (deg)')
        ax11.set_ylabel('Pitch Rate (deg/s)')
        ax11.set_title('Phase Diagram: Pitch Dynamics', fontweight='bold')
        ax11.grid(True, alpha=0.3)

        # Elevator vs Pitch Error (to verify sign convention)
        ax12 = fig.add_subplot(gs[3, 2])
        target_altitude = 30.0  # Assumed from controller
        altitude_error = target_altitude - altitude
        ax12.scatter(altitude_error, elevator, c=time, cmap='viridis', s=1, alpha=0.5)
        ax12.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax12.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax12.set_xlabel('Altitude Error (m)')
        ax12.set_ylabel('Elevator (deg)')
        ax12.set_title('Elevator Response to Altitude Error', fontweight='bold')
        ax12.grid(True, alpha=0.3)

        # Row 5: Trajectory and Control Surface Correlation
        # 3D-like trajectory view
        ax13 = fig.add_subplot(gs[4, :2])
        distance = np.cumsum(u * dt)
        ax13.plot(distance, altitude, color=colors[ep], alpha=alpha, linewidth=lw+1)
        ax13.fill_between(distance, 0, altitude, alpha=0.1, color=colors[ep])
        ax13.axhline(y=target_altitude, color='r', linestyle='--', alpha=0.3, label='Target Alt')
        ax13.set_xlabel('Distance (m)')
        ax13.set_ylabel('Altitude (m)')
        ax13.set_title('Flight Path Profile', fontweight='bold')
        ax13.grid(True, alpha=0.3)
        ax13.legend()

        # Pitch vs Elevator (verify correlation)
        ax14 = fig.add_subplot(gs[4, 2])
        ax14.scatter(pitch, elevator, c=time, cmap='plasma', s=1, alpha=0.5)
        ax14.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax14.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax14.set_xlabel('Pitch (deg)')
        ax14.set_ylabel('Elevator (deg)')
        ax14.set_title('Pitch vs Elevator Control', fontweight='bold')
        ax14.grid(True, alpha=0.3)

    plt.suptitle(f'BC Dataset Visualization - CORRECTED Elevator Sign Convention\n' +
                 f'Dataset: {Path(args.dataset).name} | Episodes: {n_plot}/{n_episodes}',
                 fontsize=14, fontweight='bold', y=0.995)

    # Save plot
    output_path = Path(__file__).parent.parent / args.output
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")

    # Print statistics
    print("\n" + "=" * 70)
    print("  DATASET STATISTICS")
    print("=" * 70)

    # Overall statistics
    all_altitude = -observations[:, StateIndex.Z]
    all_pitch = np.rad2deg(observations[:, StateIndex.THETA])
    all_elevator = np.rad2deg(actions[:, ControlIndex.ELEVATOR])
    all_airspeed = observations[:, StateIndex.U]

    print(f"\nAltitude:")
    print(f"  Mean: {all_altitude.mean():.2f} m")
    print(f"  Std:  {all_altitude.std():.2f} m")
    print(f"  Range: [{all_altitude.min():.2f}, {all_altitude.max():.2f}] m")

    print(f"\nAirspeed:")
    print(f"  Mean: {all_airspeed.mean():.2f} m/s")
    print(f"  Std:  {all_airspeed.std():.2f} m/s")
    print(f"  Range: [{all_airspeed.min():.2f}, {all_airspeed.max():.2f}] m/s")

    print(f"\nPitch:")
    print(f"  Mean: {all_pitch.mean():.2f}°")
    print(f"  Std:  {all_pitch.std():.2f}°")
    print(f"  Range: [{all_pitch.min():.2f}, {all_pitch.max():.2f}]°")

    print(f"\nElevator:")
    print(f"  Mean: {all_elevator.mean():.2f}°")
    print(f"  Std:  {all_elevator.std():.2f}°")
    print(f"  Range: [{all_elevator.min():.2f}, {all_elevator.max():.2f}]°")

    # Check for sign convention correctness
    print("\n" + "=" * 70)
    print("  ELEVATOR SIGN CONVENTION CHECK")
    print("=" * 70)

    # Correlation between altitude error and elevator
    target_alt = 30.0
    alt_error = target_alt - all_altitude
    correlation = np.corrcoef(alt_error, all_elevator)[0, 1]

    print(f"\nCorrelation(altitude_error, elevator): {correlation:.3f}")
    if correlation > 0.3:
        print("  ✓ POSITIVE correlation: altitude below target → positive elevator")
        print("  ✓ This is CORRECT for the negated elevator convention!")
    elif correlation < -0.3:
        print("  ✗ NEGATIVE correlation: altitude below target → negative elevator")
        print("  ✗ This suggests elevator sign may still be wrong")
    else:
        print("  ~ WEAK correlation: elevator response may be indirect")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
