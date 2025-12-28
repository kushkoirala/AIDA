#!/usr/bin/env python3
"""
3D Visualization of Phase 1 Ground Roll

Creates an interactive 3D plot showing the Cessna 172's ground roll trajectory.

Author: Kushal Koirala (with Claude Code)
Date: December 27, 2024
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from aida_sim.env.flight_env_cessna172 import Cessna172Env


def visualize_ground_roll(model_path, n_episodes=3, save_path=None):
    """
    Visualize ground roll trajectories in 3D.

    Args:
        model_path: Path to trained model
        n_episodes: Number of episodes to visualize
        save_path: Optional path to save plot
    """
    print("\n" + "="*60)
    print("  3D GROUND ROLL VISUALIZATION")
    print("="*60)
    print(f"\nModel: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("="*60 + "\n")

    # Create environment
    env = Cessna172Env(task='ground_roll', cruise_altitude_ft=3000.0)

    # Load model
    print(f"Loading model...")
    model = PPO.load(model_path)
    print(f"✓ Model loaded\n")

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title('3D Trajectory - Ground Roll', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m) - Lateral')
    ax1.set_ylabel('Y (m) - Distance Along Runway')
    ax1.set_zlabel('Z (m) - Altitude')

    # Top-down view (runway tracking)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Top-Down View - Runway Tracking', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m) - Lateral Position')
    ax2.set_ylabel('Y (m) - Distance Along Runway')
    ax2.grid(True, alpha=0.3)

    # Speed vs Time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('Airspeed vs Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Airspeed (KIAS)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=55, color='r', linestyle='--', label='Rotation Speed (55 KIAS)', linewidth=2)

    # Lateral Deviation vs Distance
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('Lateral Deviation vs Distance', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Distance Along Runway (m)')
    ax4.set_ylabel('Lateral Deviation (m)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=15, color='orange', linestyle='--', label='Runway Edge (15m)', alpha=0.5)
    ax4.axhline(y=-15, color='orange', linestyle='--', alpha=0.5)

    colors = plt.cm.viridis(np.linspace(0, 1, n_episodes))

    for ep in range(n_episodes):
        print(f"Simulating Episode {ep+1}/{n_episodes}...")

        obs, info = env.reset()
        done = False
        steps = 0

        # Data storage
        trajectory_x = []
        trajectory_y = []
        trajectory_z = []
        airspeeds = []
        times = []
        lateral_devs = []

        while not done and steps < 500:
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record data
            x, y, z = obs[0], obs[1], obs[2]
            trajectory_x.append(x)
            trajectory_y.append(y)
            trajectory_z.append(-z)  # Convert NED to visualization (Z up)

            airspeed_kias = info['airspeed'] * 1.944  # m/s to knots
            airspeeds.append(airspeed_kias)
            times.append(steps * env.dt)
            lateral_devs.append(abs(x))

            steps += 1

        # Convert to numpy arrays
        trajectory_x = np.array(trajectory_x)
        trajectory_y = np.array(trajectory_y)
        trajectory_z = np.array(trajectory_z)
        airspeeds = np.array(airspeeds)
        times = np.array(times)
        lateral_devs = np.array(lateral_devs)

        episode_label = f'Episode {ep+1}'

        # Plot 3D trajectory
        ax1.plot(trajectory_x, trajectory_y, trajectory_z,
                color=colors[ep], linewidth=2, label=episode_label, alpha=0.8)
        ax1.scatter(trajectory_x[0], trajectory_y[0], trajectory_z[0],
                   color='green', s=100, marker='o', label='Start' if ep == 0 else '')
        ax1.scatter(trajectory_x[-1], trajectory_y[-1], trajectory_z[-1],
                   color='red', s=100, marker='x', label='End' if ep == 0 else '')

        # Plot top-down view
        ax2.plot(trajectory_x, trajectory_y, color=colors[ep], linewidth=2,
                label=episode_label, alpha=0.8)
        ax2.scatter(trajectory_x[0], trajectory_y[0], color='green', s=100, marker='o')
        ax2.scatter(trajectory_x[-1], trajectory_y[-1], color='red', s=100, marker='x')

        # Plot airspeed
        ax3.plot(times, airspeeds, color=colors[ep], linewidth=2,
                label=episode_label, alpha=0.8)

        # Plot lateral deviation
        ax4.plot(trajectory_y, lateral_devs, color=colors[ep], linewidth=2,
                label=episode_label, alpha=0.8)

        print(f"  ✓ Episode {ep+1}: {steps} steps, max speed {airspeeds.max():.1f} KIAS")

    # Draw runway on top-down view
    runway_length = 1000  # meters
    runway_width = 30  # meters (±15m from centerline)
    runway_y = np.array([-500, 500])

    # Runway edges
    ax2.fill_betweenx(runway_y, -runway_width/2, runway_width/2,
                      color='gray', alpha=0.2, label='Runway')
    ax2.plot([-runway_width/2, -runway_width/2], runway_y, 'k--', alpha=0.5)
    ax2.plot([runway_width/2, runway_width/2], runway_y, 'k--', alpha=0.5)

    # Centerline
    ax2.plot([0, 0], runway_y, 'w--', linewidth=2, alpha=0.7, label='Centerline')

    # Set equal aspect for top-down view
    ax2.set_aspect('equal')

    # Add legends
    ax1.legend(loc='best', fontsize=9)
    ax2.legend(loc='best', fontsize=9)
    ax3.legend(loc='best', fontsize=9)
    ax4.legend(loc='best', fontsize=9)

    # Set view angle for 3D plot
    ax1.view_init(elev=20, azim=45)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        print(f"\nSaving plot to: {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved")

    print(f"\n✓ Displaying interactive plot...")
    print(f"  Rotate: Click and drag")
    print(f"  Zoom: Scroll wheel")
    print(f"  Close window when done")

    plt.show()

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Phase 1 Ground Roll')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model (default: auto-find best)')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to visualize (default: 3)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save plot to file (e.g., ground_roll.png)')
    args = parser.parse_args()

    # Auto-find model if not specified
    if args.model is None:
        # Try to find the best model
        best_model = "checkpoints/cessna172_curriculum/phase1_ground_roll/best_model.zip"
        if os.path.exists(best_model):
            model_path = best_model
            print(f"Using best model: {model_path}")
        else:
            # Try final model
            final_model = "checkpoints/cessna172_curriculum/phase1_ground_roll/phase1_ground_roll_ppo_final.zip"
            if os.path.exists(final_model):
                model_path = final_model
                print(f"Using final model: {model_path}")
            else:
                print(f"❌ No trained model found!")
                print(f"   Expected: {best_model}")
                print(f"   Or: {final_model}")
                return 1
    else:
        model_path = args.model

    # Visualize
    visualize_ground_roll(
        model_path=model_path,
        n_episodes=args.episodes,
        save_path=args.save
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
