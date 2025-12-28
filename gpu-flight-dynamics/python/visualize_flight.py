"""
Flight Dynamics Visualization

Demonstrates actual flight behavior with plots showing:
- 3D trajectory
- Attitude (roll, pitch, yaw)
- Control responses
- Multiple aircraft comparison

Author: Kushal Koirala
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from flight_dynamics import FlightSimulator, StateIndex, ControlIndex, STATE_DIM, CONTROL_DIM


def rad2deg(rad):
    return rad * 180.0 / np.pi


def run_maneuver_simulation():
    """
    Simulate different maneuvers and collect data for visualization.
    """
    dt = 0.01
    duration = 20.0  # seconds
    n_steps = int(duration / dt)
    
    # We'll simulate 4 aircraft with different control inputs
    n_aircraft = 4
    maneuver_names = [
        "Straight & Level",
        "Climbing Turn",
        "Barrel Roll",
        "Pitch Oscillation"
    ]
    
    sim = FlightSimulator(n_instances=n_aircraft, use_gpu=True, dt=dt)
    
    # Storage for time history
    time_history = np.zeros(n_steps)
    state_history = np.zeros((n_steps, n_aircraft, STATE_DIM))
    control_history = np.zeros((n_steps, n_aircraft, CONTROL_DIM))
    
    # Initial trim state (will be set by reset)
    sim.reset()
    
    # Set initial altitude and airspeed for all aircraft
    initial_states = np.zeros((n_aircraft, STATE_DIM), dtype=np.float32)
    initial_states[:, StateIndex.Z] = -1000.0  # 1000m altitude
    initial_states[:, StateIndex.U] = 50.0      # 50 m/s forward
    
    # Spread aircraft out in Y direction for visualization
    initial_states[0, StateIndex.Y] = 0.0
    initial_states[1, StateIndex.Y] = 200.0
    initial_states[2, StateIndex.Y] = 400.0
    initial_states[3, StateIndex.Y] = 600.0
    
    sim.reset(initial_state=initial_states)
    
    # Simulation loop
    for step in range(n_steps):
        t = step * dt
        time_history[step] = t
        
        # Define control inputs for each maneuver
        controls = np.zeros((n_aircraft, CONTROL_DIM), dtype=np.float32)
        
        # Aircraft 0: Straight and level (trim)
        controls[0, ControlIndex.THROTTLE] = 0.4
        controls[0, ControlIndex.ELEVATOR] = 0.02
        
        # Aircraft 1: Climbing turn (bank + pull + power)
        controls[1, ControlIndex.THROTTLE] = 0.7
        controls[1, ControlIndex.AILERON] = 0.2 if t < 5 else 0.0  # Roll into turn
        controls[1, ControlIndex.ELEVATOR] = 0.15  # Pull up
        
        # Aircraft 2: Barrel roll
        if t < 2:
            # Setup: slight climb
            controls[2, ControlIndex.THROTTLE] = 0.6
            controls[2, ControlIndex.ELEVATOR] = 0.1
        elif t < 6:
            # Roll
            controls[2, ControlIndex.THROTTLE] = 0.5
            controls[2, ControlIndex.AILERON] = 0.8
            controls[2, ControlIndex.ELEVATOR] = 0.2
        else:
            # Recovery
            controls[2, ControlIndex.THROTTLE] = 0.4
            controls[2, ControlIndex.ELEVATOR] = 0.0
        
        # Aircraft 3: Pitch oscillation (phugoid-like)
        controls[3, ControlIndex.THROTTLE] = 0.4
        controls[3, ControlIndex.ELEVATOR] = 0.15 * np.sin(0.5 * t)
        
        # Record controls
        control_history[step] = controls
        
        # Apply controls and step
        sim.set_controls(controls)
        sim.step()
        
        # Record states
        state_history[step] = sim.get_states()
    
    return time_history, state_history, control_history, maneuver_names


def plot_3d_trajectories(time_history, state_history, maneuver_names):
    """
    Plot 3D trajectories of all aircraft.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, name in enumerate(maneuver_names):
        x = state_history[:, i, StateIndex.X]
        y = state_history[:, i, StateIndex.Y]
        z = -state_history[:, i, StateIndex.Z]  # Convert to altitude (positive up)
        
        ax.plot(x, y, z, color=colors[i], label=name, linewidth=2)
        
        # Mark start and end points
        ax.scatter(x[0], y[0], z[0], color=colors[i], s=100, marker='o')
        ax.scatter(x[-1], y[-1], z[-1], color=colors[i], s=100, marker='^')
    
    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('6-DOF Flight Trajectories - GPU Accelerated Simulation')
    ax.legend(loc='upper left')
    
    # Set equal aspect ratio
    max_range = max(
        state_history[:, :, StateIndex.X].max() - state_history[:, :, StateIndex.X].min(),
        state_history[:, :, StateIndex.Y].max() - state_history[:, :, StateIndex.Y].min(),
        (-state_history[:, :, StateIndex.Z]).max() - (-state_history[:, :, StateIndex.Z]).min()
    ) / 2
    
    mid_x = (state_history[:, :, StateIndex.X].max() + state_history[:, :, StateIndex.X].min()) / 2
    mid_y = (state_history[:, :, StateIndex.Y].max() + state_history[:, :, StateIndex.Y].min()) / 2
    mid_z = ((-state_history[:, :, StateIndex.Z]).max() + (-state_history[:, :, StateIndex.Z]).min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig


def plot_attitude_history(time_history, state_history, maneuver_names):
    """
    Plot attitude angles over time.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # Roll
    ax = axes[0]
    for i, name in enumerate(maneuver_names):
        phi = rad2deg(state_history[:, i, StateIndex.PHI])
        ax.plot(time_history, phi, color=colors[i], label=name, linewidth=1.5)
    ax.set_ylabel('Roll φ (deg)')
    ax.set_title('Aircraft Attitude History')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Pitch
    ax = axes[1]
    for i, name in enumerate(maneuver_names):
        theta = rad2deg(state_history[:, i, StateIndex.THETA])
        ax.plot(time_history, theta, color=colors[i], label=name, linewidth=1.5)
    ax.set_ylabel('Pitch θ (deg)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Yaw
    ax = axes[2]
    for i, name in enumerate(maneuver_names):
        psi = rad2deg(state_history[:, i, StateIndex.PSI])
        ax.plot(time_history, psi, color=colors[i], label=name, linewidth=1.5)
    ax.set_ylabel('Yaw ψ (deg)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_control_response(time_history, state_history, control_history, aircraft_idx, name):
    """
    Plot control inputs and resulting aircraft response for one aircraft.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Control inputs (left column)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_history, control_history[:, aircraft_idx, ControlIndex.THROTTLE], 'b-', linewidth=2)
    ax1.set_ylabel('Throttle')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title(f'Control Inputs: {name}')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_history, control_history[:, aircraft_idx, ControlIndex.ELEVATOR], 'r-', linewidth=2)
    ax2.set_ylabel('Elevator')
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time_history, control_history[:, aircraft_idx, ControlIndex.AILERON], 'g-', linewidth=2)
    ax3.set_ylabel('Aileron')
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(time_history, control_history[:, aircraft_idx, ControlIndex.RUDDER], 'm-', linewidth=2)
    ax4.set_ylabel('Rudder')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylim(-1.1, 1.1)
    ax4.grid(True, alpha=0.3)
    
    # Aircraft response (right column)
    ax5 = fig.add_subplot(gs[0, 1])
    airspeed = np.sqrt(
        state_history[:, aircraft_idx, StateIndex.U]**2 +
        state_history[:, aircraft_idx, StateIndex.V]**2 +
        state_history[:, aircraft_idx, StateIndex.W]**2
    )
    ax5.plot(time_history, airspeed, 'b-', linewidth=2)
    ax5.set_ylabel('Airspeed (m/s)')
    ax5.set_title(f'Aircraft Response: {name}')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 1])
    altitude = -state_history[:, aircraft_idx, StateIndex.Z]
    ax6.plot(time_history, altitude, 'r-', linewidth=2)
    ax6.set_ylabel('Altitude (m)')
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 1])
    roll = rad2deg(state_history[:, aircraft_idx, StateIndex.PHI])
    pitch = rad2deg(state_history[:, aircraft_idx, StateIndex.THETA])
    ax7.plot(time_history, roll, 'g-', linewidth=2, label='Roll')
    ax7.plot(time_history, pitch, 'orange', linewidth=2, label='Pitch')
    ax7.set_ylabel('Attitude (deg)')
    ax7.legend(loc='upper right')
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[3, 1])
    p = rad2deg(state_history[:, aircraft_idx, StateIndex.P])
    q = rad2deg(state_history[:, aircraft_idx, StateIndex.Q])
    r = rad2deg(state_history[:, aircraft_idx, StateIndex.R])
    ax8.plot(time_history, p, 'g-', linewidth=1.5, label='p (roll rate)')
    ax8.plot(time_history, q, 'orange', linewidth=1.5, label='q (pitch rate)')
    ax8.plot(time_history, r, 'm-', linewidth=1.5, label='r (yaw rate)')
    ax8.set_ylabel('Angular Rates (deg/s)')
    ax8.set_xlabel('Time (s)')
    ax8.legend(loc='upper right')
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle(f'Control Response Analysis: {name}', fontsize=14, fontweight='bold')
    return fig


def plot_monte_carlo_spread():
    """
    Show Monte Carlo-style spread of trajectories with small initial perturbations.
    This demonstrates the parallel simulation capability.
    """
    n_aircraft = 100
    dt = 0.01
    duration = 15.0
    n_steps = int(duration / dt)
    
    print(f"Running Monte Carlo simulation with {n_aircraft} aircraft...")
    
    sim = FlightSimulator(n_instances=n_aircraft, use_gpu=True, dt=dt)
    
    # Initial states with small random perturbations
    initial_states = np.zeros((n_aircraft, STATE_DIM), dtype=np.float32)
    initial_states[:, StateIndex.Z] = -1000.0 + np.random.randn(n_aircraft) * 10  # ±10m altitude
    initial_states[:, StateIndex.U] = 50.0 + np.random.randn(n_aircraft) * 2       # ±2 m/s speed
    initial_states[:, StateIndex.PHI] = np.random.randn(n_aircraft) * 0.05         # ±3 deg roll
    initial_states[:, StateIndex.THETA] = np.random.randn(n_aircraft) * 0.02       # ±1 deg pitch
    
    sim.reset(initial_state=initial_states)
    
    # Storage
    trajectory_x = np.zeros((n_steps, n_aircraft))
    trajectory_z = np.zeros((n_steps, n_aircraft))
    
    # Same control for all (slight pitch up)
    controls = np.zeros((n_aircraft, CONTROL_DIM), dtype=np.float32)
    controls[:, ControlIndex.THROTTLE] = 0.45
    controls[:, ControlIndex.ELEVATOR] = 0.05
    
    for step in range(n_steps):
        # Add small random control perturbations
        noisy_controls = controls.copy()
        noisy_controls[:, ControlIndex.ELEVATOR] += np.random.randn(n_aircraft) * 0.01
        noisy_controls[:, ControlIndex.AILERON] += np.random.randn(n_aircraft) * 0.01
        
        sim.set_controls(noisy_controls)
        sim.step()
        
        states = sim.get_states()
        trajectory_x[step] = states[:, StateIndex.X]
        trajectory_z[step] = -states[:, StateIndex.Z]  # Altitude
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # X-Z (side view)
    ax = axes[0]
    for i in range(n_aircraft):
        alpha = 0.3 if i > 0 else 1.0
        color = 'blue' if i > 0 else 'red'
        lw = 0.5 if i > 0 else 2
        ax.plot(trajectory_x[:, i], trajectory_z[:, i], color=color, alpha=alpha, linewidth=lw)
    
    ax.set_xlabel('Distance North (m)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'Monte Carlo Trajectory Spread (N={n_aircraft})\nSide View')
    ax.grid(True, alpha=0.3)
    
    # Final position distribution
    ax = axes[1]
    final_x = trajectory_x[-1, :]
    final_z = trajectory_z[-1, :]
    ax.scatter(final_x, final_z, c='blue', alpha=0.5, s=30)
    ax.axhline(y=final_z.mean(), color='red', linestyle='--', label=f'Mean alt: {final_z.mean():.0f}m')
    ax.axvline(x=final_x.mean(), color='red', linestyle='--')
    
    # Add standard deviation ellipse
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(
        (final_x.mean(), final_z.mean()),
        width=2*final_x.std(),
        height=2*final_z.std(),
        fill=False,
        color='red',
        linestyle='-',
        linewidth=2,
        label=f'1σ: ±{final_x.std():.0f}m, ±{final_z.std():.0f}m'
    )
    ax.add_patch(ellipse)
    
    ax.set_xlabel('Final X Position (m)')
    ax.set_ylabel('Final Altitude (m)')
    ax.set_title('Final Position Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.suptitle('GPU-Accelerated Monte Carlo Flight Simulation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("GPU-Accelerated Flight Dynamics Visualization")
    print("=" * 60)
    print()
    
    # Run maneuver simulation
    print("Simulating flight maneuvers...")
    time_history, state_history, control_history, maneuver_names = run_maneuver_simulation()
    print(f"  Simulated {len(maneuver_names)} aircraft for {time_history[-1]:.1f} seconds")
    print(f"  Total simulation steps: {len(time_history)}")
    print()
    
    # Create visualizations
    print("Generating visualizations...")
    
    # 1. 3D trajectories
    fig1 = plot_3d_trajectories(time_history, state_history, maneuver_names)
    fig1.savefig('trajectory_3d.png', dpi=150, bbox_inches='tight')
    print("  Saved: trajectory_3d.png")
    
    # 2. Attitude history
    fig2 = plot_attitude_history(time_history, state_history, maneuver_names)
    fig2.savefig('attitude_history.png', dpi=150, bbox_inches='tight')
    print("  Saved: attitude_history.png")
    
    # 3. Control response for barrel roll
    fig3 = plot_control_response(time_history, state_history, control_history, 2, "Barrel Roll")
    fig3.savefig('control_response_barrel_roll.png', dpi=150, bbox_inches='tight')
    print("  Saved: control_response_barrel_roll.png")
    
    # 4. Control response for climbing turn
    fig4 = plot_control_response(time_history, state_history, control_history, 1, "Climbing Turn")
    fig4.savefig('control_response_climbing_turn.png', dpi=150, bbox_inches='tight')
    print("  Saved: control_response_climbing_turn.png")
    
    # 5. Monte Carlo spread
    fig5 = plot_monte_carlo_spread()
    fig5.savefig('monte_carlo_spread.png', dpi=150, bbox_inches='tight')
    print("  Saved: monte_carlo_spread.png")
    
    print()
    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    
    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
