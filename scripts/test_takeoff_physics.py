#!/usr/bin/env python3
"""
Test Takeoff Physics with Corrected Elevator Sign Convention

This script validates that the elevator sign fix works correctly by simulating
a proper takeoff sequence:
1. Ground roll with increasing throttle
2. Rotation at V_rotate
3. Liftoff and climb to 300 ft
4. Level off
5. Cruise

Expected behavior with CORRECTED elevator:
- During rotation: pitch increases (nose up)
- During climb: maintains positive pitch angle
- During level-off: pitch decreases to 0
- No pitch oscillations or divergence
- Smooth, stable flight

Author: Kushal Koirala (with Claude Code assistance)
Date: December 26, 2024
"""

import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add gpu-flight-dynamics to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))

from flight_dynamics import FlightSimulator, StateIndex, ControlIndex, STATE_DIM, CONTROL_DIM
from aircraft_database import get_aircraft

def main():
    print("=" * 70)
    print("  TAKEOFF PHYSICS TEST - CORRECTED ELEVATOR SIGN")
    print("=" * 70)

    # Get Udaan aircraft configuration
    config = get_aircraft('udaan')

    # Convert to params format
    from generate_bc_dataset_gpu import config_to_params
    params = config_to_params(config)

    # Create simulator (single instance for testing)
    sim = FlightSimulator(n_instances=1, params=params, dt=0.02)

    # Initialize state on runway (NED frame: down is positive Z, up is negative Z)
    # Landing gear height = 0.3m, so aircraft CG should be at Z = -0.3
    # (negative Z means above ground in NED)
    GEAR_HEIGHT = 0.3  # meters
    sim.states[0, StateIndex.Z] = -GEAR_HEIGHT  # On gear, CG above ground
    sim.states[0, StateIndex.U] = 0.0
    sim.states[0, StateIndex.V] = 0.0
    sim.states[0, StateIndex.W] = 0.0
    sim.states[0, StateIndex.THETA] = 0.0  # Level pitch
    sim.states[0, StateIndex.PHI] = 0.0    # Wings level
    sim.states[0, StateIndex.PSI] = 0.0    # Heading north

    # Simulation parameters
    V_stall = 11.2  # m/s (from PropShox)
    V_rotate = 1.15 * V_stall  # 12.88 m/s
    target_climb_alt = 91.4  # 300 ft in meters
    dt = 0.02
    max_time = 30.0

    # Storage for plotting
    times = []
    velocities = []
    altitudes = []
    pitches = []
    pitch_rates = []
    elevators = []
    thrusts = []
    climb_rates = []

    # Control vector
    control = np.zeros((1, CONTROL_DIM), dtype=np.float32)

    # Flight phase tracking
    phase = "GROUND_ROLL"
    rotation_time = None
    liftoff_time = None
    level_off_time = None

    print(f"\nSimulation parameters:")
    print(f"  V_stall = {V_stall:.1f} m/s ({V_stall * 2.237:.1f} mph)")
    print(f"  V_rotate = {V_rotate:.1f} m/s ({V_rotate * 2.237:.1f} mph)")
    print(f"  Target altitude = {target_climb_alt:.1f} m (300 ft)")
    print(f"\nStarting takeoff sequence...")

    for step in range(int(max_time / dt)):
        t = step * dt

        # Extract current state (convert from CuPy to numpy if needed)
        state = sim.states
        try:
            state_np = state.get()  # CuPy to numpy
        except:
            state_np = state  # Already numpy

        u = float(state_np[0, StateIndex.U])
        w = float(state_np[0, StateIndex.W])
        z = float(state_np[0, StateIndex.Z])
        theta = float(state_np[0, StateIndex.THETA])
        q = float(state_np[0, StateIndex.Q])

        altitude = -z  # NED to AGL
        climb_rate = -w
        V = u  # Approximate airspeed (assumes small v, w)
        pitch_deg = np.rad2deg(theta)
        q_deg = np.rad2deg(q)

        # Phase-based control
        if phase == "GROUND_ROLL":
            # Gradually increase throttle
            throttle = min(1.0, t / 2.0)
            elevator_deg = 0.0  # Neutral

            # Check for rotation speed
            if V >= V_rotate:
                phase = "ROTATION"
                rotation_time = t
                print(f"\n  t={t:.2f}s: ROTATION at V={V:.2f} m/s")

        elif phase == "ROTATION":
            # Full throttle
            throttle = 1.0

            # Target 10° pitch for rotation
            target_pitch_deg = 10.0
            pitch_error_deg = target_pitch_deg - pitch_deg

            # Simple proportional control for elevator
            # IMPORTANT: The negation is now in generate_bc_dataset_gpu.py
            # Here we command positive for nose-up, which gets negated there
            elevator_deg = pitch_error_deg * 0.5 + q_deg * 0.1
            elevator_deg = np.clip(elevator_deg, -15.0, 15.0)

            # Check for liftoff (altitude > 1m)
            if altitude > 1.0:
                phase = "CLIMB"
                liftoff_time = t
                print(f"  t={t:.2f}s: LIFTOFF at alt={altitude:.1f}m, pitch={pitch_deg:.1f}°")

        elif phase == "CLIMB":
            # Full throttle
            throttle = 1.0

            # Maintain 10° pitch for climb
            target_pitch_deg = 10.0
            pitch_error_deg = target_pitch_deg - pitch_deg
            elevator_deg = pitch_error_deg * 0.3 + q_deg * 0.05
            elevator_deg = np.clip(elevator_deg, -15.0, 15.0)

            # Check for level-off altitude
            if altitude >= target_climb_alt * 0.95:  # Start leveling at 95%
                phase = "LEVEL_OFF"
                level_off_time = t
                print(f"  t={t:.2f}s: LEVEL OFF at alt={altitude:.1f}m")

        elif phase == "LEVEL_OFF":
            # Reduce throttle slightly
            throttle = 0.7

            # Gradually reduce pitch to 0°
            target_pitch_deg = 0.0
            pitch_error_deg = target_pitch_deg - pitch_deg
            elevator_deg = pitch_error_deg * 0.2 + q_deg * 0.05
            elevator_deg = np.clip(elevator_deg, -15.0, 15.0)

            # Once level, go to cruise
            if abs(pitch_deg) < 2.0 and abs(climb_rate) < 1.0:
                phase = "CRUISE"
                print(f"  t={t:.2f}s: CRUISE at alt={altitude:.1f}m, V={V:.1f} m/s")

        else:  # CRUISE
            # Cruise throttle
            throttle = 0.6

            # Altitude hold
            alt_error = target_climb_alt - altitude
            target_climb_rate = alt_error * 0.05
            climb_error = target_climb_rate - climb_rate
            elevator_deg = climb_error * 0.1 - q_deg * 0.05
            elevator_deg = np.clip(elevator_deg, -15.0, 15.0)

        # Set controls
        # CRITICAL: Apply elevator sign correction (same as in BC controller)
        # Positive elevator_deg command (nose up) needs to be NEGATED
        # because this aircraft's Cm,δe is negative
        control[0, ControlIndex.THROTTLE] = throttle
        control[0, ControlIndex.ELEVATOR] = -np.deg2rad(elevator_deg)  # NEGATED!
        control[0, ControlIndex.AILERON] = 0.0
        control[0, ControlIndex.RUDDER] = 0.0

        # Set controls in simulator and step
        try:
            sim.set_controls(control)
            sim.step()
        except Exception as e:
            print(f"\n  ERROR at t={t:.2f}s: {e}")
            break

        # Store data
        times.append(t)
        velocities.append(V)
        altitudes.append(altitude)
        pitches.append(pitch_deg)
        pitch_rates.append(q_deg)
        elevators.append(elevator_deg)
        thrusts.append(throttle)
        climb_rates.append(climb_rate)

        # Print periodic updates
        if step % 50 == 0:  # Every 1 second
            print(f"  t={t:5.1f}s: {phase:12s} V={V:5.1f} m/s, alt={altitude:6.1f}m, "
                  f"pitch={pitch_deg:6.2f}°, de={elevator_deg:6.2f}°")

        # Safety checks
        if altitude < -1.0:
            print(f"\n  GROUND IMPACT at t={t:.2f}s")
            break
        if altitude > 500:
            print(f"\n  ALTITUDE LIMIT at t={t:.2f}s")
            break
        if abs(pitch_deg) > 60:
            print(f"\n  PITCH LIMIT at t={t:.2f}s")
            break

    # Final summary
    print(f"\n{'=' * 70}")
    print("  SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\nFinal state (t={times[-1]:.1f}s):")
    print(f"  Phase: {phase}")
    print(f"  Velocity: {velocities[-1]:.1f} m/s ({velocities[-1] * 2.237:.1f} mph)")
    print(f"  Altitude: {altitudes[-1]:.1f} m ({altitudes[-1] * 3.281:.0f} ft)")
    print(f"  Pitch: {pitches[-1]:.1f}°")
    print(f"  Climb rate: {climb_rates[-1]:.1f} m/s")

    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Velocity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, velocities, 'b-', linewidth=2)
    ax1.axhline(y=V_stall, color='r', linestyle='--', alpha=0.5, label=f'V_stall ({V_stall:.1f} m/s)')
    ax1.axhline(y=V_rotate, color='g', linestyle='--', alpha=0.5, label=f'V_rotate ({V_rotate:.1f} m/s)')
    if rotation_time:
        ax1.axvline(x=rotation_time, color='orange', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Airspeed', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Altitude
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, altitudes, 'g-', linewidth=2)
    ax2.axhline(y=target_climb_alt, color='r', linestyle='--', alpha=0.5, label=f'Target ({target_climb_alt:.0f} m)')
    if liftoff_time:
        ax2.axvline(x=liftoff_time, color='orange', linestyle=':', alpha=0.5, label='Liftoff')
    if level_off_time:
        ax2.axvline(x=level_off_time, color='purple', linestyle=':', alpha=0.5, label='Level off')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude AGL', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Pitch attitude
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(times, pitches, 'r-', linewidth=2)
    ax3.axhline(y=10, color='g', linestyle='--', alpha=0.3, label='Target climb (10°)')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Level')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Pitch (deg)')
    ax3.set_title('Pitch Attitude', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Pitch rate
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(times, pitch_rates, 'm-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Pitch Rate (deg/s)')
    ax4.set_title('Pitch Rate', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Elevator deflection
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(times, elevators, 'c-', linewidth=2)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.axhline(y=15, color='r', linestyle='--', alpha=0.3, linewidth=0.5)
    ax5.axhline(y=-15, color='r', linestyle='--', alpha=0.3, linewidth=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Elevator Command (deg)')
    ax5.set_title('Elevator (CORRECTED: negated before applying)', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Throttle
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(times, thrusts, 'orange', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Throttle')
    ax6.set_title('Throttle Setting', fontweight='bold')
    ax6.set_ylim([0, 1.1])
    ax6.grid(True, alpha=0.3)

    # Climb rate
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(times, climb_rates, 'b-', linewidth=2)
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Climb Rate (m/s)')
    ax7.set_title('Vertical Speed', fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Trajectory (altitude vs distance)
    distances = np.cumsum(np.array(velocities) * dt)
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.plot(distances, altitudes, 'b-', linewidth=2.5)
    ax8.fill_between(distances, 0, altitudes, alpha=0.2)
    ax8.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax8.axhline(y=target_climb_alt, color='r', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Distance (m)')
    ax8.set_ylabel('Altitude (m)')
    ax8.set_title('Flight Trajectory', fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # Phase diagram: pitch vs pitch rate
    ax9 = fig.add_subplot(gs[3, :])
    scatter = ax9.scatter(pitches, pitch_rates, c=times, cmap='viridis', s=10, alpha=0.6)
    ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
    ax9.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
    ax9.set_xlabel('Pitch (deg)')
    ax9.set_ylabel('Pitch Rate (deg/s)')
    ax9.set_title('Phase Diagram: Pitch vs Pitch Rate (color = time)', fontweight='bold')
    ax9.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax9, label='Time (s)')

    plt.suptitle('Takeoff Physics Test - CORRECTED Elevator Sign Convention',
                 fontsize=14, fontweight='bold')

    output_path = Path(__file__).parent.parent / 'takeoff_physics_CORRECTED.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")

    # Return success if we made it to cruise
    return phase == "CRUISE"


if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 70)
    if success:
        print("  ✓ TEST PASSED: Successful takeoff with corrected elevator!")
    else:
        print("  ✗ TEST INCOMPLETE: Did not reach cruise phase")
    print("=" * 70 + "\n")
    sys.exit(0 if success else 1)
