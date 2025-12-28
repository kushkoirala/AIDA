"""
Test Proper Takeoff Sequence with CORRECTED Elevator Understanding

CRITICAL INSIGHT from Dynamics PDF Figure 2:
- Trim elevator for takeoff: δe = +9.1° (POSITIVE)
- Trim elevator for cruise: δe = +4.22° (POSITIVE)
- At AoA=0, CM ≈ +0.01 (nose UP tendency from positive Cm0)
- To counteract: need nose DOWN moment from elevator
- With CM,δe = -1.13: positive δe gives NEGATIVE Cm contribution (nose down)
- Therefore: POSITIVE elevator = nose DOWN moment (for TRIM)

ROTATION PROBLEM:
- During rotation, we WANT nose UP
- So we need NEGATIVE elevator deflection!
- The controller was commanding +10° (nose down) when we wanted nose up
- Solution: NEGATE the elevator command for pitch control

Standard Aircraft vs This Model:
- Standard: +δe = trailing edge down = nose UP moment
- This model: +δe = produces nose DOWN moment (likely defines +δe as trailing edge UP)
- OR: The model uses stability axes where elevator sign is opposite

FIX: Invert elevator sign when using it for PITCH CONTROL (not trim)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/AIDA/gpu-flight-dynamics/python')

try:
    from flight_dynamics import FlightSimulator, StateIndex, ControlIndex
    from aircraft_database import get_aircraft
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    print("="*70)
    print("  TAKEOFF TEST WITH CORRECTED ELEVATOR SIGN")
    print("="*70)

    # Get Udaan aircraft
    config = get_aircraft('udaan')

    # Create simulator
    sim = FlightSimulator(num_envs=1, aircraft_config=config, dt=0.02)

    # Initialize on runway
    state = np.zeros((1, sim.STATE_SIZE), dtype=np.float32)
    state[0, StateIndex.POS_Z] = 0.3  # On gear
    state[0, StateIndex.VEL_U] = 0.0
    state[0, StateIndex.VEL_V] = 0.0
    state[0, StateIndex.VEL_W] = 0.0
    state[0, StateIndex.THETA] = 0.0  # Level pitch
    state[0, StateIndex.PHI] = 0.0
    state[0, StateIndex.PSI] = 0.0

    # Simulation parameters
    V_stall = 11.2  # m/s
    V_rotate = 1.15 * V_stall  # 12.88 m/s
    dt = 0.02
    max_time = 10.0

    # Storage
    times = []
    velocities = []
    altitudes = []
    pitches = []
    pitch_rates = []
    elevators = []
    thrusts = []
    load_factors = []

    # Control sequence
    control = np.zeros((1, sim.CONTROL_SIZE), dtype=np.float32)

    print("\nStarting takeoff sequence...")
    print(f"  V_stall = {V_stall:.1f} m/s")
    print(f"  V_rotate = {V_rotate:.1f} m/s")

    # Phase tracking
    phase = "GROUND_ROLL"
    rotation_started = False
    liftoff_time = None

    for step in range(int(max_time / dt)):
        t = step * dt

        # Get current state
        V = state[0, StateIndex.VEL_U]
        alt = state[0, StateIndex.POS_Z]
        pitch = np.rad2deg(state[0, StateIndex.THETA])
        q = np.rad2deg(state[0, StateIndex.Q])

        # Phase logic
        if phase == "GROUND_ROLL":
            # Gradually increase throttle
            throttle = min(1.0, t / 2.0)

            # CRITICAL FIX: Use NEGATIVE elevator for nose UP rotation
            # Based on PDF: +δe gives nose DOWN, so -δe gives nose UP
            if V >= V_rotate and not rotation_started:
                phase = "ROTATION"
                rotation_started = True
                print(f"\nt={t:.2f}s: ROTATION at V={V:.2f} m/s")

            elevator_deg = 0.0  # Neutral during ground roll

        elif phase == "ROTATION":
            throttle = 1.0

            # NEGATIVE elevator for nose UP (opposite of standard convention)
            # Target: 10° pitch attitude
            pitch_error = 10.0 - pitch
            elevator_deg = -pitch_error * 0.5  # Proportional control, NEGATED
            elevator_deg = np.clip(elevator_deg, -15.0, 15.0)

            # Check for liftoff
            if alt > 0.5 and liftoff_time is None:
                liftoff_time = t
                phase = "CLIMB"
                print(f"t={t:.2f}s: LIFTOFF at alt={alt:.2f}m, pitch={pitch:.1f}°")

        elif phase == "CLIMB":
            throttle = 1.0

            # Maintain 10° pitch for climb to 300ft (91.4m)
            pitch_error = 10.0 - pitch
            elevator_deg = -pitch_error * 0.3  # NEGATED
            elevator_deg = np.clip(elevator_deg, -15.0, 15.0)

            # Check for level-off altitude
            if alt >= 90.0:  # ~300 ft
                phase = "LEVEL_OFF"
                print(f"t={t:.2f}s: LEVEL OFF at alt={alt:.2f}m")

        elif phase == "LEVEL_OFF":
            throttle = 0.8

            # Gradually reduce pitch to 0°
            pitch_error = 0.0 - pitch
            elevator_deg = -pitch_error * 0.2  # NEGATED
            elevator_deg = np.clip(elevator_deg, -15.0, 15.0)

            # Once level, go to cruise
            if abs(pitch) < 1.0 and abs(q) < 2.0:
                phase = "CRUISE"
                print(f"t={t:.2f}s: CRUISE at alt={alt:.2f}m, V={V:.2f} m/s")

        else:  # CRUISE
            throttle = 0.6

            # Maintain level flight
            pitch_error = 0.0 - pitch
            elevator_deg = -pitch_error * 0.1  # NEGATED
            elevator_deg = np.clip(elevator_deg, -15.0, 15.0)

        # Set controls
        control[0, ControlIndex.THROTTLE] = throttle
        control[0, ControlIndex.ELEVATOR] = np.deg2rad(elevator_deg)
        control[0, ControlIndex.AILERON] = 0.0
        control[0, ControlIndex.RUDDER] = 0.0

        # Step simulation
        next_state = sim.step(state, control)

        # Store data
        times.append(t)
        velocities.append(V)
        altitudes.append(alt)
        pitches.append(pitch)
        pitch_rates.append(q)
        elevators.append(elevator_deg)
        thrusts.append(throttle)

        # Calculate load factor (from aerodynamic forces)
        # This is a simplified calculation
        load_factors.append(1.0)  # Placeholder

        # Update state
        state = next_state

        # Print periodic updates
        if step % 50 == 0:
            print(f"t={t:.1f}s: {phase:12s} V={V:5.1f} m/s, alt={alt:6.1f}m, "
                  f"pitch={pitch:6.1f}°, q={q:6.1f}°/s, de={elevator_deg:6.1f}°")

        # Check termination
        if alt < 0 or alt > 500 or abs(pitch) > 60:
            print(f"\nTERMINATED at t={t:.2f}s")
            print(f"  Reason: alt={alt:.1f}m, pitch={pitch:.1f}°")
            break

    # Plot results
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig)

    # Velocity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, velocities, 'b-', linewidth=2)
    ax1.axhline(y=V_stall, color='r', linestyle='--', label='V_stall')
    ax1.axhline(y=V_rotate, color='g', linestyle='--', label='V_rotate')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Airspeed')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Altitude
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, altitudes, 'g-', linewidth=2)
    ax2.axhline(y=91.4, color='r', linestyle='--', label='300 ft')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Pitch attitude
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(times, pitches, 'r-', linewidth=2)
    ax3.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Target climb')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Pitch (deg)')
    ax3.set_title('Pitch Attitude')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Pitch rate
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(times, pitch_rates, 'm-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Pitch Rate (deg/s)')
    ax4.set_title('Pitch Rate')
    ax4.grid(True, alpha=0.3)

    # Elevator deflection
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(times, elevators, 'c-', linewidth=2)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.axhline(y=15, color='r', linestyle='--', alpha=0.3, label='Limit')
    ax5.axhline(y=-15, color='r', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Elevator (deg)')
    ax5.set_title('Elevator Deflection (CORRECTED: -δe for nose UP)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Throttle
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(times, thrusts, 'orange', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Throttle')
    ax6.set_title('Throttle Setting')
    ax6.set_ylim([0, 1.1])
    ax6.grid(True, alpha=0.3)

    # Trajectory (altitude vs distance)
    distances = np.cumsum(np.array(velocities) * dt)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.plot(distances, altitudes, 'b-', linewidth=2)
    ax7.set_xlabel('Distance (m)')
    ax7.set_ylabel('Altitude (m)')
    ax7.set_title('Flight Trajectory')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('takeoff_CORRECTED_elevator.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: takeoff_CORRECTED_elevator.png")
    plt.close()

    print("\n" + "="*70)
    print("  TAKEOFF TEST COMPLETE")
    print("="*70)
    print(f"\nFinal state:")
    print(f"  Time: {times[-1]:.1f}s")
    print(f"  Velocity: {velocities[-1]:.1f} m/s")
    print(f"  Altitude: {altitudes[-1]:.1f} m")
    print(f"  Pitch: {pitches[-1]:.1f}°")
    print(f"  Phase: {phase}")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\nThis script requires access to the GPU flight dynamics module.")
    print("The module may be in WSL or a different environment.")
