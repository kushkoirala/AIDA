"""
F-16 Fighting Falcon Flight Dynamics Demonstration

Demonstrates the unique characteristics of a relaxed static stability fighter:
1. Unstable pitch dynamics (requires augmentation)
2. High roll rate capability
3. High-G maneuvering
4. Comparison with stable aircraft (Cessna 172)

Author: Kushal Koirala
Date: 2025

References:
    NASA TP-1538: F-16 stability derivatives
    Stevens & Lewis: Aircraft Control and Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from flight_dynamics import FlightSimulator, StateIndex, ControlIndex, STATE_DIM, CONTROL_DIM
from aircraft_database import get_aircraft, print_aircraft_info, AircraftConfig


def rad2deg(rad):
    return rad * 180.0 / np.pi


def config_to_params(config: AircraftConfig):
    """Convert AircraftConfig to the format expected by FlightSimulator"""
    from flight_dynamics import (
        AircraftParams, MassProperties, Geometry,
        LongitudinalDerivatives, LateralDerivatives, PropulsionParams
    )
    
    return AircraftParams(
        mass=MassProperties(
            mass=config.mass.mass,
            Ixx=config.mass.Ixx,
            Iyy=config.mass.Iyy,
            Izz=config.mass.Izz,
            Ixz=config.mass.Ixz,
        ),
        geom=Geometry(
            S=config.geom.S,
            b=config.geom.b,
            c=config.geom.c,
            e=config.geom.e,
        ),
        longi=LongitudinalDerivatives(
            CL0=config.longi.CL0,
            CLa=config.longi.CLa,
            CLq=config.longi.CLq,
            CLde=config.longi.CLde,
            CLmax=config.longi.CLmax,
            CLmin=config.longi.CLmin,
            CD0=config.longi.CD0,
            K=config.longi.K,
            CDa=config.longi.CDa,
            Cm0=config.longi.Cm0,
            Cma=config.longi.Cma,
            Cmq=config.longi.Cmq,
            Cmde=config.longi.Cmde,
        ),
        latdi=LateralDerivatives(
            CYb=config.latdi.CYb,
            CYp=config.latdi.CYp,
            CYr=config.latdi.CYr,
            CYda=config.latdi.CYda,
            CYdr=config.latdi.CYdr,
            Clb=config.latdi.Clb,
            Clp=config.latdi.Clp,
            Clr=config.latdi.Clr,
            Clda=config.latdi.Clda,
            Cldr=config.latdi.Cldr,
            Cnb=config.latdi.Cnb,
            Cnp=config.latdi.Cnp,
            Cnr=config.latdi.Cnr,
            Cnda=config.latdi.Cnda,
            Cndr=config.latdi.Cndr,
        ),
        prop=PropulsionParams(
            thrust_max=config.prop.thrust_max,
            thrust_min=config.prop.thrust_min,
            tau=config.prop.tau,
        ),
    )


def demo_pitch_instability():
    """
    Demonstrate pitch instability of the F-16 vs stability of Cessna.
    
    Both aircraft given a small pitch perturbation and released.
    - Cessna: Returns to trim (stable)
    - F-16: Diverges (unstable)
    """
    print("\n" + "="*60)
    print("DEMO 1: Pitch Stability Comparison")
    print("="*60)
    print("Both aircraft given 2° pitch-up perturbation, controls neutral.")
    print("Watch how they respond...\n")
    
    dt = 0.01
    duration = 5.0
    n_steps = int(duration / dt)
    
    # Get aircraft configs
    cessna_config = get_aircraft("cessna172")
    f16_config = get_aircraft("f16")  # Unstable version
    
    print(f"Cessna Cma: {cessna_config.longi.Cma:.3f} (negative = stable)")
    print(f"F-16 Cma:   {f16_config.longi.Cma:.3f} (positive = UNSTABLE)")
    
    # Create simulators
    cessna_params = config_to_params(cessna_config)
    f16_params = config_to_params(f16_config)
    
    sim_cessna = FlightSimulator(n_instances=1, params=cessna_params, dt=dt, use_gpu=True)
    sim_f16 = FlightSimulator(n_instances=1, params=f16_params, dt=dt, use_gpu=True)
    
    # Initial state: level flight with 2° pitch perturbation
    initial_state_cessna = np.zeros((1, STATE_DIM), dtype=np.float32)
    initial_state_cessna[0, StateIndex.Z] = -1000.0
    initial_state_cessna[0, StateIndex.U] = 50.0
    initial_state_cessna[0, StateIndex.THETA] = 2.0 * np.pi / 180.0  # 2° pitch up
    
    initial_state_f16 = np.zeros((1, STATE_DIM), dtype=np.float32)
    initial_state_f16[0, StateIndex.Z] = -4572.0  # 15,000 ft
    initial_state_f16[0, StateIndex.U] = 200.0     # ~Mach 0.6
    initial_state_f16[0, StateIndex.THETA] = 2.0 * np.pi / 180.0  # 2° pitch up
    
    sim_cessna.reset(initial_state=initial_state_cessna)
    sim_f16.reset(initial_state=initial_state_f16)
    
    # Neutral controls
    controls = np.zeros((1, CONTROL_DIM), dtype=np.float32)
    controls[0, ControlIndex.THROTTLE] = 0.3
    
    # Record history
    time_hist = np.zeros(n_steps)
    pitch_cessna = np.zeros(n_steps)
    pitch_f16 = np.zeros(n_steps)
    
    for step in range(n_steps):
        time_hist[step] = step * dt
        
        pitch_cessna[step] = rad2deg(sim_cessna.get_states()[0, StateIndex.THETA])
        pitch_f16[step] = rad2deg(sim_f16.get_states()[0, StateIndex.THETA])
        
        sim_cessna.set_controls(controls)
        sim_f16.set_controls(controls)
        
        sim_cessna.step()
        sim_f16.step()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time_hist, pitch_cessna, 'b-', linewidth=2, label='Cessna 172 (Cma = -0.61, stable)')
    ax.plot(time_hist, pitch_f16, 'r-', linewidth=2, label='F-16 (Cma = +0.04, UNSTABLE)')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='Initial perturbation')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Pitch Angle (deg)', fontsize=12)
    ax.set_title('Pitch Stability: F-16 vs Cessna 172\n(2° initial perturbation, hands-off)', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)
    
    # Add annotation
    ax.annotate('F-16 diverges!\n(needs flight computer)', 
                xy=(4, pitch_f16[-100]), 
                xytext=(3, 30),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red')
    
    ax.annotate('Cessna returns to trim\n(naturally stable)', 
                xy=(4, pitch_cessna[-100]), 
                xytext=(2.5, -15),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='blue'),
                color='blue')
    
    plt.tight_layout()
    fig.savefig('f16_pitch_instability.png', dpi=150, bbox_inches='tight')
    print("\nSaved: f16_pitch_instability.png")
    
    return fig


def demo_high_g_turn():
    """
    Demonstrate F-16 high-G turn capability.
    
    F-16 can pull much higher G than a Cessna due to:
    - Higher thrust-to-weight ratio
    - Structural limits
    - Aerodynamic limits
    """
    print("\n" + "="*60)
    print("DEMO 2: High-G Turn Comparison")
    print("="*60)
    print("Both aircraft attempt a 4G turn (65° bank, pull back).\n")
    
    dt = 0.01
    duration = 30.0
    n_steps = int(duration / dt)
    
    # Use stable F-16 for this demo
    cessna_config = get_aircraft("cessna172")
    f16_config = get_aircraft("f16_stable")  # Stable version for controllable sim
    
    cessna_params = config_to_params(cessna_config)
    f16_params = config_to_params(f16_config)
    
    sim_cessna = FlightSimulator(n_instances=1, params=cessna_params, dt=dt, use_gpu=True)
    sim_f16 = FlightSimulator(n_instances=1, params=f16_params, dt=dt, use_gpu=True)
    
    # Initial states
    initial_cessna = np.zeros((1, STATE_DIM), dtype=np.float32)
    initial_cessna[0, StateIndex.Z] = -2000.0
    initial_cessna[0, StateIndex.U] = 60.0  # m/s
    
    initial_f16 = np.zeros((1, STATE_DIM), dtype=np.float32)
    initial_f16[0, StateIndex.Z] = -5000.0
    initial_f16[0, StateIndex.U] = 250.0  # m/s
    
    sim_cessna.reset(initial_state=initial_cessna)
    sim_f16.reset(initial_state=initial_f16)
    
    # Storage
    time_hist = np.zeros(n_steps)
    pos_cessna = np.zeros((n_steps, 3))
    pos_f16 = np.zeros((n_steps, 3))
    g_cessna = np.zeros(n_steps)
    g_f16 = np.zeros(n_steps)
    
    for step in range(n_steps):
        t = step * dt
        time_hist[step] = t
        
        states_c = sim_cessna.get_states()
        states_f = sim_f16.get_states()
        
        pos_cessna[step] = [states_c[0, StateIndex.X], states_c[0, StateIndex.Y], -states_c[0, StateIndex.Z]]
        pos_f16[step] = [states_f[0, StateIndex.X], states_f[0, StateIndex.Y], -states_f[0, StateIndex.Z]]
        
        # Estimate G from pitch rate and bank angle (simplified)
        phi_c = states_c[0, StateIndex.PHI]
        phi_f = states_f[0, StateIndex.PHI]
        g_cessna[step] = 1.0 / max(np.cos(phi_c), 0.1)
        g_f16[step] = 1.0 / max(np.cos(phi_f), 0.1)
        
        # Controls: Roll into turn, then pull
        ctrl_cessna = np.zeros((1, CONTROL_DIM), dtype=np.float32)
        ctrl_f16 = np.zeros((1, CONTROL_DIM), dtype=np.float32)
        
        if t < 3:
            # Roll into bank
            ctrl_cessna[0] = [0.7, 0.5, 0.0, 0.0]  # throttle, aileron, elevator, rudder
            ctrl_f16[0] = [0.5, 0.5, 0.0, 0.0]
        elif t < 25:
            # Hold turn
            ctrl_cessna[0] = [0.9, 0.0, 0.3, 0.0]   # Full power, pull back
            ctrl_f16[0] = [0.6, 0.0, 0.2, 0.0]      # Less pull needed
        else:
            # Roll out
            ctrl_cessna[0] = [0.5, -0.3, 0.0, 0.0]
            ctrl_f16[0] = [0.4, -0.3, 0.0, 0.0]
        
        sim_cessna.set_controls(ctrl_cessna)
        sim_f16.set_controls(ctrl_f16)
        
        sim_cessna.step()
        sim_f16.step()
    
    # Plot
    fig = plt.figure(figsize=(14, 6))
    
    # 3D trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(pos_cessna[:, 0], pos_cessna[:, 1], pos_cessna[:, 2], 'b-', linewidth=2, label='Cessna 172')
    ax1.plot(pos_f16[:, 0], pos_f16[:, 1], pos_f16[:, 2], 'r-', linewidth=2, label='F-16')
    ax1.set_xlabel('North (m)')
    ax1.set_ylabel('East (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('Turn Trajectory Comparison')
    ax1.legend()
    
    # Top-down view
    ax2 = fig.add_subplot(122)
    ax2.plot(pos_cessna[:, 0], pos_cessna[:, 1], 'b-', linewidth=2, label='Cessna 172')
    ax2.plot(pos_f16[:, 0], pos_f16[:, 1], 'r-', linewidth=2, label='F-16')
    ax2.plot(pos_cessna[0, 0], pos_cessna[0, 1], 'bo', markersize=10)
    ax2.plot(pos_f16[0, 0], pos_f16[0, 1], 'ro', markersize=10)
    ax2.set_xlabel('North (m)')
    ax2.set_ylabel('East (m)')
    ax2.set_title('Turn Radius Comparison (Top View)')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('f16_turn_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: f16_turn_comparison.png")
    
    return fig


def demo_roll_rate():
    """
    Compare roll rate capability of F-16 vs Cessna.
    
    F-16 can achieve much higher roll rates due to:
    - Lower roll inertia relative to control power
    - Higher airspeed = more control authority
    """
    print("\n" + "="*60)
    print("DEMO 3: Roll Rate Comparison")
    print("="*60)
    print("Full aileron input, comparing roll response.\n")
    
    dt = 0.005  # Smaller timestep for fast dynamics
    duration = 4.0
    n_steps = int(duration / dt)
    
    cessna_config = get_aircraft("cessna172")
    f16_config = get_aircraft("f16_stable")
    
    cessna_params = config_to_params(cessna_config)
    f16_params = config_to_params(f16_config)
    
    sim_cessna = FlightSimulator(n_instances=1, params=cessna_params, dt=dt, use_gpu=True)
    sim_f16 = FlightSimulator(n_instances=1, params=f16_params, dt=dt, use_gpu=True)
    
    # Initial states (level flight)
    initial_cessna = np.zeros((1, STATE_DIM), dtype=np.float32)
    initial_cessna[0, StateIndex.Z] = -1500.0
    initial_cessna[0, StateIndex.U] = 55.0
    
    initial_f16 = np.zeros((1, STATE_DIM), dtype=np.float32)
    initial_f16[0, StateIndex.Z] = -5000.0
    initial_f16[0, StateIndex.U] = 200.0
    
    sim_cessna.reset(initial_state=initial_cessna)
    sim_f16.reset(initial_state=initial_f16)
    
    # Storage
    time_hist = np.zeros(n_steps)
    roll_cessna = np.zeros(n_steps)
    roll_f16 = np.zeros(n_steps)
    roll_rate_cessna = np.zeros(n_steps)
    roll_rate_f16 = np.zeros(n_steps)
    
    for step in range(n_steps):
        t = step * dt
        time_hist[step] = t
        
        states_c = sim_cessna.get_states()
        states_f = sim_f16.get_states()
        
        roll_cessna[step] = rad2deg(states_c[0, StateIndex.PHI])
        roll_f16[step] = rad2deg(states_f[0, StateIndex.PHI])
        roll_rate_cessna[step] = rad2deg(states_c[0, StateIndex.P])
        roll_rate_f16[step] = rad2deg(states_f[0, StateIndex.P])
        
        # Full aileron input at t=0.5s
        ctrl = np.zeros((1, CONTROL_DIM), dtype=np.float32)
        if t > 0.5 and t < 2.5:
            ctrl[0, ControlIndex.AILERON] = 1.0  # Full right aileron
        ctrl[0, ControlIndex.THROTTLE] = 0.4
        
        sim_cessna.set_controls(ctrl)
        sim_f16.set_controls(ctrl)
        
        sim_cessna.step()
        sim_f16.step()
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(time_hist, roll_cessna, 'b-', linewidth=2, label='Cessna 172')
    ax1.plot(time_hist, roll_f16, 'r-', linewidth=2, label='F-16')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.axvspan(0.5, 2.5, alpha=0.1, color='green', label='Aileron input')
    ax1.set_ylabel('Roll Angle (deg)', fontsize=12)
    ax1.set_title('Roll Rate Comparison: F-16 vs Cessna 172\n(Full aileron input)', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(time_hist, roll_rate_cessna, 'b-', linewidth=2, label='Cessna 172')
    ax2.plot(time_hist, roll_rate_f16, 'r-', linewidth=2, label='F-16')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.axvspan(0.5, 2.5, alpha=0.1, color='green')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Roll Rate (deg/s)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    max_rate_cessna = np.max(np.abs(roll_rate_cessna))
    max_rate_f16 = np.max(np.abs(roll_rate_f16))
    
    ax2.annotate(f'F-16 max: {max_rate_f16:.0f}°/s', 
                xy=(1.5, max_rate_f16), 
                xytext=(2.5, max_rate_f16 + 50),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red')
    
    ax2.annotate(f'Cessna max: {max_rate_cessna:.0f}°/s', 
                xy=(1.5, max_rate_cessna), 
                xytext=(2.5, max_rate_cessna + 20),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='blue'),
                color='blue')
    
    plt.tight_layout()
    fig.savefig('f16_roll_rate.png', dpi=150, bbox_inches='tight')
    print(f"\nMax roll rates:")
    print(f"  Cessna 172: {max_rate_cessna:.1f} deg/s")
    print(f"  F-16:       {max_rate_f16:.1f} deg/s")
    print("\nSaved: f16_roll_rate.png")
    
    return fig


def demo_f16_combat_maneuver():
    """
    Simulate an F-16 performing a combat maneuver sequence:
    1. High-G break turn
    2. Vertical climb
    3. Immelmann turn
    """
    print("\n" + "="*60)
    print("DEMO 4: F-16 Combat Maneuver Sequence")
    print("="*60)
    print("Break turn → Vertical climb → Immelmann\n")
    
    dt = 0.01
    duration = 40.0
    n_steps = int(duration / dt)
    
    f16_config = get_aircraft("f16_stable")
    f16_params = config_to_params(f16_config)
    
    sim = FlightSimulator(n_instances=1, params=f16_params, dt=dt, use_gpu=True)
    
    # Initial state: fast level flight
    initial = np.zeros((1, STATE_DIM), dtype=np.float32)
    initial[0, StateIndex.Z] = -3000.0  # 3000m altitude
    initial[0, StateIndex.U] = 250.0    # 250 m/s
    
    sim.reset(initial_state=initial)
    
    # Storage
    time_hist = np.zeros(n_steps)
    position = np.zeros((n_steps, 3))
    attitude = np.zeros((n_steps, 3))
    airspeed = np.zeros(n_steps)
    
    for step in range(n_steps):
        t = step * dt
        time_hist[step] = t
        
        states = sim.get_states()
        position[step] = [states[0, StateIndex.X], states[0, StateIndex.Y], -states[0, StateIndex.Z]]
        attitude[step] = [
            rad2deg(states[0, StateIndex.PHI]),
            rad2deg(states[0, StateIndex.THETA]),
            rad2deg(states[0, StateIndex.PSI])
        ]
        airspeed[step] = np.sqrt(
            states[0, StateIndex.U]**2 + 
            states[0, StateIndex.V]**2 + 
            states[0, StateIndex.W]**2
        )
        
        # Control sequence
        ctrl = np.zeros((1, CONTROL_DIM), dtype=np.float32)
        
        if t < 2:
            # Setup: level flight
            ctrl[0] = [0.5, 0.0, 0.0, 0.0]
        elif t < 7:
            # Break turn: hard left, pull back
            ctrl[0] = [0.8, -0.7, 0.4, -0.1]
        elif t < 12:
            # Roll wings level, pull into climb
            ctrl[0] = [0.95, 0.5, 0.5, 0.0]
        elif t < 18:
            # Vertical climb
            ctrl[0] = [1.0, 0.0, 0.3, 0.0]
        elif t < 25:
            # Over the top (Immelmann)
            ctrl[0] = [0.3, 0.0, 0.4, 0.0]
        elif t < 30:
            # Roll upright
            ctrl[0] = [0.4, 0.8, 0.0, 0.0]
        else:
            # Level off
            ctrl[0] = [0.5, 0.0, -0.1, 0.0]
        
        sim.set_controls(ctrl)
        sim.step()
    
    # Plot
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 3D trajectory
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Color by time
    colors = plt.cm.plasma(np.linspace(0, 1, n_steps))
    for i in range(0, n_steps-1, 20):
        ax1.plot(position[i:i+21, 0], position[i:i+21, 1], position[i:i+21, 2], 
                color=colors[i], linewidth=2)
    
    ax1.set_xlabel('North (m)')
    ax1.set_ylabel('East (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('F-16 Combat Maneuver\n(color = time progression)')
    
    # Side view (X-Z)
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(position[::10, 0], position[::10, 2], c=time_hist[::10], 
                         cmap='plasma', s=20)
    ax2.set_xlabel('North (m)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Side View')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Time (s)')
    
    # Attitude history
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_hist, attitude[:, 0], 'r-', linewidth=1.5, label='Roll')
    ax3.plot(time_hist, attitude[:, 1], 'g-', linewidth=1.5, label='Pitch')
    ax3.plot(time_hist, attitude[:, 2], 'b-', linewidth=1.5, label='Yaw')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (deg)')
    ax3.set_title('Attitude History')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add maneuver labels
    ax3.axvspan(2, 7, alpha=0.1, color='red')
    ax3.axvspan(7, 18, alpha=0.1, color='green')
    ax3.axvspan(18, 30, alpha=0.1, color='blue')
    ax3.text(4.5, ax3.get_ylim()[1]*0.9, 'Break\nTurn', ha='center', fontsize=9)
    ax3.text(12.5, ax3.get_ylim()[1]*0.9, 'Vertical\nClimb', ha='center', fontsize=9)
    ax3.text(24, ax3.get_ylim()[1]*0.9, 'Immelmann', ha='center', fontsize=9)
    
    # Airspeed and altitude
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_alt = ax4.twinx()
    
    l1, = ax4.plot(time_hist, airspeed, 'b-', linewidth=2, label='Airspeed')
    l2, = ax4_alt.plot(time_hist, position[:, 2], 'r-', linewidth=2, label='Altitude')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Airspeed (m/s)', color='blue')
    ax4_alt.set_ylabel('Altitude (m)', color='red')
    ax4.set_title('Energy State')
    ax4.tick_params(axis='y', colors='blue')
    ax4_alt.tick_params(axis='y', colors='red')
    ax4.grid(True, alpha=0.3)
    
    lines = [l1, l2]
    ax4.legend(lines, [l.get_label() for l in lines], loc='upper right')
    
    plt.suptitle('F-16 Fighting Falcon - Combat Maneuver Sequence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig('f16_combat_maneuver.png', dpi=150, bbox_inches='tight')
    print("\nSaved: f16_combat_maneuver.png")
    
    return fig


def main():
    print("\n" + "="*70)
    print("   F-16 FIGHTING FALCON - FLIGHT DYNAMICS DEMONSTRATION")
    print("   GPU-Accelerated 6-DOF Simulation")
    print("="*70)
    
    # Show aircraft info
    print("\n--- Aircraft Database ---")
    f16_config = get_aircraft("f16")
    print_aircraft_info(f16_config)
    
    cessna_config = get_aircraft("cessna172")
    print_aircraft_info(cessna_config)
    
    # Run demonstrations
    fig1 = demo_pitch_instability()
    fig2 = demo_roll_rate()
    fig3 = demo_high_g_turn()
    fig4 = demo_f16_combat_maneuver()
    
    print("\n" + "="*70)
    print("   DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  • f16_pitch_instability.png - Stability comparison")
    print("  • f16_roll_rate.png - Roll rate comparison")
    print("  • f16_turn_comparison.png - High-G turn capability")
    print("  • f16_combat_maneuver.png - Combat maneuver sequence")
    print("\nThese visualizations demonstrate:")
    print("  1. Relaxed static stability (real F-16 is unstable)")
    print("  2. High roll rate capability")
    print("  3. High-G maneuvering")
    print("  4. Complex maneuver sequences")
    print("="*70 + "\n")
    
    plt.show()


if __name__ == "__main__":
    main()
