"""
Udaan Electric UAV Flight Dynamics Demonstration

Demonstrates the flight characteristics of the Udaan research platform:
1. Takeoff and climb performance
2. Cruise flight stability
3. Electric propulsion dynamics
4. Comparison with Cessna 172 for scaling reference

Author: Kushal Koirala
Date: December 2024

Data Source:
    PropShox Final Design Report (2024)
    AIDA Flight Test Simulation Data
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


def ms_to_fps(ms):
    return ms * 3.28084


def m_to_ft(m):
    return m * 3.28084


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


def demo_cruise_flight():
    """
    Demonstrate Udaan in stable cruise flight.

    Show the aircraft maintaining level flight at cruise speed (73.5 ft/s)
    with typical control inputs.
    """
    print("\n" + "="*60)
    print("DEMO 1: Cruise Flight Stability")
    print("="*60)
    print("Udaan maintaining cruise at 73.5 ft/s (22.4 m/s)")
    print("Altitude: 100 ft AGL\n")

    dt = 0.02
    duration = 15.0
    n_steps = int(duration / dt)

    # Get Udaan config
    udaan_config = get_aircraft("udaan")
    print(f"Aircraft: {udaan_config.name}")
    print(f"Mass: {udaan_config.mass.mass:.2f} kg")
    print(f"Wing Area: {udaan_config.geom.S:.3f} m²")
    print(f"Cma: {udaan_config.longi.Cma:.3f} (negative = stable)")

    # Create simulator
    udaan_params = config_to_params(udaan_config)
    sim = FlightSimulator(n_instances=1, params=udaan_params, dt=dt, use_gpu=True)

    # Initial state: cruise flight at 100 ft AGL
    initial_state = np.zeros((1, STATE_DIM), dtype=np.float32)
    initial_state[0, StateIndex.Z] = -30.5  # 100 ft in NED (negative = up)
    initial_state[0, StateIndex.U] = 22.4   # 73.5 ft/s cruise speed
    initial_state[0, StateIndex.THETA] = np.deg2rad(2.0)  # Slight nose up for cruise

    sim.reset(initial_state=initial_state)

    # Cruise controls (trimmed for level flight)
    controls = np.zeros((1, CONTROL_DIM), dtype=np.float32)
    controls[0, ControlIndex.THROTTLE] = 0.45  # ~45% throttle for cruise
    controls[0, ControlIndex.ELEVATOR] = np.deg2rad(-1.5)  # Slight down elevator for trim

    # Record history
    time_hist = np.zeros(n_steps)
    altitude_hist = np.zeros(n_steps)
    speed_hist = np.zeros(n_steps)
    pitch_hist = np.zeros(n_steps)
    throttle_hist = np.zeros(n_steps)

    for step in range(n_steps):
        time_hist[step] = step * dt

        states = sim.get_states()
        altitude_hist[step] = -states[0, StateIndex.Z]  # Convert NED to AGL
        speed_hist[step] = ms_to_fps(states[0, StateIndex.U])
        pitch_hist[step] = rad2deg(states[0, StateIndex.THETA])
        throttle_hist[step] = controls[0, ControlIndex.THROTTLE] * 100

        sim.set_controls(controls)
        sim.step()

    # Plot
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Altitude
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_hist, m_to_ft(altitude_hist), 'b-', linewidth=2)
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Target: 100 ft')
    ax1.set_ylabel('Altitude (ft AGL)', fontsize=11)
    ax1.set_title('Udaan Cruise Flight Performance', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Airspeed
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_hist, speed_hist, 'g-', linewidth=2)
    ax2.axhline(y=73.5, color='gray', linestyle='--', alpha=0.5, label='V_cruise: 73.5 ft/s')
    ax2.set_ylabel('Airspeed (ft/s)', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Pitch angle
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_hist, pitch_hist, 'r-', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax3.set_ylabel('Pitch Angle (deg)', fontsize=11)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Throttle
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(time_hist, throttle_hist, 'm-', linewidth=2)
    ax4.set_ylabel('Throttle (%)', fontsize=11)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('udaan_cruise_flight.png', dpi=150, bbox_inches='tight')
    print("\nSaved: udaan_cruise_flight.png")

    return fig


def demo_scale_comparison():
    """
    Compare Udaan (small UAV) with Cessna 172 (GA trainer)
    to show relative sizes and performance.
    """
    print("\n" + "="*60)
    print("DEMO 2: Scale Comparison - Udaan vs Cessna 172")
    print("="*60)

    udaan = get_aircraft("udaan")
    cessna = get_aircraft("cessna172")

    # Print comparison table
    print(f"\n{'Parameter':<25} {'Udaan':<15} {'Cessna 172':<15} {'Ratio':<10}")
    print("-" * 70)

    mass_ratio = cessna.mass.mass / udaan.mass.mass
    print(f"{'Mass (kg)':<25} {udaan.mass.mass:<15.2f} {cessna.mass.mass:<15.2f} {mass_ratio:<10.1f}x")

    wing_ratio = cessna.geom.S / udaan.geom.S
    print(f"{'Wing Area (m²)':<25} {udaan.geom.S:<15.3f} {cessna.geom.S:<15.2f} {wing_ratio:<10.1f}x")

    span_ratio = cessna.geom.b / udaan.geom.b
    print(f"{'Wingspan (m)':<25} {udaan.geom.b:<15.2f} {cessna.geom.b:<15.2f} {span_ratio:<10.1f}x")

    thrust_ratio = cessna.prop.thrust_max / udaan.prop.thrust_max
    print(f"{'Max Thrust (N)':<25} {udaan.prop.thrust_max:<15.1f} {cessna.prop.thrust_max:<15.1f} {thrust_ratio:<10.1f}x")

    print(f"{'Propulsion':<25} {'Electric':<15} {'Piston':<15}")
    print(f"{'CL_alpha (1/rad)':<25} {udaan.longi.CLa:<15.2f} {cessna.longi.CLa:<15.2f}")
    print(f"{'Cma (stability)':<25} {udaan.longi.Cma:<15.3f} {cessna.longi.Cma:<15.3f}")

    print("\nKey Observations:")
    print(f"  • Cessna is ~{mass_ratio:.0f}x heavier than Udaan")
    print(f"  • Both aircraft are stable (negative Cma)")
    print(f"  • Udaan: Electric propulsion, instant response")
    print(f"  • Cessna: Piston engine, slower spool-up")
    print(f"  • Udaan designed for autonomous operation at low altitude")


def demo_parallel_simulation():
    """
    Demonstrate GPU-accelerated parallel simulation of 1000 Udaan instances.

    This shows the power of GPU acceleration for:
    - Monte Carlo rollouts
    - Behavior cloning data generation
    - PPO warmup training
    """
    print("\n" + "="*60)
    print("DEMO 3: GPU-Accelerated Parallel Simulation")
    print("="*60)
    print("Simulating 1000 Udaan instances in parallel...")
    print("Use case: Behavior cloning dataset generation for PPO warmup\n")

    dt = 0.02
    duration = 5.0
    n_steps = int(duration / dt)
    n_instances = 1000

    # Get Udaan config
    udaan_config = get_aircraft("udaan")
    udaan_params = config_to_params(udaan_config)

    # Create simulator with 1000 instances
    sim = FlightSimulator(n_instances=n_instances, params=udaan_params, dt=dt, use_gpu=True)

    # Random initial conditions (cruise flight with variations)
    initial_states = np.zeros((n_instances, STATE_DIM), dtype=np.float32)

    # Randomize altitude (20-50 m)
    initial_states[:, StateIndex.Z] = -np.random.uniform(20, 50, n_instances)

    # Randomize cruise speed (±20% variation)
    cruise_speed = 22.4  # m/s
    initial_states[:, StateIndex.U] = np.random.uniform(cruise_speed * 0.8, cruise_speed * 1.2, n_instances)

    # Small pitch variations
    initial_states[:, StateIndex.THETA] = np.random.uniform(-5, 5, n_instances) * np.pi / 180

    sim.reset(initial_state=initial_states)

    # Constant controls for all instances
    controls = np.zeros((n_instances, CONTROL_DIM), dtype=np.float32)
    controls[:, ControlIndex.THROTTLE] = 0.45

    # Timing benchmark
    import time
    start_time = time.time()

    for step in range(n_steps):
        sim.set_controls(controls)
        sim.step()

    elapsed_time = time.time() - start_time
    total_steps = n_instances * n_steps
    steps_per_second = total_steps / elapsed_time

    print(f"Simulation Complete!")
    print(f"  Total instances: {n_instances}")
    print(f"  Steps per instance: {n_steps}")
    print(f"  Total simulation steps: {total_steps:,}")
    print(f"  Elapsed time: {elapsed_time:.2f} s")
    print(f"  Throughput: {steps_per_second:,.0f} steps/sec")
    print(f"  Speedup vs real-time: {(n_instances * duration) / elapsed_time:.1f}x")

    # Get final states
    final_states = sim.get_states()

    # Plot distribution of final altitudes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Altitude distribution
    altitudes = -final_states[:, StateIndex.Z]
    ax1.hist(m_to_ft(altitudes), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Final Altitude (ft AGL)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'Altitude Distribution ({n_instances} instances)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Airspeed distribution
    airspeeds = final_states[:, StateIndex.U]
    ax2.hist(ms_to_fps(airspeeds), bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Final Airspeed (ft/s)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Airspeed Distribution ({n_instances} instances)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('udaan_parallel_simulation.png', dpi=150, bbox_inches='tight')
    print("\nSaved: udaan_parallel_simulation.png")

    print("\nThis data can be used for:")
    print("  • Behavior cloning (BC) warmstart for PPO")
    print("  • Flight envelope exploration")
    print("  • Monte Carlo uncertainty quantification")
    print("  • Controller validation across diverse conditions")

    return fig


def main():
    """Run all Udaan demonstrations"""
    print("\n" + "="*60)
    print("  UDAAN FLIGHT DYNAMICS DEMONSTRATION")
    print("  AIDA Autonomous Flight Research Platform")
    print("="*60)

    # Print aircraft info
    udaan_config = get_aircraft("udaan")
    print_aircraft_info(udaan_config)

    # Run demos
    try:
        demo_cruise_flight()
        demo_scale_comparison()
        demo_parallel_simulation()

        print("\n" + "="*60)
        print("  All demonstrations complete!")
        print("="*60)
        print("\nGenerated files:")
        print("  • udaan_cruise_flight.png")
        print("  • udaan_parallel_simulation.png")
        print("\nNext steps:")
        print("  • Use parallel simulation for BC dataset generation")
        print("  • Warm-start PPO training in AIDA environment")
        print("  • Validate with 3D viewer visualization")
        print("="*60 + "\n")

        plt.show()

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
