"""
Udaan Physics Tuning Tool

Interactive tool to:
1. Compare wind tunnel data with current model
2. Adjust aerodynamic coefficients
3. Test stability and trim conditions
4. Validate controller performance

Author: Kushal Koirala
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

# Add AIDA paths
sys.path.insert(0, '/home/AIDA/gpu-flight-dynamics/python')

from flight_dynamics import FlightSimulator, StateIndex, ControlIndex
from aircraft_database import get_aircraft, AircraftConfig


def plot_trim_analysis(aircraft_name='udaan'):
    """
    Analyze trim conditions at different airspeeds.
    Shows if the aircraft can maintain level flight across its speed range.
    """
    print("\n" + "="*60)
    print("  TRIM CONDITION ANALYSIS")
    print("="*60)

    # Get aircraft
    config = get_aircraft(aircraft_name)

    # Speed range to test
    V_stall = 11.2  # m/s (from PropShox)
    V_max = 26.6    # m/s
    speeds = np.linspace(V_stall * 1.1, V_max * 0.9, 20)

    trim_alpha = []
    trim_elevator = []
    trim_thrust = []

    for V in speeds:
        # Calculate trim alpha (angle of attack for level flight)
        # L = W → CL * 0.5 * rho * V^2 * S = m * g
        rho = 1.225  # kg/m³ (sea level)
        W = config.mass.mass * 9.81
        CL_required = (2 * W) / (rho * V**2 * config.geom.S)

        # Alpha from CL (using linear lift curve)
        if CL_required > config.longi.CLmax:
            print(f"  WARNING: V={V:.1f} m/s requires CL={CL_required:.3f} > CLmax")
            continue

        alpha = (CL_required - config.longi.CL0) / config.longi.CLa
        trim_alpha.append(np.rad2deg(alpha))

        # Calculate elevator for trim (Cm = 0)
        # Cm = Cm0 + Cma*alpha + Cmde*de = 0
        Cm_alpha = config.longi.Cm0 + config.longi.Cma * alpha
        delta_e = -Cm_alpha / config.longi.Cmde
        trim_elevator.append(np.rad2deg(delta_e))

        # Calculate thrust required
        CD = config.longi.CD0 + config.longi.K * CL_required**2
        D = 0.5 * rho * V**2 * config.geom.S * CD
        trim_thrust.append(D)

    # Plot results
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)

    # Trim alpha
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(speeds[:len(trim_alpha)], trim_alpha, 'b-', linewidth=2, label='Trim α')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axhline(y=10, color='r', linestyle='--', alpha=0.3, label='~Stall')
    ax1.set_xlabel('Airspeed (m/s)', fontsize=11)
    ax1.set_ylabel('Trim Angle of Attack (deg)', fontsize=11)
    ax1.set_title('Trim Alpha vs Speed', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Trim elevator
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(speeds[:len(trim_elevator)], trim_elevator, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axhline(y=15, color='r', linestyle='--', alpha=0.3, label='Max deflection')
    ax2.axhline(y=-15, color='r', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Airspeed (m/s)', fontsize=11)
    ax2.set_ylabel('Trim Elevator (deg)', fontsize=11)
    ax2.set_title('Trim Elevator vs Speed', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Thrust required
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(speeds[:len(trim_thrust)], trim_thrust, 'r-', linewidth=2, label='Thrust Required')
    ax3.axhline(y=28, color='g', linestyle='--', alpha=0.5, label='Max Thrust (28N)')
    ax3.set_xlabel('Airspeed (m/s)', fontsize=11)
    ax3.set_ylabel('Thrust (N)', fontsize=11)
    ax3.set_title('Thrust Required vs Speed', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Power required
    ax4 = fig.add_subplot(gs[1, 1])
    power_required = np.array(trim_thrust) * speeds[:len(trim_thrust)]
    ax4.plot(speeds[:len(power_required)], power_required, 'm-', linewidth=2, label='Power Required')
    ax4.axhline(y=630, color='g', linestyle='--', alpha=0.5, label='Max Power (630W)')
    ax4.set_xlabel('Airspeed (m/s)', fontsize=11)
    ax4.set_ylabel('Power (W)', fontsize=11)
    ax4.set_title('Power Required vs Speed', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('udaan_trim_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: udaan_trim_analysis.png")

    return speeds[:len(trim_alpha)], trim_alpha, trim_elevator, trim_thrust


def test_stability_margins(aircraft_name='udaan'):
    """
    Test static and dynamic stability margins.
    """
    print("\n" + "="*60)
    print("  STABILITY MARGIN ANALYSIS")
    print("="*60)

    config = get_aircraft(aircraft_name)

    print(f"\n{config.name} Stability Derivatives:")
    print(f"  Longitudinal:")
    print(f"    Cmα = {config.longi.Cma:.4f} /rad  {'✓ STABLE' if config.longi.Cma < 0 else '✗ UNSTABLE'}")
    print(f"    Cmq = {config.longi.Cmq:.4f} /rad  (damping)")
    print(f"    CLα = {config.longi.CLa:.4f} /rad")

    print(f"\n  Lateral-Directional:")
    print(f"    Cnβ = {config.latdi.Cnb:.4f} /rad  {'✓ STABLE' if config.latdi.Cnb > 0 else '✗ UNSTABLE'}")
    print(f"    Clβ = {config.latdi.Clb:.4f} /rad  {'✓ STABLE' if config.latdi.Clb < 0 else '✗ UNSTABLE'}")
    print(f"    Cnr = {config.latdi.Cnr:.4f} /rad  (yaw damping)")
    print(f"    Clp = {config.latdi.Clp:.4f} /rad  (roll damping)")

    # Check for sufficient damping
    print(f"\n  Damping Assessment:")
    if abs(config.longi.Cmq) > 5.0:
        print(f"    ✓ Pitch damping GOOD (|Cmq| = {abs(config.longi.Cmq):.1f} > 5)")
    else:
        print(f"    ⚠ Pitch damping LOW (|Cmq| = {abs(config.longi.Cmq):.1f} < 5)")

    if abs(config.latdi.Clp) > 1.0:
        print(f"    ✓ Roll damping GOOD (|Clp| = {abs(config.latdi.Clp):.1f} > 1)")
    else:
        print(f"    ⚠ Roll damping LOW (|Clp| = {abs(config.latdi.Clp):.1f} < 1)")

    if abs(config.latdi.Cnr) > 0.15:
        print(f"    ✓ Yaw damping GOOD (|Cnr| = {abs(config.latdi.Cnr):.1f} > 0.15)")
    else:
        print(f"    ⚠ Yaw damping LOW (|Cnr| = {abs(config.latdi.Cnr):.1f} < 0.15)")


def compare_with_wind_tunnel_data():
    """
    Template for comparing model with wind tunnel data.
    User fills in their wind tunnel measurements.
    """
    print("\n" + "="*60)
    print("  WIND TUNNEL DATA COMPARISON")
    print("="*60)

    config = get_aircraft('udaan')

    print("\nCurrent Model Values:")
    print(f"  CL0   = {config.longi.CL0:.4f}")
    print(f"  CLα   = {config.longi.CLa:.4f} /rad ({np.rad2deg(config.longi.CLa):.4f} /deg)")
    print(f"  CLmax = {config.longi.CLmax:.4f}")
    print(f"  CD0   = {config.longi.CD0:.4f}")
    print(f"  Cm0   = {config.longi.Cm0:.4f}")
    print(f"  Cmα   = {config.longi.Cma:.4f} /rad")

    print("\n" + "-"*60)
    print("Enter your wind tunnel data:")
    print("-"*60)
    print("\nLift Coefficients:")
    print("  CL0 (zero-alpha lift):  ______")
    print("  CLα (lift slope, /deg): ______")
    print("  CLmax (stall):          ______")
    print("  α_stall (deg):          ______")

    print("\nDrag Coefficients:")
    print("  CD0 (parasitic):        ______")
    print("  e (Oswald efficiency):  ______ (current: 0.75)")

    print("\nMoment Coefficients:")
    print("  Cm0 (zero-alpha):       ______")
    print("  Cmα (/deg):             ______")
    print("  Neutral point (%MAC):   ______")

    print("\n" + "="*60)
    print("\nOnce you provide the data, I'll:")
    print("  1. Update the aircraft_database.py with correct values")
    print("  2. Regenerate BC dataset with accurate physics")
    print("  3. Show comparison plots")
    print("="*60)


def test_controller_step_response():
    """
    Test controller response to step inputs.
    Shows if controller gains are properly tuned.
    """
    print("\n" + "="*60)
    print("  CONTROLLER STEP RESPONSE TEST")
    print("="*60)

    from generate_bc_dataset_gpu import ClassicalFlightController

    # This would test altitude hold, heading hold, etc.
    print("\n(Implement step response test here)")
    print("Will test:")
    print("  - Altitude hold step response")
    print("  - Heading hold step response")
    print("  - Airspeed hold response")
    print("  - Overshoot, settling time, steady-state error")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  UDAAN PHYSICS TUNING TOOL")
    print("  AIDA Autonomous Flight Research")
    print("="*60)

    # Run all analyses
    test_stability_margins('udaan')
    speeds, alphas, elevators, thrusts = plot_trim_analysis('udaan')
    compare_with_wind_tunnel_data()

    print("\n" + "="*60)
    print("  Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - udaan_trim_analysis.png")
    print("\nNext steps:")
    print("  1. Review the trim analysis plots")
    print("  2. Enter your wind tunnel data above")
    print("  3. Update aircraft_database.py with correct coefficients")
    print("  4. Re-run BC dataset generation")
    print("="*60 + "\n")
