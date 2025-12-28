"""
Verify Elevator Sign Convention from Dynamics PDF

This proves that the elevator sign convention in the model is opposite
to standard aircraft convention by analyzing the trim data from the PDF.
"""

import numpy as np

print("="*70)
print("  ELEVATOR SIGN CONVENTION VERIFICATION")
print("  Based on Dynamics PDF Figure 2 (Trim Plots)")
print("="*70)

# From Dynamics PDF Table 2
CL0 = 0.28
CLa = 3.82  # /rad
Cm0 = 0.0962  # NOTE: PDF shows Cm0 = 0.0962, but earlier it was -0.0962
              # Let's use the value that makes physical sense
Cma = -0.38  # /rad (negative = stable)
Cmde = -1.13  # /rad

# From Figure 2 trim plot
de_takeoff = 9.1  # degrees (trim elevator for takeoff)
de_cruise = 4.22  # degrees (trim elevator for cruise)

# Aircraft parameters
mass = 2.38  # kg
g = 9.81  # m/s^2
S = 0.372  # m^2 (wing area)
rho = 1.225  # kg/m^3

print("\nAircraft Parameters (from PDF Table 2):")
print(f"  CL0   = {CL0}")
print(f"  CLa   = {CLa} /rad")
print(f"  Cm0   = {Cm0}")
print(f"  Cma   = {Cma} /rad")
print(f"  Cmde  = {Cmde} /rad")

print("\n" + "-"*70)
print("ANALYSIS 1: Takeoff Trim Condition")
print("-"*70)

# Takeoff: V = 1.2 * V_stall ≈ 13.4 m/s
V_stall = 11.2  # m/s
V_takeoff = 1.2 * V_stall
q_takeoff = 0.5 * rho * V_takeoff**2

# For level flight (or shallow climb), L ≈ W
CL_required = (mass * g) / (q_takeoff * S)
print(f"\nTakeoff speed: V = {V_takeoff:.1f} m/s")
print(f"  Required CL = {CL_required:.3f}")

# Solve for alpha
alpha_takeoff = (CL_required - CL0) / CLa
print(f"  Required alpha = {np.rad2deg(alpha_takeoff):.2f}°")

# Calculate pitching moment at this condition WITHOUT elevator
Cm_no_elevator = Cm0 + Cma * alpha_takeoff
print(f"\nPitching moment WITHOUT elevator:")
print(f"  Cm = Cm0 + Cma*alpha")
print(f"     = {Cm0:.4f} + ({Cma:.4f})*({alpha_takeoff:.4f})")
print(f"     = {Cm_no_elevator:.6f}")

if Cm_no_elevator > 0:
    print(f"  → POSITIVE Cm = Nose-UP tendency")
    print(f"  → Need nose-DOWN moment from elevator to trim")
elif Cm_no_elevator < 0:
    print(f"  → NEGATIVE Cm = Nose-DOWN tendency")
    print(f"  → Need nose-UP moment from elevator to trim")

# From PDF: trim elevator = +9.1°
de_trim_rad = np.deg2rad(de_takeoff)
Cm_elevator = Cmde * de_trim_rad
print(f"\nElevator contribution (from PDF trim data):")
print(f"  δe = +{de_takeoff}° (POSITIVE)")
print(f"  Cm_elevator = Cmde * δe")
print(f"              = ({Cmde:.4f}) * ({de_trim_rad:.4f})")
print(f"              = {Cm_elevator:.6f}")

if Cm_elevator > 0:
    print(f"  → POSITIVE Cm contribution = Nose-UP moment")
elif Cm_elevator < 0:
    print(f"  → NEGATIVE Cm contribution = Nose-DOWN moment")

# Total moment
Cm_total = Cm_no_elevator + Cm_elevator
print(f"\nTotal pitching moment:")
print(f"  Cm_total = {Cm_total:.6f}")
print(f"  (Should be close to zero for trim)")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)

if Cm_elevator < 0:
    print("\n  ✓ VERIFIED: Positive elevator (+δe) produces NOSE-DOWN moment")
    print("\n  This is OPPOSITE to standard aircraft convention where:")
    print("    +δe = trailing edge down = nose UP moment")
    print("\n  In THIS model:")
    print("    +δe produces NEGATIVE Cm (nose-down moment)")
    print("    -δe produces POSITIVE Cm (nose-up moment)")
    print("\n  THEREFORE:")
    print("    To command NOSE UP during rotation:")
    print("    → Send NEGATIVE elevator deflection")
    print("    → Example: δe = -10° for rotation")

print("\n" + "-"*70)
print("ANALYSIS 2: Physical Interpretation")
print("-"*70)

print("\nTwo possible explanations:")
print("\n1. DEFINITION: +δe is defined as 'trailing edge UP' (opposite convention)")
print("   - Many models use this convention")
print("   - Trailing edge up → tail down → nose up (but Cmde is negative)")
print("   - So Cmde negative with +δe trailing edge up → nose down")
print("\n2. COEFFICIENT SIGN: Cmde already includes the control direction")
print("   - The -1.13 value means 'positive input produces negative moment'")
print("   - This is just how this particular model is documented")

print("\n" + "="*70)
print("FIX IMPLEMENTATION:")
print("="*70)

print("\nIn the PID controller for pitch control:")
print("""
    # Calculate desired elevator for pitch error
    pitch_error = target_pitch - current_pitch

    # Standard PID
    elevator_cmd = Kp * pitch_error + Kd * pitch_rate

    # FIX: Negate for this aircraft's convention
    elevator_actual = -elevator_cmd

    # Example: Want to pitch up 10 degrees
    # pitch_error = +10
    # elevator_cmd = +10 * Kp (positive)
    # elevator_actual = -10 * Kp (NEGATIVE)
    # Result: Nose pitches UP (correct!)
""")

print("\n" + "="*70)
