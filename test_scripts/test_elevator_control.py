"""
Test Elevator Control Sign Convention

This script tests whether positive elevator deflection produces nose-up or nose-down
pitching moment, to debug the backwards control issue discovered during takeoff testing.

Based on the summary:
- Current Cmδe = -1.13 (from Dynamics PDF Table 2)
- Positive elevator deflection (+10°) causes NOSE DOWN instead of NOSE UP
- This suggests a sign error somewhere in the control chain
"""

import numpy as np

# Aircraft parameters from Udaan (Dynamics PDF Table 2)
Cm0 = -0.0962
Cm_alpha = -0.38  # /rad - negative = stable
Cm_q = -12.0      # /rad - damping
Cm_de = -1.13     # /rad - control power

print("="*70)
print("  ELEVATOR CONTROL SIGN CONVENTION TEST")
print("="*70)

print("\nCurrent Udaan Parameters:")
print(f"  Cm0     = {Cm0}")
print(f"  Cm_alpha = {Cm_alpha} /rad (negative = stable)")
print(f"  Cm_q     = {Cm_q} /rad (damping)")
print(f"  Cm_de    = {Cm_de} /rad")

print("\n" + "-"*70)
print("TEST 1: Level Flight (alpha=0, q=0, de=0)")
print("-"*70)
alpha = 0.0
q = 0.0
elevator = 0.0
Cm = Cm0 + Cm_alpha * alpha + Cm_q * q + Cm_de * elevator
print(f"  alpha = {alpha} deg, q = {q} deg/s, de = {elevator} deg")
print(f"  Cm = {Cm:.6f}")
print(f"  Result: {'Nose UP moment' if Cm > 0 else 'Nose DOWN moment' if Cm < 0 else 'No moment'}")

print("\n" + "-"*70)
print("TEST 2: Positive Elevator (+10 deg) - Should cause NOSE UP")
print("-"*70)
alpha = 0.0  # deg
q = 0.0      # deg/s
elevator = np.deg2rad(10.0)  # +10 deg in radians

Cm = Cm0 + Cm_alpha * alpha + Cm_q * q + Cm_de * elevator
print(f"  alpha = {np.rad2deg(alpha):.1f} deg, q = {np.rad2deg(q):.1f} deg/s, de = +10 deg")
print(f"  Cm_de * de = {Cm_de} * {elevator:.4f} = {Cm_de * elevator:.6f}")
print(f"  Total Cm = {Cm:.6f}")

if Cm > 0:
    print(f"  OK Result: POSITIVE Cm -> Nose UP moment (CORRECT)")
else:
    print(f"  BAD Result: NEGATIVE Cm -> Nose DOWN moment (WRONG!)")

print("\n" + "-"*70)
print("TEST 3: Understanding the Sign Convention")
print("-"*70)
print("\nStandard Aircraft Convention:")
print("  - Positive elevator deflection (de > 0) = Trailing edge DOWN")
print("  - Trailing edge down -> Tail pitches UP -> Nose pitches UP")
print("  - Therefore: de > 0 should produce Cm > 0 (nose-up moment)")
print("\nCurrent Model Behavior:")
print("  - Cm_de = -1.13 (NEGATIVE)")
print("  - de = +10 deg (positive)")
print("  - Contribution: -1.13 x (+0.1745) = -0.197 (NEGATIVE)")
print("  - Negative Cm -> Nose DOWN moment")
print("\n  BAD PROBLEM: Positive elevator causes nose DOWN!")

print("\n" + "-"*70)
print("SOLUTION OPTIONS:")
print("-"*70)
print("\nOption 1: Flip the sign of Cm_de")
print(f"  Change: Cm_de = -1.13 -> Cm_de = +1.13")
print(f"  Then: de = +10 deg gives Cm = +0.197 (nose UP) OK")

print("\nOption 2: Flip the elevator input sign")
print(f"  Keep: Cm_de = -1.13")
print(f"  Apply: elevator_actual = -elevator_commanded")
print(f"  Then: de_cmd = +10 deg -> de_actual = -10 deg -> Cm = +0.197 (nose UP) OK")

print("\nOption 3: Check if the sign convention definition is backwards")
print(f"  Verify: Does the model define +de as trailing edge UP instead of DOWN?")
print(f"  If so, the coefficient sign is correct for that convention")

print("\n" + "-"*70)
print("RECOMMENDED ACTION:")
print("-"*70)
print("\n1. Check the Dynamics PDF to see how elevator deflection is defined")
print("2. Check the flight_dynamics.py source to see how elevator is applied")
print("3. Most likely fix: NEGATE the elevator input before calculating moments")
print("   (This preserves the Cmδe value from the reference document)")

print("\n" + "="*70)
print("  Let's check what the reference document says about elevator sign...")
print("="*70 + "\n")
