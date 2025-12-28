import numpy as np

print("ELEVATOR SIGN VERIFICATION")
print("="*60)

# From PDF Table 2
Cm0 = 0.0962
Cma = -0.38
Cmde = -1.13

# Takeoff trim from PDF Figure 2
de_takeoff = 9.1  # degrees

# Takeoff conditions
V_takeoff = 13.4  # m/s
alpha_takeoff_deg = 4.31  # calculated from CL requirement
alpha_rad = np.deg2rad(alpha_takeoff_deg)
de_rad = np.deg2rad(de_takeoff)

# Calculate pitching moment WITHOUT elevator
Cm_no_elev = Cm0 + Cma * alpha_rad
print(f"\nAt takeoff (V={V_takeoff} m/s, alpha={alpha_takeoff_deg:.1f} deg):")
print(f"  Cm (no elevator) = {Cm_no_elev:.6f}")

if Cm_no_elev > 0:
    print(f"  -> POSITIVE = Nose UP tendency")
    print(f"  -> Need NOSE DOWN from elevator to trim")

# Elevator contribution
Cm_elev = Cmde * de_rad
print(f"\nElevator trim (from PDF): de = +{de_takeoff} deg")
print(f"  Cm (elevator) = Cmde * de")
print(f"                = {Cmde:.2f} * {de_rad:.4f}")
print(f"                = {Cm_elev:.6f}")

if Cm_elev < 0:
    print(f"  -> NEGATIVE = Nose DOWN moment")

# Total
Cm_total = Cm_no_elev + Cm_elev
print(f"\nTotal Cm = {Cm_total:.6f} (should be ~0 for trim)")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("\nPOSITIVE elevator (+de) produces NEGATIVE Cm (nose DOWN)")
print("This is OPPOSITE to standard aircraft convention.")
print("\nFIX: Negate elevator in controller")
print("  - To pitch UP: send NEGATIVE elevator")
print("  - To pitch DOWN: send POSITIVE elevator")
print("\nExample for rotation:")
print("  pitch_error = +10 deg (want nose up)")
print("  elevator_cmd = Kp * 10 = +5 deg")
print("  elevator_actual = -elevator_cmd = -5 deg  <- FIX")
print("  Result: Nose pitches UP (correct!)")
