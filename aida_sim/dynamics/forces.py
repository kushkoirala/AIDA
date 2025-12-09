import numpy as np

# Stall angle in radians (~15 degrees)
STALL_ALPHA = np.deg2rad(15.0)

# Aerodynamic force and moment placeholders. Replace with calibrated models.
def aero_forces_moments(alpha: float, beta: float, rates: np.ndarray, dyn_pressure: float, wing_area: float,
                        cl_alpha: float = 3.0, cm_alpha: float = -0.3, cd0: float = 0.04,
                        k_induced: float = 0.08):
    # Clamp alpha to valid aerodynamic range and model stall
    alpha_clamped = np.clip(alpha, -STALL_ALPHA, STALL_ALPHA)
    
    # Beyond stall, lift drops sharply - use flat plate model
    if abs(alpha) > STALL_ALPHA:
        # Post-stall: lift ~ sin(2*alpha), drag ~ sin^2(alpha)
        cl = np.sign(alpha) * 0.8 * np.sin(2 * abs(alpha_clamped))
        cd = cd0 + 1.2 * np.sin(abs(alpha_clamped))**2
    else:
        # Normal flight: linear lift model
        cl = cl_alpha * alpha_clamped
        cd = cd0 + k_induced * cl * cl
    
    cm = cm_alpha * alpha_clamped
    lift = dyn_pressure * wing_area * cl
    drag = dyn_pressure * wing_area * cd
    
    # Side force neglected in placeholder; moments only pitch for now.
    moments = np.array([0.0, dyn_pressure * wing_area * cm, 0.0], dtype=np.float32)
    forces_body = np.array([
        -drag,     # X forward (negative drag)
        0.0,       # Y right
        lift       # Z up (positive lift)
    ], dtype=np.float32)
    return forces_body, moments


def thrust_force(throttle: float, airspeed: float, max_thrust: float = 40.0):
    # Moderate thrust for takeoff capability (T/W ~ 1.2)
    # 40N max thrust vs 33.4N weight = 1.2 T/W ratio
    thrust_available = max_thrust * throttle
    speed_factor = 1.0 / (1.0 + 0.03 * airspeed)  # More thrust taper with speed
    return np.array([thrust_available * speed_factor, 0.0, 0.0], dtype=np.float32)
