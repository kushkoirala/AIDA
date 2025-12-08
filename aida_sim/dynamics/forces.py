import numpy as np

# Aerodynamic force and moment placeholders. Replace with calibrated models.
def aero_forces_moments(alpha: float, beta: float, rates: np.ndarray, dyn_pressure: float, wing_area: float,
                        cl_alpha: float = 3.82, cm_alpha: float = -0.38, cd0: float = 0.02,
                        k_induced: float = 0.08):
    cl = cl_alpha * alpha
    cm = cm_alpha * alpha
    cd = cd0 + k_induced * cl * cl
    lift = dyn_pressure * wing_area * cl
    drag = dyn_pressure * wing_area * cd
    # Side force neglected in placeholder; moments only pitch for now.
    moments = np.array([0.0, dyn_pressure * wing_area * cm, 0.0], dtype=np.float32)
    forces_body = np.array([
        -drag,     # X forward (negative drag)
        0.0,       # Y right
        -lift      # Z down (negative lift)
    ], dtype=np.float32)
    return forces_body, moments


def thrust_force(throttle: float, airspeed: float, max_thrust: float = 2.62 * 4.44822):
    # Simple thrust curve: flat at low speed, taper with airspeed.
    thrust_available = max_thrust * throttle
    speed_factor = 1.0 / (1.0 + 0.04 * airspeed)
    return np.array([thrust_available * speed_factor, 0.0, 0.0], dtype=np.float32)
