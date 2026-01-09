import numpy as np
from dataclasses import dataclass


@dataclass
class AeroParams:
    """Compact set of aerodynamic derivatives."""
    wing_area: float = 0.4803087      # m^2 (PropShox Table 1)
    wing_span: float = 1.3716         # m (4.5 ft span)
    mean_chord: float = 0.35018       # m (wing_area/span)
    alpha_limit: float = np.deg2rad(25.0)
    CL_max: float = 1.006

    CL0: float = 0.278
    CL_alpha: float = 0.058 * (180.0 / np.pi)
    CL_q: float = 6.5
    CL_de: float = 0.9

    CD0: float = 0.019
    CD_alpha2: float = 0.30
    CD_q: float = 0.01
    CD_de: float = 0.02
    CDi: float = 0.04

    # Flap/spoiler effects (normalized 0-1 input)
    # Flaps increase both lift and drag
    dCL_flap: float = 0.4             # CL increment at full flaps
    dCD_flap: float = 0.06            # CD increment at full flaps
    dCm_flap: float = -0.10           # Pitching moment change (nose down)
    # Spoilers decrease lift and increase drag significantly
    dCL_spoiler: float = -0.3         # CL decrement at full spoilers
    dCD_spoiler: float = 0.08         # CD increment at full spoilers

    CY_beta: float = -0.9
    CY_da: float = 0.06
    CY_dr: float = 0.17

    Cl_beta: float = -0.12
    Cl_p: float = -1.25
    Cl_r: float = 0.18
    Cl_da: float = 0.085
    Cl_dr: float = 0.01

    Cm0: float = -0.096
    Cm_alpha: float = -0.38
    Cm_q: float = -9.0
    Cm_de: float = -1.13

    Cn_beta: float = 0.25
    Cn_p: float = -0.05
    Cn_r: float = -0.18
    Cn_da: float = 0.01
    Cn_dr: float = -0.18


def _calc_alpha_beta(vel_body: np.ndarray):
    u, v, w = vel_body
    V = float(np.linalg.norm(vel_body))
    if V < 1e-3:
        return 0.0, 0.0, V
    alpha = np.arctan2(w, max(1e-3, u))
    beta = np.arcsin(np.clip(v / V, -1.0, 1.0))
    return alpha, beta, V


def aero_forces_moments(
    vel_body: np.ndarray,
    body_rates: np.ndarray,
    surfaces: np.ndarray,
    air_density: float,
    params: AeroParams,
    flap: float = 0.0,
    spoiler: float = 0.0,
):
    """
    Compute aerodynamic forces/moments using linearized stability derivatives.
    Returns body-frame forces/moments (N, N-m).
    
    Args:
        vel_body: Body-frame velocity [u, v, w]
        body_rates: Angular rates [p, q, r]
        surfaces: Control surfaces [elevator, aileron, rudder]
        air_density: Air density (kg/m^3)
        params: Aerodynamic parameters
        flap: Flap deflection 0-1 (0=retracted, 1=full)
        spoiler: Spoiler deflection 0-1 (0=retracted, 1=full)
    """
    alpha, beta, airspeed = _calc_alpha_beta(vel_body)
    if airspeed < 1e-3:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    alpha = np.clip(alpha, -params.alpha_limit, params.alpha_limit)
    elevator, aileron, rudder = surfaces
    p, q, r = body_rates
    q_bar = 0.5 * air_density * airspeed * airspeed
    span = params.wing_span
    chord = params.mean_chord
    inv_2V = 1.0 / (2.0 * airspeed)
    p_hat = p * span * inv_2V
    q_hat = q * chord * inv_2V
    r_hat = r * span * inv_2V

    # Clamp flap/spoiler to valid range
    flap = np.clip(flap, 0.0, 1.0)
    spoiler = np.clip(spoiler, 0.0, 1.0)

    # Coefficients with flap/spoiler effects
    CL_linear = (
        params.CL0 
        + params.CL_alpha * alpha 
        + params.CL_q * q_hat 
        + params.CL_de * elevator
        + params.dCL_flap * flap           # Flaps increase lift
        + params.dCL_spoiler * spoiler      # Spoilers decrease lift
    )
    CL = np.clip(CL_linear, -params.CL_max, params.CL_max)
    
    CD = (
        params.CD0
        + params.CD_alpha2 * alpha * alpha
        + params.CD_q * abs(q_hat)
        + params.CD_de * abs(elevator)
        + params.CDi * CL * CL
        + params.dCD_flap * flap           # Flaps increase drag
        + params.dCD_spoiler * spoiler      # Spoilers increase drag significantly
    )
    CY = params.CY_beta * beta + params.CY_da * aileron + params.CY_dr * rudder

    Cl = (
        params.Cl_beta * beta
        + params.Cl_p * p_hat
        + params.Cl_r * r_hat
        + params.Cl_da * aileron
        + params.Cl_dr * rudder
    )
    Cm = (
        params.Cm0
        + params.Cm_alpha * alpha
        + params.Cm_q * q_hat
        + params.Cm_de * elevator
        + params.dCm_flap * flap            # Flaps cause nose-down moment
    )
    Cn = (
        params.Cn_beta * beta
        + params.Cn_p * p_hat
        + params.Cn_r * r_hat
        + params.Cn_da * aileron
        + params.Cn_dr * rudder
    )

    lift = q_bar * params.wing_area * CL
    drag = q_bar * params.wing_area * CD
    side = q_bar * params.wing_area * CY

    # Convert lift/drag to body axes (X forward, Y right, Z up)
    ca, sa = np.cos(alpha), np.sin(alpha)
    Fx = -drag * ca + lift * sa
    Fz = -lift * ca - drag * sa
    forces = np.array([Fx, side, Fz], dtype=np.float32)

    moments = np.array(
        [
            q_bar * params.wing_area * span * Cl,
            q_bar * params.wing_area * chord * Cm,
            q_bar * params.wing_area * span * Cn,
        ],
        dtype=np.float32,
    )
    return forces, moments


ELECTRICAL_MAX_POWER_W = 630.0  # 315 W per motor from PropShox Sec. 3.2
PROP_EFFICIENCY = 0.50          # Average prop efficiency reported
PROP_DIAMETER_M = 0.2286   # 9 in prop
PROP_AREA = np.pi * (PROP_DIAMETER_M * 0.5) ** 2
NUM_PROPS = 2


def _solve_induced_velocity(power_air: float, V: float, rho: float) -> float:
    """Solve for induced velocity using actuator disk model."""
    if power_air <= 1e-6:
        return 0.0
    # Initial guess: hover solution
    vi = (power_air / (2.0 * rho * PROP_AREA)) ** (1.0 / 3.0)
    for _ in range(8):
        total = V + vi
        if total <= 1e-3:
            total = 1e-3
        f = 2.0 * rho * PROP_AREA * vi * total * total - power_air
        df = 2.0 * rho * PROP_AREA * (total * total + 2.0 * vi * total)
        if abs(df) < 1e-6:
            break
        vi = max(0.0, vi - f / df)
    return max(0.0, vi)


def propulsion_model(throttle: float, airspeed: float, rho: float = 1.225):
    throttle = np.clip(throttle, 0.0, 1.0)
    electrical_power = throttle * ELECTRICAL_MAX_POWER_W
    air_power = electrical_power * PROP_EFFICIENCY
    vi = _solve_induced_velocity(air_power, airspeed, rho)
    total_velocity = airspeed + vi
    thrust = 2.0 * rho * PROP_AREA * vi * total_velocity * NUM_PROPS
    return np.array([thrust, 0.0, 0.0], dtype=np.float32), electrical_power


def thrust_force(throttle: float, airspeed: float, rho: float = 1.225):
    thrust_vec, _ = propulsion_model(throttle, airspeed, rho)
    return thrust_vec
