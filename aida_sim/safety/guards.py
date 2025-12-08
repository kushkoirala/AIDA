import numpy as np

MAX_G = 4.0
MAX_CURRENT_A = 40.0


def clamp_actions(throttle: float, surfaces: np.ndarray, surface_limits: np.ndarray):
    throttle_clamped = float(np.clip(throttle, 0.0, 1.0))
    surfaces_clamped = np.clip(surfaces, -surface_limits, surface_limits)
    return throttle_clamped, surfaces_clamped


def enforce_limits(load_factor: float, current_a: float):
    over_g = load_factor > MAX_G
    over_current = current_a > MAX_CURRENT_A
    return over_g, over_current
