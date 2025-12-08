import numpy as np
from .state import VehicleState


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ], dtype=np.float32)


def _integrate_quat(q: np.ndarray, body_rates: np.ndarray, dt: float) -> np.ndarray:
    # Small-angle quaternion integration; adequate for placeholder dynamics.
    p, q_rate, r = body_rates
    omega = np.array([0.0, p, q_rate, r], dtype=np.float32)
    # Quaternion multiplication q_dot = 0.5 * q ⊗ omega
    w, x, y, z = q
    ow, ox, oy, oz = omega
    q_dot = 0.5 * np.array([
        -x * ox - y * oy - z * oz,
        w * ox + y * oz - z * oy,
        w * oy - x * oz + z * ox,
        w * oz + x * oy - y * ox,
    ], dtype=np.float32)
    q_next = q + q_dot * dt
    return _quat_normalize(q_next)


# Semi-implicit Euler integrator with quaternion update and diagonal inertia.
def integrate_step(state: VehicleState, forces_body: np.ndarray, moments_body: np.ndarray, mass: float,
                   inertia_diag: np.ndarray, dt: float) -> VehicleState:
    R_bw = _quat_to_rotmat(state.orientation)  # body->world rotation
    accel_world = (R_bw @ (forces_body / mass)).astype(np.float32)

    next_state = VehicleState(
        position=state.position.copy(),
        velocity=state.velocity.copy(),
        orientation=state.orientation.copy(),
        body_rates=state.body_rates.copy(),
        surfaces=state.surfaces.copy(),
        soc=state.soc,
        voltage=state.voltage,
        load_factor=state.load_factor,
    )

    # Linear motion
    next_state.velocity += accel_world * dt
    next_state.position += next_state.velocity * dt

    # Angular motion (diagonal inertia approximation)
    ang_accel = moments_body / inertia_diag
    next_state.body_rates += ang_accel * dt
    next_state.orientation = _integrate_quat(state.orientation, next_state.body_rates, dt)

    # Load factor from total acceleration magnitude
    next_state.load_factor = float(np.linalg.norm(accel_world) / 9.81)
    return next_state
