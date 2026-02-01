"""
Entropy-Modulated Control Barrier Functions for Flight Envelope Protection

Implements the AIDA assurance layer (Chapter 5 of thesis):

    h_AIDA(x) = h_nom(x) + eta * (1 - H(P(w)))

Where H(P(w)) is the entropy of the BIRL reward posterior:
    - High entropy (ambiguous intent) -> h_AIDA ~ h_nom (conservative)
    - Low entropy (clear intent)      -> h_AIDA = h_nom + eta (relaxed)

The CBF-QP finds the minimum-deviation safe control:
    min  ||u - u_des||^2
    s.t. dh_i/dt + alpha(h_i) >= 0   for all active barriers
         u_min <= u <= u_max          (actuator limits)

Author: Kushal Koirala
Date: 2026
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# Unit conversions
M_TO_FT = 3.28084
FT_TO_M = 0.3048
KTS_TO_MPS = 0.514444
MPS_TO_KTS = 1.94384

# State indices (matching flight_simulator.py)
X, Y, Z = 0, 1, 2
U, V, W = 3, 4, 5
PHI, THETA, PSI = 6, 7, 8
P, Q, R = 9, 10, 11

# Control indices
THROTTLE, AILERON, ELEVATOR, RUDDER = 0, 1, 2, 3
FLAP, SPOILER, BRAKE = 4, 5, 6

# Actuator limits
U_MIN = np.array([0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0])
U_MAX = np.array([1.0,  1.0,  1.0,  1.0, 1.0, 1.0, 1.0])


@dataclass
class BarrierConstraint:
    """A single barrier function evaluation."""
    name: str
    h: float              # Barrier value (>= 0 is safe)
    dh_dx: np.ndarray     # Gradient w.r.t. state (12-element)
    alpha_h: float        # class-K function value alpha(h)
    margin: float         # Activation margin (only enforce QP when h < margin)
    active: bool = False  # Whether this constraint is active in the QP


@dataclass
class CBFResult:
    """Result of CBF safety filtering."""
    u_safe: np.ndarray
    intervened: bool
    control_deviation: float
    barrier_values: Dict[str, float]
    active_barriers: List[str]
    entropy: float
    solver_success: bool
    solver_iterations: int = 0


@dataclass
class CBFMetrics:
    """Accumulated metrics for a flight."""
    total_interventions: int = 0
    intervention_times: List[float] = field(default_factory=list)
    barrier_violations_prevented: Dict[str, int] = field(default_factory=lambda: {})
    control_deviations: List[float] = field(default_factory=list)
    max_control_deviation: float = 0.0
    entropy_timeline: List[Tuple[float, float]] = field(default_factory=list)

    def record(self, sim_time: float, result: CBFResult):
        """Record a CBF result."""
        if result.intervened:
            self.total_interventions += 1
            self.intervention_times.append(sim_time)
            for name in result.active_barriers:
                self.barrier_violations_prevented[name] = \
                    self.barrier_violations_prevented.get(name, 0) + 1
        self.control_deviations.append(result.control_deviation)
        self.max_control_deviation = max(self.max_control_deviation, result.control_deviation)
        self.entropy_timeline.append((sim_time, result.entropy))

    def summary(self) -> dict:
        devs = self.control_deviations
        return {
            "total_interventions": self.total_interventions,
            "barrier_violations_prevented": dict(self.barrier_violations_prevented),
            "avg_control_deviation": float(np.mean(devs)) if devs else 0.0,
            "max_control_deviation": self.max_control_deviation,
            "n_frames": len(devs),
        }


class FlightEnvelopeCBF:
    """
    Entropy-Modulated Control Barrier Functions for flight envelope protection.

    Six barrier functions define the safe set:
        1. Altitude floor (min AGL)
        2. Altitude ceiling (service ceiling)
        3. Stall speed (minimum airspeed)
        4. Overspeed (VNE)
        5. Bank angle limit
        6. Pitch angle limits

    Each barrier is modulated by the BIRL entropy:
        h_AIDA(x) = h_nom(x) + eta * (1 - H)
    """

    def __init__(
        self,
        min_altitude_ft: float = 200.0,
        max_altitude_ft: float = 14000.0,
        min_airspeed_kts: float = 52.0,
        max_airspeed_kts: float = 155.0,
        max_bank_deg: float = 45.0,
        max_pitch_deg: float = 20.0,
        min_pitch_deg: float = -15.0,
        eta: float = 0.3,
        alpha_cbf: float = 1.0,
        barrier_activation_margin: float = 0.5,
    ):
        # Envelope limits (stored in metric)
        self.min_alt_m = min_altitude_ft * FT_TO_M
        self.max_alt_m = max_altitude_ft * FT_TO_M
        self.min_airspeed = min_airspeed_kts * KTS_TO_MPS
        self.max_airspeed = max_airspeed_kts * KTS_TO_MPS
        self.max_bank = np.radians(max_bank_deg)
        self.max_pitch = np.radians(max_pitch_deg)
        self.min_pitch = np.radians(min_pitch_deg)

        # CBF parameters
        self.eta = eta
        self.alpha_cbf = alpha_cbf
        self.margin = barrier_activation_margin

        # Approximate control effectiveness for Cessna 172
        # Maps control inputs to state rate changes
        # These are rough estimates — CBF only needs approximate sign/magnitude
        self._aileron_to_pdot = 3.0    # rad/s^2 per unit aileron
        self._elevator_to_qdot = -2.5  # rad/s^2 per unit elevator (negative = nose up)
        self._rudder_to_rdot = 0.8     # rad/s^2 per unit rudder
        self._throttle_to_udot = 3.0   # m/s^2 per unit throttle (approximate)

        # Roll damping and pitch damping
        self._p_damping = -0.5
        self._q_damping = -0.3

        # Metrics
        self.metrics = CBFMetrics()

    def compute_barriers(
        self, state: np.ndarray, entropy: float
    ) -> List[BarrierConstraint]:
        """
        Compute all barrier function values and gradients.

        Args:
            state: 12-element state in meters/radians
            entropy: BIRL posterior entropy in [0, 1]

        Returns:
            List of BarrierConstraint for each envelope limit.
        """
        barriers = []
        eta_mod = self.eta * (1.0 - entropy)  # Relaxation term

        alt_m = -state[Z]  # altitude = -z in NED
        airspeed = np.sqrt(state[U]**2 + state[V]**2 + state[W]**2)
        airspeed = max(airspeed, 0.1)  # prevent division by zero
        phi = state[PHI]
        theta = state[THETA]

        dh_dx = np.zeros(12)

        # --- 1. Altitude floor ---
        h_alt_floor = alt_m - self.min_alt_m + eta_mod * self.min_alt_m * 0.1
        dh_dx_floor = np.zeros(12)
        dh_dx_floor[Z] = -1.0  # d(alt)/d(z) = -1
        barriers.append(BarrierConstraint(
            name="altitude_floor",
            h=float(h_alt_floor),
            dh_dx=dh_dx_floor,
            alpha_h=self.alpha_cbf * h_alt_floor,
            margin=self.min_alt_m * 1.5,  # activate when within 1.5x min alt
        ))

        # --- 2. Altitude ceiling ---
        h_alt_ceil = self.max_alt_m - alt_m + eta_mod * self.max_alt_m * 0.01
        dh_dx_ceil = np.zeros(12)
        dh_dx_ceil[Z] = 1.0  # d(-alt)/d(z) = 1
        barriers.append(BarrierConstraint(
            name="altitude_ceiling",
            h=float(h_alt_ceil),
            dh_dx=dh_dx_ceil,
            alpha_h=self.alpha_cbf * h_alt_ceil,
            margin=self.max_alt_m * 0.1,
        ))

        # --- 3. Stall speed ---
        speed_margin = eta_mod * 5.0  # relax by up to 5 m/s when clear intent
        h_stall = airspeed - self.min_airspeed + speed_margin
        dh_dx_stall = np.zeros(12)
        dh_dx_stall[U] = state[U] / airspeed
        dh_dx_stall[V] = state[V] / airspeed
        dh_dx_stall[W] = state[W] / airspeed
        barriers.append(BarrierConstraint(
            name="stall_speed",
            h=float(h_stall),
            dh_dx=dh_dx_stall,
            alpha_h=self.alpha_cbf * h_stall,
            margin=10.0,  # activate within 10 m/s of stall
        ))

        # --- 4. Overspeed ---
        h_overspeed = self.max_airspeed - airspeed + speed_margin
        dh_dx_overspeed = np.zeros(12)
        dh_dx_overspeed[U] = -state[U] / airspeed
        dh_dx_overspeed[V] = -state[V] / airspeed
        dh_dx_overspeed[W] = -state[W] / airspeed
        barriers.append(BarrierConstraint(
            name="overspeed",
            h=float(h_overspeed),
            dh_dx=dh_dx_overspeed,
            alpha_h=self.alpha_cbf * h_overspeed,
            margin=10.0,
        ))

        # --- 5. Bank angle ---
        bank_margin = eta_mod * np.radians(5.0)  # relax by up to 5 deg
        h_bank = self.max_bank - abs(phi) + bank_margin
        dh_dx_bank = np.zeros(12)
        dh_dx_bank[PHI] = -np.sign(phi) if abs(phi) > 0.01 else 0.0
        barriers.append(BarrierConstraint(
            name="bank_angle",
            h=float(h_bank),
            dh_dx=dh_dx_bank,
            alpha_h=self.alpha_cbf * h_bank,
            margin=np.radians(15.0),  # activate within 15 deg of limit
        ))

        # --- 6. Pitch limits (quadratic barrier) ---
        pitch_range = self.max_pitch - self.min_pitch
        pitch_margin = eta_mod * np.radians(3.0)
        # Quadratic: h = (max_pitch + margin - theta) * (theta - min_pitch + margin) / range^2
        h_pitch = ((self.max_pitch + pitch_margin - theta) *
                    (theta - self.min_pitch + pitch_margin)) / (pitch_range**2)
        dh_dx_pitch = np.zeros(12)
        # d/dtheta of (a - theta)(theta - b) = -2*theta + a + b
        a = self.max_pitch + pitch_margin
        b = self.min_pitch - pitch_margin
        dh_dx_pitch[THETA] = (-2.0 * theta + a + b) / (pitch_range**2)
        barriers.append(BarrierConstraint(
            name="pitch_limits",
            h=float(h_pitch),
            dh_dx=dh_dx_pitch,
            alpha_h=self.alpha_cbf * h_pitch,
            margin=0.3,  # normalized, activate when approaching limits
        ))

        # Mark active barriers
        for b in barriers:
            b.active = b.h < b.margin

        return barriers

    def _estimate_state_derivative(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        Estimate dx/dt from current state and proposed control input.

        Uses kinematic equations + simplified control effectiveness model.
        The CBF only needs approximate sign and magnitude of dh/dt.
        """
        x_dot = np.zeros(12)

        phi, theta, psi = state[PHI], state[THETA], state[PSI]
        u, v, w = state[U], state[V], state[W]
        p, q, r = state[P], state[Q], state[R]

        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        # Prevent division by zero at theta = +/- 90 deg
        ctheta = max(abs(ctheta), 0.01) * np.sign(ctheta) if abs(ctheta) < 0.01 else ctheta

        # Position derivatives (body velocity -> NED)
        x_dot[X] = ctheta*cpsi*u + (sphi*stheta*cpsi - cphi*spsi)*v + (cphi*stheta*cpsi + sphi*spsi)*w
        x_dot[Y] = ctheta*spsi*u + (sphi*stheta*spsi + cphi*cpsi)*v + (cphi*stheta*spsi - sphi*cpsi)*w
        x_dot[Z] = -stheta*u + sphi*ctheta*v + cphi*ctheta*w

        # Body velocity derivatives (approximate: gravity + thrust + damping)
        g = 9.81
        throttle = control[THROTTLE]
        x_dot[U] = self._throttle_to_udot * throttle - g * stheta - 0.05 * u  # thrust - gravity component - drag
        x_dot[V] = -0.1 * v  # sideslip damping
        x_dot[W] = g * cphi * ctheta - g + control[ELEVATOR] * 1.0  # approximate

        # Euler angle derivatives from body rates
        x_dot[PHI] = p + (q*sphi + r*cphi) * stheta / ctheta
        x_dot[THETA] = q*cphi - r*sphi
        x_dot[PSI] = (q*sphi + r*cphi) / ctheta

        # Body rate derivatives from control inputs
        aileron = control[AILERON]
        elevator = control[ELEVATOR]
        rudder = control[RUDDER]

        x_dot[P] = self._aileron_to_pdot * aileron + self._p_damping * p
        x_dot[Q] = self._elevator_to_qdot * elevator + self._q_damping * q
        x_dot[R] = self._rudder_to_rdot * rudder - 0.2 * r

        return x_dot

    def _apply_fallback(
        self, u_des: np.ndarray, state: np.ndarray, barriers: List[BarrierConstraint]
    ) -> np.ndarray:
        """
        Simple clamp-based fallback when QP solver fails.
        Directly overrides controls to steer away from violated barriers.
        """
        u_safe = u_des.copy()

        for b in barriers:
            if b.h < 0:  # Barrier violated
                if b.name == "altitude_floor":
                    # Pitch up, add throttle
                    u_safe[ELEVATOR] = min(u_safe[ELEVATOR], -0.3)
                    u_safe[THROTTLE] = max(u_safe[THROTTLE], 0.8)
                    u_safe[SPOILER] = 0.0
                elif b.name == "altitude_ceiling":
                    # Pitch down, reduce throttle
                    u_safe[ELEVATOR] = max(u_safe[ELEVATOR], 0.2)
                    u_safe[THROTTLE] = min(u_safe[THROTTLE], 0.3)
                elif b.name == "stall_speed":
                    # Add throttle, reduce pitch up, retract spoilers
                    u_safe[THROTTLE] = max(u_safe[THROTTLE], 0.9)
                    u_safe[ELEVATOR] = max(u_safe[ELEVATOR], -0.1)
                    u_safe[SPOILER] = 0.0
                elif b.name == "overspeed":
                    # Reduce throttle, deploy spoilers
                    u_safe[THROTTLE] = min(u_safe[THROTTLE], 0.2)
                    u_safe[SPOILER] = max(u_safe[SPOILER], 0.5)
                elif b.name == "bank_angle":
                    # Reduce bank
                    if state[PHI] > 0:
                        u_safe[AILERON] = min(u_safe[AILERON], -0.3)
                    else:
                        u_safe[AILERON] = max(u_safe[AILERON], 0.3)
                elif b.name == "pitch_limits":
                    # Center elevator
                    u_safe[ELEVATOR] = u_safe[ELEVATOR] * 0.5

        # Enforce actuator limits
        u_safe = np.clip(u_safe, U_MIN, U_MAX)
        return u_safe

    def solve_safe_control(
        self,
        u_des: np.ndarray,
        state: np.ndarray,
        entropy: float,
        dt: float = 0.02,
        phase: str = "",
    ) -> CBFResult:
        """
        Solve the CBF-QP to find the safe control closest to desired.

        min  ||u - u_des||^2
        s.t. dh_i/dt + alpha(h_i) >= 0  for active barriers
             u_min <= u <= u_max

        Args:
            u_des: 7-element desired control from controller
            state: 12-element state in meters/radians
            entropy: BIRL posterior entropy in [0, 1]
            dt: Physics timestep

        Returns:
            CBFResult with safe control and diagnostics
        """
        barriers = self.compute_barriers(state, entropy)

        # Disable irrelevant barriers based on flight phase
        phase_upper = phase.upper()
        ground_phases = {"GROUND_ROLL", "ROTATION", "ROLLOUT", "LANDING", "LANDED"}
        takeoff_phases = {"GROUND_ROLL", "ROTATION", "INITIAL_CLIMB"}
        landing_phases = {"SHORT_FINAL", "FLARE"}
        if phase_upper in ground_phases:
            for b in barriers:
                if b.name in ("stall_speed", "overspeed"):
                    b.active = False
        if phase_upper in takeoff_phases or phase_upper in ground_phases or phase_upper in landing_phases:
            for b in barriers:
                if b.name == "altitude_floor":
                    b.active = False

        # Check if any barriers need QP enforcement
        active = [b for b in barriers if b.active]
        barrier_values = {b.name: round(b.h, 4) for b in barriers}

        if not active:
            # No barriers active — pass through unchanged
            return CBFResult(
                u_safe=u_des.copy(),
                intervened=False,
                control_deviation=0.0,
                barrier_values=barrier_values,
                active_barriers=[],
                entropy=entropy,
                solver_success=True,
            )

        # Build QP constraints
        constraints = []
        for b in active:
            def make_cbf_constraint(barrier=b):
                def constraint_fn(u):
                    x_dot = self._estimate_state_derivative(state, u)
                    dh_dt = np.dot(barrier.dh_dx, x_dot)
                    return dh_dt + barrier.alpha_h
                return constraint_fn

            constraints.append({
                'type': 'ineq',
                'fun': make_cbf_constraint(b),
            })

        # Objective: min ||u - u_des||^2
        def objective(u):
            diff = u - u_des
            return 0.5 * np.dot(diff, diff)

        def objective_jac(u):
            return u - u_des

        bounds = list(zip(U_MIN, U_MAX))

        try:
            result = minimize(
                objective,
                u_des.copy(),
                jac=objective_jac,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 20, 'ftol': 1e-6},
            )

            if result.success:
                u_safe = np.clip(result.x, U_MIN, U_MAX)
                deviation = np.linalg.norm(u_safe - u_des)
                intervened = deviation > 0.01

                return CBFResult(
                    u_safe=u_safe,
                    intervened=intervened,
                    control_deviation=float(deviation),
                    barrier_values=barrier_values,
                    active_barriers=[b.name for b in active],
                    entropy=entropy,
                    solver_success=True,
                    solver_iterations=result.nit if hasattr(result, 'nit') else 0,
                )
            else:
                # QP infeasible or didn't converge — use fallback
                raise RuntimeError("SLSQP did not converge")

        except Exception:
            # Fallback: direct clamp-based override
            u_safe = self._apply_fallback(u_des, state, barriers)
            deviation = np.linalg.norm(u_safe - u_des)

            return CBFResult(
                u_safe=u_safe,
                intervened=deviation > 0.01,
                control_deviation=float(deviation),
                barrier_values=barrier_values,
                active_barriers=[b.name for b in active],
                entropy=entropy,
                solver_success=False,
            )

    def reset_metrics(self):
        """Reset accumulated metrics (start of new flight)."""
        self.metrics = CBFMetrics()
