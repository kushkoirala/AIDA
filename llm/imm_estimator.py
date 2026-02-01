"""
AIDA Interacting Multiple Model (IMM) Estimator

Runs multiple intent models in parallel and blends their outputs using
Bayesian model probabilities. This addresses the single-model limitation
where maneuvering phases (takeoff, climb, approach) produce low confidence
because the aircraft deviates from a single tight prior.

Models:
    1. Tracking     - Aircraft following assigned heading/altitude (tight priors)
    2. Maneuvering  - Aircraft in a turn or altitude change (wide priors)
    3. Anomalous    - Erratic or unexpected behavior (very wide priors)

The IMM cycle each frame:
    1. Mixing:      Compute mixed initial conditions from all models
    2. Filtering:   Each model runs its own Kalman-style update
    3. Mode Update: Reweight model probabilities using likelihoods
    4. Combination:  Blend outputs for a single confidence estimate
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Mode definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModeModel:
    """A single mode hypothesis with its own state estimate and noise model."""
    name: str

    # Heading: Von Mises concentration (higher = tighter)
    heading_kappa: float

    # Altitude: Gaussian std (ft)
    altitude_std: float

    # Airspeed: Gaussian std (kts)
    airspeed_std: float

    # Process noise multiplier (how fast uncertainty grows between updates)
    process_noise_scale: float

    # Current state estimate (mean values)
    heading_mean: float = 0.0
    altitude_mean: float = 0.0
    airspeed_mean: float = 0.0

    # Current residuals (innovation) from last update
    heading_residual: float = 0.0
    altitude_residual: float = 0.0
    airspeed_residual: float = 0.0

    # Log-likelihood from last measurement update
    log_likelihood: float = 0.0


# Phase-specific mode parameters
# Each phase has 3 models: tracking, maneuvering, anomalous
# The key difference is how wide the priors are

def _make_modes(
    track_hdg_kappa: float, track_alt_std: float,
    maneuver_hdg_kappa: float, maneuver_alt_std: float,
    track_spd_std: float = 5.0, maneuver_spd_std: float = 15.0,
) -> List[ModeModel]:
    return [
        ModeModel(
            name="tracking",
            heading_kappa=track_hdg_kappa,
            altitude_std=track_alt_std,
            airspeed_std=track_spd_std,
            process_noise_scale=1.0,
        ),
        ModeModel(
            name="maneuvering",
            heading_kappa=maneuver_hdg_kappa,
            altitude_std=maneuver_alt_std,
            airspeed_std=maneuver_spd_std,
            process_noise_scale=3.0,
        ),
        ModeModel(
            name="anomalous",
            heading_kappa=0.5,       # Nearly uniform on circle
            altitude_std=3000.0,     # Very wide
            airspeed_std=30.0,
            process_noise_scale=10.0,
        ),
    ]


PHASE_MODES: Dict[str, List[ModeModel]] = {
    "ground":        _make_modes(50.0, 10.0, 5.0, 50.0, 3.0, 10.0),
    "takeoff":       _make_modes(10.0, 300.0, 1.5, 800.0, 8.0, 20.0),
    "initial_climb": _make_modes(15.0, 400.0, 2.0, 1000.0, 8.0, 18.0),
    "climb":         _make_modes(8.0, 500.0, 1.5, 1500.0, 8.0, 18.0),
    "cruise":        _make_modes(20.0, 200.0, 3.0, 800.0, 5.0, 15.0),
    "cruise_to_tp":  _make_modes(15.0, 300.0, 2.0, 1000.0, 5.0, 15.0),
    "turn_to_intercept": _make_modes(3.0, 400.0, 1.0, 800.0, 8.0, 18.0),
    "intercept_leg": _make_modes(20.0, 300.0, 3.0, 600.0, 5.0, 15.0),
    "descent":       _make_modes(10.0, 500.0, 2.0, 1200.0, 8.0, 18.0),
    "approach":      _make_modes(8.0, 400.0, 1.5, 1000.0, 8.0, 18.0),
    "final_approach": _make_modes(25.0, 200.0, 3.0, 500.0, 5.0, 12.0),
    "final":         _make_modes(30.0, 150.0, 5.0, 400.0, 5.0, 12.0),
    "short_final":   _make_modes(40.0, 80.0, 8.0, 200.0, 4.0, 10.0),
    "flare":         _make_modes(60.0, 30.0, 10.0, 100.0, 3.0, 8.0),
    "rollout":       _make_modes(60.0, 10.0, 10.0, 50.0, 5.0, 15.0),
    "landing":       _make_modes(40.0, 100.0, 5.0, 300.0, 5.0, 12.0),
}

# Default Markov transition matrix between modes
# Rows = from, Cols = to  [tracking, maneuvering, anomalous]
DEFAULT_TRANSITION = np.array([
    [0.90, 0.09, 0.01],   # tracking    -> mostly stays tracking
    [0.15, 0.80, 0.05],   # maneuvering -> tends to stay, can return to tracking
    [0.05, 0.15, 0.80],   # anomalous   -> sticky but can recover
], dtype=np.float64)

# Initial mode probabilities
DEFAULT_INIT_PROBS = np.array([0.6, 0.35, 0.05], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# IMM Estimator
# ─────────────────────────────────────────────────────────────────────────────

class IMMEstimator:
    """
    Interacting Multiple Model estimator for flight intent.

    Maintains 3 parallel mode hypotheses (tracking, maneuvering, anomalous)
    and blends their confidence outputs using Bayesian model probabilities
    updated each frame from measurement residuals.
    """

    def __init__(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        init_probs: Optional[np.ndarray] = None,
    ):
        self.transition = transition_matrix if transition_matrix is not None else DEFAULT_TRANSITION.copy()
        self.mode_probs = init_probs.copy() if init_probs is not None else DEFAULT_INIT_PROBS.copy()
        self.n_modes = len(self.mode_probs)

        # Current mode models (set when phase changes)
        self.modes: List[ModeModel] = []
        self.phase: str = ""

        # History for diagnostics
        self.prob_history: List[np.ndarray] = []
        self.confidence_history: List[float] = []

        # Frame counter
        self.frame = 0

    def set_phase(self, phase: str):
        """Update the flight phase, loading new mode noise parameters.

        Crucially, we preserve the current state estimates (heading_mean,
        altitude_mean, airspeed_mean) across phase transitions so that the
        IMM doesn't see a spike in residuals at the boundary.
        """
        phase_lower = phase.lower().replace(" ", "_")
        if phase_lower == self.phase:
            return

        self.phase = phase_lower
        templates = PHASE_MODES.get(phase_lower, PHASE_MODES.get("cruise", []))

        if not self.modes:
            # First call: create modes from scratch
            self.modes = []
            for t in templates:
                m = ModeModel(
                    name=t.name,
                    heading_kappa=t.heading_kappa,
                    altitude_std=t.altitude_std,
                    airspeed_std=t.airspeed_std,
                    process_noise_scale=t.process_noise_scale,
                )
                self.modes.append(m)
        else:
            # Phase change: update noise parameters but KEEP state estimates
            for mode, t in zip(self.modes, templates):
                mode.heading_kappa = t.heading_kappa
                mode.altitude_std = t.altitude_std
                mode.airspeed_std = t.airspeed_std
                mode.process_noise_scale = t.process_noise_scale

    def update(
        self,
        heading_deg: float,
        altitude_ft: float,
        airspeed_kts: float,
        commanded_heading: Optional[float] = None,
        commanded_altitude: Optional[float] = None,
        commanded_airspeed: Optional[float] = None,
        dt: float = 0.02,
    ) -> Dict:
        """
        Run one IMM cycle: mix -> filter -> mode update -> combine.

        Args:
            heading_deg:  Current aircraft heading (0-360)
            altitude_ft:  Current altitude in feet
            airspeed_kts: Current airspeed in knots
            commanded_*:  Commanded values (if available). If None, we use
                          the current state as the "expected" value (tracking
                          mode assumes steady state).
            dt:           Time step in seconds

        Returns:
            Dict with blended confidence, mode probabilities, dominant mode, etc.
        """
        if not self.modes:
            self.set_phase("cruise")

        self.frame += 1

        # ── Step 1: Mixing ───────────────────────────────────────────────
        # Compute mixing probabilities: P(mode_j at t-1 | mode_i at t)
        # c_j = sum_i( pi_ij * mu_i(t-1) )
        c_bar = self.transition.T @ self.mode_probs  # predicted mode probs
        c_bar = np.maximum(c_bar, 1e-12)

        # Mixing weights: mu_{i|j} = pi_ij * mu_i / c_j
        mixing_weights = np.zeros((self.n_modes, self.n_modes))
        for j in range(self.n_modes):
            for i in range(self.n_modes):
                mixing_weights[i, j] = self.transition[i, j] * self.mode_probs[i] / c_bar[j]

        # ── Step 2: Filtering (measurement update per model) ─────────────
        # Each model computes its own innovation and log-likelihood
        for idx, mode in enumerate(self.modes):
            # Reference values: commanded if available, else current state
            ref_heading = commanded_heading if commanded_heading is not None else mode.heading_mean
            ref_altitude = commanded_altitude if commanded_altitude is not None else mode.altitude_mean
            ref_airspeed = commanded_airspeed if commanded_airspeed is not None else mode.airspeed_mean

            # If this is the first frame, initialize mode states from observation
            if self.frame == 1:
                mode.heading_mean = heading_deg
                mode.altitude_mean = altitude_ft
                mode.airspeed_mean = airspeed_kts
                ref_heading = heading_deg
                ref_altitude = altitude_ft
                ref_airspeed = airspeed_kts

            # Mixed initial state (from mixing step)
            mixed_heading = 0.0
            mixed_altitude = 0.0
            mixed_airspeed = 0.0
            for i in range(self.n_modes):
                w = mixing_weights[i, idx]
                mixed_heading += w * self.modes[i].heading_mean
                mixed_altitude += w * self.modes[i].altitude_mean
                mixed_airspeed += w * self.modes[i].airspeed_mean

            # Innovation (residual): observation - mixed prediction
            hdg_residual = self._wrap_angle(heading_deg - mixed_heading)
            alt_residual = altitude_ft - mixed_altitude
            spd_residual = airspeed_kts - mixed_airspeed

            mode.heading_residual = hdg_residual
            mode.altitude_residual = alt_residual
            mode.airspeed_residual = spd_residual

            # Log-likelihood for this model
            # Heading: Von Mises log-pdf approximated as Gaussian with
            # variance ≈ 1/kappa (in radians), converted to degrees
            hdg_std_deg = np.degrees(1.0 / np.sqrt(max(mode.heading_kappa, 0.01)))
            # Scale by process noise for uncertainty growth
            effective_hdg_std = hdg_std_deg * (1.0 + mode.process_noise_scale * dt * 5.0)
            effective_alt_std = mode.altitude_std * (1.0 + mode.process_noise_scale * dt * 2.0)
            effective_spd_std = mode.airspeed_std * (1.0 + mode.process_noise_scale * dt * 2.0)

            ll_hdg = -0.5 * (hdg_residual / max(effective_hdg_std, 0.1)) ** 2 \
                     - np.log(max(effective_hdg_std, 0.1))
            ll_alt = -0.5 * (alt_residual / max(effective_alt_std, 0.1)) ** 2 \
                     - np.log(max(effective_alt_std, 0.1))
            ll_spd = -0.5 * (spd_residual / max(effective_spd_std, 0.1)) ** 2 \
                     - np.log(max(effective_spd_std, 0.1))

            mode.log_likelihood = ll_hdg + ll_alt + ll_spd

            # State update: Kalman-style blending of observation into mode estimate
            # Gain depends inversely on model noise (tight model trusts prediction more)
            alpha = min(0.3, dt * 5.0)  # Update rate
            mode.heading_mean = self._wrap_to_360(
                mixed_heading + alpha * hdg_residual
            )
            mode.altitude_mean = mixed_altitude + alpha * alt_residual
            mode.airspeed_mean = mixed_airspeed + alpha * spd_residual

        # ── Step 3: Mode probability update ──────────────────────────────
        # mu_j(t) = c_j * L_j / sum_k(c_k * L_k)
        log_likes = np.array([m.log_likelihood for m in self.modes])

        # Numerical stability: subtract max before exp
        max_ll = np.max(log_likes)
        scaled_likes = np.exp(log_likes - max_ll)

        updated_probs = c_bar * scaled_likes
        prob_sum = np.sum(updated_probs)
        if prob_sum > 0:
            self.mode_probs = updated_probs / prob_sum
        else:
            self.mode_probs = DEFAULT_INIT_PROBS.copy()

        # Clamp minimum probability to prevent mode death
        self.mode_probs = np.maximum(self.mode_probs, 0.01)
        self.mode_probs /= np.sum(self.mode_probs)

        # ── Step 4: Combination ──────────────────────────────────────────
        # Blended state estimate (probability-weighted)
        blended_heading = sum(
            self.mode_probs[i] * self.modes[i].heading_mean
            for i in range(self.n_modes)
        )
        blended_altitude = sum(
            self.mode_probs[i] * self.modes[i].altitude_mean
            for i in range(self.n_modes)
        )
        blended_airspeed = sum(
            self.mode_probs[i] * self.modes[i].airspeed_mean
            for i in range(self.n_modes)
        )

        # IMM confidence: 1 - P(anomalous)
        # High P(tracking) or P(maneuvering) = high confidence
        # High P(anomalous) = low confidence
        p_normal = self.mode_probs[0] + self.mode_probs[1]  # tracking + maneuvering
        p_anomalous = self.mode_probs[2]

        # Confidence is weighted: tracking contributes full confidence,
        # maneuvering contributes partial (it's expected but less certain),
        # anomalous contributes near-zero
        confidence = (
            self.mode_probs[0] * 1.0 +    # tracking: full confidence
            self.mode_probs[1] * 0.7 +     # maneuvering: still confident, but noted
            self.mode_probs[2] * 0.05      # anomalous: near-zero
        )

        # Dominant mode
        dominant_idx = int(np.argmax(self.mode_probs))
        dominant_mode = self.modes[dominant_idx].name

        # Store history
        self.prob_history.append(self.mode_probs.copy())
        self.confidence_history.append(confidence)

        return {
            "confidence": float(confidence),
            "mode_probs": {
                "tracking": float(self.mode_probs[0]),
                "maneuvering": float(self.mode_probs[1]),
                "anomalous": float(self.mode_probs[2]),
            },
            "dominant_mode": dominant_mode,
            "blended_state": {
                "heading": float(blended_heading),
                "altitude": float(blended_altitude),
                "airspeed": float(blended_airspeed),
            },
            "residuals": {
                m.name: {
                    "heading": float(m.heading_residual),
                    "altitude": float(m.altitude_residual),
                    "airspeed": float(m.airspeed_residual),
                }
                for m in self.modes
            },
            "frame": self.frame,
            "phase": self.phase,
        }

    def get_confidence(self) -> float:
        """Return the current blended confidence."""
        if self.confidence_history:
            return self.confidence_history[-1]
        return 0.6  # default

    def get_mode_summary(self) -> Dict:
        """Get a summary suitable for telemetry/display."""
        return {
            "tracking": float(self.mode_probs[0]),
            "maneuvering": float(self.mode_probs[1]),
            "anomalous": float(self.mode_probs[2]),
            "dominant": self.modes[int(np.argmax(self.mode_probs))].name if self.modes else "unknown",
            "confidence": self.get_confidence(),
            "phase": self.phase,
            "frame": self.frame,
        }

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _wrap_angle(deg: float) -> float:
        """Wrap angle difference to [-180, 180]."""
        while deg > 180:
            deg -= 360
        while deg < -180:
            deg += 360
        return deg

    @staticmethod
    def _wrap_to_360(deg: float) -> float:
        """Wrap angle to [0, 360)."""
        return deg % 360


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

_imm_instance: Optional[IMMEstimator] = None


def get_imm_estimator() -> IMMEstimator:
    global _imm_instance
    if _imm_instance is None:
        _imm_instance = IMMEstimator()
    return _imm_instance


def reset_imm_estimator():
    global _imm_instance
    _imm_instance = None
