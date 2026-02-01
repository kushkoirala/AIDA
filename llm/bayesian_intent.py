"""
AIDA Bayesian Intent Inference

Core decision-making layer for the Autonomous Intelligent Decision Architecture (AIDA).
This module provides probabilistic reasoning over pilot intent, enabling autonomous
flight decisions grounded in principled Bayesian inference.

AIDA Stack:
    ┌─────────────────────────────────────────────┐
    │  Natural Language Interface (LLM + xLAM)   │  ← User commands
    ├─────────────────────────────────────────────┤
    │  BAYESIAN INTENT INFERENCE  ← THIS MODULE  │  ← Decision layer
    ├─────────────────────────────────────────────┤
    │  Intent Validation & Constraints            │  ← Safety gates
    ├─────────────────────────────────────────────┤
    │  Expert Pilot (Classical + NN)              │  ← Execution layer
    ├─────────────────────────────────────────────┤
    │  Flight Dynamics (Cessna 172 Physics)       │  ← Simulation
    └─────────────────────────────────────────────┘

Components:
    1. Intent State Space - Parameters with uncertainty (mean, variance)
    2. Prior Distributions - Phase-dependent beliefs about likely commands
    3. Likelihood Model - P(observation | intent)
    4. Posterior Inference - Updated beliefs after observing command
    5. Temporal Propagation - Kalman-like uncertainty evolution
    6. Multi-Step Tracking - Sequence coherence and pattern detection

Key Insight: Instead of asking "is this command valid?", we ask
"what is the probability distribution over possible intents given
what we observed, and does the commanded intent fall in a high-probability region?"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from enum import Enum
from scipy import stats
import json
import time


# =============================================================================
# Intent State Space
# =============================================================================

@dataclass
class GaussianBelief:
    """Represents belief about a continuous parameter as Gaussian."""
    mean: float
    variance: float

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)

    def pdf(self, x: float) -> float:
        """Probability density at x."""
        return stats.norm.pdf(x, self.mean, self.std)

    def log_pdf(self, x: float) -> float:
        """Log probability density at x."""
        return stats.norm.logpdf(x, self.mean, self.std)

    def cdf(self, x: float) -> float:
        """Cumulative distribution at x."""
        return stats.norm.cdf(x, self.mean, self.std)

    def credible_interval(self, alpha: float = 0.95) -> Tuple[float, float]:
        """Return alpha credible interval."""
        z = stats.norm.ppf((1 + alpha) / 2)
        return (self.mean - z * self.std, self.mean + z * self.std)

    def mahalanobis(self, x: float) -> float:
        """Mahalanobis distance (z-score) of x from this distribution."""
        return abs(x - self.mean) / self.std if self.std > 0 else float('inf')


@dataclass
class WrappedGaussianBelief:
    """Gaussian belief for circular quantities (heading, bearing)."""
    mean: float  # 0-360
    concentration: float  # kappa parameter (higher = more concentrated)

    def pdf(self, x: float) -> float:
        """Von Mises probability density."""
        # Convert to radians
        x_rad = np.deg2rad(x)
        mu_rad = np.deg2rad(self.mean)
        return stats.vonmises.pdf(x_rad, self.concentration, loc=mu_rad)

    def log_pdf(self, x: float) -> float:
        x_rad = np.deg2rad(x)
        mu_rad = np.deg2rad(self.mean)
        return stats.vonmises.logpdf(x_rad, self.concentration, loc=mu_rad)

    @property
    def circular_std_deg(self) -> float:
        """Circular standard deviation in degrees."""
        # Approximation for von Mises
        if self.concentration > 0:
            return np.rad2deg(1.0 / np.sqrt(self.concentration))
        return 180.0


@dataclass
class IntentState:
    """
    Full intent state with uncertainty on all parameters.

    Each parameter is a belief distribution, not a point estimate.
    """
    # Continuous state beliefs
    heading: WrappedGaussianBelief = field(default_factory=lambda: WrappedGaussianBelief(0, 1))
    altitude: GaussianBelief = field(default_factory=lambda: GaussianBelief(5500, 500**2))
    airspeed: GaussianBelief = field(default_factory=lambda: GaussianBelief(110, 20**2))
    vertical_rate: GaussianBelief = field(default_factory=lambda: GaussianBelief(0, 300**2))

    # Discrete state (categorical)
    phase_probs: Dict[str, float] = field(default_factory=dict)
    goal_probs: Dict[str, float] = field(default_factory=dict)

    # Timestamp for temporal tracking
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "heading": {"mean": self.heading.mean, "concentration": self.heading.concentration},
            "altitude": {"mean": self.altitude.mean, "std": self.altitude.std},
            "airspeed": {"mean": self.airspeed.mean, "std": self.airspeed.std},
            "vertical_rate": {"mean": self.vertical_rate.mean, "std": self.vertical_rate.std},
            "phase_probs": self.phase_probs,
            "goal_probs": self.goal_probs,
            "timestamp": self.timestamp
        }

    def copy(self) -> "IntentState":
        """Create a deep copy of this state."""
        return IntentState(
            heading=WrappedGaussianBelief(self.heading.mean, self.heading.concentration),
            altitude=GaussianBelief(self.altitude.mean, self.altitude.variance),
            airspeed=GaussianBelief(self.airspeed.mean, self.airspeed.variance),
            vertical_rate=GaussianBelief(self.vertical_rate.mean, self.vertical_rate.variance),
            phase_probs=self.phase_probs.copy(),
            goal_probs=self.goal_probs.copy(),
            timestamp=self.timestamp
        )


# =============================================================================
# Temporal Dynamics Model
# =============================================================================

@dataclass
class ProcessNoise:
    """
    Process noise for temporal propagation (Kalman-style).

    These represent the uncertainty growth rates when propagating
    state estimates forward without new observations.
    """
    # Heading drift rate (deg/s standard deviation)
    heading_drift: float = 0.5

    # Altitude uncertainty growth (ft/s standard deviation)
    altitude_drift: float = 10.0

    # Airspeed uncertainty growth (kts/s standard deviation)
    airspeed_drift: float = 2.0

    # Vertical rate uncertainty growth (fpm/s standard deviation)
    vertical_rate_drift: float = 50.0


@dataclass
class IntentTransition:
    """Represents a transition between intent states."""
    from_state: IntentState
    to_state: IntentState
    command: dict
    timestamp: float
    dt: float  # Time since last transition

    # Computed metrics
    coherence_score: float = 1.0  # How coherent this transition is
    anomaly_flag: bool = False    # Whether this transition is anomalous


# =============================================================================
# Prior Distributions by Flight Phase
# =============================================================================

class FlightPhase(Enum):
    GROUND = "ground"
    TAKEOFF = "takeoff"
    INITIAL_CLIMB = "initial_climb"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    FINAL = "final"
    FLARE = "flare"
    ROLLOUT = "rollout"
    LANDING = "landing"
    EMERGENCY = "emergency"


@dataclass
class PhasePriors:
    """Prior distributions for a specific flight phase."""

    # Expected heading change (degrees) - centered at 0 (no change)
    heading_change_std: float  # Standard deviation of expected heading changes

    # Expected altitude (ft)
    altitude_mean: float
    altitude_std: float

    # Expected airspeed (kts)
    airspeed_mean: float
    airspeed_std: float

    # Expected vertical rate (fpm)
    vertical_rate_mean: float
    vertical_rate_std: float

    # Probability of various goals being commanded
    goal_priors: Dict[str, float] = field(default_factory=dict)


# Phase-specific priors based on typical GA flight operations
PHASE_PRIORS: Dict[FlightPhase, PhasePriors] = {
    FlightPhase.GROUND: PhasePriors(
        heading_change_std=5.0,  # Small taxi corrections
        altitude_mean=0, altitude_std=10,
        airspeed_mean=10, airspeed_std=5,
        vertical_rate_mean=0, vertical_rate_std=50,
        goal_priors={"taxi": 0.7, "takeoff": 0.2, "hold": 0.1}
    ),
    FlightPhase.TAKEOFF: PhasePriors(
        heading_change_std=10.0,  # Runway alignment, departure turn
        altitude_mean=500, altitude_std=300,
        airspeed_mean=75, airspeed_std=10,
        vertical_rate_mean=700, vertical_rate_std=200,
        goal_priors={"climb": 0.8, "turn": 0.15, "abort": 0.05}
    ),
    FlightPhase.INITIAL_CLIMB: PhasePriors(
        heading_change_std=15.0,  # Departure turn possible
        altitude_mean=1500, altitude_std=500,
        airspeed_mean=80, airspeed_std=10,
        vertical_rate_mean=600, vertical_rate_std=200,
        goal_priors={"continue_climb": 0.7, "turn": 0.2, "level_off": 0.1}
    ),
    FlightPhase.CLIMB: PhasePriors(
        heading_change_std=20.0,  # Departure procedures, vectors
        altitude_mean=4000, altitude_std=2000,
        airspeed_mean=85, airspeed_std=10,
        vertical_rate_mean=500, vertical_rate_std=200,
        goal_priors={"continue_climb": 0.6, "level_off": 0.2, "turn": 0.15, "descend": 0.05}
    ),
    FlightPhase.CRUISE: PhasePriors(
        heading_change_std=15.0,  # Course corrections, vectors
        altitude_mean=5500, altitude_std=1000,
        airspeed_mean=110, airspeed_std=15,
        vertical_rate_mean=0, vertical_rate_std=100,
        goal_priors={"maintain": 0.5, "turn": 0.2, "climb": 0.1, "descend": 0.15, "divert": 0.05}
    ),
    FlightPhase.DESCENT: PhasePriors(
        heading_change_std=25.0,  # Approach vectors
        altitude_mean=3000, altitude_std=1500,
        airspeed_mean=100, airspeed_std=15,
        vertical_rate_mean=-500, vertical_rate_std=200,
        goal_priors={"continue_descent": 0.5, "level_off": 0.2, "turn": 0.2, "go_around": 0.1}
    ),
    FlightPhase.APPROACH: PhasePriors(
        heading_change_std=30.0,  # Pattern work, final alignment
        altitude_mean=1500, altitude_std=500,
        airspeed_mean=85, airspeed_std=10,
        vertical_rate_mean=-400, vertical_rate_std=150,
        goal_priors={"continue_approach": 0.6, "go_around": 0.15, "turn": 0.2, "land": 0.05}
    ),
    FlightPhase.FINAL: PhasePriors(
        heading_change_std=5.0,  # Runway alignment
        altitude_mean=500, altitude_std=300,
        airspeed_mean=75, airspeed_std=8,
        vertical_rate_mean=-400, vertical_rate_std=150,
        goal_priors={"continue_approach": 0.5, "land": 0.3, "go_around": 0.2}
    ),
    FlightPhase.FLARE: PhasePriors(
        heading_change_std=3.0,  # Very small corrections only
        altitude_mean=20, altitude_std=20,
        airspeed_mean=60, airspeed_std=5,
        vertical_rate_mean=-100, vertical_rate_std=50,
        goal_priors={"land": 0.9, "go_around": 0.1}
    ),
    FlightPhase.ROLLOUT: PhasePriors(
        heading_change_std=3.0,  # Runway centerline only
        altitude_mean=0, altitude_std=5,
        airspeed_mean=40, airspeed_std=15,
        vertical_rate_mean=0, vertical_rate_std=20,
        goal_priors={"stop": 0.8, "taxi": 0.2}
    ),
    FlightPhase.LANDING: PhasePriors(
        heading_change_std=5.0,  # Runway alignment only
        altitude_mean=100, altitude_std=100,
        airspeed_mean=65, airspeed_std=10,
        vertical_rate_mean=-300, vertical_rate_std=100,
        goal_priors={"land": 0.7, "go_around": 0.3}
    ),
    FlightPhase.EMERGENCY: PhasePriors(
        heading_change_std=90.0,  # Anything possible
        altitude_mean=3000, altitude_std=2000,
        airspeed_mean=90, airspeed_std=30,
        vertical_rate_mean=0, vertical_rate_std=500,
        goal_priors={"divert": 0.4, "land_immediate": 0.3, "return": 0.2, "hold": 0.1}
    ),
}


# =============================================================================
# Bayesian Inference Engine
# =============================================================================

class BayesianIntentInference:
    """
    Performs Bayesian inference over flight intent.

    The key equations:
        Prior:      P(intent)           - What we expect before observation
        Likelihood: P(command | intent) - How likely is this command given intent
        Posterior:  P(intent | command) ∝ P(command | intent) × P(intent)

    Temporal Model:
        x(t+dt) = x(t) + process_noise * sqrt(dt)
        Uncertainty grows with time since last observation.

    Multi-Step Tracking:
        Maintains sliding window of intent transitions.
        Computes sequence coherence and detects patterns.
    """

    # Window size for multi-step tracking
    HISTORY_WINDOW = 10

    def __init__(self):
        self.current_state = IntentState()
        self.phase = FlightPhase.CRUISE
        self.history: List[IntentState] = []

        # Temporal tracking
        self.last_update_time: float = time.time()
        self.process_noise = ProcessNoise()

        # Multi-step intent tracking (sliding window)
        self.intent_window: Deque[IntentTransition] = deque(maxlen=self.HISTORY_WINDOW)
        self.predicted_state: Optional[IntentState] = None

        # Aircraft performance limits (hard constraints)
        self.limits = {
            "min_altitude_ft": 0,
            "max_altitude_ft": 14000,  # Service ceiling
            "min_airspeed_kts": 48,    # Vs0
            "max_airspeed_kts": 163,   # Vne
            "max_bank_deg": 60,
            "max_climb_fpm": 1000,
            "max_descent_fpm": 1500,
        }

        # Expected intent patterns (for pattern recognition)
        self.intent_patterns = {
            "approach_sequence": ["descend", "turn", "descend", "land"],
            "holding_pattern": ["turn", "maintain", "turn", "maintain"],
            "departure": ["climb", "turn", "climb", "maintain"],
            "emergency_divert": ["turn", "descend", "land"],
        }

    def set_phase(self, phase: FlightPhase):
        """Update the flight phase, which changes the prior."""
        self.phase = phase
        self._update_priors_from_phase()

    def _update_priors_from_phase(self):
        """Set priors based on current flight phase."""
        priors = PHASE_PRIORS.get(self.phase, PHASE_PRIORS[FlightPhase.CRUISE])

        self.current_state.altitude = GaussianBelief(
            priors.altitude_mean,
            priors.altitude_std ** 2
        )
        self.current_state.airspeed = GaussianBelief(
            priors.airspeed_mean,
            priors.airspeed_std ** 2
        )
        self.current_state.vertical_rate = GaussianBelief(
            priors.vertical_rate_mean,
            priors.vertical_rate_std ** 2
        )
        self.current_state.goal_probs = priors.goal_priors.copy()

    def update_from_telemetry(self, telemetry: dict):
        """
        Update beliefs from current aircraft state (telemetry).

        This anchors our prior to the actual current state.
        """
        if "heading" in telemetry:
            # Center heading prior on current heading
            self.current_state.heading = WrappedGaussianBelief(
                mean=telemetry["heading"],
                concentration=10.0  # Fairly concentrated around current
            )

        if "altitude" in telemetry:
            # Update altitude belief - narrow variance around current
            self.current_state.altitude = GaussianBelief(
                mean=telemetry["altitude"],
                variance=200 ** 2  # ±200ft uncertainty
            )

        if "airspeed" in telemetry:
            self.current_state.airspeed = GaussianBelief(
                mean=telemetry["airspeed"],
                variance=10 ** 2
            )

        if "phase" in telemetry:
            phase_str = telemetry["phase"].lower()
            for p in FlightPhase:
                if p.value in phase_str:
                    self.set_phase(p)
                    break

    def compute_likelihood(self, command: dict) -> dict:
        """
        Compute likelihood P(command | intent) for different intent hypotheses.

        Returns dict with likelihood values and analysis.
        """
        action = command.get("action", "")
        value = command.get("value")

        result = {
            "action": action,
            "value": value,
            "likelihoods": {},
            "anomaly_score": 0.0,
        }

        priors = PHASE_PRIORS.get(self.phase, PHASE_PRIORS[FlightPhase.CRUISE])

        if action == "heading" and value is not None:
            # Compute likelihood under heading change prior
            current_hdg = self.current_state.heading.mean
            change = self._normalize_heading_change(value - current_hdg)

            # Derive heading std from concentration (Von Mises -> approx Gaussian)
            # For Von Mises with concentration kappa, variance ~ 1/kappa (in radians)
            # Convert to degrees: std ~ degrees(1/sqrt(kappa))
            kappa = self.current_state.heading.concentration
            if kappa > 0:
                heading_std = np.rad2deg(1.0 / np.sqrt(kappa))
            else:
                heading_std = priors.heading_change_std  # fallback

            # Likelihood: heading changes are normally distributed around 0
            likelihood = stats.norm.pdf(change, 0, heading_std)
            max_likelihood = stats.norm.pdf(0, 0, heading_std)
            normalized = likelihood / max_likelihood if max_likelihood > 0 else 0

            result["likelihoods"]["heading_change"] = {
                "value": float(normalized),
                "change_deg": float(change),
                "expected_std": heading_std,
                "z_score": abs(change) / heading_std if heading_std > 0 else float('inf')
            }
            result["anomaly_score"] = abs(change) / heading_std if heading_std > 0 else float('inf')

        elif action == "altitude" and value is not None:
            # Likelihood under altitude prior
            likelihood = self.current_state.altitude.pdf(value)
            max_likelihood = self.current_state.altitude.pdf(self.current_state.altitude.mean)
            normalized = likelihood / max_likelihood if max_likelihood > 0 else 0

            z_score = self.current_state.altitude.mahalanobis(value)

            # Also check against hard limits
            within_limits = (self.limits["min_altitude_ft"] <= value <= self.limits["max_altitude_ft"])

            result["likelihoods"]["altitude"] = {
                "value": float(normalized),
                "target_ft": float(value),
                "prior_mean": self.current_state.altitude.mean,
                "prior_std": self.current_state.altitude.std,
                "z_score": float(z_score),
                "within_limits": within_limits
            }
            result["anomaly_score"] = z_score

        elif action == "land":
            # Landing intent - check if expected for current phase
            land_prior = self.current_state.goal_probs.get("land", 0.1)
            result["likelihoods"]["land"] = {
                "value": land_prior,
                "phase": self.phase.value,
                "expected": self.phase in [FlightPhase.APPROACH, FlightPhase.LANDING]
            }
            # Low anomaly if we're already in approach/landing
            result["anomaly_score"] = 0.5 if land_prior < 0.3 else 0.1

        return result

    def infer_posterior(self, command: dict, telemetry: Optional[dict] = None) -> dict:
        """
        Main inference: compute posterior P(intent | command, telemetry).

        Returns:
            Dictionary with posterior beliefs and confidence metrics.
        """
        # Update from telemetry if provided
        if telemetry:
            self.update_from_telemetry(telemetry)

        # Compute likelihood
        likelihood_result = self.compute_likelihood(command)

        # Extract action-specific posterior
        action = command.get("action", "")
        value = command.get("value")

        posterior = {
            "action": action,
            "value": value,
            "prior": {},
            "likelihood": likelihood_result["likelihoods"],
            "posterior": {},
            "confidence": {},
            "anomaly_score": likelihood_result["anomaly_score"],
            "credible_intervals": {},
            "hard_constraint_violations": [],
        }

        # Compute posterior for specific actions
        if action == "heading" and value is not None:
            posterior["prior"]["heading"] = {
                "mean": self.current_state.heading.mean,
                "std": self.current_state.heading.circular_std_deg
            }

            # Posterior heading is the commanded value with some uncertainty
            # (we're fairly confident about what was commanded)
            posterior["posterior"]["heading"] = {
                "mean": value,
                "std": 2.0  # Small uncertainty on commanded value
            }

            # Confidence based on how likely this command was under the prior
            z_score = likelihood_result["likelihoods"].get("heading_change", {}).get("z_score", 3.0)
            posterior["confidence"]["heading"] = self._z_to_confidence(z_score)

        elif action == "altitude" and value is not None:
            posterior["prior"]["altitude"] = {
                "mean": self.current_state.altitude.mean,
                "std": self.current_state.altitude.std
            }

            # Check hard constraints
            if value < self.limits["min_altitude_ft"]:
                posterior["hard_constraint_violations"].append(
                    f"Below minimum altitude ({self.limits['min_altitude_ft']} ft)"
                )
            if value > self.limits["max_altitude_ft"]:
                posterior["hard_constraint_violations"].append(
                    f"Above service ceiling ({self.limits['max_altitude_ft']} ft)"
                )

            # Posterior altitude
            posterior["posterior"]["altitude"] = {
                "mean": value,
                "std": 50.0  # Small uncertainty on commanded value
            }

            # Credible interval for expected altitude
            ci = self.current_state.altitude.credible_interval(0.95)
            posterior["credible_intervals"]["altitude_95"] = {
                "lower": ci[0],
                "upper": ci[1],
                "commanded_within": ci[0] <= value <= ci[1]
            }

            z_score = likelihood_result["likelihoods"].get("altitude", {}).get("z_score", 3.0)
            posterior["confidence"]["altitude"] = self._z_to_confidence(z_score)

            # Hard constraint violation kills confidence
            if posterior["hard_constraint_violations"]:
                posterior["confidence"]["altitude"] = 0.0

        elif action == "land":
            land_likelihood = likelihood_result["likelihoods"].get("land", {}).get("value", 0.1)
            posterior["confidence"]["land"] = land_likelihood
            posterior["posterior"]["goal"] = "land"

        # Overall confidence is minimum of component confidences (conservative)
        if posterior["confidence"]:
            posterior["overall_confidence"] = min(posterior["confidence"].values())
        else:
            posterior["overall_confidence"] = 0.5  # Default uncertainty

        # Store in history
        new_state = IntentState(
            heading=WrappedGaussianBelief(self.current_state.heading.mean, self.current_state.heading.concentration),
            altitude=GaussianBelief(self.current_state.altitude.mean, self.current_state.altitude.variance),
            airspeed=GaussianBelief(self.current_state.airspeed.mean, self.current_state.airspeed.variance),
            vertical_rate=GaussianBelief(self.current_state.vertical_rate.mean, self.current_state.vertical_rate.variance),
            phase_probs={self.phase.value: 1.0},
            goal_probs=self.current_state.goal_probs.copy(),
            timestamp=command.get("timestamp", time.time())
        )
        self.history.append(new_state)

        # Record transition for multi-step tracking
        self.record_transition(command, posterior)

        # Add sequence coherence to posterior
        sequence_analysis = self.get_sequence_coherence()
        posterior["sequence_coherence"] = sequence_analysis["overall_coherence"]
        posterior["sequence_analysis"] = sequence_analysis

        # Update last update time for temporal tracking
        self.last_update_time = time.time()

        return posterior

    def _normalize_heading_change(self, change: float) -> float:
        """Normalize heading change to [-180, 180]."""
        while change > 180:
            change -= 360
        while change < -180:
            change += 360
        return change

    def _z_to_confidence(self, z_score: float) -> float:
        """
        Convert z-score to confidence value.

        z=0 -> 1.0 (exactly expected)
        z=1 -> 0.84 (within 1 std)
        z=2 -> 0.5 (within 2 std)
        z=3 -> 0.15 (3 std - unusual)
        z>4 -> ~0 (very anomalous)
        """
        # Use survival function of chi-squared with 1 df
        # This gives P(|Z| > z) which we invert
        return float(1.0 - stats.chi2.cdf(z_score ** 2, df=1))

    def propagate(self, dt: float = None):
        """
        Propagate uncertainty forward in time (Kalman prediction step).

        As time passes without new observations, uncertainty grows according
        to the process noise model. This is the "predict" step in Kalman filtering.

        Args:
            dt: Time delta in seconds. If None, uses time since last update.
        """
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time

        if dt <= 0:
            return

        # Process noise growth (variance grows linearly with time)
        # σ²(t+dt) = σ²(t) + q² * dt  where q is process noise rate

        # Heading: concentration decreases (uncertainty increases)
        # Concentration κ relates to variance roughly as σ² ≈ 1/κ
        # So decreasing κ increases uncertainty
        decay_rate = 0.1  # concentration decay per second
        self.current_state.heading.concentration *= np.exp(-decay_rate * dt)
        self.current_state.heading.concentration = max(
            self.current_state.heading.concentration, 0.1  # Minimum concentration
        )

        # Altitude uncertainty growth
        self.current_state.altitude.variance += (self.process_noise.altitude_drift ** 2) * dt

        # Airspeed uncertainty growth
        self.current_state.airspeed.variance += (self.process_noise.airspeed_drift ** 2) * dt

        # Vertical rate uncertainty growth
        self.current_state.vertical_rate.variance += (self.process_noise.vertical_rate_drift ** 2) * dt

        # Update timestamp
        self.current_state.timestamp += dt

    def predict_state(self, dt: float) -> IntentState:
        """
        Predict the intent state at time t + dt.

        This creates a forward projection of what we expect the intent
        to be in the future, accounting for current dynamics.

        Args:
            dt: Time horizon in seconds

        Returns:
            Predicted IntentState with propagated uncertainty
        """
        # Start from copy of current state
        predicted = self.current_state.copy()

        # Propagate mean values based on current dynamics
        # Altitude changes according to vertical rate
        vr_fpm = predicted.vertical_rate.mean
        altitude_change = vr_fpm * (dt / 60.0)  # Convert fpm to ft for dt seconds
        predicted.altitude.mean += altitude_change

        # Propagate uncertainties (same as propagate but on the copy)
        predicted.heading.concentration *= np.exp(-0.1 * dt)
        predicted.heading.concentration = max(predicted.heading.concentration, 0.1)
        predicted.altitude.variance += (self.process_noise.altitude_drift ** 2) * dt
        predicted.airspeed.variance += (self.process_noise.airspeed_drift ** 2) * dt
        predicted.vertical_rate.variance += (self.process_noise.vertical_rate_drift ** 2) * dt

        # Update timestamp
        predicted.timestamp = self.current_state.timestamp + dt

        self.predicted_state = predicted
        return predicted

    def get_prediction_horizon(self, horizons: List[float] = None) -> List[IntentState]:
        """
        Get predictions at multiple time horizons.

        Useful for anticipating intent evolution.

        Args:
            horizons: List of time deltas in seconds (default: [5, 15, 30, 60])

        Returns:
            List of predicted IntentStates at each horizon
        """
        if horizons is None:
            horizons = [5.0, 15.0, 30.0, 60.0]

        return [self.predict_state(h) for h in horizons]

    # =========================================================================
    # Multi-Step Intent Tracking
    # =========================================================================

    def record_transition(self, command: dict, posterior: dict):
        """
        Record an intent transition in the sliding window.

        Called after infer_posterior to track the sequence of intents.
        """
        current_time = time.time()

        # Compute time since last transition
        if self.intent_window:
            dt = current_time - self.intent_window[-1].timestamp
        else:
            dt = 0.0

        # Create new state from posterior
        new_state = self.current_state.copy()
        new_state.timestamp = current_time

        # Compute coherence with previous state
        coherence = self._compute_transition_coherence(command)

        # Create transition record
        transition = IntentTransition(
            from_state=self.history[-1] if self.history else self.current_state.copy(),
            to_state=new_state,
            command=command,
            timestamp=current_time,
            dt=dt,
            coherence_score=coherence,
            anomaly_flag=coherence < 0.3  # Flag if very incoherent
        )

        self.intent_window.append(transition)

    def _compute_transition_coherence(self, command: dict) -> float:
        """
        Compute how coherent a command transition is with the recent history.

        High coherence = consistent with recent intent pattern
        Low coherence = sudden change or inconsistent behavior
        """
        if len(self.intent_window) < 2:
            return 1.0  # No history, assume coherent

        action = command.get("action", "")
        value = command.get("value")

        # Factor 1: Action consistency
        # Check if action type follows a logical pattern
        recent_actions = [t.command.get("action", "") for t in self.intent_window][-3:]
        action_coherence = self._action_sequence_coherence(recent_actions + [action])

        # Factor 2: Value continuity
        # Check if commanded values are continuous with history
        value_coherence = 1.0
        if value is not None and action in ["heading", "altitude", "airspeed"]:
            value_coherence = self._value_continuity_score(action, value)

        # Factor 3: Timing coherence
        # Very rapid commands might indicate instability, but don't heavily penalize
        if self.intent_window:
            last_dt = time.time() - self.intent_window[-1].timestamp
            # Soft penalty for very rapid commands (< 1s)
            timing_coherence = min(1.0, 0.5 + last_dt / 2.0)
        else:
            timing_coherence = 1.0

        # Combine factors - weighted average, not geometric mean
        # Value coherence is most important for detecting erratic behavior
        weights = {"action": 0.2, "value": 0.6, "timing": 0.2}
        coherence = (
            weights["action"] * action_coherence +
            weights["value"] * value_coherence +
            weights["timing"] * timing_coherence
        )
        return float(coherence)

    def _action_sequence_coherence(self, actions: List[str]) -> float:
        """
        Score how coherent a sequence of actions is.

        Looks for known patterns and penalizes erratic switching.
        """
        if len(actions) < 2:
            return 1.0

        # Check for oscillation (rapid back-and-forth)
        if len(actions) >= 3:
            # A-B-A pattern is suspicious
            if actions[-3] == actions[-1] and actions[-3] != actions[-2]:
                return 0.5  # Oscillation detected

        # Check if sequence matches known patterns
        for pattern_name, pattern in self.intent_patterns.items():
            if self._matches_pattern(actions, pattern):
                return 1.0  # Matches expected pattern

        # Default: slight penalty for changing action type
        if len(actions) >= 2 and actions[-1] != actions[-2]:
            return 0.8

        return 1.0

    def _matches_pattern(self, actions: List[str], pattern: List[str]) -> bool:
        """Check if recent actions match a known intent pattern."""
        if len(actions) < len(pattern):
            return False

        # Simple substring matching
        actions_str = ",".join(actions[-len(pattern):])
        pattern_str = ",".join(pattern)

        # Allow partial matches for heading->turn, altitude->climb/descend
        action_mapping = {
            "heading": "turn",
            "altitude": ["climb", "descend"],
        }

        return actions_str == pattern_str

    def _value_continuity_score(self, action: str, value: float) -> float:
        """
        Score how continuous a commanded value is with history.

        Considers both magnitude of change AND consistency of direction.
        A consistent descent/climb is more coherent than oscillations.
        """
        # Get historical values for this action type
        historical_values = []
        for t in self.intent_window:
            if t.command.get("action") == action:
                v = t.command.get("value")
                if v is not None:
                    historical_values.append(v)

        if not historical_values:
            return 1.0

        # Compute rate of change from last value
        last_value = historical_values[-1]

        if action == "heading":
            # Heading is circular
            change = abs(self._normalize_heading_change(value - last_value))
            # Expect max ~30 deg change normally
            base_score = max(0, 1.0 - change / 60.0)

            # Check for consistent turn direction
            if len(historical_values) >= 2:
                prev_change = self._normalize_heading_change(historical_values[-1] - historical_values[-2])
                curr_change = self._normalize_heading_change(value - last_value)
                # Same direction = bonus, opposite direction = penalty
                if np.sign(prev_change) == np.sign(curr_change) and abs(curr_change) > 5:
                    base_score = min(1.0, base_score + 0.2)  # Consistent turn bonus
                elif np.sign(prev_change) != np.sign(curr_change) and abs(prev_change) > 10 and abs(curr_change) > 10:
                    base_score *= 0.7  # Oscillation penalty

            return base_score

        elif action == "altitude":
            change = value - last_value  # Signed change
            abs_change = abs(change)

            # Base score from magnitude
            base_score = max(0, 1.0 - abs_change / 2000.0)

            # Check for consistent climb/descent direction
            if len(historical_values) >= 2:
                prev_change = historical_values[-1] - historical_values[-2]

                # Same direction (both climbing or both descending)
                if np.sign(prev_change) == np.sign(change) and abs(change) > 100:
                    # Consistent trend - high coherence even for large changes
                    trend_bonus = 0.3
                    base_score = min(1.0, base_score + trend_bonus)
                elif np.sign(prev_change) != np.sign(change) and abs(prev_change) > 500 and abs(change) > 500:
                    # Oscillation (climb-descend-climb) - penalize
                    base_score *= 0.5

            return max(0.1, base_score)  # Floor to prevent zero score for valid commands

        elif action == "airspeed":
            change = abs(value - last_value)
            # Expect max ~20kt change normally
            return max(0, 1.0 - change / 40.0)

        return 1.0

    def get_sequence_coherence(self) -> dict:
        """
        Analyze the coherence of the full intent sequence in the window.

        Returns:
            Dictionary with sequence analysis metrics
        """
        if len(self.intent_window) < 2:
            return {
                "overall_coherence": 1.0,
                "transitions": 0,
                "anomalies": 0,
                "pattern_detected": None,
                "trend": "stable"
            }

        # Compute average coherence
        coherences = [t.coherence_score for t in self.intent_window]
        avg_coherence = np.mean(coherences)

        # Count anomalies
        anomalies = sum(1 for t in self.intent_window if t.anomaly_flag)

        # Detect pattern
        actions = [t.command.get("action", "") for t in self.intent_window]
        detected_pattern = None
        for pattern_name, pattern in self.intent_patterns.items():
            if self._matches_pattern(actions, pattern):
                detected_pattern = pattern_name
                break

        # Analyze trend (are values moving consistently in one direction?)
        trend = self._analyze_trend()

        return {
            "overall_coherence": float(avg_coherence),
            "transitions": len(self.intent_window),
            "anomalies": anomalies,
            "anomaly_rate": anomalies / len(self.intent_window),
            "pattern_detected": detected_pattern,
            "trend": trend,
            "recent_coherences": coherences[-5:],  # Last 5 coherence scores
        }

    def _analyze_trend(self) -> str:
        """
        Analyze the trend of commanded values.

        Returns: "climbing", "descending", "turning_left", "turning_right",
                 "accelerating", "decelerating", "stable", "erratic"
        """
        if len(self.intent_window) < 3:
            return "stable"

        # Extract altitude changes
        altitude_commands = [
            t.command.get("value") for t in self.intent_window
            if t.command.get("action") == "altitude" and t.command.get("value") is not None
        ]

        if len(altitude_commands) >= 2:
            changes = [altitude_commands[i+1] - altitude_commands[i]
                      for i in range(len(altitude_commands)-1)]
            if all(c > 0 for c in changes):
                return "climbing"
            elif all(c < 0 for c in changes):
                return "descending"

        # Extract heading changes
        heading_commands = [
            t.command.get("value") for t in self.intent_window
            if t.command.get("action") == "heading" and t.command.get("value") is not None
        ]

        if len(heading_commands) >= 2:
            changes = [self._normalize_heading_change(heading_commands[i+1] - heading_commands[i])
                      for i in range(len(heading_commands)-1)]
            if all(c > 5 for c in changes):
                return "turning_right"
            elif all(c < -5 for c in changes):
                return "turning_left"

        return "stable"

    def get_intent_summary(self) -> dict:
        """
        Get a comprehensive summary of current intent state and history.

        Useful for displaying to operators or logging.
        """
        sequence_analysis = self.get_sequence_coherence()

        # Prediction horizon
        predictions = {}
        for horizon in [5, 15, 30]:
            pred = self.predict_state(float(horizon))
            predictions[f"{horizon}s"] = {
                "altitude": {"mean": pred.altitude.mean, "std": pred.altitude.std},
                "heading": {"mean": pred.heading.mean, "std": pred.heading.circular_std_deg},
            }

        return {
            "current_state": self.current_state.to_dict(),
            "phase": self.phase.value,
            "sequence_analysis": sequence_analysis,
            "predictions": predictions,
            "time_since_update": time.time() - self.last_update_time,
            "history_length": len(self.intent_window),
        }


# =============================================================================
# Integration with existing FlightIntent system
# =============================================================================

# Global inference engine for stateful tracking across calls
_inference_engine: Optional[BayesianIntentInference] = None


def get_inference_engine() -> BayesianIntentInference:
    """Get or create the global inference engine."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = BayesianIntentInference()
    return _inference_engine


def reset_inference_engine():
    """Reset the global inference engine (e.g., for new flight)."""
    global _inference_engine
    _inference_engine = None


def bayesian_validate_intent(
    action: str,
    value: Optional[float],
    target: Optional[str],
    telemetry: Optional[dict] = None,
    phase: str = "cruise",
    use_stateful: bool = True
) -> dict:
    """
    Validate a flight command using Bayesian inference.

    This is the main entry point for integration with the LLM command server.

    Args:
        action: Command action (heading, altitude, land, etc.)
        value: Numeric value for the command (if applicable)
        target: Target identifier (e.g., airport code)
        telemetry: Current aircraft telemetry
        phase: Current flight phase
        use_stateful: If True, uses persistent engine for multi-step tracking

    Returns:
        Dictionary compatible with the existing intent validation format,
        but with probabilistically-grounded confidence scores.
    """
    # Use stateful engine for multi-step tracking, or create fresh one
    if use_stateful:
        inference = get_inference_engine()
    else:
        inference = BayesianIntentInference()

    # Set phase
    phase_map = {
        "ground": FlightPhase.GROUND,
        "takeoff": FlightPhase.TAKEOFF,
        "initial_climb": FlightPhase.INITIAL_CLIMB,
        "climb": FlightPhase.CLIMB,
        "cruise": FlightPhase.CRUISE,
        "descent": FlightPhase.DESCENT,
        "approach": FlightPhase.APPROACH,
        "final": FlightPhase.FINAL,
        "flare": FlightPhase.FLARE,
        "rollout": FlightPhase.ROLLOUT,
        "landing": FlightPhase.LANDING,
        "emergency": FlightPhase.EMERGENCY,
    }
    inference.set_phase(phase_map.get(phase.lower(), FlightPhase.CRUISE))

    # Build command dict
    command = {
        "action": action,
        "value": value,
        "target": target,
        "timestamp": time.time(),
    }

    # Run inference
    posterior = inference.infer_posterior(command, telemetry)

    # Get sequence analysis for multi-step coherence
    sequence_analysis = posterior.get("sequence_analysis", {})

    # Format for compatibility with existing system
    result = {
        "validated": len(posterior["hard_constraint_violations"]) == 0,
        "confidence": posterior["overall_confidence"],
        "hard_gates": {
            "schema_valid": True,  # Command parsed successfully
            "constraints_feasible": len(posterior["hard_constraint_violations"]) == 0,
            "goals_achievable": posterior["overall_confidence"] > 0.1,
        },
        "soft_scores": {
            "safety_margin": max(0, 1.0 - posterior["anomaly_score"] / 3.0),
            "semantic_coherence": posterior["overall_confidence"],
            "specificity": 0.9 if value is not None else 0.5,
            "sequence_coherence": sequence_analysis.get("overall_coherence", 1.0),
        },
        "bayesian": {
            "prior": posterior["prior"],
            "likelihood": posterior["likelihood"],
            "posterior": posterior["posterior"],
            "credible_intervals": posterior["credible_intervals"],
            "anomaly_score": posterior["anomaly_score"],
        },
        "temporal": {
            "sequence_coherence": sequence_analysis.get("overall_coherence", 1.0),
            "anomaly_rate": sequence_analysis.get("anomaly_rate", 0.0),
            "pattern_detected": sequence_analysis.get("pattern_detected"),
            "trend": sequence_analysis.get("trend", "stable"),
            "history_length": sequence_analysis.get("transitions", 0),
        },
        "issues": posterior["hard_constraint_violations"],
    }

    # Add anomaly warning if sequence is incoherent
    if sequence_analysis.get("overall_coherence", 1.0) < 0.5:
        result["issues"].append(
            f"Low sequence coherence ({sequence_analysis.get('overall_coherence', 0):.0%}) - "
            f"intent may be erratic"
        )

    # BIRL integration: if a BIRL engine is attached, include reward posterior
    # and modulate confidence by entropy (high entropy = less certain intent)
    birl = getattr(inference, '_birl_engine', None)
    if birl is not None:
        birl_summary = birl.get_reward_posterior_summary()
        result["birl"] = birl_summary
        # Entropy-modulated confidence: penalize ambiguous intent
        entropy = birl_summary["entropy"]
        result["confidence"] *= (1.0 - 0.5 * entropy)
        result["soft_scores"]["birl_entropy"] = entropy
        result["soft_scores"]["birl_dominant_intent"] = birl_summary["dominant_intent"]
        if entropy > 0.7:
            result["issues"].append(
                f"High BIRL entropy ({entropy:.2f}) - intent is ambiguous"
            )

    return result


# =============================================================================
# Example / Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Bayesian Intent Inference - Test")
    print("=" * 70)

    # Reset engine for clean test
    reset_inference_engine()

    # Simulate flight context
    telemetry = {
        "heading": 90,
        "altitude": 5500,
        "airspeed": 110,
        "phase": "cruise"
    }

    print("\n" + "-" * 70)
    print("PART 1: Basic Inference Tests")
    print("-" * 70)

    # Test cases
    test_commands = [
        ("heading", 100, None, "Small heading change - should be high confidence"),
        ("heading", 270, None, "Large heading change - lower confidence"),
        ("altitude", 6000, None, "Reasonable altitude change"),
        ("altitude", 20000, None, "Above service ceiling - should fail"),
    ]

    for action, value, target, description in test_commands:
        print(f"\n--- {description} ---")
        print(f"Command: {action} = {value}")

        result = bayesian_validate_intent(action, value, target, telemetry, "cruise", use_stateful=False)

        print(f"Valid: {result['validated']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Anomaly Score: {result['bayesian']['anomaly_score']:.2f}")

        if result['bayesian'].get('credible_intervals'):
            ci = result['bayesian']['credible_intervals'].get('altitude_95', {})
            if ci:
                print(f"95% CI: [{ci['lower']:.0f}, {ci['upper']:.0f}] ft")
                print(f"Commanded within CI: {ci['commanded_within']}")

        if result['issues']:
            print(f"Issues: {result['issues']}")

    print("\n" + "-" * 70)
    print("PART 2: Temporal Propagation Test")
    print("-" * 70)

    # Create fresh engine for temporal test
    engine = BayesianIntentInference()
    engine.update_from_telemetry(telemetry)
    engine.set_phase(FlightPhase.CRUISE)

    print(f"\nInitial state:")
    print(f"  Altitude: {engine.current_state.altitude.mean:.0f} ft ± {engine.current_state.altitude.std:.0f}")
    print(f"  Heading: {engine.current_state.heading.mean:.0f}° (κ={engine.current_state.heading.concentration:.1f})")

    # Simulate time passing
    for dt in [5, 15, 30]:
        predicted = engine.predict_state(float(dt))
        print(f"\nPredicted state at t+{dt}s:")
        print(f"  Altitude: {predicted.altitude.mean:.0f} ft ± {predicted.altitude.std:.0f}")
        print(f"  Heading: {predicted.heading.mean:.0f}° (κ={predicted.heading.concentration:.1f})")

    print("\n" + "-" * 70)
    print("PART 3: Multi-Step Intent Tracking")
    print("-" * 70)

    # Reset for multi-step test
    reset_inference_engine()

    # Simulate a coherent approach sequence
    print("\n--- Coherent Approach Sequence ---")
    approach_commands = [
        ("altitude", 4500, None, {"heading": 90, "altitude": 5500, "airspeed": 110, "phase": "descent"}),
        ("heading", 120, None, {"heading": 90, "altitude": 4500, "airspeed": 105, "phase": "descent"}),
        ("altitude", 3500, None, {"heading": 120, "altitude": 4500, "airspeed": 100, "phase": "approach"}),
        ("altitude", 2500, None, {"heading": 120, "altitude": 3500, "airspeed": 95, "phase": "approach"}),
    ]

    for i, (action, value, target, telem) in enumerate(approach_commands):
        print(f"\nCommand {i+1}: {action} = {value}")
        result = bayesian_validate_intent(action, value, target, telem, telem["phase"])
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Sequence Coherence: {result['temporal']['sequence_coherence']:.3f}")
        print(f"  Trend: {result['temporal']['trend']}")
        if result['temporal']['pattern_detected']:
            print(f"  Pattern Detected: {result['temporal']['pattern_detected']}")

    # Now test erratic sequence
    reset_inference_engine()
    print("\n--- Erratic Command Sequence ---")
    erratic_commands = [
        ("altitude", 6000, None, {"heading": 90, "altitude": 5500, "airspeed": 110, "phase": "cruise"}),
        ("altitude", 3000, None, {"heading": 90, "altitude": 6000, "airspeed": 110, "phase": "cruise"}),
        ("altitude", 7000, None, {"heading": 90, "altitude": 3000, "airspeed": 110, "phase": "cruise"}),
        ("altitude", 2000, None, {"heading": 90, "altitude": 7000, "airspeed": 110, "phase": "cruise"}),
    ]

    for i, (action, value, target, telem) in enumerate(erratic_commands):
        print(f"\nCommand {i+1}: {action} = {value}")
        result = bayesian_validate_intent(action, value, target, telem, telem["phase"])
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Sequence Coherence: {result['temporal']['sequence_coherence']:.3f}")
        print(f"  Trend: {result['temporal']['trend']}")
        if result['issues']:
            print(f"  Issues: {result['issues']}")

    print("\n" + "-" * 70)
    print("PART 4: Intent Summary")
    print("-" * 70)

    engine = get_inference_engine()
    summary = engine.get_intent_summary()
    print(f"\nCurrent Phase: {summary['phase']}")
    print(f"History Length: {summary['history_length']}")
    print(f"Time Since Update: {summary['time_since_update']:.2f}s")
    print(f"Sequence Analysis:")
    print(f"  Overall Coherence: {summary['sequence_analysis']['overall_coherence']:.3f}")
    print(f"  Anomaly Rate: {summary['sequence_analysis']['anomaly_rate']:.3f}")
    print(f"  Trend: {summary['sequence_analysis']['trend']}")

    print("\nPredictions:")
    for horizon, pred in summary['predictions'].items():
        print(f"  {horizon}: altitude={pred['altitude']['mean']:.0f}±{pred['altitude']['std']:.0f} ft, "
              f"heading={pred['heading']['mean']:.0f}±{pred['heading']['std']:.0f}°")

    print("\n" + "=" * 70)
    print("Test complete")
    print("=" * 70)
