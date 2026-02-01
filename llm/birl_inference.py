"""
Bayesian Inverse Reinforcement Learning (BIRL) for Pilot Intent Inference

Implements the Policy Walk algorithm (Ramachandran & Amir, IJCAI 2007) to recover
the latent reward function driving pilot control inputs. The reward posterior
P(w|O) provides both a distribution over intents and an entropy measure H(P(w))
that quantifies ambiguity — the key input to the entropy-modulated CBF.

Reward model: R(x, u; w) = w^T * Phi(x, u)
Features:
    0: trajectory_tracking  — how well controls track commanded targets
    1: safety               — normalized margin from envelope boundaries
    2: comfort              — G-load deviation from 1g (inverted)
    3: effort               — control surface deflection magnitude (inverted)

Author: Kushal Koirala
Date: 2026
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# Feature indices
FEAT_TRACKING = 0
FEAT_SAFETY = 1
FEAT_COMFORT = 2
FEAT_EFFORT = 3
N_FEATURES = 4

FEATURE_NAMES = ["tracking", "safety", "comfort", "effort"]

# Unit conversions
M_TO_FT = 3.28084
MPS_TO_KTS = 1.94384


def extract_features(
    state: np.ndarray,
    action: np.ndarray,
    phase: str,
    target_state: dict,
    limits: Optional[dict] = None,
) -> np.ndarray:
    """
    Extract the 4-dimensional feature vector Phi(x, u) from state and control.

    Args:
        state: 12-element [x,y,z,u,v,w,phi,theta,psi,p,q,r] in meters/radians
        action: 7-element [throttle, aileron, elevator, rudder, flap, spoiler, brake]
        phase: Current flight phase name (e.g. 'CRUISE_TO_TP')
        target_state: dict with 'altitude_ft', 'heading_deg', 'airspeed_kts'
        limits: Aircraft envelope limits (optional, uses defaults)

    Returns:
        4-element feature vector, each in [0, 1]
    """
    if limits is None:
        limits = {
            "min_altitude_ft": 0.0,
            "max_altitude_ft": 14000.0,
            "min_airspeed_kts": 48.0,
            "max_airspeed_kts": 163.0,
        }

    # Current state in imperial
    alt_ft = -state[2] * M_TO_FT  # z is down in NED
    airspeed_mps = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2)
    airspeed_kts = airspeed_mps * MPS_TO_KTS
    heading_deg = np.degrees(state[8]) % 360.0
    phi = state[6]  # roll/bank angle

    # Targets
    tgt_alt = target_state.get("altitude_ft", 5500.0)
    tgt_hdg = target_state.get("heading_deg", heading_deg)
    tgt_spd = target_state.get("airspeed_kts", 110.0)

    # --- Feature 0: Trajectory Tracking ---
    alt_err = abs(alt_ft - tgt_alt) / 1000.0  # normalized by 1000ft
    hdg_err_raw = (heading_deg - tgt_hdg + 180.0) % 360.0 - 180.0
    hdg_err = abs(hdg_err_raw) / 45.0  # normalized by 45 deg
    spd_err = abs(airspeed_kts - tgt_spd) / 30.0  # normalized by 30 kts
    tracking = 1.0 - np.clip((alt_err + hdg_err + spd_err) / 3.0, 0.0, 1.0)

    # --- Feature 1: Safety (min margin from envelope) ---
    alt_floor_margin = alt_ft / 500.0  # 0 at ground, 1 at 500ft
    alt_ceil_margin = (limits["max_altitude_ft"] - alt_ft) / limits["max_altitude_ft"]
    spd_low_margin = (airspeed_kts - limits["min_airspeed_kts"]) / 20.0
    spd_high_margin = (limits["max_airspeed_kts"] - airspeed_kts) / 30.0
    safety = np.clip(min(alt_floor_margin, alt_ceil_margin, spd_low_margin, spd_high_margin), 0.0, 1.0)

    # --- Feature 2: Comfort (G-load from bank angle) ---
    cos_phi = np.cos(phi)
    if abs(cos_phi) > 0.01:
        load_factor = 1.0 / cos_phi
    else:
        load_factor = 10.0  # extreme bank
    comfort = 1.0 - np.clip(abs(load_factor - 1.0) / 2.0, 0.0, 1.0)

    # --- Feature 3: Effort (control deflection magnitude) ---
    surfaces = action[1:4]  # aileron, elevator, rudder
    rms_deflection = np.sqrt(np.mean(surfaces**2))
    effort = 1.0 - np.clip(rms_deflection, 0.0, 1.0)

    return np.array([tracking, safety, comfort, effort], dtype=np.float64)


def _project_simplex(w: np.ndarray) -> np.ndarray:
    """Project weight vector onto the probability simplex (non-negative, sum to 1)."""
    # Softmax projection ensures non-negative and sums to 1
    w_shifted = w - np.max(w)  # numerical stability
    exp_w = np.exp(w_shifted)
    return exp_w / np.sum(exp_w)


@dataclass
class BIRLPosteriorSample:
    """A single MCMC sample from the reward posterior."""
    weights: np.ndarray
    log_probability: float


class BIRLInference:
    """
    Bayesian Inverse Reinforcement Learning using MCMC (Policy Walk).

    Maintains a posterior distribution P(w | O) over reward weights w,
    where O is a set of observed (state, action) feature vectors from the
    expert (pilot/controller). The entropy of this posterior quantifies
    how ambiguous the inferred intent is.

    The posterior is computed via Metropolis-Hastings:
        P(w | O) proportional to P(O | w) * P(w)
    where:
        P(O | w) = prod_i exp(beta * w^T * phi_i) / Z(w)
        P(w) = N(w; prior_mean, prior_std^2 I)
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_samples: int = 200,
        n_burnin: int = 50,
        step_size: float = 0.1,
        beta: float = 5.0,
        prior_mean: Optional[np.ndarray] = None,
        prior_std: float = 1.0,
    ):
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.step_size = step_size
        self.beta = beta
        self.prior_std = prior_std

        # Default prior: slightly favoring tracking and safety
        if prior_mean is not None:
            self.prior_mean = np.asarray(prior_mean, dtype=np.float64)
        else:
            self.prior_mean = np.array([0.30, 0.40, 0.15, 0.15])

        # Current state
        self.current_weights = self.prior_mean.copy()
        self.weight_samples: List[np.ndarray] = []
        self.observation_buffer: deque = deque(maxlen=500)

        # Cached posterior statistics
        self._posterior_mean = self.prior_mean.copy()
        self._posterior_entropy = 1.0  # Start at max entropy (no info)
        self._posterior_std = np.ones(n_features) * 0.25

        # Diagnostics
        self.total_updates = 0
        self.acceptance_rate = 0.0

    def add_observation(self, features: np.ndarray):
        """Add a feature vector observation to the buffer."""
        self.observation_buffer.append(np.asarray(features, dtype=np.float64))

    def _log_likelihood(self, weights: np.ndarray, observations: np.ndarray) -> float:
        """
        Compute log P(O | w) under Boltzmann rationality.

        P(action | state, w) proportional to exp(beta * w^T * phi(s, a))
        Log-likelihood = sum_i beta * w^T * phi_i
        (The partition function Z(w) cancels in the MH ratio for same observation set.)
        """
        # observations: (N, n_features)
        rewards = observations @ weights  # (N,)
        return self.beta * np.sum(rewards)

    def _log_prior(self, weights: np.ndarray) -> float:
        """Log P(w) under Gaussian prior."""
        diff = weights - self.prior_mean
        return -0.5 * np.sum(diff**2) / (self.prior_std**2)

    def _log_posterior(self, weights: np.ndarray, observations: np.ndarray) -> float:
        """Log P(w | O) proportional to log P(O | w) + log P(w)."""
        return self._log_likelihood(weights, observations) + self._log_prior(weights)

    def update_posterior(self) -> float:
        """
        Run Metropolis-Hastings MCMC to sample from P(w | observations).

        Returns:
            Current posterior entropy H(P(w)) in [0, 1].
        """
        if len(self.observation_buffer) < 10:
            return self._posterior_entropy

        observations = np.array(list(self.observation_buffer))  # (N, 4)

        # Initialize from current best
        w_current = self.current_weights.copy()
        log_p_current = self._log_posterior(w_current, observations)

        samples = []
        n_accepted = 0

        for i in range(self.n_samples + self.n_burnin):
            # Propose: perturb in unconstrained space, then project to simplex
            w_proposal_raw = w_current + np.random.randn(self.n_features) * self.step_size
            w_proposal = _project_simplex(w_proposal_raw)

            log_p_proposal = self._log_posterior(w_proposal, observations)

            # Metropolis-Hastings acceptance
            log_alpha = log_p_proposal - log_p_current
            if np.log(np.random.rand()) < log_alpha:
                w_current = w_proposal
                log_p_current = log_p_proposal
                n_accepted += 1

            # Store after burn-in
            if i >= self.n_burnin:
                samples.append(w_current.copy())

        self.weight_samples = samples
        self.current_weights = w_current
        self.acceptance_rate = n_accepted / (self.n_samples + self.n_burnin)

        # Compute posterior statistics
        samples_arr = np.array(samples)  # (n_samples, n_features)
        self._posterior_mean = np.mean(samples_arr, axis=0)
        self._posterior_std = np.std(samples_arr, axis=0)

        # Compute entropy
        self._posterior_entropy = self._compute_entropy(samples_arr)

        self.total_updates += 1
        return self._posterior_entropy

    def _compute_entropy(self, samples: np.ndarray) -> float:
        """
        Compute normalized entropy of the posterior from MCMC samples.

        Uses differential entropy approximation via average marginal variance.
        Normalized to [0, 1] where 1 = maximum ambiguity (uniform on simplex).

        For a Gaussian approximation: H = 0.5 * ln(det(2*pi*e*Sigma))
        Normalized by H_max = 0.5 * n * ln(2*pi*e * sigma_max^2)
        """
        if len(samples) < 10:
            return 1.0

        # Covariance of posterior samples
        cov = np.cov(samples.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])

        # Differential entropy of multivariate Gaussian
        det_cov = np.linalg.det(cov)
        if det_cov <= 0:
            # Degenerate — use product of marginal variances
            marginal_vars = np.var(samples, axis=0)
            det_cov = np.prod(np.maximum(marginal_vars, 1e-10))

        n = self.n_features
        h_posterior = 0.5 * np.log(np.maximum(det_cov, 1e-30)) + 0.5 * n * np.log(2 * np.pi * np.e)

        # Maximum entropy on simplex: uniform distribution
        # For a Dirichlet(1,...,1) on n-simplex: H_max ~ (n-1) * 0.5 * ln(2*pi*e / n)
        sigma_max = 1.0 / np.sqrt(self.n_features)  # std of uniform on simplex
        h_max = 0.5 * n * np.log(2 * np.pi * np.e * sigma_max**2)

        # Normalize to [0, 1]
        if h_max <= 0:
            return 1.0

        entropy_normalized = np.clip(h_posterior / h_max, 0.0, 1.0)
        return float(entropy_normalized)

    def get_entropy(self) -> float:
        """Get cached posterior entropy in [0, 1]. 1 = max ambiguity."""
        return self._posterior_entropy

    def get_posterior_mean(self) -> np.ndarray:
        """Get posterior mean reward weights."""
        return self._posterior_mean.copy()

    def get_posterior_std(self) -> np.ndarray:
        """Get posterior std of reward weights."""
        return self._posterior_std.copy()

    def get_dominant_intent(self) -> str:
        """Get the feature with highest posterior weight."""
        return FEATURE_NAMES[int(np.argmax(self._posterior_mean))]

    def get_reward_posterior_summary(self) -> dict:
        """
        Get a summary of the current reward posterior for integration
        with the Bayesian intent validation pipeline.
        """
        return {
            "reward_weights": self._posterior_mean.tolist(),
            "reward_std": self._posterior_std.tolist(),
            "entropy": float(self._posterior_entropy),
            "dominant_intent": self.get_dominant_intent(),
            "n_observations": len(self.observation_buffer),
            "n_samples": len(self.weight_samples),
            "acceptance_rate": float(self.acceptance_rate),
            "total_updates": self.total_updates,
        }

    def warmstart_from_observations(self, observations: List[np.ndarray]):
        """
        Initialize the posterior from a batch of historical observations.
        Useful for starting with data from the training pipeline.
        """
        for obs in observations:
            self.add_observation(obs)
        if len(self.observation_buffer) >= 10:
            self.update_posterior()

    def reset(self):
        """Reset to prior state (start of new flight)."""
        self.current_weights = self.prior_mean.copy()
        self.weight_samples = []
        self.observation_buffer.clear()
        self._posterior_mean = self.prior_mean.copy()
        self._posterior_entropy = 1.0
        self._posterior_std = np.ones(self.n_features) * 0.25
        self.total_updates = 0
        self.acceptance_rate = 0.0
