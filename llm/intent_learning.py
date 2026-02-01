"""
AIDA Intent Learning Module

Collects intent observations and learns empirical priors for Bayesian inference.
This module provides:
1. Data collection - Logging intent observations to JSONL files
2. Empirical Bayes - Estimating phase-dependent priors from data
3. Prior loading - Integration with bayesian_intent.py

Training Pipeline:
    1. Run flights with logging enabled → intent_observations.jsonl
    2. Run estimate_priors() → learned_priors.json
    3. BayesianIntentInference loads learned priors on init

Usage:
    # In llm_command_server.py:
    from intent_learning import log_intent_observation, get_flight_outcome

    # After validation:
    log_intent_observation(command, telemetry, phase, validation_result)

    # To train:
    python intent_learning.py --estimate-priors
"""

import json
import os
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from enum import Enum

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "intent_observations"
OBSERVATIONS_FILE = DATA_DIR / "intent_observations.jsonl"
LEARNED_PRIORS_FILE = DATA_DIR / "learned_priors.json"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class IntentObservation:
    """Single observation for prior learning."""
    timestamp: float
    phase: str
    telemetry: dict  # heading, altitude, airspeed, vertical_rate
    command: dict    # action, value, target
    validation: dict # confidence, anomaly_score, etc.
    outcome: str     # "executed", "rejected", "anomaly_flagged"

    # Optional metadata
    session_id: Optional[str] = None
    flight_id: Optional[str] = None

    # BIRL / CBF fields (Chapter 3 & 5 of thesis)
    birl_features: Optional[list] = None       # 4-element feature vector [tracking, safety, comfort, effort]
    birl_entropy: Optional[float] = None       # BIRL posterior entropy at time of observation
    birl_weights: Optional[list] = None        # BIRL posterior mean weights
    cbf_intervened: Optional[bool] = None      # Whether CBF modified the control at this timestep

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IntentObservation":
        # Filter to only known fields (backward compat with old observations)
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# Data Collection
# =============================================================================

_current_session_id: Optional[str] = None
_current_flight_id: Optional[str] = None


def start_session(flight_id: Optional[str] = None):
    """Start a new logging session."""
    global _current_session_id, _current_flight_id
    _current_session_id = f"session_{int(time.time())}"
    _current_flight_id = flight_id or f"flight_{int(time.time())}"

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[IntentLearning] Started session {_current_session_id}, flight {_current_flight_id}")


def log_intent_observation(
    command: dict,
    telemetry: dict,
    phase: str,
    validation_result: dict,
    outcome: str = "executed",
):
    """
    Log an intent observation for later prior learning.

    Args:
        command: The flight command {action, value, target}
        telemetry: Current telemetry {heading, altitude, airspeed, ...}
        phase: Current flight phase
        validation_result: Result from bayesian_validate_intent()
        outcome: "executed", "rejected", or "anomaly_flagged"
    """
    global _current_session_id, _current_flight_id

    # Auto-start session if needed
    if _current_session_id is None:
        start_session()

    # Extract relevant telemetry fields
    telem_subset = {
        "heading": telemetry.get("heading"),
        "altitude": telemetry.get("altitude"),
        "airspeed": telemetry.get("airspeed"),
        "vertical_rate": telemetry.get("vertical_rate", 0),
    }

    # Extract relevant validation fields
    validation_subset = {
        "confidence": validation_result.get("confidence"),
        "validated": validation_result.get("validated"),
        "anomaly_score": validation_result.get("bayesian", {}).get("anomaly_score"),
        "sequence_coherence": validation_result.get("temporal", {}).get("sequence_coherence"),
    }

    obs = IntentObservation(
        timestamp=time.time(),
        phase=phase,
        telemetry=telem_subset,
        command=command,
        validation=validation_subset,
        outcome=outcome,
        session_id=_current_session_id,
        flight_id=_current_flight_id,
    )

    # Append to JSONL file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OBSERVATIONS_FILE, "a") as f:
        f.write(json.dumps(obs.to_dict()) + "\n")


def load_observations(min_timestamp: Optional[float] = None) -> List[IntentObservation]:
    """Load all observations from the JSONL file."""
    if not OBSERVATIONS_FILE.exists():
        return []

    observations = []
    with open(OBSERVATIONS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                obs = IntentObservation.from_dict(d)
                if min_timestamp is None or obs.timestamp >= min_timestamp:
                    observations.append(obs)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[IntentLearning] Skipping malformed line: {e}")

    return observations


# =============================================================================
# Empirical Prior Estimation
# =============================================================================

def normalize_heading_change(change: float) -> float:
    """Normalize heading change to [-180, 180]."""
    while change > 180:
        change -= 360
    while change < -180:
        change += 360
    return change


def estimate_von_mises_concentration(heading_changes: List[float]) -> float:
    """
    Estimate Von Mises concentration parameter κ from heading changes.

    Uses MLE approximation from Mardia & Jupp (2000).

    Args:
        heading_changes: List of heading changes in degrees

    Returns:
        Estimated κ (concentration parameter)
    """
    if len(heading_changes) < 3:
        return 5.0  # Default moderate concentration

    # Convert to radians
    angles_rad = np.deg2rad(heading_changes)

    # Compute mean resultant length R̄
    # R̄ = |mean(exp(i*θ))| measures concentration
    complex_mean = np.mean(np.exp(1j * angles_rad))
    R_bar = np.abs(complex_mean)

    # MLE approximation for κ from R̄
    # From Mardia & Jupp, Directional Statistics (2000)
    if R_bar < 0.53:
        kappa = 2 * R_bar + R_bar**3 + 5/6 * R_bar**5
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / (1 - R_bar)
    else:
        kappa = 1 / (R_bar**3 - 4*R_bar**2 + 3*R_bar + 1e-6)

    # Clamp to reasonable range
    kappa = max(0.5, min(100.0, kappa))

    return float(kappa)


def estimate_gaussian_params(values: List[float]) -> Tuple[float, float]:
    """
    Estimate Gaussian parameters (mean, std) from observed values.

    For command changes (altitude change, airspeed change), we model
    the distribution of changes, not absolute values.

    Returns:
        (mean, std) tuple
    """
    if len(values) < 3:
        return 0.0, 500.0  # Default

    mean = float(np.mean(values))
    std = float(np.std(values))

    # Minimum std to prevent overconfident priors
    std = max(std, 50.0)

    return mean, std


def estimate_phase_priors(observations: List[IntentObservation]) -> dict:
    """
    Estimate phase-dependent priors from observed command distributions.

    For each phase, fits:
    - Von Mises concentration κ for heading changes
    - Gaussian (μ, σ) for altitude changes
    - Gaussian (μ, σ) for airspeed changes

    Args:
        observations: List of IntentObservations

    Returns:
        Dictionary of phase -> prior parameters
    """
    # Group observations by phase
    phase_data = defaultdict(lambda: {
        "heading_changes": [],
        "altitude_changes": [],
        "airspeed_changes": [],
        "count": 0,
    })

    for obs in observations:
        phase = obs.phase.lower()
        cmd = obs.command
        telem = obs.telemetry

        if cmd.get("action") == "heading" and cmd.get("value") is not None:
            if telem.get("heading") is not None:
                change = normalize_heading_change(cmd["value"] - telem["heading"])
                phase_data[phase]["heading_changes"].append(change)

        elif cmd.get("action") == "altitude" and cmd.get("value") is not None:
            if telem.get("altitude") is not None:
                change = cmd["value"] - telem["altitude"]
                phase_data[phase]["altitude_changes"].append(change)

        elif cmd.get("action") == "airspeed" and cmd.get("value") is not None:
            if telem.get("airspeed") is not None:
                change = cmd["value"] - telem["airspeed"]
                phase_data[phase]["airspeed_changes"].append(change)

        phase_data[phase]["count"] += 1

    # Estimate priors for each phase
    priors = {}

    # Default priors (hand-tuned, used when insufficient data)
    default_priors = {
        "ground": {"heading_concentration": 50.0, "altitude_std": 10.0, "airspeed_std": 5.0},
        "takeoff": {"heading_concentration": 20.0, "altitude_std": 200.0, "airspeed_std": 10.0},
        "climb": {"heading_concentration": 10.0, "altitude_std": 500.0, "airspeed_std": 15.0},
        "cruise": {"heading_concentration": 5.0, "altitude_std": 300.0, "airspeed_std": 10.0},
        "descent": {"heading_concentration": 8.0, "altitude_std": 400.0, "airspeed_std": 15.0},
        "approach": {"heading_concentration": 15.0, "altitude_std": 200.0, "airspeed_std": 10.0},
        "landing": {"heading_concentration": 30.0, "altitude_std": 50.0, "airspeed_std": 5.0},
        # XC phases (from generalized_xc_controller.py)
        "initial_climb": {"heading_concentration": 15.0, "altitude_std": 300.0, "airspeed_std": 10.0},
        "cruise_climb": {"heading_concentration": 8.0, "altitude_std": 400.0, "airspeed_std": 10.0},
        "enroute": {"heading_concentration": 5.0, "altitude_std": 300.0, "airspeed_std": 10.0},
        "descent_planning": {"heading_concentration": 10.0, "altitude_std": 500.0, "airspeed_std": 15.0},
        "pattern_entry": {"heading_concentration": 12.0, "altitude_std": 200.0, "airspeed_std": 10.0},
        "downwind": {"heading_concentration": 20.0, "altitude_std": 150.0, "airspeed_std": 8.0},
        "base": {"heading_concentration": 25.0, "altitude_std": 100.0, "airspeed_std": 8.0},
        "final": {"heading_concentration": 40.0, "altitude_std": 50.0, "airspeed_std": 5.0},
        "flare": {"heading_concentration": 50.0, "altitude_std": 20.0, "airspeed_std": 3.0},
        "rollout": {"heading_concentration": 50.0, "altitude_std": 5.0, "airspeed_std": 5.0},
    }

    for phase, data in phase_data.items():
        prior = {}

        # Heading concentration
        if len(data["heading_changes"]) >= 5:
            prior["heading_concentration"] = estimate_von_mises_concentration(data["heading_changes"])
            prior["heading_concentration_source"] = "learned"
            prior["heading_sample_size"] = len(data["heading_changes"])
        else:
            default = default_priors.get(phase, default_priors["cruise"])
            prior["heading_concentration"] = default["heading_concentration"]
            prior["heading_concentration_source"] = "default"

        # Altitude std
        if len(data["altitude_changes"]) >= 5:
            _, std = estimate_gaussian_params(data["altitude_changes"])
            prior["altitude_std"] = std
            prior["altitude_std_source"] = "learned"
            prior["altitude_sample_size"] = len(data["altitude_changes"])
        else:
            default = default_priors.get(phase, default_priors["cruise"])
            prior["altitude_std"] = default["altitude_std"]
            prior["altitude_std_source"] = "default"

        # Airspeed std
        if len(data["airspeed_changes"]) >= 5:
            _, std = estimate_gaussian_params(data["airspeed_changes"])
            prior["airspeed_std"] = std
            prior["airspeed_std_source"] = "learned"
            prior["airspeed_sample_size"] = len(data["airspeed_changes"])
        else:
            default = default_priors.get(phase, default_priors["cruise"])
            prior["airspeed_std"] = default["airspeed_std"]
            prior["airspeed_std_source"] = "default"

        prior["total_observations"] = data["count"]
        priors[phase] = prior

    # Add default priors for phases with no data
    for phase, default in default_priors.items():
        if phase not in priors:
            priors[phase] = {
                "heading_concentration": default["heading_concentration"],
                "heading_concentration_source": "default",
                "altitude_std": default["altitude_std"],
                "altitude_std_source": "default",
                "airspeed_std": default["airspeed_std"],
                "airspeed_std_source": "default",
                "total_observations": 0,
            }

    return priors


def save_learned_priors(priors: dict):
    """Save learned priors to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": time.time(),
        "version": "1.0",
        "priors": priors,
    }

    with open(LEARNED_PRIORS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[IntentLearning] Saved learned priors to {LEARNED_PRIORS_FILE}")


def load_learned_priors() -> Optional[dict]:
    """Load learned priors from JSON file."""
    if not LEARNED_PRIORS_FILE.exists():
        return None

    try:
        with open(LEARNED_PRIORS_FILE, "r") as f:
            data = json.load(f)
        return data.get("priors")
    except (json.JSONDecodeError, IOError) as e:
        print(f"[IntentLearning] Failed to load priors: {e}")
        return None


# =============================================================================
# Process Noise Estimation
# =============================================================================

def estimate_process_noise(observations: List[IntentObservation]) -> dict:
    """
    Estimate process noise parameters from temporal patterns.

    Process noise represents how quickly intent uncertainty grows
    between observations. We estimate this from the variance of
    command changes over different time intervals.

    Returns:
        Dictionary of process noise parameters
    """
    # Group observations by session to compute within-session dynamics
    sessions = defaultdict(list)
    for obs in observations:
        sessions[obs.session_id].append(obs)

    # Sort each session by timestamp
    for session_id in sessions:
        sessions[session_id].sort(key=lambda x: x.timestamp)

    # Collect time deltas and value changes
    dt_heading_changes = []  # (dt, heading_change)
    dt_altitude_changes = []  # (dt, altitude_change)

    for session_id, obs_list in sessions.items():
        for i in range(1, len(obs_list)):
            prev = obs_list[i-1]
            curr = obs_list[i]

            dt = curr.timestamp - prev.timestamp
            if dt <= 0 or dt > 300:  # Skip invalid or very long gaps
                continue

            # Heading changes between consecutive commands
            if (prev.command.get("action") == "heading" and
                curr.command.get("action") == "heading" and
                prev.command.get("value") is not None and
                curr.command.get("value") is not None):
                change = abs(normalize_heading_change(
                    curr.command["value"] - prev.command["value"]
                ))
                dt_heading_changes.append((dt, change))

            # Altitude changes
            if (prev.command.get("action") == "altitude" and
                curr.command.get("action") == "altitude" and
                prev.command.get("value") is not None and
                curr.command.get("value") is not None):
                change = abs(curr.command["value"] - prev.command["value"])
                dt_altitude_changes.append((dt, change))

    # Estimate drift rates (change per second)
    process_noise = {}

    if len(dt_heading_changes) >= 5:
        dts, changes = zip(*dt_heading_changes)
        # Fit linear model: variance ~ drift_rate^2 * dt
        # Simplified: drift_rate ≈ mean(change / sqrt(dt))
        drift_rates = [c / np.sqrt(max(dt, 0.1)) for dt, c in dt_heading_changes]
        process_noise["heading_drift"] = float(np.median(drift_rates))
    else:
        process_noise["heading_drift"] = 0.5  # Default

    if len(dt_altitude_changes) >= 5:
        drift_rates = [c / np.sqrt(max(dt, 0.1)) for dt, c in dt_altitude_changes]
        process_noise["altitude_drift"] = float(np.median(drift_rates))
    else:
        process_noise["altitude_drift"] = 10.0  # Default

    # Airspeed and vertical rate - use defaults for now
    process_noise["airspeed_drift"] = 2.0
    process_noise["vertical_rate_drift"] = 50.0

    return process_noise


# =============================================================================
# Coherence Weight Optimization
# =============================================================================

def estimate_coherence_weights(observations: List[IntentObservation]) -> dict:
    """
    Estimate optimal weights for coherence computation.

    Uses the validation outcomes to find weights that best separate
    good commands (high confidence) from anomalous ones.

    Returns:
        Dictionary of coherence weights
    """
    # This is a simplified version - full optimization would use gradient descent
    # For now, analyze the correlation between different factors and confidence

    # Default weights
    weights = {
        "action": 0.2,
        "value": 0.6,
        "timing": 0.2,
    }

    # TODO: Implement weight optimization based on validation outcomes
    # This would involve:
    # 1. Computing action/value/timing coherence for each observation
    # 2. Correlating with validation confidence
    # 3. Finding weights that maximize discrimination

    return weights


# =============================================================================
# Integration with BayesianIntentInference
# =============================================================================

def apply_learned_priors_to_engine(engine):
    """
    Apply learned priors to a BayesianIntentInference engine.

    Args:
        engine: BayesianIntentInference instance
    """
    priors = load_learned_priors()
    if priors is None:
        print("[IntentLearning] No learned priors found, using defaults")
        return False

    # Store learned priors on engine for phase-dependent updates
    engine._learned_priors = priors

    # Override _update_priors_from_phase to use learned priors
    original_update = engine._update_priors_from_phase

    def learned_update():
        # First apply defaults (creates GaussianBelief objects)
        original_update()

        # Then overwrite with learned values
        phase_name = engine.phase.value.lower()
        learned = priors.get(phase_name)
        if learned is None:
            return

        if learned.get("heading_concentration_source") == "learned":
            engine.current_state.heading.concentration = learned["heading_concentration"]

        if learned.get("altitude_std_source") == "learned":
            engine.current_state.altitude.variance = learned["altitude_std"] ** 2

        if learned.get("airspeed_std_source") == "learned":
            engine.current_state.airspeed.variance = learned["airspeed_std"] ** 2

    engine._update_priors_from_phase = learned_update

    # Also patch update_from_telemetry so it uses learned concentration/variance
    # instead of the hardcoded values (concentration=10, altitude_var=200^2)
    original_telemetry_update = engine.update_from_telemetry

    def learned_telemetry_update(telemetry):
        try:
            from bayesian_intent import WrappedGaussianBelief, GaussianBelief, FlightPhase
        except ImportError:
            from llm.bayesian_intent import WrappedGaussianBelief, GaussianBelief, FlightPhase

        phase_name = engine.phase.value.lower()
        learned = priors.get(phase_name)

        if "heading" in telemetry:
            # Use learned concentration, fall back to default 10.0
            conc = 10.0
            if learned and learned.get("heading_concentration_source") == "learned":
                conc = learned["heading_concentration"]
            engine.current_state.heading = WrappedGaussianBelief(
                mean=telemetry["heading"],
                concentration=conc,
            )

        if "altitude" in telemetry:
            # Use learned std, fall back to default 200
            var = 200 ** 2
            if learned and learned.get("altitude_std_source") == "learned":
                var = learned["altitude_std"] ** 2
            engine.current_state.altitude = GaussianBelief(
                mean=telemetry["altitude"],
                variance=var,
            )

        if "airspeed" in telemetry:
            # Use learned std, fall back to default 10
            var = 10 ** 2
            if learned and learned.get("airspeed_std_source") == "learned":
                var = learned["airspeed_std"] ** 2
            engine.current_state.airspeed = GaussianBelief(
                mean=telemetry["airspeed"],
                variance=var,
            )

        if "phase" in telemetry:
            phase_str = telemetry["phase"].lower()
            for p in FlightPhase:
                if p.value in phase_str:
                    engine.set_phase(p)
                    break

    engine.update_from_telemetry = learned_telemetry_update

    print(f"[IntentLearning] Applied learned priors for {len(priors)} phases")
    return True


# =============================================================================
# Statistics and Reporting
# =============================================================================

def print_observation_stats(observations: List[IntentObservation]):
    """Print statistics about collected observations."""
    print(f"\n{'='*60}")
    print("Intent Observation Statistics")
    print(f"{'='*60}")
    print(f"Total observations: {len(observations)}")

    # By phase
    phase_counts = defaultdict(int)
    for obs in observations:
        phase_counts[obs.phase] += 1

    print(f"\nBy phase:")
    for phase, count in sorted(phase_counts.items(), key=lambda x: -x[1]):
        print(f"  {phase}: {count}")

    # By action
    action_counts = defaultdict(int)
    for obs in observations:
        action_counts[obs.command.get("action", "unknown")] += 1

    print(f"\nBy action:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count}")

    # By outcome
    outcome_counts = defaultdict(int)
    for obs in observations:
        outcome_counts[obs.outcome] += 1

    print(f"\nBy outcome:")
    for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        print(f"  {outcome}: {count}")

    # Sessions
    sessions = set(obs.session_id for obs in observations)
    print(f"\nUnique sessions: {len(sessions)}")

    # Time range
    if observations:
        timestamps = [obs.timestamp for obs in observations]
        print(f"Time range: {time.ctime(min(timestamps))} to {time.ctime(max(timestamps))}")


def print_learned_priors(priors: dict):
    """Print learned priors in a readable format."""
    print(f"\n{'='*60}")
    print("Learned Priors")
    print(f"{'='*60}")

    for phase, params in sorted(priors.items()):
        print(f"\n{phase}:")
        print(f"  Heading κ: {params['heading_concentration']:.2f} "
              f"({params.get('heading_concentration_source', 'unknown')}, "
              f"n={params.get('heading_sample_size', 0)})")
        print(f"  Altitude σ: {params['altitude_std']:.1f} ft "
              f"({params.get('altitude_std_source', 'unknown')}, "
              f"n={params.get('altitude_sample_size', 0)})")
        print(f"  Airspeed σ: {params['airspeed_std']:.1f} kts "
              f"({params.get('airspeed_std_source', 'unknown')}, "
              f"n={params.get('airspeed_sample_size', 0)})")
        print(f"  Total obs: {params.get('total_observations', 0)}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="AIDA Intent Learning")
    parser.add_argument("--estimate-priors", action="store_true",
                        help="Estimate priors from collected observations")
    parser.add_argument("--stats", action="store_true",
                        help="Print observation statistics")
    parser.add_argument("--show-priors", action="store_true",
                        help="Show learned priors")
    parser.add_argument("--min-observations", type=int, default=10,
                        help="Minimum observations required per phase")
    args = parser.parse_args()

    if args.stats or args.estimate_priors:
        observations = load_observations()
        print_observation_stats(observations)

    if args.estimate_priors:
        observations = load_observations()
        if len(observations) < 10:
            print(f"\nInsufficient observations ({len(observations)}). "
                  f"Need at least 10 to estimate priors.")
            print("Run flights with logging enabled to collect more data.")
            return

        print("\nEstimating priors...")
        priors = estimate_phase_priors(observations)
        print_learned_priors(priors)

        save_learned_priors(priors)

        print("\nEstimating process noise...")
        process_noise = estimate_process_noise(observations)
        print(f"Process noise: {process_noise}")

    if args.show_priors:
        priors = load_learned_priors()
        if priors:
            print_learned_priors(priors)
        else:
            print("No learned priors found. Run --estimate-priors first.")


if __name__ == "__main__":
    main()
