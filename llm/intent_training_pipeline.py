#!/usr/bin/env python3
"""
AIDA Intent Training Pipeline

Automated training pipeline that:
1. Runs multiple simulated flights in parallel (GPU-accelerated when available)
2. Collects intent observations from each flight
3. Periodically estimates priors from accumulated data
4. Saves checkpoints and training metrics

Leverages existing AIDA infrastructure:
- FlightEnvRL for physics simulation
- GPU acceleration via PyTorch
- Parallel environments with vectorized operations

Usage:
    # Run 10 parallel flights, train after every 100 observations
    python intent_training_pipeline.py --parallel 10 --train-every 100

    # Run until 1000 observations collected
    python intent_training_pipeline.py --target-observations 1000

    # Quick test with 3 flights
    python intent_training_pipeline.py --parallel 3 --flights-per-worker 2

    # Use real sim environment
    python intent_training_pipeline.py --use-sim --parallel 4
"""

import asyncio
import json
import random
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))

# Check for GPU/PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"[Pipeline] GPU available: {torch.cuda.get_device_name(0)}")
        print(f"[Pipeline] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    print("[Pipeline] PyTorch not available, using CPU simulation")

# Try to import the real flight environment
try:
    from aida_sim.env.flight_env_rl import FlightEnvRL
    SIM_ENV_AVAILABLE = True
    print("[Pipeline] AIDA FlightEnvRL available for realistic simulation")
except ImportError:
    SIM_ENV_AVAILABLE = False
    print("[Pipeline] FlightEnvRL not available, using simplified simulation")

from intent_learning import (
    IntentObservation,
    log_intent_observation,
    load_observations,
    estimate_phase_priors,
    estimate_process_noise,
    save_learned_priors,
    load_learned_priors,
    apply_learned_priors_to_engine,
    print_observation_stats,
    print_learned_priors,
    DATA_DIR,
    OBSERVATIONS_FILE,
)

from bayesian_intent import (
    bayesian_validate_intent,
    reset_inference_engine,
    get_inference_engine,
    FlightPhase,
)


# =============================================================================
# Flight Simulation Scenarios
# =============================================================================

@dataclass
class FlightScenario:
    """Defines a simulated flight scenario."""
    name: str
    phases: List[str]
    commands: List[dict]  # List of {action, value, delay}
    initial_telemetry: dict


# Realistic flight scenarios based on typical GA operations
FLIGHT_SCENARIOS = [
    # Pattern work - multiple touch and go
    FlightScenario(
        name="pattern_work",
        phases=["takeoff", "climb", "cruise", "descent", "approach", "landing"] * 3,
        commands=[
            # Takeoff and initial climb
            {"action": "altitude", "value": 1500, "delay": 2},
            {"action": "heading", "value": 270, "delay": 5},  # Crosswind
            {"action": "altitude", "value": 2000, "delay": 3},
            {"action": "heading", "value": 180, "delay": 5},  # Downwind
            {"action": "altitude", "value": 2000, "delay": 2},
            {"action": "heading", "value": 90, "delay": 5},   # Base
            {"action": "altitude", "value": 1500, "delay": 3},
            {"action": "heading", "value": 0, "delay": 5},    # Final
            {"action": "altitude", "value": 1000, "delay": 3},
            {"action": "altitude", "value": 500, "delay": 3},
            # Touch and go, repeat
            {"action": "altitude", "value": 1500, "delay": 2},
            {"action": "heading", "value": 270, "delay": 5},
            {"action": "altitude", "value": 2000, "delay": 3},
            {"action": "heading", "value": 180, "delay": 5},
        ],
        initial_telemetry={"heading": 0, "altitude": 1000, "airspeed": 65, "vertical_rate": 500},
    ),

    # Cross-country with multiple waypoints
    FlightScenario(
        name="cross_country",
        phases=["takeoff", "climb", "climb", "cruise", "cruise", "cruise", "descent", "approach", "landing"],
        commands=[
            {"action": "altitude", "value": 3500, "delay": 3},
            {"action": "heading", "value": 45, "delay": 5},
            {"action": "altitude", "value": 5500, "delay": 5},
            {"action": "heading", "value": 90, "delay": 8},
            {"action": "altitude", "value": 6500, "delay": 5},
            {"action": "heading", "value": 120, "delay": 10},
            {"action": "altitude", "value": 6500, "delay": 3},
            {"action": "heading", "value": 150, "delay": 8},
            {"action": "altitude", "value": 5000, "delay": 5},
            {"action": "heading", "value": 180, "delay": 5},
            {"action": "altitude", "value": 3500, "delay": 5},
            {"action": "heading", "value": 200, "delay": 5},
            {"action": "altitude", "value": 2000, "delay": 5},
        ],
        initial_telemetry={"heading": 0, "altitude": 1000, "airspeed": 80, "vertical_rate": 700},
    ),

    # Steep turns practice (lots of heading changes)
    FlightScenario(
        name="steep_turns",
        phases=["cruise"] * 10,
        commands=[
            {"action": "heading", "value": 90, "delay": 3},
            {"action": "heading", "value": 180, "delay": 3},
            {"action": "heading", "value": 270, "delay": 3},
            {"action": "heading", "value": 0, "delay": 3},
            {"action": "heading", "value": 270, "delay": 3},  # Reverse
            {"action": "heading", "value": 180, "delay": 3},
            {"action": "heading", "value": 90, "delay": 3},
            {"action": "heading", "value": 0, "delay": 3},
            {"action": "altitude", "value": 4500, "delay": 2},  # Minor altitude correction
            {"action": "heading", "value": 45, "delay": 3},
        ],
        initial_telemetry={"heading": 0, "altitude": 4500, "airspeed": 100, "vertical_rate": 0},
    ),

    # Slow flight and stall practice
    FlightScenario(
        name="slow_flight",
        phases=["cruise", "cruise", "cruise", "climb", "cruise"],
        commands=[
            {"action": "altitude", "value": 5000, "delay": 3},
            {"action": "airspeed", "value": 70, "delay": 5},
            {"action": "heading", "value": 30, "delay": 4},
            {"action": "airspeed", "value": 60, "delay": 5},
            {"action": "heading", "value": 60, "delay": 4},
            {"action": "altitude", "value": 5500, "delay": 3},  # Recovery climb
            {"action": "airspeed", "value": 90, "delay": 3},
            {"action": "heading", "value": 90, "delay": 4},
        ],
        initial_telemetry={"heading": 0, "altitude": 5000, "airspeed": 110, "vertical_rate": 0},
    ),

    # Emergency descent
    FlightScenario(
        name="emergency_descent",
        phases=["cruise", "descent", "descent", "descent", "approach"],
        commands=[
            {"action": "altitude", "value": 6000, "delay": 2},
            {"action": "altitude", "value": 4000, "delay": 3},
            {"action": "heading", "value": 180, "delay": 2},
            {"action": "altitude", "value": 2500, "delay": 3},
            {"action": "altitude", "value": 1500, "delay": 3},
            {"action": "heading", "value": 200, "delay": 2},
            {"action": "altitude", "value": 1000, "delay": 3},
        ],
        initial_telemetry={"heading": 0, "altitude": 8000, "airspeed": 120, "vertical_rate": 0},
    ),

    # Holding pattern
    FlightScenario(
        name="holding_pattern",
        phases=["cruise"] * 8,
        commands=[
            {"action": "heading", "value": 90, "delay": 5},
            {"action": "heading", "value": 180, "delay": 3},
            {"action": "heading", "value": 270, "delay": 5},
            {"action": "heading", "value": 0, "delay": 3},
            {"action": "heading", "value": 90, "delay": 5},
            {"action": "heading", "value": 180, "delay": 3},
            {"action": "altitude", "value": 5000, "delay": 2},  # Altitude change during hold
            {"action": "heading", "value": 270, "delay": 5},
        ],
        initial_telemetry={"heading": 0, "altitude": 5500, "airspeed": 100, "vertical_rate": 0},
    ),

    # Approach with go-around
    FlightScenario(
        name="go_around",
        phases=["descent", "approach", "approach", "climb", "cruise", "descent", "approach", "landing"],
        commands=[
            {"action": "altitude", "value": 3000, "delay": 3},
            {"action": "heading", "value": 270, "delay": 4},
            {"action": "altitude", "value": 2000, "delay": 3},
            {"action": "altitude", "value": 1500, "delay": 3},
            # Go around!
            {"action": "altitude", "value": 2500, "delay": 2},
            {"action": "heading", "value": 270, "delay": 3},
            {"action": "altitude", "value": 3000, "delay": 3},
            {"action": "heading", "value": 180, "delay": 4},
            # Second approach
            {"action": "altitude", "value": 2000, "delay": 3},
            {"action": "heading", "value": 270, "delay": 4},
            {"action": "altitude", "value": 1000, "delay": 3},
        ],
        initial_telemetry={"heading": 90, "altitude": 4000, "airspeed": 100, "vertical_rate": -500},
    ),
]


# =============================================================================
# Telemetry Simulation
# =============================================================================

def simulate_telemetry_update(current: dict, command: dict, dt: float) -> dict:
    """
    Simulate how telemetry changes in response to a command.

    This is a simplified model - real telemetry would come from the sim.
    """
    new_telem = current.copy()

    action = command.get("action")
    value = command.get("value")

    if action == "heading" and value is not None:
        # Gradually turn toward target heading
        current_hdg = current.get("heading", 0)
        diff = value - current_hdg
        while diff > 180: diff -= 360
        while diff < -180: diff += 360

        # Turn rate ~3 deg/sec standard rate
        max_change = 3.0 * dt
        actual_change = np.clip(diff, -max_change, max_change)
        new_telem["heading"] = (current_hdg + actual_change) % 360

    elif action == "altitude" and value is not None:
        # Climb/descend toward target
        current_alt = current.get("altitude", 5000)
        diff = value - current_alt

        # Climb rate ~500-700 fpm, descent ~700-1000 fpm
        if diff > 0:
            rate = min(700, diff / (dt / 60))
        else:
            rate = max(-1000, diff / (dt / 60))

        new_telem["altitude"] = current_alt + rate * (dt / 60)
        new_telem["vertical_rate"] = rate

    elif action == "airspeed" and value is not None:
        # Adjust airspeed
        current_spd = current.get("airspeed", 100)
        diff = value - current_spd

        # Acceleration ~2-5 kts/sec
        max_change = 3.0 * dt
        actual_change = np.clip(diff, -max_change, max_change)
        new_telem["airspeed"] = current_spd + actual_change

    return new_telem


def add_telemetry_noise(telem: dict) -> dict:
    """Add realistic noise to telemetry readings."""
    noisy = telem.copy()

    # Heading: ±1 degree noise
    if "heading" in noisy:
        noisy["heading"] = (noisy["heading"] + random.gauss(0, 1)) % 360

    # Altitude: ±20 ft noise
    if "altitude" in noisy:
        noisy["altitude"] = noisy["altitude"] + random.gauss(0, 20)

    # Airspeed: ±2 kt noise
    if "airspeed" in noisy:
        noisy["airspeed"] = noisy["airspeed"] + random.gauss(0, 2)

    # Vertical rate: ±50 fpm noise
    if "vertical_rate" in noisy:
        noisy["vertical_rate"] = noisy["vertical_rate"] + random.gauss(0, 50)

    return noisy


# =============================================================================
# Flight Worker
# =============================================================================

def run_single_flight(
    scenario: FlightScenario,
    flight_id: str,
    add_noise: bool = True,
    add_variations: bool = True,
) -> List[IntentObservation]:
    """
    Run a single simulated flight and collect observations.

    Args:
        scenario: Flight scenario to simulate
        flight_id: Unique identifier for this flight
        add_noise: Add telemetry noise
        add_variations: Add random variations to commands

    Returns:
        List of IntentObservation objects
    """
    observations = []

    # Reset the Bayesian engine for this flight
    reset_inference_engine()

    # Initialize telemetry
    telemetry = scenario.initial_telemetry.copy()

    # Run through commands
    phase_idx = 0

    for i, cmd_template in enumerate(scenario.commands):
        # Get current phase
        if phase_idx < len(scenario.phases):
            phase = scenario.phases[phase_idx]
        else:
            phase = scenario.phases[-1]

        # Add variations to command
        cmd = cmd_template.copy()
        if add_variations:
            if cmd["action"] == "heading":
                cmd["value"] = (cmd["value"] + random.gauss(0, 5)) % 360
            elif cmd["action"] == "altitude":
                cmd["value"] = cmd["value"] + random.gauss(0, 100)
            elif cmd["action"] == "airspeed":
                cmd["value"] = cmd["value"] + random.gauss(0, 3)

        # Add noise to telemetry
        telem_for_validation = add_telemetry_noise(telemetry) if add_noise else telemetry

        # Validate intent
        result = bayesian_validate_intent(
            action=cmd["action"],
            value=cmd["value"],
            target=cmd.get("target"),
            telemetry=telem_for_validation,
            phase=phase,
            use_stateful=True,
        )

        # Determine outcome
        if not result.get("validated", True):
            outcome = "anomaly_flagged"
        elif result.get("confidence", 1.0) < 0.3:
            outcome = "low_confidence"
        else:
            outcome = "executed"

        # Create observation
        obs = IntentObservation(
            timestamp=time.time() + i * cmd.get("delay", 1),  # Simulated time
            phase=phase,
            telemetry={
                "heading": telem_for_validation.get("heading"),
                "altitude": telem_for_validation.get("altitude"),
                "airspeed": telem_for_validation.get("airspeed"),
                "vertical_rate": telem_for_validation.get("vertical_rate", 0),
            },
            command={
                "action": cmd["action"],
                "value": cmd["value"],
                "target": cmd.get("target"),
            },
            validation={
                "confidence": result.get("confidence"),
                "validated": result.get("validated"),
                "anomaly_score": result.get("bayesian", {}).get("anomaly_score"),
                "sequence_coherence": result.get("temporal", {}).get("sequence_coherence"),
            },
            outcome=outcome,
            session_id=f"auto_train_{int(time.time())}",
            flight_id=flight_id,
        )
        observations.append(obs)

        # Update telemetry based on command
        telemetry = simulate_telemetry_update(telemetry, cmd, cmd.get("delay", 1))

        # Advance phase occasionally
        if random.random() < 0.3:
            phase_idx = min(phase_idx + 1, len(scenario.phases) - 1)

    return observations


def worker_run_flights(
    worker_id: int,
    num_flights: int,
    scenarios: List[dict],
) -> List[dict]:
    """
    Worker function that runs multiple flights.

    Args:
        worker_id: Unique worker identifier
        num_flights: Number of flights to run
        scenarios: List of scenario dicts (serialized FlightScenario)

    Returns:
        List of observation dicts
    """
    all_observations = []

    # Reconstruct scenarios from dicts
    flight_scenarios = []
    for s in scenarios:
        flight_scenarios.append(FlightScenario(**s))

    for i in range(num_flights):
        # Pick random scenario
        scenario = random.choice(flight_scenarios)
        flight_id = f"worker{worker_id}_flight{i}_{int(time.time())}"

        try:
            obs_list = run_single_flight(scenario, flight_id)
            all_observations.extend([obs.to_dict() for obs in obs_list])
        except Exception as e:
            print(f"[Worker {worker_id}] Flight {i} failed: {e}")

    return all_observations


# =============================================================================
# Training Pipeline
# =============================================================================

class IntentTrainingPipeline:
    """
    Manages the automated training pipeline.
    """

    def __init__(
        self,
        parallel_workers: int = 4,
        flights_per_worker: int = 5,
        train_every: int = 100,
        target_observations: int = 1000,
        checkpoint_dir: Optional[Path] = None,
        use_sim: bool = False,
        device: str = "cpu",
        use_gpu: bool = False,
    ):
        self.parallel_workers = parallel_workers
        self.flights_per_worker = flights_per_worker
        self.train_every = train_every
        self.target_observations = target_observations
        self.checkpoint_dir = checkpoint_dir or DATA_DIR / "checkpoints"
        self.use_sim = use_sim and SIM_ENV_AVAILABLE
        self.device = device
        self.use_gpu = use_gpu

        # Training state
        self.total_observations = 0
        self.training_rounds = 0
        self.metrics_history = []

        # Ensure directories exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Serialize scenarios for multiprocessing
        self.scenarios_serialized = [asdict(s) for s in FLIGHT_SCENARIOS]

        # PPO model path
        self.ppo_model_path = None

        # Initialize sim runner if using real simulation
        self.sim_runner = None
        if self.use_sim:
            print(f"[Pipeline] Initializing SimEnvFlightRunner on {device}")
            self.sim_runner = SimEnvFlightRunner(
                num_envs=parallel_workers,
                device=device,
                ppo_model_path=self.ppo_model_path,
            )

    def run(self):
        """Run the full training pipeline."""
        print("=" * 70)
        print("AIDA Intent Training Pipeline")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Parallel workers: {self.parallel_workers}")
        print(f"  Flights per worker: {self.flights_per_worker}")
        print(f"  Train every: {self.train_every} observations")
        print(f"  Target observations: {self.target_observations}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Use sim environment: {self.use_sim}")
        print(f"  Device: {self.device}")
        if TORCH_AVAILABLE and self.device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 70)

        start_time = time.time()

        # Load existing observations
        existing_obs = load_observations()
        self.total_observations = len(existing_obs)
        print(f"\nExisting observations: {self.total_observations}")

        batch_num = 0

        while self.total_observations < self.target_observations:
            batch_num += 1
            print(f"\n{'─' * 50}")
            print(f"Batch {batch_num}: Running {self.parallel_workers} parallel workers...")

            # Run parallel flights
            new_observations = self._run_parallel_flights()

            # Save observations
            self._save_observations(new_observations)
            self.total_observations += len(new_observations)

            print(f"  Collected {len(new_observations)} new observations")
            print(f"  Total: {self.total_observations}/{self.target_observations}")

            # Train if threshold reached
            if self.total_observations >= (self.training_rounds + 1) * self.train_every:
                self._train_priors()
                self.training_rounds += 1

            # Progress update
            elapsed = time.time() - start_time
            rate = self.total_observations / elapsed if elapsed > 0 else 0
            remaining = (self.target_observations - self.total_observations) / rate if rate > 0 else 0
            print(f"  Rate: {rate:.1f} obs/sec, ETA: {remaining:.0f}s")

        # Final training
        print(f"\n{'=' * 70}")
        print("Final Training")
        print("=" * 70)
        self._train_priors()

        # Summary
        total_time = time.time() - start_time
        print(f"\n{'=' * 70}")
        print("Training Complete!")
        print("=" * 70)
        print(f"Total observations: {self.total_observations}")
        print(f"Training rounds: {self.training_rounds}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Observations per second: {self.total_observations / total_time:.1f}")

        # Print final priors
        from intent_learning import load_learned_priors
        priors = load_learned_priors()
        if priors:
            print_learned_priors(priors)

    def _run_parallel_flights(self) -> List[IntentObservation]:
        """Run flights in parallel using GPU batched, CPU ProcessPool, or scenario sim."""

        # Use sim environment if available and requested
        if self.use_sim and self.sim_runner is not None:
            num_flights = self.parallel_workers * self.flights_per_worker
            # GPU-batched: all flights in single CUDA kernel
            if self.use_gpu:
                return self.sim_runner.run_gpu_batched_flights(num_flights)
            # CPU parallel: ProcessPoolExecutor
            return self.sim_runner.run_parallel_flights(num_flights)

        # Fall back to simplified scenario-based simulation with multiprocessing
        all_observations = []

        # Use multiprocessing for true parallelism
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = []
            for worker_id in range(self.parallel_workers):
                future = executor.submit(
                    worker_run_flights,
                    worker_id,
                    self.flights_per_worker,
                    self.scenarios_serialized,
                )
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                try:
                    obs_dicts = future.result()
                    observations = [IntentObservation.from_dict(d) for d in obs_dicts]
                    all_observations.extend(observations)
                except Exception as e:
                    print(f"  Worker failed: {e}")

        return all_observations

    def _save_observations(self, observations: List[IntentObservation]):
        """Append observations to the JSONL file."""
        with open(OBSERVATIONS_FILE, "a") as f:
            for obs in observations:
                f.write(json.dumps(obs.to_dict()) + "\n")

    def _train_priors(self):
        """Estimate priors from all observations."""
        print(f"\n  Training priors (round {self.training_rounds + 1})...")

        # Load all observations
        observations = load_observations()

        if len(observations) < 10:
            print(f"  Skipping - insufficient observations ({len(observations)})")
            return

        # Estimate priors
        priors = estimate_phase_priors(observations)
        process_noise = estimate_process_noise(observations)

        # Save priors
        save_learned_priors(priors)

        # Save checkpoint
        checkpoint_file = self.checkpoint_dir / f"priors_round_{self.training_rounds + 1}.json"
        with open(checkpoint_file, "w") as f:
            json.dump({
                "round": self.training_rounds + 1,
                "timestamp": time.time(),
                "num_observations": len(observations),
                "priors": priors,
                "process_noise": process_noise,
            }, f, indent=2)

        # Compute and save metrics
        metrics = self._compute_metrics(observations, priors)
        self.metrics_history.append(metrics)

        # Save metrics history
        metrics_file = self.checkpoint_dir / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        print(f"  Saved checkpoint: {checkpoint_file.name}")
        print(f"  Learned priors for {len(priors)} phases")

        # Print summary
        learned_count = sum(
            1 for p in priors.values()
            if p.get("heading_concentration_source") == "learned"
        )
        print(f"  Phases with learned heading priors: {learned_count}/{len(priors)}")

    def _compute_metrics(self, observations: List[IntentObservation], priors: dict) -> dict:
        """Compute training metrics."""
        # Count by outcome
        outcome_counts = {}
        for obs in observations:
            outcome_counts[obs.outcome] = outcome_counts.get(obs.outcome, 0) + 1

        # Count by phase
        phase_counts = {}
        for obs in observations:
            phase_counts[obs.phase] = phase_counts.get(obs.phase, 0) + 1

        # Compute average confidence
        confidences = [
            obs.validation.get("confidence", 0)
            for obs in observations
            if obs.validation.get("confidence") is not None
        ]
        avg_confidence = np.mean(confidences) if confidences else 0

        # Count learned vs default priors
        learned_heading = sum(
            1 for p in priors.values()
            if p.get("heading_concentration_source") == "learned"
        )
        learned_altitude = sum(
            1 for p in priors.values()
            if p.get("altitude_std_source") == "learned"
        )

        return {
            "round": self.training_rounds + 1,
            "timestamp": time.time(),
            "total_observations": len(observations),
            "outcome_counts": outcome_counts,
            "phase_counts": phase_counts,
            "avg_confidence": float(avg_confidence),
            "learned_heading_phases": learned_heading,
            "learned_altitude_phases": learned_altitude,
            "total_phases": len(priors),
        }


# =============================================================================
# GPU-Accelerated Sim Flight Runner (Expert + PPO Policy)
# =============================================================================

# XCPhase -> Bayesian intent phase mapping
XCPHASE_TO_INTENT = {
    "GROUND_ROLL": "ground",
    "ROTATION": "takeoff",
    "INITIAL_CLIMB": "initial_climb",
    "CLIMB": "climb",
    "CRUISE_TO_TP": "cruise",
    "TURN_TO_INTERCEPT": "approach",
    "INTERCEPT_LEG": "approach",
    "FINAL_APPROACH": "final",
    "SHORT_FINAL": "final",
    "FLARE": "flare",
    "ROLLOUT": "rollout",
    "LANDING": "landing",
    "LANDED": "ground",
}

# Phase-appropriate commands: what a pilot would realistically command in each phase
PHASE_COMMAND_TEMPLATES = {
    "GROUND_ROLL": [
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-2, 2)) % 360},
    ],
    "ROTATION": [
        {"action": "altitude", "value_fn": lambda t: t["altitude"] + random.uniform(500, 1500)},
    ],
    "INITIAL_CLIMB": [
        {"action": "altitude", "value_fn": lambda t: t["altitude"] + random.uniform(500, 2000)},
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-10, 10)) % 360},
    ],
    "CLIMB": [
        {"action": "altitude", "value_fn": lambda t: t["altitude"] + random.uniform(1000, 3000)},
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-30, 30)) % 360},
    ],
    "CRUISE_TO_TP": [
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-45, 45)) % 360},
        {"action": "altitude", "value_fn": lambda t: t["altitude"] + random.uniform(-200, 200)},
    ],
    "TURN_TO_INTERCEPT": [
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-90, 90)) % 360},
    ],
    "INTERCEPT_LEG": [
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-15, 15)) % 360},
        {"action": "altitude", "value_fn": lambda t: t["altitude"] - random.uniform(200, 800)},
    ],
    "FINAL_APPROACH": [
        {"action": "altitude", "value_fn": lambda t: t["altitude"] - random.uniform(100, 500)},
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-5, 5)) % 360},
    ],
    "SHORT_FINAL": [
        {"action": "altitude", "value_fn": lambda t: max(0, t["altitude"] - random.uniform(50, 200))},
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-3, 3)) % 360},
    ],
    "FLARE": [
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-2, 2)) % 360},
    ],
    "ROLLOUT": [
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-2, 2)) % 360},
    ],
    "LANDING": [
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-1, 1)) % 360},
    ],
    "LANDED": [
        {"action": "heading", "value_fn": lambda t: (t["heading"] + random.uniform(-1, 1)) % 360},
    ],
}


class SimEnvFlightRunner:
    """
    Run realistic flights using the AIDA simulation + Generalized XC Controller.

    Uses the GeneralizedXCController (expert) to fly complete cross-country flights
    across multiple airport pairs (SN65->KHUT, SN65->KICT, SN65->KAAO, etc.)
    for diverse phase coverage.

    Generates intent observations by issuing phase-appropriate commands at each
    phase transition and periodically during steady phases.
    """

    M_TO_FT = 3.28084
    MS_TO_KTS = 1.94384
    MS_TO_FPM = 196.85

    # Available route pairs for training diversity
    ROUTE_PAIRS = [
        ("SN65", "KHUT"),   # ~35nm NW, Rwy 314
        ("SN65", "KICT"),   # ~16nm NE, Rwy 014
        ("SN65", "KAAO"),   # ~21nm NE, Rwy 180
    ]

    def __init__(self, num_envs: int = 4, device: str = "cpu", ppo_model_path: str = None):
        self.num_envs = num_envs
        self.device = device
        self.ppo_model = None
        # CPU mode is ~36x faster than GPU for single-instance sim loops
        # due to CuPy kernel launch overhead per step
        self.use_gpu_sim = False
        # Larger dt = fewer steps for same sim time (stable up to 0.1)
        self.sim_dt = 0.05

        # Try to load flight dynamics and generalized XC controller
        try:
            from flight_dynamics import FlightSimulator, StateIndex
            from generalized_xc_controller import (
                GeneralizedXCController, create_controller,
                XCPhase, KANSAS_AIRPORTS, AirportConfig
            )
            self._FlightSimulator = FlightSimulator
            self._StateIndex = StateIndex
            self._create_controller = create_controller
            self._XCPhase = XCPhase
            self._KANSAS_AIRPORTS = KANSAS_AIRPORTS
            self.gpu_sim_available = True
            n_routes = len(self.ROUTE_PAIRS)
            print(f"[SimRunner] Flight dynamics + XC controller loaded "
                  f"(CPU mode, dt={self.sim_dt}, {n_routes} routes)")
        except ImportError as e:
            self.gpu_sim_available = False
            print(f"[SimRunner] Flight dynamics not available: {e}")
            return

        # Load PPO policy if provided
        if ppo_model_path:
            try:
                from stable_baselines3 import PPO
                self.ppo_model = PPO.load(ppo_model_path)
                action_dim = self.ppo_model.action_space.shape[0]
                print(f"[SimRunner] PPO policy loaded ({action_dim} actions): {ppo_model_path}")
            except Exception as e:
                print(f"[SimRunner] PPO policy not loaded: {e}")

        # Residual scales (from run_residual_telemetry_v2.py)
        self.residual_scales = np.array([0.15, 0.10, 0.15, 0.10, 0.10, 0.15, 0.10], dtype=np.float32)

        print(f"[SimRunner] Ready ({num_envs} envs, {mp.cpu_count()} CPU cores)")

    def _extract_telemetry(self, state: np.ndarray) -> dict:
        """Extract telemetry from 12-dim state vector [x,y,z,u,v,w,phi,theta,psi,p,q,r]."""
        alt_m = -state[2]  # NED: z negative is up
        u, v, w = state[3], state[4], state[5]
        airspeed_ms = np.sqrt(u**2 + v**2 + w**2)
        vr_ms = -w  # NED: w negative is climb

        return {
            "altitude": float(alt_m * self.M_TO_FT),
            "airspeed": float(airspeed_ms * self.MS_TO_KTS),
            "vertical_rate": float(vr_ms * self.MS_TO_FPM),
            "heading": float(np.rad2deg(state[8]) % 360),
            "roll": float(np.rad2deg(state[6])),
            "pitch": float(np.rad2deg(state[7])),
        }

    def _build_obs_v2(self, state, expert_action, phase, khut_x, khut_y, cruise_alt_m):
        """Build 25-dim observation for PPO policy."""
        x, y, z = state[0], state[1], state[2]
        psi = state[8]
        alt = -z

        alt_error = (cruise_alt_m - alt) / 1000.0
        dx = khut_x - x
        dy = khut_y - y
        bearing = np.arctan2(dy, dx)
        heading_error = bearing - psi
        while heading_error > np.pi: heading_error -= 2 * np.pi
        while heading_error < -np.pi: heading_error += 2 * np.pi
        dist = np.sqrt(dx**2 + dy**2) / 10000.0

        # Phase encoding [takeoff, cruise, approach]
        XCPhase = self._XCPhase
        if phase in [XCPhase.GROUND_ROLL, XCPhase.ROTATION, XCPhase.INITIAL_CLIMB]:
            phase_enc = np.array([1, 0, 0], dtype=np.float32)
        elif phase in [XCPhase.CLIMB, XCPhase.CRUISE_TO_TP]:
            phase_enc = np.array([0, 1, 0], dtype=np.float32)
        else:
            phase_enc = np.array([0, 0, 1], dtype=np.float32)

        return np.concatenate([
            state.astype(np.float32),
            expert_action[:7].astype(np.float32),
            np.array([alt_error, heading_error, dist], dtype=np.float32),
            phase_enc,
        ])

    def run_flight(self, flight_id: str, cruise_alt_ft: float = 5500.0,
                   origin: str = "SN65", destination: str = "KHUT") -> List[IntentObservation]:
        """
        Run a complete XC flight and collect intent observations.

        Uses GeneralizedXCController for the specified airport pair.
        At phase transitions and periodically, issues realistic pilot commands
        and validates them through the Bayesian engine.
        """
        if not self.gpu_sim_available:
            return []

        observations = []
        FlightSimulator = self._FlightSimulator

        try:
            # Create sim and controller (CPU mode for speed)
            sim = FlightSimulator(n_instances=1, dt=self.sim_dt, use_gpu=self.use_gpu_sim)
            expert = self._create_controller(origin, destination, cruise_altitude_ft=cruise_alt_ft)

            # Reset
            initial_state = np.zeros((1, 12), dtype=np.float32)
            initial_state[0, 0] = -400.0  # Start on runway
            initial_state[0, 3] = 5.0     # Small forward velocity
            sim.reset(initial_state)
            expert.reset()
            reset_inference_engine()

            khut_x, khut_y = 52800.0, -21300.0
            cruise_alt_m = cruise_alt_ft * 0.3048

            sim_time = 0.0
            dt = self.sim_dt
            last_phase = None
            last_cmd_time = 0.0
            cmd_interval = random.uniform(5.0, 15.0)  # Seconds between commands
            # Full XC route to KHUT is ~53km, needs ~45 min at cruise speed
            max_steps = int(2700.0 / dt)  # 45 min of sim time

            for step in range(max_steps):
                state = sim.get_states()[0]
                sim_time = step * dt

                # Expert action
                expert_action = expert.compute_action(state, sim_time)
                phase = expert.phase
                phase_name = phase.name

                # Apply PPO residual if available
                if self.ppo_model is not None:
                    obs = self._build_obs_v2(state, expert_action, phase, khut_x, khut_y, cruise_alt_m)
                    residual, _ = self.ppo_model.predict(obs, deterministic=True)
                    action = np.clip(
                        expert_action[:7] + self.residual_scales * residual,
                        np.array([0, -1, -1, -1, 0, 0, 0]),
                        np.array([1, 1, 1, 1, 1, 1, 1]),
                    )
                else:
                    action = expert_action[:7]

                # Apply controls and step
                controls = np.array([action], dtype=np.float32)
                sim.set_controls(controls)
                sim.step()

                # Extract telemetry
                telem = self._extract_telemetry(state)
                intent_phase = XCPHASE_TO_INTENT.get(phase_name, "cruise")

                # Generate commands at phase transitions or periodically
                should_command = False
                if phase_name != last_phase:
                    should_command = True
                    last_phase = phase_name
                    cmd_interval = random.uniform(3.0, 10.0)
                elif sim_time - last_cmd_time >= cmd_interval:
                    should_command = True
                    cmd_interval = random.uniform(5.0, 15.0)

                if should_command and phase_name in PHASE_COMMAND_TEMPLATES:
                    templates = PHASE_COMMAND_TEMPLATES[phase_name]
                    if templates:
                        # Pick a random command appropriate for this phase
                        template = random.choice(templates)
                        cmd_value = template["value_fn"](telem)

                        # Validate through Bayesian engine
                        result = bayesian_validate_intent(
                            action=template["action"],
                            value=cmd_value,
                            target=None,
                            telemetry=telem,
                            phase=intent_phase,
                            use_stateful=True,
                        )

                        # Determine outcome
                        if not result.get("validated", True):
                            outcome = "anomaly_flagged"
                        elif result.get("confidence", 1.0) < 0.3:
                            outcome = "low_confidence"
                        else:
                            outcome = "executed"

                        obs_entry = IntentObservation(
                            timestamp=time.time(),
                            phase=intent_phase,
                            telemetry=telem,
                            command={"action": template["action"], "value": float(cmd_value), "target": None},
                            validation={
                                "confidence": result.get("confidence"),
                                "validated": result.get("validated"),
                                "anomaly_score": result.get("bayesian", {}).get("anomaly_score"),
                                "sequence_coherence": result.get("temporal", {}).get("sequence_coherence"),
                            },
                            outcome=outcome,
                            session_id=f"gpu_train_{int(time.time())}",
                            flight_id=flight_id,
                        )
                        observations.append(obs_entry)
                        last_cmd_time = sim_time

                # Check if flight complete
                if phase_name == "LANDED":
                    break

        except Exception as e:
            print(f"[SimRunner] Flight {flight_id} error: {e}")
            import traceback
            traceback.print_exc()

        return observations

    def run_parallel_flights(self, num_flights: int) -> List[IntentObservation]:
        """Run multiple flights in parallel using process pool (CPU sim)."""
        all_observations = []

        # Vary cruise altitudes and routes for diversity
        altitudes = [3500, 4500, 5500, 6500, 7500]
        n_workers = min(num_flights, mp.cpu_count())

        print(f"    [SimRunner] Running {num_flights} flights across {n_workers} CPU workers "
              f"({len(self.ROUTE_PAIRS)} routes)")

        # Build flight configs with diverse routes
        flight_configs = []
        for i in range(num_flights):
            origin, dest = random.choice(self.ROUTE_PAIRS)
            flight_configs.append({
                "flight_id": f"sim_{origin}_{dest}_{i}_{int(time.time())}",
                "cruise_alt_ft": random.choice(altitudes),
                "sim_dt": self.sim_dt,
                "origin": origin,
                "destination": dest,
                "ppo_model_path": None,  # PPO model can't be pickled across processes
            })

        # Run in parallel using process pool
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_sim_flight_worker, cfg): cfg
                for cfg in flight_configs
            }
            completed = 0
            for future in as_completed(futures):
                try:
                    obs_dicts = future.result()
                    observations = [IntentObservation.from_dict(d) for d in obs_dicts]
                    all_observations.extend(observations)
                    completed += 1
                    if completed % max(1, num_flights // 5) == 0:
                        print(f"    [SimRunner] {completed}/{num_flights} flights done, "
                              f"{len(all_observations)} obs so far")
                except Exception as e:
                    completed += 1
                    print(f"    [SimRunner] Flight failed: {e}")

        print(f"    [SimRunner] Total: {len(all_observations)} observations from {num_flights} flights")
        return all_observations

    def run_gpu_batched_flights(self, num_flights: int,
                                substeps: int = 10) -> List[IntentObservation]:
        """
        Run flights using native CUDA kernel for physics + CPU controllers.

        Architecture:
          - Single CudaFlightSim(n_instances=N) on GPU (fused RK4 kernel)
          - N GeneralizedXCControllers on CPU
          - Per outer step: GPU runs `substeps` physics steps (fused),
            then CPU updates N controllers once
          - substeps=10 means controller runs at 2Hz (every 0.5s at dt=0.05)
            instead of 20Hz, reducing Python overhead by 10x
          - GPU physics: ~16ms for 1000 steps × 1000 instances (essentially free)
        """
        if not self.gpu_sim_available:
            print("    [GPU] Flight dynamics not available, falling back to CPU")
            return self.run_parallel_flights(num_flights)

        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))
            from cuda_flight_sim import CudaFlightSim
        except ImportError as e:
            print(f"    [GPU] Native CUDA sim not available ({e}), falling back to CPU")
            return self.run_parallel_flights(num_flights)

        all_observations = []
        altitudes = [3500, 4500, 5500, 6500, 7500]

        print(f"    [GPU] Running {num_flights} flights on native CUDA kernel "
              f"(dt={self.sim_dt}, substeps={substeps}, {len(self.ROUTE_PAIRS)} routes)")

        try:
            # Create single GPU sim for all flights
            sim = CudaFlightSim(n_instances=num_flights, dt=self.sim_dt)

            # Create N controllers (CPU)
            controllers = []
            flight_ids = []
            import io, contextlib
            for i in range(num_flights):
                origin, dest = random.choice(self.ROUTE_PAIRS)
                cruise_alt = random.choice(altitudes)
                with contextlib.redirect_stdout(io.StringIO()):
                    ctrl = self._create_controller(origin, dest, cruise_altitude_ft=cruise_alt)
                controllers.append(ctrl)
                flight_ids.append(f"gpu_{origin}_{dest}_{i}_{int(time.time())}")

            # Reset all instances to runway start
            init_state = np.zeros(12, dtype=np.float32)
            init_state[0] = -400.0  # Start on runway
            init_state[3] = 5.0     # Small forward velocity
            sim.reset(init_state)

            # Per-flight tracking
            active = np.ones(num_flights, dtype=bool)
            last_phases = [None] * num_flights
            last_cmd_times = [0.0] * num_flights
            cmd_intervals = [random.uniform(5.0, 15.0) for _ in range(num_flights)]
            flight_observations = [[] for _ in range(num_flights)]

            # Reset all inference engines (one per flight via separate sessions)
            reset_inference_engine()

            controls = np.zeros((num_flights, 7), dtype=np.float32)
            max_steps = int(2700.0 / self.sim_dt)  # 45 min
            # Outer loop steps = total / substeps
            outer_steps = max_steps // substeps
            outer_dt = substeps * self.sim_dt  # Time per outer step

            start_time = time.time()
            total_sim_steps = 0

            for outer in range(outer_steps):
                if not np.any(active):
                    break

                sim_time = outer * outer_dt

                # Get all states from GPU (single transfer)
                states = sim.get_states()  # [N, 12]

                # Compute controls for each active flight (CPU)
                for i in range(num_flights):
                    if not active[i]:
                        continue

                    state = states[i]
                    expert_action = controllers[i].compute_action(state, sim_time)
                    phase = controllers[i].phase
                    phase_name = phase.name
                    controls[i, :] = expert_action[:7]

                    # Generate intent observations at phase transitions or periodically
                    should_command = False
                    if phase_name != last_phases[i]:
                        should_command = True
                        last_phases[i] = phase_name
                        cmd_intervals[i] = random.uniform(3.0, 10.0)
                    elif sim_time - last_cmd_times[i] >= cmd_intervals[i]:
                        should_command = True
                        cmd_intervals[i] = random.uniform(5.0, 15.0)

                    if should_command and phase_name in PHASE_COMMAND_TEMPLATES:
                        templates = PHASE_COMMAND_TEMPLATES[phase_name]
                        if templates:
                            alt_m = -state[2]
                            u, v, w = state[3], state[4], state[5]
                            airspeed_ms = np.sqrt(u**2 + v**2 + w**2)
                            telem = {
                                "altitude": float(alt_m * self.M_TO_FT),
                                "airspeed": float(airspeed_ms * self.MS_TO_KTS),
                                "vertical_rate": float(-w * self.MS_TO_FPM),
                                "heading": float(np.rad2deg(state[8]) % 360),
                                "roll": float(np.rad2deg(state[6])),
                                "pitch": float(np.rad2deg(state[7])),
                            }
                            intent_phase = XCPHASE_TO_INTENT.get(phase_name, "cruise")

                            template = random.choice(templates)
                            cmd_value = template["value_fn"](telem)

                            result = bayesian_validate_intent(
                                action=template["action"],
                                value=cmd_value,
                                target=None,
                                telemetry=telem,
                                phase=intent_phase,
                                use_stateful=True,
                            )

                            if not result.get("validated", True):
                                outcome = "anomaly_flagged"
                            elif result.get("confidence", 1.0) < 0.3:
                                outcome = "low_confidence"
                            else:
                                outcome = "executed"

                            obs_entry = IntentObservation(
                                timestamp=time.time(),
                                phase=intent_phase,
                                telemetry=telem,
                                command={"action": template["action"],
                                         "value": float(cmd_value), "target": None},
                                validation={
                                    "confidence": result.get("confidence"),
                                    "validated": result.get("validated"),
                                    "anomaly_score": result.get("bayesian", {}).get("anomaly_score"),
                                    "sequence_coherence": result.get("temporal", {}).get("sequence_coherence"),
                                },
                                outcome=outcome,
                                session_id=f"gpu_train_{int(time.time())}",
                                flight_id=flight_ids[i],
                            )
                            flight_observations[i].append(obs_entry)
                            last_cmd_times[i] = sim_time

                    if phase_name == "LANDED":
                        active[i] = False

                # Set controls and run substeps on GPU (single set_controls + step_n)
                sim.set_controls(controls)
                sim.step_n(substeps)
                total_sim_steps += substeps

                # Progress reporting
                if outer > 0 and total_sim_steps % 10000 < substeps:
                    n_active = np.sum(active)
                    n_obs = sum(len(o) for o in flight_observations)
                    elapsed = time.time() - start_time
                    print(f"    [GPU] Step {total_sim_steps}/{max_steps} | "
                          f"{n_active}/{num_flights} active | "
                          f"{n_obs} obs | {elapsed:.1f}s "
                          f"({substeps} substeps/ctrl)")

            # Collect all observations
            for obs_list in flight_observations:
                all_observations.extend(obs_list)

            elapsed = time.time() - start_time
            n_landed = num_flights - np.sum(active)
            print(f"    [GPU] Done: {len(all_observations)} obs from "
                  f"{n_landed}/{num_flights} completed flights in {elapsed:.1f}s")

        except Exception as e:
            print(f"    [GPU] Error: {e}")
            import traceback
            traceback.print_exc()
            if not all_observations:
                print("    [GPU] Falling back to CPU parallel flights")
                return self.run_parallel_flights(num_flights)

        return all_observations


def _run_sim_flight_worker(config: dict) -> List[dict]:
    """
    Worker function for parallel sim flights (runs in separate process).
    Each worker creates its own sim instance on CPU with GeneralizedXCController.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))

    try:
        from flight_dynamics import FlightSimulator
        from generalized_xc_controller import create_controller, XCPhase
    except ImportError as e:
        print(f"[Worker] Import failed: {e}")
        return []

    flight_id = config["flight_id"]
    cruise_alt_ft = config["cruise_alt_ft"]
    sim_dt = config["sim_dt"]
    origin = config.get("origin", "SN65")
    destination = config.get("destination", "KHUT")

    M_TO_FT = 3.28084
    MS_TO_KTS = 1.94384
    MS_TO_FPM = 196.85

    observations = []

    try:
        sim = FlightSimulator(n_instances=1, dt=sim_dt, use_gpu=False)
        # Suppress verbose controller config output during batch training
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            expert = create_controller(origin, destination, cruise_altitude_ft=cruise_alt_ft)

        initial_state = np.zeros((1, 12), dtype=np.float32)
        initial_state[0, 0] = -400.0
        initial_state[0, 3] = 5.0
        sim.reset(initial_state)
        expert.reset()
        reset_inference_engine()

        max_steps = int(2700.0 / sim_dt)  # 45 min for full XC route
        last_phase = None
        last_cmd_time = 0.0
        cmd_interval = random.uniform(5.0, 15.0)

        for step in range(max_steps):
            state = sim.get_states()[0]
            sim_time = step * sim_dt

            expert_action = expert.compute_action(state, sim_time)
            phase = expert.phase
            phase_name = phase.name

            controls = np.array([expert_action[:7]], dtype=np.float32)
            sim.set_controls(controls)
            sim.step()

            # Extract telemetry
            alt_m = -state[2]
            u, v, w = state[3], state[4], state[5]
            airspeed_ms = np.sqrt(u**2 + v**2 + w**2)
            telem = {
                "altitude": float(alt_m * M_TO_FT),
                "airspeed": float(airspeed_ms * MS_TO_KTS),
                "vertical_rate": float(-w * MS_TO_FPM),
                "heading": float(np.rad2deg(state[8]) % 360),
                "roll": float(np.rad2deg(state[6])),
                "pitch": float(np.rad2deg(state[7])),
            }
            intent_phase = XCPHASE_TO_INTENT.get(phase_name, "cruise")

            # Generate commands at phase transitions or periodically
            should_command = False
            if phase_name != last_phase:
                should_command = True
                last_phase = phase_name
                cmd_interval = random.uniform(3.0, 10.0)
            elif sim_time - last_cmd_time >= cmd_interval:
                should_command = True
                cmd_interval = random.uniform(5.0, 15.0)

            if should_command and phase_name in PHASE_COMMAND_TEMPLATES:
                templates = PHASE_COMMAND_TEMPLATES[phase_name]
                if templates:
                    template = random.choice(templates)
                    cmd_value = template["value_fn"](telem)

                    result = bayesian_validate_intent(
                        action=template["action"],
                        value=cmd_value,
                        target=None,
                        telemetry=telem,
                        phase=intent_phase,
                        use_stateful=True,
                    )

                    if not result.get("validated", True):
                        outcome = "anomaly_flagged"
                    elif result.get("confidence", 1.0) < 0.3:
                        outcome = "low_confidence"
                    else:
                        outcome = "executed"

                    obs_entry = IntentObservation(
                        timestamp=time.time(),
                        phase=intent_phase,
                        telemetry=telem,
                        command={"action": template["action"], "value": float(cmd_value), "target": None},
                        validation={
                            "confidence": result.get("confidence"),
                            "validated": result.get("validated"),
                            "anomaly_score": result.get("bayesian", {}).get("anomaly_score"),
                            "sequence_coherence": result.get("temporal", {}).get("sequence_coherence"),
                        },
                        outcome=outcome,
                        session_id=f"sim_train_{int(time.time())}",
                        flight_id=flight_id,
                    )
                    observations.append(obs_entry)
                    last_cmd_time = sim_time

            if phase_name == "LANDED":
                break

    except Exception as e:
        import traceback
        print(f"[Worker] Flight {flight_id} error: {e}")
        traceback.print_exc()

    return [obs.to_dict() for obs in observations]


# =============================================================================
# Quick Test Mode
# =============================================================================

def run_quick_test():
    """Run a quick test to verify the pipeline works."""
    print("=" * 70)
    print("Quick Test Mode")
    print("=" * 70)

    # Run single flight synchronously
    scenario = random.choice(FLIGHT_SCENARIOS)
    print(f"\nRunning scenario: {scenario.name}")

    observations = run_single_flight(scenario, "quick_test")

    print(f"\nCollected {len(observations)} observations:")
    for i, obs in enumerate(observations[:5]):
        print(f"  {i+1}. {obs.command['action']} = {obs.command['value']:.1f} "
              f"| phase={obs.phase} | conf={obs.validation.get('confidence', 0):.2f} "
              f"| outcome={obs.outcome}")

    if len(observations) > 5:
        print(f"  ... and {len(observations) - 5} more")

    # Estimate priors from just this flight
    print("\nEstimating priors from test data...")
    priors = estimate_phase_priors(observations)

    print(f"\nPriors for {len(priors)} phases:")
    for phase, params in sorted(priors.items()):
        if params.get("total_observations", 0) > 0:
            print(f"  {phase}: κ={params['heading_concentration']:.1f} "
                  f"({params.get('heading_concentration_source', 'default')})")

    print("\n✓ Quick test passed!")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AIDA Intent Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python intent_training_pipeline.py --test

  # Run 4 parallel workers, each doing 5 flights
  python intent_training_pipeline.py --parallel 4 --flights-per-worker 5

  # Collect 500 observations, train every 50
  python intent_training_pipeline.py --target 500 --train-every 50

  # Full training run
  python intent_training_pipeline.py --parallel 8 --target 2000 --train-every 100

  # Use real AIDA sim environment (GPU accelerated)
  python intent_training_pipeline.py --use-sim --parallel 4 --device cuda

  # Fast training with sim environment
  python intent_training_pipeline.py --use-sim -p 8 -n 1000 -t 100
        """
    )

    parser.add_argument("--test", action="store_true",
                        help="Run quick test mode")
    parser.add_argument("--parallel", "-p", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--flights-per-worker", "-f", type=int, default=5,
                        help="Flights per worker per batch (default: 5)")
    parser.add_argument("--train-every", "-t", type=int, default=100,
                        help="Train priors every N observations (default: 100)")
    parser.add_argument("--target", "-n", type=int, default=500,
                        help="Target total observations (default: 500)")
    parser.add_argument("--stats", action="store_true",
                        help="Print statistics from existing observations")
    parser.add_argument("--use-sim", action="store_true",
                        help="Use GPU-accelerated AIDA simulation (expert controller)")
    parser.add_argument("--ppo-model", type=str, default=None,
                        help="Path to residual PPO model (.zip) for realistic flights")
    parser.add_argument("--device", type=str, default=DEVICE,
                        choices=["cpu", "cuda"],
                        help=f"Device for simulation (default: {DEVICE})")
    parser.add_argument("--test-sim", action="store_true",
                        help="Test GPU sim flight runner")
    parser.add_argument("--iterative", action="store_true",
                        help="Run iterative training (collect -> train -> repeat)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of iterative training rounds (default: 5)")
    parser.add_argument("--reset", action="store_true",
                        help="Clear existing observations before training")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation test: default vs learned priors")
    parser.add_argument("--gpu", action="store_true",
                        help="Use native CUDA kernel for batched GPU training (requires libflightdynamics.so)")

    args = parser.parse_args()

    if args.test:
        run_quick_test()
        return

    if args.validate:
        run_validation_test()
        return

    if args.test_sim:
        run_sim_test(ppo_model_path=args.ppo_model)
        return

    if args.stats:
        observations = load_observations()
        print_observation_stats(observations)
        return

    if args.reset:
        if OBSERVATIONS_FILE.exists():
            OBSERVATIONS_FILE.unlink()
            print(f"Cleared {OBSERVATIONS_FILE}")

    if args.iterative:
        run_iterative_training(
            iterations=args.iterations,
            obs_per_iteration=args.target,
            parallel=args.parallel,
            flights_per_worker=args.flights_per_worker,
            use_sim=args.use_sim,
            ppo_model_path=args.ppo_model,
            device=args.device,
            use_gpu=args.gpu,
        )
        return

    # Run full pipeline
    pipeline = IntentTrainingPipeline(
        parallel_workers=args.parallel,
        flights_per_worker=args.flights_per_worker,
        train_every=args.train_every,
        target_observations=args.target,
        use_sim=args.use_sim,
        device=args.device,
        use_gpu=args.gpu,
    )
    pipeline.run()


def run_iterative_training(
    iterations: int = 5,
    obs_per_iteration: int = 500,
    parallel: int = 8,
    flights_per_worker: int = 5,
    use_sim: bool = False,
    ppo_model_path: str = None,
    device: str = "cpu",
    use_gpu: bool = False,
):
    """
    Run iterative training: collect data -> train priors -> repeat.

    Each iteration:
    1. Applies current learned priors to the Bayesian engine
    2. Runs flights and collects observations WITH current priors
    3. Re-estimates priors from ALL accumulated data
    4. Saves checkpoint and metrics

    This converges the priors toward the true distribution of pilot commands.
    """
    print("=" * 70)
    print("AIDA Iterative Intent Training")
    print("=" * 70)
    print(f"  Iterations: {iterations}")
    print(f"  Observations per iteration: {obs_per_iteration}")
    print(f"  Parallel workers: {parallel}")
    print(f"  Use sim: {use_sim}")
    print(f"  GPU batched: {use_gpu}")
    print(f"  PPO model: {ppo_model_path or 'None'}")
    print("=" * 70)

    checkpoint_dir = DATA_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    iteration_metrics = []
    start_time = time.time()

    for iteration in range(1, iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"  ITERATION {iteration}/{iterations}")
        print(f"{'=' * 60}")

        # Step 1: Apply current learned priors (if any)
        try:
            reset_inference_engine()
            engine = get_inference_engine()
            priors = load_learned_priors()
            if priors is not None:
                apply_learned_priors_to_engine(engine)
                print(f"  Applied learned priors from previous iteration")
            else:
                print(f"  Using default priors (no learned priors yet)")
        except Exception as e:
            print(f"  Could not apply priors: {e}")

        # Step 2: Run flights and collect observations
        print(f"  Collecting ~{obs_per_iteration} observations...")
        pipeline = IntentTrainingPipeline(
            parallel_workers=parallel,
            flights_per_worker=flights_per_worker,
            train_every=obs_per_iteration + 1,  # Don't train mid-collection
            target_observations=obs_per_iteration,
            use_sim=use_sim,
            device=device,
            use_gpu=use_gpu,
        )
        if use_sim and pipeline.sim_runner is not None and ppo_model_path:
            pipeline.sim_runner.ppo_model_path = ppo_model_path

        # Override to not auto-train (we train after collection)
        pipeline.total_observations = 0
        pipeline.training_rounds = 999  # Prevent auto-train

        # Collect flights
        new_observations = pipeline._run_parallel_flights()
        pipeline._save_observations(new_observations)

        total_obs = len(load_observations())
        print(f"  Collected {len(new_observations)} new ({total_obs} total)")

        # Step 3: Estimate priors from ALL accumulated data
        print(f"  Training priors from {total_obs} observations...")
        all_obs = load_observations()
        priors = estimate_phase_priors(all_obs)
        process_noise = estimate_process_noise(all_obs)
        save_learned_priors(priors)

        # Step 4: Save checkpoint
        checkpoint = {
            "iteration": iteration,
            "timestamp": time.time(),
            "total_observations": total_obs,
            "new_observations": len(new_observations),
            "priors": priors,
            "process_noise": process_noise,
        }
        cp_file = checkpoint_dir / f"iterative_round_{iteration}.json"
        with open(cp_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

        # Compute metrics
        confidences = [
            obs.validation.get("confidence", 0)
            for obs in new_observations
            if obs.validation.get("confidence") is not None
        ]
        avg_conf = np.mean(confidences) if confidences else 0
        outcome_counts = {}
        for obs in new_observations:
            outcome_counts[obs.outcome] = outcome_counts.get(obs.outcome, 0) + 1
        phase_counts = {}
        for obs in new_observations:
            phase_counts[obs.phase] = phase_counts.get(obs.phase, 0) + 1

        learned_count = sum(1 for p in priors.values() if p.get("heading_concentration_source") == "learned")

        metrics = {
            "iteration": iteration,
            "total_observations": total_obs,
            "new_observations": len(new_observations),
            "avg_confidence": float(avg_conf),
            "outcome_counts": outcome_counts,
            "phase_counts": phase_counts,
            "learned_phases": learned_count,
            "total_phases": len(priors),
        }
        iteration_metrics.append(metrics)

        print(f"  Avg confidence: {avg_conf:.1%}")
        print(f"  Outcomes: {outcome_counts}")
        print(f"  Phases: {phase_counts}")
        print(f"  Learned priors: {learned_count}/{len(priors)} phases")

    # Save full metrics
    metrics_file = checkpoint_dir / "iterative_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(iteration_metrics, f, indent=2)

    # Print convergence summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print("Iterative Training Complete!")
    print(f"{'=' * 70}")
    print(f"Total time: {total_time:.1f}s")
    print(f"\nConvergence:")
    print(f"{'Iter':>5} {'Obs':>8} {'Avg Conf':>10} {'Learned':>8} {'Executed':>10} {'Low Conf':>10}")
    print("-" * 60)
    for m in iteration_metrics:
        executed = m["outcome_counts"].get("executed", 0)
        low_conf = m["outcome_counts"].get("low_confidence", 0)
        print(f"{m['iteration']:>5} {m['total_observations']:>8} {m['avg_confidence']:>9.1%} "
              f"{m['learned_phases']:>7}/{m['total_phases']} "
              f"{executed:>10} {low_conf:>10}")

    # Print final learned priors
    final_priors = load_learned_priors()
    if final_priors:
        print_learned_priors(final_priors)


def run_validation_test():
    """
    Compare default vs learned priors on realistic flight commands.

    Runs the same set of phase-appropriate commands through the Bayesian engine
    twice: once with default priors, once with learned priors. Shows a
    side-by-side comparison of confidence scores.
    """
    print("=" * 80)
    print("  BAYESIAN INTENT VALIDATION TEST: Default vs Learned Priors")
    print("=" * 80)

    # Test scenarios: (phase, telemetry, commands)
    # Each command is (action, value, description)
    test_scenarios = [
        {
            "phase": "ground",
            "telemetry": {"heading": 314, "altitude": 1600, "airspeed": 0},
            "commands": [
                ("heading", 314, "Runway heading (on-phase, normal)"),
                ("heading", 180, "Opposite heading (off-phase, anomalous)"),
                ("altitude", 1600, "Field elevation (on-phase, normal)"),
            ],
        },
        {
            "phase": "takeoff",
            "telemetry": {"heading": 314, "altitude": 1800, "airspeed": 70},
            "commands": [
                ("heading", 320, "Near runway heading (normal departure)"),
                ("heading", 134, "Reverse course (anomalous)"),
                ("altitude", 2500, "Initial climb target (normal)"),
                ("altitude", 14000, "Service ceiling (anomalous for takeoff)"),
            ],
        },
        {
            "phase": "climb",
            "telemetry": {"heading": 330, "altitude": 3500, "airspeed": 85},
            "commands": [
                ("heading", 340, "Small course correction (normal)"),
                ("heading", 150, "Large heading change (unusual)"),
                ("altitude", 5500, "Cruise altitude target (normal)"),
                ("altitude", 500, "Descend during climb (anomalous)"),
            ],
        },
        {
            "phase": "cruise",
            "telemetry": {"heading": 300, "altitude": 5500, "airspeed": 110},
            "commands": [
                ("heading", 305, "Minor course correction (normal)"),
                ("heading", 120, "Major turn (unusual but valid)"),
                ("altitude", 5500, "Maintain altitude (normal)"),
                ("altitude", 7500, "Step climb (normal)"),
            ],
        },
        {
            "phase": "approach",
            "telemetry": {"heading": 134, "altitude": 2500, "airspeed": 90},
            "commands": [
                ("heading", 134, "Runway heading (on-approach, normal)"),
                ("heading", 314, "Opposite direction (anomalous)"),
                ("altitude", 2000, "Descend on approach (normal)"),
                ("altitude", 8000, "Climb during approach (anomalous)"),
            ],
        },
        {
            "phase": "landing",
            "telemetry": {"heading": 134, "altitude": 200, "airspeed": 65},
            "commands": [
                ("heading", 134, "Runway heading (normal)"),
                ("heading", 270, "Cross-wind heading (anomalous)"),
                ("altitude", 100, "Continue descent (normal)"),
                ("land", None, "Land command (normal for phase)"),
            ],
        },
    ]

    # Run each scenario with default and learned engines
    results = []

    for scenario in test_scenarios:
        phase = scenario["phase"]
        telemetry = scenario["telemetry"]

        for action, value, description in scenario["commands"]:
            # --- Default priors ---
            reset_inference_engine()
            engine_default = get_inference_engine()
            result_default = bayesian_validate_intent(
                action=action,
                value=value,
                target=None,
                telemetry=telemetry,
                phase=phase,
                use_stateful=False,  # Fresh engine each time
            )

            # --- Learned priors ---
            reset_inference_engine()
            engine_learned = get_inference_engine()
            applied = apply_learned_priors_to_engine(engine_learned)
            result_learned = bayesian_validate_intent(
                action=action,
                value=value,
                target=None,
                telemetry=telemetry,
                phase=phase,
                use_stateful=True,  # Uses the global engine we just patched
            )

            results.append({
                "phase": phase,
                "action": action,
                "value": value,
                "description": description,
                "default_confidence": result_default["confidence"],
                "learned_confidence": result_learned["confidence"],
                "default_anomaly": result_default["bayesian"]["anomaly_score"],
                "learned_anomaly": result_learned["bayesian"]["anomaly_score"],
                "default_validated": result_default["validated"],
                "learned_validated": result_learned["validated"],
            })

    # Print results
    print(f"\n{'Phase':>12} | {'Command':>40} | {'Default':>8} | {'Learned':>8} | {'Delta':>7} | {'Anomaly D':>9} | {'Anomaly L':>9}")
    print("-" * 120)

    total_default = 0
    total_learned = 0
    improvements = 0
    regressions = 0

    current_phase = None
    for r in results:
        if r["phase"] != current_phase:
            current_phase = r["phase"]
            if r != results[0]:
                print("-" * 120)

        cmd_str = f"{r['action']}={r['value']}" if r['value'] is not None else r['action']
        desc = r['description'][:35]
        dc = r['default_confidence']
        lc = r['learned_confidence']
        delta = lc - dc
        da = r['default_anomaly']
        la = r['learned_anomaly']

        arrow = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")

        print(f"{r['phase']:>12} | {cmd_str:>12} {desc:<27} | {dc:>7.1%} | {lc:>7.1%} | {arrow}{abs(delta):>5.1%} | {da:>9.2f} | {la:>9.2f}")

        total_default += dc
        total_learned += lc
        if delta > 0.01:
            improvements += 1
        elif delta < -0.01:
            regressions += 1

    n = len(results)
    print("=" * 120)
    print(f"{'AVERAGE':>12} | {'':>40} | {total_default/n:>7.1%} | {total_learned/n:>7.1%} | "
          f"{'+'if (total_learned-total_default)/n > 0 else ''}{(total_learned-total_default)/n:>5.1%} |")
    print(f"\nSummary: {improvements} improved, {regressions} regressed, {n - improvements - regressions} unchanged out of {n} commands")

    # Detailed analysis by phase
    print(f"\n{'=' * 80}")
    print("  Per-Phase Average Confidence")
    print(f"{'=' * 80}")
    phase_stats = {}
    for r in results:
        p = r["phase"]
        if p not in phase_stats:
            phase_stats[p] = {"default": [], "learned": []}
        phase_stats[p]["default"].append(r["default_confidence"])
        phase_stats[p]["learned"].append(r["learned_confidence"])

    print(f"{'Phase':>12} | {'Default Avg':>12} | {'Learned Avg':>12} | {'Delta':>8} | {'N':>3}")
    print("-" * 55)
    for phase, stats_dict in phase_stats.items():
        d_avg = np.mean(stats_dict["default"])
        l_avg = np.mean(stats_dict["learned"])
        delta = l_avg - d_avg
        print(f"{phase:>12} | {d_avg:>11.1%} | {l_avg:>11.1%} | {delta:>+7.1%} | {len(stats_dict['default']):>3}")

    print(f"\nValidation test complete.")
    return results


def run_sim_test(ppo_model_path=None):
    """Test the sim flight runner with GeneralizedXCController."""
    print("=" * 70)
    print("Sim Flight Runner Test (GeneralizedXCController)")
    print("=" * 70)
    print(f"CPU cores available: {mp.cpu_count()}")

    runner = SimEnvFlightRunner(num_envs=1, device=DEVICE, ppo_model_path=ppo_model_path)

    if not runner.gpu_sim_available:
        print("ERROR: Flight dynamics not available!")
        print("Make sure flight_dynamics and generalized_xc_controller are importable.")
        return

    # Test each route
    for origin, dest in runner.ROUTE_PAIRS:
        print(f"\nRunning XC flight: {origin} -> {dest}...")
        start = time.time()
        observations = runner.run_flight(f"test_{origin}_{dest}",
                                          cruise_alt_ft=5500.0,
                                          origin=origin, destination=dest)
        elapsed = time.time() - start
        phase_counts = {}
        for obs in observations:
            phase_counts[obs.phase] = phase_counts.get(obs.phase, 0) + 1
        print(f"  {len(observations)} obs in {elapsed:.1f}s | Phases: {dict(sorted(phase_counts.items()))}")

    # Test parallel
    print(f"\nRunning 6 parallel flights across all routes...")
    start = time.time()
    observations = runner.run_parallel_flights(6)
    elapsed = time.time() - start

    print(f"\nCollected {len(observations)} observations in {elapsed:.1f}s")

    if observations:
        # Phase distribution
        phase_counts = {}
        for obs in observations:
            phase_counts[obs.phase] = phase_counts.get(obs.phase, 0) + 1
        print(f"\nPhase distribution:")
        for phase, count in sorted(phase_counts.items(), key=lambda x: -x[1]):
            print(f"  {phase}: {count}")

        print(f"\nSample observations:")
        for i, obs in enumerate(observations[:8]):
            conf = obs.validation.get("confidence", 0)
            print(f"  {i+1}. [{obs.phase:>15}] {obs.command['action']:>8} = {obs.command.get('value', 0):>8.1f} "
                  f"| conf={conf:.2f} | {obs.outcome}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
