#!/usr/bin/env python3
"""
Test Route Generation Transformer

This script tests the trained route generation transformer by:
1. Loading the trained model
2. Generating controller parameters for various airport pairs
3. Comparing generated parameters against expert baselines
4. Optionally running a simulated flight with generated parameters

Usage:
    python test_route_generation.py --checkpoint checkpoints/route_gen/best_model.pt
    python test_route_generation.py --run-flight  # Run actual flight simulation

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpu-flight-dynamics' / 'python'))

from aida_sim.models.unified_transformer import (
    UnifiedTransformer, ModelConfig, TaskType, denormalize_controller_params
)

# Test airport pairs
TEST_ROUTES = [
    # (origin_name, origin_lat, origin_lon, origin_elev, origin_rwy,
    #  dest_name, dest_lat, dest_lon, dest_elev, dest_rwy, expected_distance_nm)
    ("KHUT", 38.066, -97.860, 1542, 314, "KAAO", 37.747, -97.221, 1421, 180, 36),
    ("SN65", 37.594, -97.616, 1448, 4, "KICT", 37.650, -97.433, 1333, 14, 12),
    ("KAAO", 37.747, -97.221, 1421, 180, "KEWK", 38.058, -97.276, 1533, 174, 21),
    ("KICT", 37.650, -97.433, 1333, 14, "KHUT", 38.066, -97.860, 1542, 314, 35),
    ("SN65", 37.594, -97.616, 1448, 4, "KHUT", 38.066, -97.860, 1542, 314, 35),
]


def create_route_spec(
    origin_lat: float, origin_lon: float, origin_elev: float, origin_rwy: float,
    dest_lat: float, dest_lon: float, dest_elev: float, dest_rwy: float,
    cruise_alt: float = 3500, distance_nm: float = 30
) -> torch.Tensor:
    """Create normalized route specification tensor."""
    return torch.tensor([[
        origin_lat / 90.0,
        origin_lon / 180.0,
        origin_elev / 10000.0,
        origin_rwy / 360.0,
        dest_lat / 90.0,
        dest_lon / 180.0,
        dest_elev / 10000.0,
        dest_rwy / 360.0,
        cruise_alt / 10000.0,
        distance_nm / 200.0,
    ]], dtype=torch.float32)


def evaluate_parameters(params: Dict[str, float], distance_nm: float) -> Dict[str, float]:
    """Evaluate quality of generated parameters.

    Returns scores for each parameter (0-1, higher is better).
    """
    scores = {}

    # Turn point distance (should scale with route distance)
    ideal_tp = min(15, max(5, distance_nm * 0.2))
    tp_error = abs(params['tp_distance_nm'] - ideal_tp) / ideal_tp
    scores['tp_distance'] = max(0, 1 - tp_error)

    # Glideslope (3° is standard)
    gs_error = abs(params['glideslope_deg'] - 3.0) / 2.0
    scores['glideslope'] = max(0, 1 - gs_error)

    # Pattern altitude (1000 ft AGL is standard)
    pat_error = abs(params['pattern_alt_ft'] - 1000) / 500
    scores['pattern_alt'] = max(0, 1 - pat_error)

    # Cruise altitude (should scale with distance)
    ideal_cruise = min(10000, max(2500, 1500 + distance_nm * 80))
    cruise_error = abs(params['cruise_alt_ft'] - ideal_cruise) / ideal_cruise
    scores['cruise_alt'] = max(0, 1 - cruise_error)

    # Approach speed (65-75 kts is typical for light aircraft)
    v_app_error = abs(params['v_approach_kts'] - 70) / 20
    scores['approach_speed'] = max(0, 1 - v_app_error)

    # Cruise speed (should be reasonable)
    ideal_cruise_spd = min(130, max(90, 80 + distance_nm * 0.5))
    v_cruise_error = abs(params['v_cruise_kts'] - ideal_cruise_spd) / 30
    scores['cruise_speed'] = max(0, 1 - v_cruise_error)

    # Overall score
    scores['overall'] = np.mean(list(scores.values()))

    return scores


def test_model(model: UnifiedTransformer, device: torch.device, verbose: bool = True):
    """Test model on all test routes."""
    model.eval()
    results = []

    print("\n" + "=" * 80)
    print("ROUTE GENERATION TEST RESULTS")
    print("=" * 80)

    for route in TEST_ROUTES:
        origin_name, o_lat, o_lon, o_elev, o_rwy = route[:5]
        dest_name, d_lat, d_lon, d_elev, d_rwy, distance = route[5:]

        # Determine appropriate cruise altitude
        cruise_alt = min(8000, max(2500, 1500 + distance * 80))

        # Create route spec
        route_spec = create_route_spec(
            o_lat, o_lon, o_elev, o_rwy,
            d_lat, d_lon, d_elev, d_rwy,
            cruise_alt, distance
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model(TaskType.ROUTE_GENERATION, route_spec=route_spec)

        # Denormalize parameters
        ctrl_norm = outputs['controller_params'].cpu()
        params = denormalize_controller_params(ctrl_norm)

        # Convert to dict of floats
        params_dict = {k: v.item() for k, v in params.items()}

        # Evaluate
        scores = evaluate_parameters(params_dict, distance)

        # Store result
        results.append({
            'route': f"{origin_name} -> {dest_name}",
            'distance_nm': distance,
            'params': params_dict,
            'scores': scores,
        })

        if verbose:
            print(f"\n{origin_name} -> {dest_name} ({distance} nm)")
            print("-" * 40)
            print(f"  Turn Point:     {params_dict['tp_distance_nm']:6.1f} nm  (score: {scores['tp_distance']:.2f})")
            print(f"  Glideslope:     {params_dict['glideslope_deg']:6.1f}°   (score: {scores['glideslope']:.2f})")
            print(f"  Pattern Alt:    {params_dict['pattern_alt_ft']:6.0f} ft  (score: {scores['pattern_alt']:.2f})")
            print(f"  Cruise Alt:     {params_dict['cruise_alt_ft']:6.0f} ft  (score: {scores['cruise_alt']:.2f})")
            print(f"  Approach Spd:   {params_dict['v_approach_kts']:6.0f} kts (score: {scores['approach_speed']:.2f})")
            print(f"  Cruise Spd:     {params_dict['v_cruise_kts']:6.0f} kts (score: {scores['cruise_speed']:.2f})")
            print(f"  Bank Angle:     {params_dict['turn_bank_deg']:6.0f}°")
            print(f"  Triangle Pat:   {params_dict['use_triangle_pattern']:.2f}")
            print(f"  OVERALL SCORE:  {scores['overall']:.2f}")

    # Summary
    avg_score = np.mean([r['scores']['overall'] for r in results])
    print("\n" + "=" * 80)
    print(f"AVERAGE OVERALL SCORE: {avg_score:.2f}")
    print("=" * 80)

    return results


def test_waypoint_generation(model: UnifiedTransformer, device: torch.device):
    """Test waypoint generation quality."""
    model.eval()

    print("\n" + "=" * 80)
    print("WAYPOINT GENERATION TEST")
    print("=" * 80)

    # Test on KHUT -> KAAO route
    route_spec = create_route_spec(
        38.066, -97.860, 1542, 314,  # KHUT
        37.747, -97.221, 1421, 180,  # KAAO
        3500, 36
    ).to(device)

    with torch.no_grad():
        outputs = model(TaskType.ROUTE_GENERATION, route_spec=route_spec)

    waypoints = outputs['waypoints'].cpu().squeeze(0).numpy()
    validity = outputs['validity'].cpu().squeeze(0).numpy()
    phase_logits = outputs['phase_logits'].cpu().squeeze(0).numpy()

    print("\nKHUT -> KAAO Waypoints:")
    print("-" * 70)
    print(f"{'WP':>3} | {'Valid':>5} | {'Lat':>8} | {'Lon':>8} | {'Alt':>8} | {'Spd':>6} | {'Hdg':>6} | Phase")
    print("-" * 70)

    phase_names = ['Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach', 'Landing']

    for i in range(len(waypoints)):
        valid = validity[i, 0] > 0.5
        lat_delta = waypoints[i, 0] * 2  # Denormalize
        lon_delta = waypoints[i, 1] * 2
        alt = waypoints[i, 2] * 10000
        speed = waypoints[i, 3] * 200
        hdg = np.rad2deg(waypoints[i, 4])
        phase_idx = np.argmax(phase_logits[i])
        phase = phase_names[phase_idx]

        valid_str = "YES" if valid else "no"
        print(f"{i+1:3d} | {valid_str:>5} | {lat_delta:+8.4f} | {lon_delta:+8.4f} | {alt:8.0f} | {speed:6.0f} | {hdg:6.1f} | {phase}")

    print("-" * 70)
    active_count = (validity > 0.5).sum()
    print(f"Active waypoints: {active_count}/8")


def run_flight_test(model: UnifiedTransformer, device: torch.device):
    """Run a simulated flight using generated parameters."""
    try:
        from generalized_xc_controller import (
            GeneralizedXCController, AirportConfig, KANSAS_AIRPORTS
        )
        from flight_dynamics import FlightSimulator, StateIndex
    except ImportError as e:
        print(f"\nCannot run flight test: {e}")
        print("Flight dynamics module not available.")
        return

    model.eval()

    print("\n" + "=" * 80)
    print("FLIGHT SIMULATION TEST")
    print("=" * 80)

    # Generate parameters for KHUT -> KAAO
    origin = KANSAS_AIRPORTS["KHUT"]
    dest = KANSAS_AIRPORTS["KAAO"]

    route_spec = create_route_spec(
        origin.lat, origin.lon, origin.elevation_ft, origin.runway_heading_deg,
        dest.lat, dest.lon, dest.elevation_ft, dest.runway_heading_deg,
        3500, 36
    ).to(device)

    with torch.no_grad():
        outputs = model(TaskType.ROUTE_GENERATION, route_spec=route_spec)

    params = denormalize_controller_params(outputs['controller_params'].cpu())

    print(f"\nGenerated parameters for KHUT -> KAAO:")
    print(f"  Cruise altitude: {params['cruise_alt_ft'].item():.0f} ft")
    print(f"  Turn point: {params['tp_distance_nm'].item():.1f} nm")
    print(f"  Pattern altitude: {params['pattern_alt_ft'].item():.0f} ft")

    # Create controller with generated parameters
    controller = GeneralizedXCController(
        origin=origin,
        destination=dest,
        cruise_altitude_ft=params['cruise_alt_ft'].item(),
        pattern_altitude_ft=params['pattern_alt_ft'].item(),
        tp_distance_nm=params['tp_distance_nm'].item()
    )

    # Run brief simulation
    dt = 0.02
    sim = FlightSimulator(n_instances=1, dt=dt, use_gpu=False)

    # Initialize state on runway
    initial_state = np.zeros(12)
    initial_state[StateIndex.X] = origin.x_ft * 0.3048
    initial_state[StateIndex.Y] = origin.y_ft * 0.3048
    initial_state[StateIndex.Z] = -10.0  # 10m AGL
    initial_state[StateIndex.PSI] = np.deg2rad(origin.runway_heading_deg)

    sim.reset(np.array([initial_state]))

    print(f"\nRunning 30 seconds of simulation...")

    sim_time = 0.0
    last_phase = None

    for step in range(int(30.0 / dt)):
        state = sim.get_states()[0]
        action = controller.compute_action(state, sim_time)
        sim.set_controls(action.reshape(1, -1))
        sim.step()
        sim_time += dt

        if controller.phase != last_phase:
            alt_ft = -state[StateIndex.Z] * 3.28084
            spd_kts = state[StateIndex.U] * 1.944
            print(f"  [{sim_time:5.1f}s] {controller.phase.name:18s} | Alt: {alt_ft:5.0f}ft | Spd: {spd_kts:5.1f}kts")
            last_phase = controller.phase

    print(f"\nFinal state after 30s:")
    state = sim.get_states()[0]
    print(f"  Altitude: {-state[StateIndex.Z] * 3.28084:.0f} ft AGL")
    print(f"  Airspeed: {state[StateIndex.U] * 1.944:.1f} kts")
    print(f"  Phase: {controller.phase.name}")
    print("\nFlight simulation test complete!")


def main():
    parser = argparse.ArgumentParser(description="Test route generation transformer")
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/route_gen/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--run-flight", action="store_true",
                       help="Run actual flight simulation test")
    parser.add_argument("--test-waypoints", action="store_true",
                       help="Test waypoint generation")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Route Generation Transformer Test")
    print("=" * 80)
    print(f"Device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent.parent / args.checkpoint

    config = ModelConfig(
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )
    model = UnifiedTransformer(config)

    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Val loss: {checkpoint.get('val_losses', {}).get('loss', 'N/A')}")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("Using untrained model (random weights)")

    model = model.to(device)

    # Run tests
    test_model(model, device)

    if args.test_waypoints:
        test_waypoint_generation(model, device)

    if args.run_flight:
        run_flight_test(model, device)

    print("\n" + "=" * 80)
    print("All tests complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
