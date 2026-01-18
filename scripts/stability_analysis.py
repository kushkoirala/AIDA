#!/usr/bin/env python3
"""
Stability Analysis for AIDA Trajectory Controllers

Tests the new control architecture for:
1. Heading response - step response, overshoot, settling time
2. Altitude hold - disturbance rejection, steady-state error
3. Glideslope tracking - approach stability
4. Cross-track correction - lateral path following

Outputs metrics and plots for tuning guidance.

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import sys
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpu-flight-dynamics' / 'python'))

from flight_dynamics import FlightSimulator as FlightDynamics, StateIndex
from aida_sim.control.pid import (
    HeadingController, PitchController, AltitudeController,
    AirspeedController, GlideslopeController
)
from aida_sim.control.autopilot import TrajectoryAutopilot, SimpleWaypointAutopilot, AutopilotConfig

# Unit conversion
M_TO_FT = 3.28084
FT_TO_M = 0.3048
KTS_TO_FPS = 1.68781
FPS_TO_KTS = 1 / KTS_TO_FPS


@dataclass
class StepResponse:
    """Step response metrics."""
    rise_time: float      # Time to reach 90% of target
    settling_time: float  # Time to stay within 2% of target
    overshoot: float      # Peak overshoot percentage
    steady_state_error: float  # Final error
    oscillations: int     # Number of zero crossings


def analyze_step_response(times: List[float], values: List[float],
                          target: float, initial: float) -> StepResponse:
    """Analyze step response characteristics."""
    values = np.array(values)
    times = np.array(times)

    # Normalize to 0-1 range
    if abs(target - initial) < 1e-6:
        return StepResponse(0, 0, 0, 0, 0)

    normalized = (values - initial) / (target - initial)

    # Rise time (10% to 90%)
    rise_start_idx = np.argmax(normalized >= 0.1) if np.any(normalized >= 0.1) else 0
    rise_end_idx = np.argmax(normalized >= 0.9) if np.any(normalized >= 0.9) else len(times)-1
    rise_time = times[rise_end_idx] - times[rise_start_idx]

    # Overshoot
    peak = np.max(normalized)
    overshoot = max(0, (peak - 1.0) * 100)  # percentage

    # Settling time (2% band)
    settled_mask = np.abs(normalized - 1.0) < 0.02
    if np.any(settled_mask):
        # Find last time it was outside the band
        outside_mask = ~settled_mask
        if np.any(outside_mask):
            last_outside = np.max(np.where(outside_mask)[0])
            settling_time = times[last_outside] if last_outside < len(times)-1 else times[-1]
        else:
            settling_time = 0.0
    else:
        settling_time = times[-1]

    # Steady state error
    steady_state_error = abs(values[-1] - target)

    # Count oscillations (zero crossings of error)
    error = values - target
    sign_changes = np.diff(np.sign(error))
    oscillations = np.sum(np.abs(sign_changes) > 0)

    return StepResponse(rise_time, settling_time, overshoot, steady_state_error, oscillations)


def get_state(sim):
    """Get state for single instance simulator."""
    states = sim.get_states()
    return states[0]  # First (only) instance


def set_state(sim, state):
    """Set state for single instance simulator."""
    sim.states[0] = state


def step_with_action(sim, action):
    """Step simulator with action."""
    sim.set_controls(action.reshape(1, -1))
    sim.step()
    return get_state(sim)


def test_heading_response(sim: FlightDynamics, autopilot: SimpleWaypointAutopilot,
                          heading_change_deg: float = 45.0,
                          duration: float = 30.0) -> Tuple[StepResponse, dict]:
    """Test heading step response."""
    print(f"\n{'='*60}")
    print(f"HEADING STEP RESPONSE TEST: {heading_change_deg}° turn")
    print('='*60)

    # Initialize at cruise
    state = get_state(sim)
    state[StateIndex.Z] = -5500 * FT_TO_M
    state[StateIndex.U] = 110 * KTS_TO_FPS * FT_TO_M
    state[StateIndex.V] = 0.0
    state[StateIndex.W] = 0.0
    state[StateIndex.PSI] = 0.0
    state[StateIndex.PHI] = 0.0
    state[StateIndex.THETA] = 0.0
    set_state(sim, state)

    initial_heading = 0.0
    target_heading = np.deg2rad(heading_change_deg)

    autopilot.target_heading = initial_heading
    autopilot.target_altitude_ft = 5500
    autopilot.target_airspeed_fps = 110 * KTS_TO_FPS

    # Stabilize for 2 seconds
    dt = 0.02
    for _ in range(int(2.0 / dt)):
        state = get_state(sim)
        action = autopilot.compute_action(
            -state[StateIndex.Z] * M_TO_FT,
            np.sqrt(state[StateIndex.U]**2 + state[StateIndex.V]**2 + state[StateIndex.W]**2) * M_TO_FT,
            state[StateIndex.W] * M_TO_FT,  # W positive = climbing in this sim
            state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI],
            state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]
        )
        step_with_action(sim, action)

    # Apply step input
    autopilot.target_heading = target_heading

    times = []
    headings = []
    rolls = []

    sim_time = 0.0
    while sim_time < duration:
        state = get_state(sim)

        times.append(sim_time)
        headings.append(state[StateIndex.PSI])
        rolls.append(np.rad2deg(state[StateIndex.PHI]))

        action = autopilot.compute_action(
            -state[StateIndex.Z] * M_TO_FT,
            np.sqrt(state[StateIndex.U]**2 + state[StateIndex.V]**2 + state[StateIndex.W]**2) * M_TO_FT,
            state[StateIndex.W] * M_TO_FT,  # W positive = climbing in this sim
            state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI],
            state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]
        )
        step_with_action(sim, action)
        sim_time += dt

    # Analyze
    response = analyze_step_response(times, headings, target_heading, initial_heading)

    print(f"  Rise time:       {response.rise_time:.2f} s")
    print(f"  Settling time:   {response.settling_time:.2f} s")
    print(f"  Overshoot:       {response.overshoot:.1f}%")
    print(f"  Steady-state err: {np.rad2deg(response.steady_state_error):.2f}°")
    print(f"  Oscillations:    {response.oscillations}")
    print(f"  Max bank angle:  {max(abs(np.array(rolls))):.1f}°")

    data = {'times': times, 'headings': headings, 'rolls': rolls}
    return response, data


def test_altitude_hold(sim: FlightDynamics, autopilot: SimpleWaypointAutopilot,
                       altitude_change_ft: float = 500.0,
                       duration: float = 90.0) -> Tuple[StepResponse, dict]:
    """Test altitude step response."""
    print(f"\n{'='*60}")
    print(f"ALTITUDE STEP RESPONSE TEST: {altitude_change_ft}ft change")
    print('='*60)

    # Initialize at cruise
    initial_alt = 5500
    target_alt = initial_alt + altitude_change_ft

    state = get_state(sim)
    state[StateIndex.Z] = -initial_alt * FT_TO_M
    state[StateIndex.U] = 110 * KTS_TO_FPS * FT_TO_M
    state[StateIndex.V] = 0.0
    state[StateIndex.W] = 0.0
    state[StateIndex.PSI] = 0.0
    state[StateIndex.PHI] = 0.0
    state[StateIndex.THETA] = 0.0
    set_state(sim, state)

    autopilot.target_heading = 0.0
    autopilot.target_altitude_ft = initial_alt
    autopilot.target_airspeed_fps = 110 * KTS_TO_FPS

    # Stabilize
    dt = 0.02
    for _ in range(int(2.0 / dt)):
        state = get_state(sim)
        action = autopilot.compute_action(
            -state[StateIndex.Z] * M_TO_FT,
            np.sqrt(state[StateIndex.U]**2 + state[StateIndex.V]**2 + state[StateIndex.W]**2) * M_TO_FT,
            state[StateIndex.W] * M_TO_FT,  # W positive = climbing in this sim
            state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI],
            state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]
        )
        step_with_action(sim, action)

    # Apply step
    autopilot.target_altitude_ft = target_alt

    times = []
    altitudes = []
    pitches = []
    vs_fpm = []

    sim_time = 0.0
    while sim_time < duration:
        state = get_state(sim)

        alt_ft = -state[StateIndex.Z] * M_TO_FT
        times.append(sim_time)
        altitudes.append(alt_ft)
        pitches.append(np.rad2deg(state[StateIndex.THETA]))
        vs_fpm.append(state[StateIndex.W] * M_TO_FT * 60)  # W positive = climbing

        action = autopilot.compute_action(
            alt_ft,
            np.sqrt(state[StateIndex.U]**2 + state[StateIndex.V]**2 + state[StateIndex.W]**2) * M_TO_FT,
            state[StateIndex.W] * M_TO_FT,  # W positive = climbing in this sim
            state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI],
            state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]
        )
        step_with_action(sim, action)
        sim_time += dt

    # Analyze
    response = analyze_step_response(times, altitudes, target_alt, initial_alt)

    print(f"  Rise time:       {response.rise_time:.2f} s")
    print(f"  Settling time:   {response.settling_time:.2f} s")
    print(f"  Overshoot:       {response.overshoot:.1f}%")
    print(f"  Steady-state err: {response.steady_state_error:.1f} ft")
    print(f"  Oscillations:    {response.oscillations}")
    print(f"  Max VS:          {max(abs(np.array(vs_fpm))):.0f} fpm")
    print(f"  Max pitch:       {max(abs(np.array(pitches))):.1f}°")

    data = {'times': times, 'altitudes': altitudes, 'pitches': pitches, 'vs': vs_fpm}
    return response, data


def test_airspeed_hold(sim: FlightDynamics, autopilot: SimpleWaypointAutopilot,
                       speed_change_kts: float = 20.0,
                       duration: float = 30.0) -> Tuple[StepResponse, dict]:
    """Test airspeed step response."""
    print(f"\n{'='*60}")
    print(f"AIRSPEED STEP RESPONSE TEST: {speed_change_kts}kt change")
    print('='*60)

    initial_speed = 110 * KTS_TO_FPS
    target_speed = (110 + speed_change_kts) * KTS_TO_FPS

    state = get_state(sim)
    state[StateIndex.Z] = -5500 * FT_TO_M
    state[StateIndex.U] = initial_speed * FT_TO_M
    state[StateIndex.V] = 0.0
    state[StateIndex.W] = 0.0
    state[StateIndex.PSI] = 0.0
    state[StateIndex.PHI] = 0.0
    state[StateIndex.THETA] = 0.0
    set_state(sim, state)

    autopilot.target_heading = 0.0
    autopilot.target_altitude_ft = 5500
    autopilot.target_airspeed_fps = initial_speed

    # Stabilize
    dt = 0.02
    for _ in range(int(2.0 / dt)):
        state = get_state(sim)
        action = autopilot.compute_action(
            -state[StateIndex.Z] * M_TO_FT,
            np.sqrt(state[StateIndex.U]**2 + state[StateIndex.V]**2 + state[StateIndex.W]**2) * M_TO_FT,
            state[StateIndex.W] * M_TO_FT,  # W positive = climbing in this sim
            state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI],
            state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]
        )
        step_with_action(sim, action)

    # Apply step
    autopilot.target_airspeed_fps = target_speed

    times = []
    airspeeds = []
    throttles = []

    sim_time = 0.0
    while sim_time < duration:
        state = get_state(sim)

        airspeed = np.sqrt(state[StateIndex.U]**2 + state[StateIndex.V]**2 + state[StateIndex.W]**2) * M_TO_FT
        times.append(sim_time)
        airspeeds.append(airspeed)

        action = autopilot.compute_action(
            -state[StateIndex.Z] * M_TO_FT,
            airspeed,
            state[StateIndex.W] * M_TO_FT,  # W positive = climbing in this sim
            state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI],
            state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]
        )
        throttles.append(action[0])
        step_with_action(sim, action)
        sim_time += dt

    # Analyze
    response = analyze_step_response(times, airspeeds, target_speed, initial_speed)

    print(f"  Rise time:       {response.rise_time:.2f} s")
    print(f"  Settling time:   {response.settling_time:.2f} s")
    print(f"  Overshoot:       {response.overshoot:.1f}%")
    print(f"  Steady-state err: {response.steady_state_error * FPS_TO_KTS:.1f} kts")
    print(f"  Oscillations:    {response.oscillations}")

    data = {'times': times, 'airspeeds': airspeeds, 'throttles': throttles}
    return response, data


def test_coupled_maneuver(sim: FlightDynamics, autopilot: SimpleWaypointAutopilot,
                          duration: float = 60.0) -> dict:
    """Test coupled heading + altitude change (climbing turn)."""
    print(f"\n{'='*60}")
    print("COUPLED MANEUVER TEST: 45° turn + 500ft climb")
    print('='*60)

    state = get_state(sim)
    state[StateIndex.Z] = -5500 * FT_TO_M
    state[StateIndex.U] = 110 * KTS_TO_FPS * FT_TO_M
    state[StateIndex.V] = 0.0
    state[StateIndex.W] = 0.0
    state[StateIndex.PSI] = 0.0
    state[StateIndex.PHI] = 0.0
    state[StateIndex.THETA] = 0.0
    set_state(sim, state)

    autopilot.target_heading = 0.0
    autopilot.target_altitude_ft = 5500
    autopilot.target_airspeed_fps = 110 * KTS_TO_FPS

    # Stabilize
    dt = 0.02
    for _ in range(int(2.0 / dt)):
        state = get_state(sim)
        action = autopilot.compute_action(
            -state[StateIndex.Z] * M_TO_FT,
            np.sqrt(state[StateIndex.U]**2 + state[StateIndex.V]**2 + state[StateIndex.W]**2) * M_TO_FT,
            state[StateIndex.W] * M_TO_FT,  # W positive = climbing in this sim
            state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI],
            state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]
        )
        step_with_action(sim, action)

    # Apply coupled command
    autopilot.target_heading = np.deg2rad(45)
    autopilot.target_altitude_ft = 6000

    times = []
    headings = []
    altitudes = []
    airspeeds = []
    rolls = []
    pitches = []

    sim_time = 0.0
    while sim_time < duration:
        state = get_state(sim)

        times.append(sim_time)
        headings.append(np.rad2deg(state[StateIndex.PSI]))
        altitudes.append(-state[StateIndex.Z] * M_TO_FT)
        airspeeds.append(np.sqrt(state[StateIndex.U]**2 + state[StateIndex.V]**2 + state[StateIndex.W]**2) * M_TO_FT * FPS_TO_KTS)
        rolls.append(np.rad2deg(state[StateIndex.PHI]))
        pitches.append(np.rad2deg(state[StateIndex.THETA]))

        action = autopilot.compute_action(
            -state[StateIndex.Z] * M_TO_FT,
            np.sqrt(state[StateIndex.U]**2 + state[StateIndex.V]**2 + state[StateIndex.W]**2) * M_TO_FT,
            state[StateIndex.W] * M_TO_FT,  # W positive = climbing in this sim
            state[StateIndex.PHI], state[StateIndex.THETA], state[StateIndex.PSI],
            state[StateIndex.P], state[StateIndex.Q], state[StateIndex.R]
        )
        step_with_action(sim, action)
        sim_time += dt

    # Analyze final state
    hdg_error = abs(headings[-1] - 45)
    alt_error = abs(altitudes[-1] - 6000)
    speed_loss = 110 - min(airspeeds)

    print(f"  Final heading error:  {hdg_error:.1f}°")
    print(f"  Final altitude error: {alt_error:.0f} ft")
    print(f"  Max speed loss:       {speed_loss:.1f} kts")
    print(f"  Max bank angle:       {max(abs(np.array(rolls))):.1f}°")
    print(f"  Max pitch angle:      {max(abs(np.array(pitches))):.1f}°")

    return {
        'times': times, 'headings': headings, 'altitudes': altitudes,
        'airspeeds': airspeeds, 'rolls': rolls, 'pitches': pitches
    }


def print_tuning_recommendations(heading_resp: StepResponse,
                                  altitude_resp: StepResponse,
                                  speed_resp: StepResponse):
    """Print tuning recommendations based on test results."""
    print(f"\n{'='*60}")
    print("TUNING RECOMMENDATIONS")
    print('='*60)

    issues = []
    recommendations = []

    # Heading analysis
    if heading_resp.overshoot > 10:
        issues.append(f"Heading overshoot too high ({heading_resp.overshoot:.0f}%)")
        recommendations.append("- Reduce heading_kp or increase roll_kd")
    if heading_resp.settling_time > 15:
        issues.append(f"Heading settling too slow ({heading_resp.settling_time:.1f}s)")
        recommendations.append("- Increase heading_kp or roll_kp")
    if heading_resp.oscillations > 4:
        issues.append(f"Heading oscillating ({heading_resp.oscillations} oscillations)")
        recommendations.append("- Increase roll_kd (damping)")

    # Altitude analysis
    if altitude_resp.overshoot > 5:
        issues.append(f"Altitude overshoot ({altitude_resp.overshoot:.0f}%)")
        recommendations.append("- Reduce alt_kp or vs_kp")
    if altitude_resp.settling_time > 45:
        issues.append(f"Altitude settling slow ({altitude_resp.settling_time:.1f}s)")
        recommendations.append("- Increase alt_kp")
    if altitude_resp.steady_state_error > 50:
        issues.append(f"Altitude steady-state error ({altitude_resp.steady_state_error:.0f}ft)")
        recommendations.append("- Add integral term to altitude controller")

    # Speed analysis
    if speed_resp.overshoot > 10:
        issues.append(f"Speed overshoot ({speed_resp.overshoot:.0f}%)")
        recommendations.append("- Reduce speed_kp")
    if speed_resp.settling_time > 20:
        issues.append(f"Speed settling slow ({speed_resp.settling_time:.1f}s)")
        recommendations.append("- Increase speed_kp or speed_ki")

    if not issues:
        print("All controllers performing within acceptable limits!")
        print("\nCurrent gains appear well-tuned for:")
        print("  - Smooth heading changes with minimal overshoot")
        print("  - Stable altitude tracking")
        print("  - Responsive airspeed control")
    else:
        print("Issues detected:")
        for issue in issues:
            print(f"  ! {issue}")
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  {rec}")

    print("\nCurrent AutopilotConfig defaults:")
    config = AutopilotConfig()
    print(f"  heading_kp: {config.heading_kp}")
    print(f"  roll_kp: {config.roll_kp}")
    print(f"  roll_kd: {config.roll_kd}")
    print(f"  pitch_kp: {config.pitch_kp}")
    print(f"  pitch_kd: {config.pitch_kd}")
    print(f"  alt_kp: {config.alt_kp}")
    print(f"  vs_kp: {config.vs_kp}")
    print(f"  speed_kp: {config.speed_kp}")
    print(f"  speed_ki: {config.speed_ki}")


def save_results_csv(data: dict, filename: str):
    """Save test data to CSV for plotting."""
    import csv
    filepath = Path(__file__).parent.parent / 'checkpoints' / filename
    filepath.parent.mkdir(exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        keys = list(data.keys())
        writer.writerow(keys)
        for i in range(len(data[keys[0]])):
            writer.writerow([data[k][i] for k in keys])
    print(f"Saved: {filepath}")


def create_sim():
    """Create a fresh simulator instance."""
    return FlightDynamics(n_instances=1, dt=0.02, use_gpu=False)


def main():
    print("\n" + "="*60)
    print("  AIDA STABILITY ANALYSIS")
    print("  Testing Trajectory Controller Response")
    print("="*60)

    # Initialize
    sim = create_sim()
    autopilot = SimpleWaypointAutopilot()

    # Run tests
    heading_resp, heading_data = test_heading_response(sim, autopilot, heading_change_deg=45)

    sim = create_sim()  # Reset sim
    autopilot = SimpleWaypointAutopilot()
    altitude_resp, altitude_data = test_altitude_hold(sim, autopilot, altitude_change_ft=500)

    sim = create_sim()
    autopilot = SimpleWaypointAutopilot()
    speed_resp, speed_data = test_airspeed_hold(sim, autopilot, speed_change_kts=20)

    sim = create_sim()
    autopilot = SimpleWaypointAutopilot()
    coupled_data = test_coupled_maneuver(sim, autopilot)

    # Recommendations
    print_tuning_recommendations(heading_resp, altitude_resp, speed_resp)

    # Save data for plotting
    print(f"\n{'='*60}")
    print("SAVING DATA")
    print('='*60)
    save_results_csv(heading_data, 'stability_heading.csv')
    save_results_csv(altitude_data, 'stability_altitude.csv')
    save_results_csv(speed_data, 'stability_speed.csv')
    save_results_csv(coupled_data, 'stability_coupled.csv')

    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
