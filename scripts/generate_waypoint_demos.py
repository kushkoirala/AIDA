#!/usr/bin/env python3
"""
Generate Waypoint-Labeled Expert Demonstrations

Uses the classical triangle_controller to fly missions,
but records data WITH waypoint labels for training the NN.

Output format:
    observations: aircraft state (12D)
    actions: expert controls (4D)
    waypoints: current waypoint command (7D)
        [target_x, target_y, target_alt, target_speed, wp_type, dist_to_wp, bearing_error]
        bearing_error = bearing_to_wp - current_heading (normalized to [-pi, pi])

This teaches the NN:
    "When the aircraft is in state X and needs to reach waypoint Y, use controls Z"

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpu-flight-dynamics' / 'python'))
sys.path.insert(0, str(Path(__file__).parent))  # Add scripts folder for triangle_controller

from flight_dynamics import FlightSimulator, StateIndex, ControlIndex, STATE_DIM, CONTROL_DIM
from aircraft_database import get_aircraft

# Import classical controller (from scripts folder)
from triangle_controller import TriangleInterceptController, XCPhase

# Unit conversions
M_TO_FT = 3.28084
FT_TO_M = 0.3048
KTS_TO_MS = 0.514444


class WaypointDemoGenerator:
    """
    Generates expert demonstrations with waypoint labels.

    Extracts implicit waypoints from the classical controller's behavior:
    - Ground roll → takeoff waypoint
    - Climb → altitude waypoint
    - Cruise to TP → turning point waypoint
    - Turn to intercept → intercept waypoint
    - Final approach → runway waypoint
    - Landing → touchdown waypoint
    """

    def __init__(self, controller: TriangleInterceptController):
        self.controller = controller

        # Define waypoints based on controller's flight plan
        self.phase_waypoints = self._define_waypoints()

    def _define_waypoints(self):
        """
        Define waypoints for each flight phase.
        These are extracted from the controller's internal targets.
        """
        c = self.controller

        waypoints = {
            XCPhase.GROUND_ROLL: {
                'x': 0.0,  # Runway end
                'y': 0.0,
                'altitude': 0.0,
                'speed': c.v_rotate * FT_TO_M,  # Convert ft/s to m/s
                'type': 4,  # TAKEOFF
            },
            XCPhase.ROTATION: {
                'x': 200.0,  # ~200m down runway
                'y': 0.0,
                'altitude': 30.0,  # 100 ft
                'speed': c.v_climb * FT_TO_M,
                'type': 0,  # FLYOVER
            },
            XCPhase.INITIAL_CLIMB: {
                'x': 1000.0,
                'y': 0.0,
                'altitude': 300.0,  # ~1000 ft
                'speed': c.v_climb * FT_TO_M,
                'type': 0,
            },
            XCPhase.CLIMB: {
                'x': 5000.0,  # 5km out
                'y': 0.0,
                'altitude': c.cruise_altitude,
                'speed': c.v_climb * FT_TO_M,
                'type': 0,
            },
            XCPhase.CRUISE_TO_TP: {
                'x': c.tp_x,  # Turning point
                'y': c.tp_y,
                'altitude': c.cruise_altitude,
                'speed': c.v_cruise * FT_TO_M,
                'type': 1,  # FLYBY (turn before reaching)
            },
            XCPhase.TURN_TO_INTERCEPT: {
                'x': c.threshold_x,  # Aim at runway
                'y': c.threshold_y,
                'altitude': c.cruise_altitude,
                'speed': c.v_cruise * FT_TO_M,
                'type': 0,
            },
            XCPhase.INTERCEPT_LEG: {
                'x': c.threshold_x,
                'y': c.threshold_y,
                'altitude': c.pattern_altitude,
                'speed': c.v_approach * FT_TO_M,
                'type': 0,
            },
            XCPhase.FINAL_APPROACH: {
                'x': c.aimpoint_x,
                'y': c.aimpoint_y,
                'altitude': 30.0,  # 100 ft
                'speed': c.v_approach * FT_TO_M,
                'type': 0,
            },
            XCPhase.SHORT_FINAL: {
                'x': c.aimpoint_x,
                'y': c.aimpoint_y,
                'altitude': 10.0,  # 30 ft
                'speed': c.v_touchdown * FT_TO_M,
                'type': 3,  # LAND
            },
            XCPhase.LANDING: {
                'x': c.aimpoint_x + 100.0,  # Touchdown point
                'y': c.aimpoint_y,
                'altitude': 0.0,
                'speed': c.v_touchdown * FT_TO_M * 0.8,
                'type': 3,  # LAND
            },
            XCPhase.LANDED: {
                'x': c.aimpoint_x + 200.0,
                'y': c.aimpoint_y,
                'altitude': 0.0,
                'speed': 0.0,
                'type': 3,
            },
        }

        return waypoints

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi: angle -= 2*np.pi
        while angle < -np.pi: angle += 2*np.pi
        return angle

    def get_current_waypoint(self, phase: XCPhase, state: np.ndarray):
        """
        Get the current waypoint command based on flight phase and state.

        Returns (7D):
            [target_x, target_y, target_alt, target_speed, wp_type, dist_to_wp, target_heading_error]

        target_heading_error is the difference between target heading (what expert tracks)
        and current heading, normalized to [-pi, pi]. Positive = turn right, negative = turn left.

        IMPORTANT: This uses the SAME target heading logic as the expert controller!
        """
        c = self.controller
        wp = self.phase_waypoints.get(phase, self.phase_waypoints[XCPhase.CRUISE_TO_TP])

        # Calculate distance to waypoint
        x, y = state[StateIndex.X], state[StateIndex.Y]
        dx = wp['x'] - x
        dy = wp['y'] - y
        dist = np.sqrt(dx**2 + dy**2)

        # Current heading (psi)
        current_heading = state[StateIndex.PSI]

        # Determine target heading based on phase (MATCHING the expert controller logic!)
        if phase == XCPhase.GROUND_ROLL:
            target_heading = 0.0  # Runway heading
        elif phase == XCPhase.ROTATION:
            target_heading = c.departure_heading
        elif phase == XCPhase.INITIAL_CLIMB:
            target_heading = c.departure_heading
        elif phase == XCPhase.CLIMB:
            # Expert uses cruise_heading after turn altitude, departure before
            alt = -state[StateIndex.Z]
            turn_alt_m = c.turn_to_cruise_altitude_ft * 0.3048  # ft to m
            target_heading = c.cruise_heading if alt > turn_alt_m else c.departure_heading
        elif phase == XCPhase.CRUISE_TO_TP:
            # Expert tracks bearing to TP
            target_heading = np.arctan2(c.tp_y - y, c.tp_x - x)
        elif phase == XCPhase.TURN_TO_INTERCEPT:
            target_heading = c.runway_heading
        elif phase in (XCPhase.INTERCEPT_LEG, XCPhase.FINAL_APPROACH, XCPhase.SHORT_FINAL):
            # Expert uses runway heading with cross-track correction
            target_heading = c.runway_heading
        elif phase == XCPhase.LANDING:
            target_heading = c.runway_heading
        else:
            target_heading = current_heading  # Default: maintain heading

        # Heading error: positive = need to turn right, negative = turn left
        heading_error = self._normalize_angle(target_heading - current_heading)

        return np.array([
            wp['x'],
            wp['y'],
            wp['altitude'],
            wp['speed'],
            float(wp['type']),
            dist,
            heading_error,
        ], dtype=np.float32)


def generate_demos(num_episodes=50, output_dir=None):
    """
    Generate waypoint-labeled expert demonstrations.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'bc_data'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"waypoint_demos_{timestamp}.npz"

    import builtins
    import functools
    _print = functools.partial(builtins.print, flush=True)

    _print("=" * 70)
    _print("  GENERATING WAYPOINT-LABELED EXPERT DEMOS")
    _print("=" * 70)
    _print(f"Episodes: {num_episodes}")
    _print(f"Output: {output_path}")
    _print()

    # Setup simulator - use default params (works with triangle controller)
    dt = 0.02
    sim = FlightSimulator(n_instances=1, dt=dt, use_gpu=False)

    # Setup controller and waypoint generator
    controller = TriangleInterceptController()
    wp_generator = WaypointDemoGenerator(controller)

    # Storage
    all_observations = []
    all_actions = []
    all_waypoints = []
    all_phases = []

    successful_episodes = 0

    for ep in range(num_episodes):
        # Reset - use same initial conditions as run_xc_sn65_khut.py
        initial_state = np.zeros((1, 12), dtype=np.float32)
        initial_state[0, 0] = -400.0  # X position (runway)
        initial_state[0, 3] = 5.0     # U (taxi speed)

        sim.reset(initial_state)
        controller.reset()

        ep_obs = []
        ep_actions = []
        ep_waypoints = []
        ep_phases = []

        sim_time = 0.0
        max_steps = 60000  # 20 minutes (full SN65->KHUT ~15 minutes)

        for step in range(max_steps):
            # Get state (CPU mode returns numpy directly)
            state = sim.get_states()[0]

            # Get expert action
            action = controller.compute_action(state, sim_time)

            # Get current waypoint command
            waypoint = wp_generator.get_current_waypoint(controller.phase, state)

            # Store
            ep_obs.append(state.copy())
            ep_actions.append(action[:4].copy())  # throttle, aileron, elevator, rudder
            ep_waypoints.append(waypoint.copy())
            ep_phases.append(controller.phase.value)

            # Apply action (reshape for set_controls)
            sim.set_controls(action.reshape(1, -1))
            sim.step()

            sim_time += dt

            # Check if landed
            if controller.phase == XCPhase.LANDED:
                successful_episodes += 1
                break

            # Check for crash
            altitude = -state[StateIndex.Z]
            if altitude < -10:
                break

        all_observations.extend(ep_obs)
        all_actions.extend(ep_actions)
        all_waypoints.extend(ep_waypoints)
        all_phases.extend(ep_phases)

        status = "LANDED" if controller.phase == XCPhase.LANDED else controller.phase.name
        _print(f"  Episode {ep+1}/{num_episodes}: {len(ep_obs)} steps, {status}")

    # Convert to arrays
    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    waypoints = np.array(all_waypoints, dtype=np.float32)
    phases = np.array(all_phases, dtype=np.int32)

    # Save
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        waypoints=waypoints,
        phases=phases,
        num_episodes=num_episodes,
        successful_episodes=successful_episodes,
        dt=dt,
        # Metadata
        waypoint_format=['target_x', 'target_y', 'target_alt', 'target_speed', 'wp_type', 'dist_to_wp'],
    )

    _print()
    _print(f"Generated {len(observations):,} transitions")
    _print(f"Successful landings: {successful_episodes}/{num_episodes}")
    _print(f"Saved to: {output_path}")

    # Print waypoint statistics
    _print()
    _print("Waypoint type distribution:")
    unique, counts = np.unique(waypoints[:, 4], return_counts=True)
    wp_types = ['FLYOVER', 'FLYBY', 'HOLD', 'LAND', 'TAKEOFF']
    for t, c in zip(unique, counts):
        _print(f"  {wp_types[int(t)]}: {c:,} ({100*c/len(waypoints):.1f}%)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate waypoint-labeled demos')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    generate_demos(
        num_episodes=args.episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
