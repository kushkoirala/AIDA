#!/usr/bin/env python3
"""
End-to-End Scenario Dataset Generator for Behavioral Cloning

Generates expert demonstrations from complete flight scenarios using
the classical controller and real-world airport data.

Scenarios include:
  - Traffic patterns at SN65 and KHUT
  - Cross-country flights between airports
  - Go-around maneuvers
  - Straight-out departures

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent if Path(__file__).parent.name == 'scripts' else Path(__file__).parent))

# Import AIDA modules
from aida_sim.env.flight_env_cessna172 import Cessna172Env, StateIndex
from aida_sim.platform.airports import get_airport, list_airports
from aida_sim.platform.world import World, WorldOrigin
from aida_sim.scenarios.scenarios import (
    ScenarioConfig, Scenario, Waypoint, FlightPhase,
    create_traffic_pattern_scenario,
    create_cross_country_scenario,
    create_go_around_scenario,
    create_straight_out_departure_scenario,
    SCENARIO_BUILDERS
)


class ScenarioController:
    """
    Classical controller for end-to-end scenario flight.

    Uses waypoint following with phase-appropriate control laws.
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.world = scenario.world
        self.waypoints = scenario.waypoints
        self.current_waypoint_idx = 0

        # Track if we're still in initial ground roll
        self.in_ground_roll = True

        # Speed targets (m/s)
        self.v_rotate = 28.0      # 55 KIAS
        self.v_climb = 38.0       # 75 KIAS
        self.v_cruise = 51.0      # 100 KIAS
        self.v_pattern = 46.0     # 90 KIAS
        self.v_approach = 32.0    # 62 KIAS
        self.v_touchdown = 26.0   # 50 KIAS

        # Control gains
        self.kp_pitch = 1.5
        self.kd_pitch = 0.5
        self.kp_roll = 1.2
        self.kd_roll = 0.4
        self.kp_heading = 0.8
        self.kd_heading = 0.3
        self.kp_altitude = 0.02
        self.kp_speed = 0.05

        # Bank angle for turns
        self.turn_bank = np.deg2rad(20.0)

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None

    @property
    def phase(self) -> FlightPhase:
        if self.in_ground_roll:
            return FlightPhase.GROUND_ROLL
        wp = self.current_waypoint
        return wp.phase if wp else FlightPhase.ROLLOUT

    def reset(self):
        """Reset to first waypoint."""
        self.current_waypoint_idx = 0
        self.in_ground_roll = True

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute control action based on current state and target waypoint.

        Returns: [throttle, aileron, elevator, rudder, flap, spoiler]
        """
        # Extract state
        x = obs[StateIndex.X]
        y = obs[StateIndex.Y]
        z = obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi = obs[StateIndex.PHI]
        theta = obs[StateIndex.THETA]
        psi = obs[StateIndex.PSI]
        p, q, r = obs[StateIndex.P], obs[StateIndex.Q], obs[StateIndex.R]

        # Derived quantities
        position = np.array([x, y, z])
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        altitude = -z  # NED: negative Z is up

        # Get current waypoint
        wp = self.current_waypoint
        if wp is None:
            return self._idle_control()

        # Check waypoint capture
        self._check_waypoint_capture(position, altitude, airspeed)
        wp = self.current_waypoint
        if wp is None:
            return self._idle_control()

        # Handle initial ground roll before first waypoint capture
        # The first waypoint is typically ROTATION, but we need ground roll first
        if self.in_ground_roll and altitude < 1.0:
            if airspeed < self.v_rotate:
                return self._ground_roll_control(airspeed, phi, psi, p, r, wp)
            else:
                # Start rotation when we reach Vr
                pass  # Fall through to normal phase handling
        elif self.in_ground_roll and altitude >= 1.0:
            self.in_ground_roll = False  # Now airborne

        # Compute control based on phase
        phase = wp.phase

        if phase == FlightPhase.GROUND_ROLL:
            return self._ground_roll_control(airspeed, phi, psi, p, r, wp)
        elif phase == FlightPhase.ROTATION:
            return self._rotation_control(phi, theta, psi, p, q, r, wp)
        elif phase in [FlightPhase.INITIAL_CLIMB, FlightPhase.CLIMB]:
            return self._climb_control(altitude, airspeed, phi, theta, psi, p, q, r, wp)
        elif phase in [FlightPhase.CRUISE, FlightPhase.CROSSWIND, FlightPhase.DOWNWIND]:
            return self._cruise_control(altitude, airspeed, phi, theta, psi, p, q, r, wp, position)
        elif phase in [FlightPhase.BASE, FlightPhase.DESCENT]:
            return self._descent_control(altitude, airspeed, phi, theta, psi, p, q, r, wp, position)
        elif phase in [FlightPhase.FINAL, FlightPhase.APPROACH]:
            return self._approach_control(altitude, airspeed, phi, theta, psi, p, q, r, wp, position)
        elif phase == FlightPhase.FLARE:
            return self._flare_control(altitude, airspeed, phi, theta, psi, p, q, r, wp)
        elif phase == FlightPhase.ROLLOUT:
            return self._rollout_control(airspeed, phi, psi, p, r)
        elif phase == FlightPhase.GO_AROUND:
            return self._go_around_control(altitude, airspeed, phi, theta, psi, p, q, r, wp)
        else:
            return self._cruise_control(altitude, airspeed, phi, theta, psi, p, q, r, wp, position)

    def _check_waypoint_capture(self, position: np.ndarray, altitude: float, airspeed: float):
        """Check if current waypoint has been captured."""
        wp = self.current_waypoint
        if wp is None:
            return

        # Different capture criteria by phase
        phase = wp.phase

        if phase == FlightPhase.GROUND_ROLL:
            # Capture when reaching rotation speed
            target_speed = wp.airspeed_kts * 0.5144  # Convert to m/s
            if airspeed >= target_speed * 0.95:
                self._advance_waypoint()

        elif phase == FlightPhase.ROTATION:
            # Capture when airborne
            if altitude > 3.0:
                self._advance_waypoint()

        elif phase == FlightPhase.ROLLOUT:
            # Capture when stopped
            if airspeed < 5.0:
                self._advance_waypoint()

        else:
            # Position-based capture
            wp_pos = wp.position
            dist_2d = np.sqrt((position[0] - wp_pos[0])**2 + (position[1] - wp_pos[1])**2)

            if dist_2d < wp.tolerance_m:
                self._advance_waypoint()

    def _advance_waypoint(self):
        """Move to next waypoint."""
        if self.current_waypoint_idx < len(self.waypoints) - 1:
            self.current_waypoint_idx += 1

    def _bearing_to_waypoint(self, position: np.ndarray, wp: Waypoint) -> float:
        """Calculate bearing to waypoint in radians."""
        dx = wp.position[0] - position[0]  # North
        dy = wp.position[1] - position[1]  # East
        return np.arctan2(dy, dx)

    def _distance_to_waypoint(self, position: np.ndarray, wp: Waypoint) -> float:
        """Calculate 2D distance to waypoint."""
        dx = wp.position[0] - position[0]
        dy = wp.position[1] - position[1]
        return np.sqrt(dx**2 + dy**2)

    # =========================================================================
    # Phase Control Methods
    # =========================================================================

    def _ground_roll_control(self, airspeed, phi, psi, p, r, wp):
        """Ground roll: full throttle, track runway heading."""
        throttle = 1.0
        flap = 0.1
        spoiler = 0.0

        # Gradual elevator pull as approaching Vr
        if wp.heading_deg is not None:
            target_heading = np.deg2rad(wp.heading_deg)
        else:
            target_heading = 0.0

        if airspeed < 0.85 * self.v_rotate:
            elevator = 0.0
        else:
            blend = (airspeed - 0.85 * self.v_rotate) / (0.15 * self.v_rotate)
            elevator = -blend * 0.2

        aileron = self._wings_level(phi, p)
        rudder = self._heading_hold(psi, r, target_heading)

        return np.array([throttle, aileron, elevator, rudder, flap, spoiler], dtype=np.float32)

    def _rotation_control(self, phi, theta, psi, p, q, r, wp):
        """Rotation: pitch up to liftoff attitude."""
        throttle = 1.0
        flap = 0.1
        spoiler = 0.0

        pitch_target = np.deg2rad(10.0)
        pitch_error = pitch_target - theta
        elevator = -(self.kp_pitch * pitch_error - self.kd_pitch * q)
        elevator = np.clip(elevator, -0.8, 0.5)

        if wp.heading_deg is not None:
            target_heading = np.deg2rad(wp.heading_deg)
        else:
            target_heading = psi

        aileron = self._wings_level(phi, p)
        rudder = self._heading_hold(psi, r, target_heading) * 0.7

        return np.array([throttle, aileron, elevator, rudder, flap, spoiler], dtype=np.float32)

    def _climb_control(self, altitude, airspeed, phi, theta, psi, p, q, r, wp):
        """Climb: maintain climb pitch and heading to waypoint."""
        throttle = 1.0
        flap = 0.0
        spoiler = 0.0

        # Target altitude from waypoint
        target_alt = wp.altitude_msl_ft * 0.3048
        alt_error = target_alt - altitude

        # Pitch for climb
        if alt_error > 50:
            pitch_target = np.deg2rad(8.0)
        else:
            blend = alt_error / 50.0
            pitch_target = blend * np.deg2rad(8.0) + (1-blend) * np.deg2rad(2.0)

        pitch_error = pitch_target - theta
        elevator = -(self.kp_pitch * pitch_error - self.kd_pitch * q)
        elevator = np.clip(elevator, -0.6, 0.5)

        if wp.heading_deg is not None:
            target_heading = np.deg2rad(wp.heading_deg)
        else:
            target_heading = psi

        aileron = self._wings_level(phi, p)
        rudder = self._heading_hold(psi, r, target_heading) * 0.5

        return np.array([throttle, aileron, elevator, rudder, flap, spoiler], dtype=np.float32)

    def _cruise_control(self, altitude, airspeed, phi, theta, psi, p, q, r, wp, position):
        """Cruise/pattern legs: maintain altitude and track to waypoint."""
        flap = 0.0
        spoiler = 0.0

        # Speed control
        target_speed = wp.airspeed_kts * 0.5144
        speed_error = target_speed - airspeed
        throttle = 0.7 + self.kp_speed * speed_error
        throttle = np.clip(throttle, 0.3, 1.0)

        # Altitude hold
        target_alt = wp.altitude_msl_ft * 0.3048
        alt_error = target_alt - altitude
        pitch_target = np.deg2rad(2.0) + self.kp_altitude * alt_error
        pitch_target = np.clip(pitch_target, np.deg2rad(-5.0), np.deg2rad(10.0))

        pitch_error = pitch_target - theta
        elevator = -(self.kp_pitch * pitch_error - self.kd_pitch * q)
        elevator = np.clip(elevator, -0.4, 0.4)

        # Heading to waypoint or specified heading
        if wp.heading_deg is not None:
            target_heading = np.deg2rad(wp.heading_deg)
        else:
            target_heading = self._bearing_to_waypoint(position, wp)

        heading_error = self._wrap_angle(target_heading - psi)

        # Bank for turn if needed
        if abs(heading_error) > np.deg2rad(10.0):
            target_bank = np.sign(heading_error) * self.turn_bank
        else:
            target_bank = 0.0

        roll_error = target_bank - phi
        aileron = self.kp_roll * roll_error - self.kd_roll * p
        aileron = np.clip(aileron, -0.6, 0.6)

        # Coordinated rudder
        rudder = 0.3 * np.sin(phi)

        return np.array([throttle, aileron, elevator, rudder, flap, spoiler], dtype=np.float32)

    def _descent_control(self, altitude, airspeed, phi, theta, psi, p, q, r, wp, position):
        """Descent: controlled descent to waypoint."""
        flap = 0.2
        spoiler = 0.3

        # Reduced throttle
        target_speed = wp.airspeed_kts * 0.5144
        speed_error = target_speed - airspeed
        throttle = 0.4 + self.kp_speed * speed_error
        throttle = np.clip(throttle, 0.2, 0.7)

        # Descent pitch
        target_alt = wp.altitude_msl_ft * 0.3048
        alt_error = target_alt - altitude

        if alt_error < -20:
            pitch_target = np.deg2rad(-3.0)
        else:
            pitch_target = np.deg2rad(-1.0)

        pitch_error = pitch_target - theta
        elevator = -(self.kp_pitch * pitch_error - self.kd_pitch * q)
        elevator = np.clip(elevator, -0.5, 0.5)

        # Track to waypoint
        if wp.heading_deg is not None:
            target_heading = np.deg2rad(wp.heading_deg)
        else:
            target_heading = self._bearing_to_waypoint(position, wp)

        heading_error = self._wrap_angle(target_heading - psi)

        if abs(heading_error) > np.deg2rad(10.0):
            target_bank = np.sign(heading_error) * self.turn_bank * 0.8
        else:
            target_bank = 0.0

        roll_error = target_bank - phi
        aileron = self.kp_roll * roll_error - self.kd_roll * p
        aileron = np.clip(aileron, -0.5, 0.5)

        rudder = 0.3 * np.sin(phi)

        return np.array([throttle, aileron, elevator, rudder, flap, spoiler], dtype=np.float32)

    def _approach_control(self, altitude, airspeed, phi, theta, psi, p, q, r, wp, position):
        """Final approach: stabilized approach."""
        flap = 0.4
        spoiler = 0.2

        # Approach speed
        target_speed = wp.airspeed_kts * 0.5144
        speed_error = target_speed - airspeed
        throttle = 0.35 + self.kp_speed * speed_error * 0.5
        throttle = np.clip(throttle, 0.15, 0.5)

        # Approach pitch
        pitch_target = np.deg2rad(-4.0)
        pitch_error = pitch_target - theta
        elevator = -(self.kp_pitch * pitch_error - self.kd_pitch * q)
        elevator = np.clip(elevator, -0.5, 0.5)

        # Track runway
        if wp.heading_deg is not None:
            target_heading = np.deg2rad(wp.heading_deg)
        else:
            target_heading = self._bearing_to_waypoint(position, wp)

        aileron = self._wings_level(phi, p)
        rudder = self._heading_hold(psi, r, target_heading) * 0.4

        return np.array([throttle, aileron, elevator, rudder, flap, spoiler], dtype=np.float32)

    def _flare_control(self, altitude, airspeed, phi, theta, psi, p, q, r, wp):
        """Landing flare: reduce sink rate."""
        throttle = 0.0
        flap = 0.5
        spoiler = 0.5

        # Progressive pitch up
        flare_alt = 5.0
        flare_progress = 1.0 - min(altitude / flare_alt, 1.0)

        pitch_target = np.deg2rad(-4.0) + flare_progress * np.deg2rad(9.0)
        pitch_error = pitch_target - theta
        elevator = -(self.kp_pitch * pitch_error - self.kd_pitch * q)
        elevator = np.clip(elevator, -0.6, 0.3)

        if wp.heading_deg is not None:
            target_heading = np.deg2rad(wp.heading_deg)
        else:
            target_heading = psi

        aileron = self._wings_level(phi, p) * 1.5
        rudder = self._heading_hold(psi, r, target_heading) * 0.6

        return np.array([throttle, aileron, elevator, rudder, flap, spoiler], dtype=np.float32)

    def _rollout_control(self, airspeed, phi, psi, p, r):
        """Post-landing rollout: decelerate."""
        throttle = 0.0
        flap = 0.0
        spoiler = 1.0  # Full spoiler for braking

        aileron = self._wings_level(phi, p)
        elevator = 0.1  # Slight nose up
        rudder = 0.0

        return np.array([throttle, aileron, elevator, rudder, flap, spoiler], dtype=np.float32)

    def _go_around_control(self, altitude, airspeed, phi, theta, psi, p, q, r, wp):
        """Go-around: full power, climb out."""
        throttle = 1.0
        flap = 0.1
        spoiler = 0.0

        pitch_target = np.deg2rad(10.0)
        pitch_error = pitch_target - theta
        elevator = -(self.kp_pitch * pitch_error - self.kd_pitch * q)
        elevator = np.clip(elevator, -0.7, 0.5)

        if wp.heading_deg is not None:
            target_heading = np.deg2rad(wp.heading_deg)
        else:
            target_heading = psi

        aileron = self._wings_level(phi, p)
        rudder = self._heading_hold(psi, r, target_heading) * 0.5

        return np.array([throttle, aileron, elevator, rudder, flap, spoiler], dtype=np.float32)

    def _idle_control(self):
        """Idle control when scenario complete."""
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _wings_level(self, phi, p):
        """Wings level controller."""
        roll_error = 0.0 - phi
        aileron = self.kp_roll * roll_error - self.kd_roll * p
        return np.clip(aileron, -1.0, 1.0)

    def _heading_hold(self, psi, r, target_heading):
        """Heading hold controller."""
        heading_error = self._wrap_angle(target_heading - psi)
        rudder = self.kp_heading * heading_error - self.kd_heading * r
        return np.clip(rudder, -1.0, 1.0)

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def generate_scenario_demonstrations(
    scenario_type: str = 'traffic_pattern',
    airport: str = 'SN65',
    runway: str = '35',
    dest_airport: str = 'KHUT',
    dest_runway: str = '31',
    num_episodes: int = 10,
    max_steps: int = 6000,
    dt: float = 0.02,
    output_dir: str = None,
    add_noise: bool = True,
    noise_std: float = 0.02,
):
    """
    Generate expert demonstrations for an end-to-end scenario.
    """
    if output_dir is None:
        output_dir = Path('/home/AIDA/data/scenario_demos')
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create scenario
    print("="*60)
    print("  END-TO-END SCENARIO DEMONSTRATION GENERATOR")
    print("="*60)
    print(f"Scenario: {scenario_type}")
    print(f"Departure: {airport} RWY {runway}")

    if scenario_type == 'traffic_pattern':
        scenario = create_traffic_pattern_scenario(
            airport=airport,
            runway=runway,
            pattern_altitude_agl_ft=1000.0,
            full_stop=True
        )
    elif scenario_type == 'cross_country':
        print(f"Destination: {dest_airport} RWY {dest_runway}")
        scenario = create_cross_country_scenario(
            dep_airport=airport,
            dep_runway=runway,
            arr_airport=dest_airport,
            arr_runway=dest_runway,
            cruise_altitude_ft=5500.0
        )
    elif scenario_type == 'go_around':
        scenario = create_go_around_scenario(
            airport=airport,
            runway=runway,
            go_around_altitude_ft=200.0
        )
    elif scenario_type == 'straight_out':
        scenario = create_straight_out_departure_scenario(
            airport=airport,
            runway=runway,
            cruise_altitude_ft=3500.0,
            cruise_distance_nm=5.0
        )
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    print(f"Waypoints: {len(scenario.waypoints)}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps: {max_steps} ({max_steps*dt:.1f}s)")
    print()

    # Initialize environment and controller
    # Use full_mission task for longest episodes (60s instead of 20s)
    env = Cessna172Env(task='full_mission', dt=dt)
    # Override the max_episode_steps to allow longer episodes
    env.max_episode_steps = max_steps
    controller = ScenarioController(scenario)

    # Storage
    all_observations = []
    all_actions = []
    all_phases = []
    all_waypoint_indices = []
    episode_lengths = []
    successful_episodes = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        controller.reset()

        ep_observations = []
        ep_actions = []
        ep_phases = []
        ep_waypoints = []

        for step in range(max_steps):
            # Get expert action
            action = controller.compute_action(obs)

            # Add noise
            if add_noise:
                noise = np.random.normal(0, noise_std, size=action.shape)
                if controller.phase == FlightPhase.GROUND_ROLL:
                    noise[0] = 0
                action_noisy = np.clip(action + noise, -1.0, 1.0)
            else:
                action_noisy = action

            # Store
            ep_observations.append(obs.copy())
            ep_actions.append(action.copy())
            ep_phases.append(controller.phase.value)
            ep_waypoints.append(controller.current_waypoint_idx)

            # Step
            obs, reward, terminated, truncated, info = env.step(action_noisy)

            # Check completion
            if controller.current_waypoint_idx >= len(scenario.waypoints) - 1:
                if controller.phase == FlightPhase.ROLLOUT:
                    airspeed = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)
                    if airspeed < 5.0:
                        successful_episodes += 1
                        break

            if terminated or truncated:
                break

        # Store episode
        all_observations.extend(ep_observations)
        all_actions.extend(ep_actions)
        all_phases.extend(ep_phases)
        all_waypoint_indices.extend(ep_waypoints)
        episode_lengths.append(len(ep_observations))

        # Progress
        if (ep + 1) % 5 == 0 or ep == 0:
            success_rate = successful_episodes / (ep + 1) * 100
            print(f"  Episode {ep+1:3d}/{num_episodes}: "
                  f"{len(ep_observations):5d} steps, "
                  f"WP {controller.current_waypoint_idx}/{len(scenario.waypoints)}, "
                  f"Phase={controller.phase.name:12s}, "
                  f"Success: {success_rate:.0f}%")

    env.close()

    # Convert to arrays
    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    phases = np.array(all_phases, dtype=np.int32)
    waypoint_indices = np.array(all_waypoint_indices, dtype=np.int32)

    # Metadata
    metadata = {
        'scenario_type': scenario_type,
        'scenario_name': scenario.config.name,
        'departure_airport': airport,
        'departure_runway': runway,
        'num_episodes': num_episodes,
        'num_transitions': len(observations),
        'successful_episodes': successful_episodes,
        'success_rate': successful_episodes / num_episodes,
        'episode_lengths': np.array(episode_lengths),
        'mean_episode_length': np.mean(episode_lengths),
        'num_waypoints': len(scenario.waypoints),
        'waypoint_names': [wp.name for wp in scenario.waypoints],
        'dt': dt,
        'obs_dim': observations.shape[1],
        'action_dim': actions.shape[1],
        'noise_std': noise_std if add_noise else 0.0,
        'generated_at': datetime.now().isoformat(),
    }

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{scenario_type}_{airport}_{runway}_{timestamp}.npz"

    np.savez_compressed(
        filename,
        observations=observations,
        actions=actions,
        phases=phases,
        waypoint_indices=waypoint_indices,
        **metadata
    )

    print()
    print("="*60)
    print("  GENERATION COMPLETE")
    print("="*60)
    print(f"Total transitions: {len(observations)}")
    print(f"Successful episodes: {successful_episodes}/{num_episodes} ({metadata['success_rate']*100:.1f}%)")
    print(f"Mean episode length: {metadata['mean_episode_length']:.1f} steps")
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Saved to: {filename}")

    # Phase distribution
    print("\nPhase distribution:")
    for phase in FlightPhase:
        count = np.sum(phases == phase.value)
        if count > 0:
            pct = count / len(phases) * 100
            print(f"  {phase.name:15s}: {count:6d} ({pct:5.1f}%)")

    return {
        'observations': observations,
        'actions': actions,
        'phases': phases,
        'waypoint_indices': waypoint_indices,
        'metadata': metadata,
        'filename': filename,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate end-to-end scenario demonstrations')
    parser.add_argument('--scenario', type=str, default='traffic_pattern',
                        choices=['traffic_pattern', 'cross_country', 'go_around', 'straight_out'],
                        help='Scenario type')
    parser.add_argument('--airport', type=str, default='SN65',
                        help='Departure airport ICAO code')
    parser.add_argument('--runway', type=str, default='35',
                        help='Departure runway designator')
    parser.add_argument('--dest-airport', type=str, default='KHUT',
                        help='Destination airport (for cross-country)')
    parser.add_argument('--dest-runway', type=str, default='31',
                        help='Destination runway (for cross-country)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=6000,
                        help='Max steps per episode')
    parser.add_argument('--dt', type=float, default=0.02,
                        help='Timestep')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--no-noise', action='store_true',
                        help='Disable action noise')
    parser.add_argument('--noise-std', type=float, default=0.02,
                        help='Action noise std')

    args = parser.parse_args()

    generate_scenario_demonstrations(
        scenario_type=args.scenario,
        airport=args.airport,
        runway=args.runway,
        dest_airport=args.dest_airport,
        dest_runway=args.dest_runway,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        dt=args.dt,
        output_dir=args.output,
        add_noise=not args.no_noise,
        noise_std=args.noise_std,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
