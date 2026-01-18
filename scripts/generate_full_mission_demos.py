#!/usr/bin/env python3
"""
Generate Full Mission Expert Demonstrations for End-to-End NN Training

CLASSICAL CONTROLLER → EXPERT DEMONSTRATIONS → NEURAL NETWORK TRAINING
======================================================================
Uses the classical triangle_controller.py to generate expert flight
demonstrations covering the full mission: takeoff → climb → cruise.

These demos are then used for:
1. Behavioral Cloning (BC) - Warm-start the NN policy
2. PPO Fine-tuning - Improve beyond imitation

This keeps the classical controller intact as the expert demonstrator,
while training a separate neural network to learn the same behavior.

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from aida_sim.env.flight_env_cessna172 import Cessna172Env, StateIndex, CONTROL_DIM


class FullMissionController:
    """
    Classical controller for full mission: takeoff → climb → cruise

    Simplified version of triangle_controller for straight-ahead flight.
    Uses PID-style control for each phase.
    """

    def __init__(self, cruise_altitude_m=914.4, v_rotate=28.0, v_climb=38.0, v_cruise=56.0):
        self.cruise_altitude = cruise_altitude_m  # 3000 ft default
        self.v_rotate = v_rotate
        self.v_climb = v_climb
        self.v_cruise = v_cruise

        # Phase definitions
        self.GROUND_ROLL = 0
        self.ROTATION = 1
        self.INITIAL_CLIMB = 2
        self.CLIMB = 3
        self.CRUISE = 4

        self.phase = self.GROUND_ROLL
        self.phase_names = ['GROUND_ROLL', 'ROTATION', 'INITIAL_CLIMB', 'CLIMB', 'CRUISE']

    def compute_action(self, obs):
        """Compute control action based on current state and phase."""
        # Extract state
        x, y, z = obs[StateIndex.X], obs[StateIndex.Y], obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi, theta, psi = obs[StateIndex.PHI], obs[StateIndex.THETA], obs[StateIndex.PSI]
        p, q, r = obs[StateIndex.P], obs[StateIndex.Q], obs[StateIndex.R]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        climb_rate = -w

        # Update phase based on state
        self._update_phase(altitude, airspeed, climb_rate)

        # Compute controls based on phase
        if self.phase == self.GROUND_ROLL:
            return self._ground_roll_control(airspeed, psi, phi, theta)
        elif self.phase == self.ROTATION:
            return self._rotation_control(airspeed, altitude, theta, phi, psi)
        elif self.phase == self.INITIAL_CLIMB:
            return self._initial_climb_control(airspeed, altitude, theta, phi, psi, climb_rate)
        elif self.phase == self.CLIMB:
            return self._climb_control(airspeed, altitude, theta, phi, psi, climb_rate)
        else:  # CRUISE
            return self._cruise_control(airspeed, altitude, theta, phi, psi)

    def _update_phase(self, altitude, airspeed, climb_rate):
        """Update flight phase based on current state."""
        if self.phase == self.GROUND_ROLL:
            if airspeed >= self.v_rotate * 0.95:
                self.phase = self.ROTATION

        elif self.phase == self.ROTATION:
            if altitude > 3.0:  # 10 ft
                self.phase = self.INITIAL_CLIMB

        elif self.phase == self.INITIAL_CLIMB:
            if altitude > 152.4:  # 500 ft
                self.phase = self.CLIMB

        elif self.phase == self.CLIMB:
            if altitude >= self.cruise_altitude * 0.95:
                self.phase = self.CRUISE

    def _ground_roll_control(self, airspeed, psi, phi, theta):
        """Ground roll: full throttle, maintain heading, keep nose down."""
        throttle = 1.0  # Full power

        # Rudder for heading (runway heading = 0)
        heading_error = psi
        rudder = -2.0 * heading_error - 0.5 * phi
        rudder = np.clip(rudder, -0.3, 0.3)

        # Elevator: keep nose down during ground roll
        elevator = -0.3  # Nose down

        # Aileron: wings level
        aileron = -2.0 * phi
        aileron = np.clip(aileron, -0.2, 0.2)

        # Convert to normalized action [-1, 1]
        return np.array([
            throttle * 2.0 - 1.0,  # throttle: [0,1] → [-1,1]
            aileron,
            elevator,
            rudder
        ], dtype=np.float32)

    def _rotation_control(self, airspeed, altitude, theta, phi, psi):
        """Rotation: pitch up for liftoff."""
        throttle = 1.0

        # Target pitch for rotation
        target_pitch = np.deg2rad(10.0)
        pitch_error = target_pitch - theta
        elevator = 1.5 * pitch_error
        elevator = np.clip(elevator, -0.5, 0.5)

        # Wings level
        aileron = -3.0 * phi
        aileron = np.clip(aileron, -0.3, 0.3)

        # Heading
        rudder = -1.0 * psi
        rudder = np.clip(rudder, -0.2, 0.2)

        return np.array([
            throttle * 2.0 - 1.0,
            aileron,
            elevator,
            rudder
        ], dtype=np.float32)

    def _initial_climb_control(self, airspeed, altitude, theta, phi, psi, climb_rate):
        """Initial climb: establish stable climb to 500 ft."""
        throttle = 1.0

        # Target climb pitch
        target_pitch = np.deg2rad(12.0)
        pitch_error = target_pitch - theta
        elevator = 1.0 * pitch_error
        elevator = np.clip(elevator, -0.4, 0.4)

        # Wings level
        aileron = -3.0 * phi
        aileron = np.clip(aileron, -0.3, 0.3)

        # Heading
        rudder = -1.0 * psi - 0.3 * phi
        rudder = np.clip(rudder, -0.2, 0.2)

        return np.array([
            throttle * 2.0 - 1.0,
            aileron,
            elevator,
            rudder
        ], dtype=np.float32)

    def _climb_control(self, airspeed, altitude, theta, phi, psi, climb_rate):
        """Climb: climb to cruise altitude at V_climb."""
        # Throttle for climb
        throttle = 0.9

        # Speed-based pitch control
        speed_error = self.v_climb - airspeed
        target_pitch = np.deg2rad(8.0) + 0.02 * speed_error
        target_pitch = np.clip(target_pitch, np.deg2rad(3.0), np.deg2rad(15.0))

        pitch_error = target_pitch - theta
        elevator = 0.8 * pitch_error
        elevator = np.clip(elevator, -0.3, 0.3)

        # Wings level
        aileron = -3.0 * phi
        aileron = np.clip(aileron, -0.3, 0.3)

        # Heading
        rudder = -1.0 * psi - 0.3 * phi
        rudder = np.clip(rudder, -0.2, 0.2)

        return np.array([
            throttle * 2.0 - 1.0,
            aileron,
            elevator,
            rudder
        ], dtype=np.float32)

    def _cruise_control(self, airspeed, altitude, theta, phi, psi):
        """Cruise: maintain altitude and speed."""
        # Altitude hold
        alt_error = self.cruise_altitude - altitude
        target_pitch = 0.01 * alt_error
        target_pitch = np.clip(target_pitch, np.deg2rad(-5.0), np.deg2rad(5.0))

        pitch_error = target_pitch - theta
        elevator = 0.5 * pitch_error
        elevator = np.clip(elevator, -0.2, 0.2)

        # Speed-based throttle
        speed_error = self.v_cruise - airspeed
        throttle = 0.65 + 0.02 * speed_error
        throttle = np.clip(throttle, 0.3, 0.9)

        # Wings level
        aileron = -3.0 * phi
        aileron = np.clip(aileron, -0.2, 0.2)

        # Heading
        rudder = -0.5 * psi
        rudder = np.clip(rudder, -0.1, 0.1)

        return np.array([
            throttle * 2.0 - 1.0,
            aileron,
            elevator,
            rudder
        ], dtype=np.float32)

    def reset(self):
        """Reset controller to initial state."""
        self.phase = self.GROUND_ROLL


def generate_full_mission_demos(
    num_episodes: int = 50,
    max_steps: int = 3000,
    dt: float = 0.02,
    cruise_altitude_ft: float = 3000.0,
    output_dir: str = None,
    add_noise: bool = True,
    noise_std: float = 0.01,
):
    """Generate expert demonstrations for full mission."""

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "bc_data"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cruise_altitude_m = cruise_altitude_ft * 0.3048

    # Create environment and controller
    env = Cessna172Env(task='full_mission', dt=dt, cruise_altitude_ft=cruise_altitude_ft)
    controller = FullMissionController(
        cruise_altitude_m=cruise_altitude_m,
        v_rotate=env.V_ROTATE,
        v_climb=env.V_CLIMB,
        v_cruise=env.V_CRUISE
    )

    print("="*70)
    print("  FULL MISSION EXPERT DEMONSTRATION GENERATOR")
    print("="*70)
    print("  Using CLASSICAL controller to generate expert demonstrations")
    print("  These will train a NEURAL NETWORK to fly autonomously")
    print("="*70)
    print(f"\nEpisodes: {num_episodes}")
    print(f"Max steps: {max_steps} ({max_steps*dt:.1f}s)")
    print(f"Cruise altitude: {cruise_altitude_ft:.0f} ft ({cruise_altitude_m:.0f} m)")
    print(f"V_rotate: {env.V_ROTATE:.1f} m/s ({env.V_ROTATE*1.944:.1f} KIAS)")
    print(f"V_climb: {env.V_CLIMB:.1f} m/s ({env.V_CLIMB*1.944:.1f} KIAS)")
    print(f"V_cruise: {env.V_CRUISE:.1f} m/s ({env.V_CRUISE*1.944:.1f} KIAS)")
    print(f"Action noise: {'enabled' if add_noise else 'disabled'} (std={noise_std})")
    print(f"Output: {output_dir}")
    print()

    all_observations = []
    all_actions = []
    all_phases = []
    episode_lengths = []
    successful_episodes = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        controller.reset()

        ep_observations = []
        ep_actions = []
        ep_phases = []

        max_altitude = 0.0
        reached_cruise = False

        for step in range(max_steps):
            # Get expert action from classical controller
            action = controller.compute_action(obs)

            # Add small noise for robustness (helps NN generalize)
            if add_noise:
                noise = np.random.normal(0, noise_std, size=action.shape)
                # Don't add noise to throttle during ground roll
                if controller.phase == controller.GROUND_ROLL:
                    noise[0] = 0
                action_noisy = np.clip(action + noise, -1.0, 1.0)
            else:
                action_noisy = action

            # Store demonstration (clean action, not noisy)
            ep_observations.append(obs.copy())
            ep_actions.append(action.copy())
            ep_phases.append(controller.phase)

            # Execute (potentially noisy) action
            obs, reward, terminated, truncated, info = env.step(action_noisy)

            # Track progress
            altitude = -obs[StateIndex.Z]
            max_altitude = max(max_altitude, altitude)

            if controller.phase == controller.CRUISE:
                reached_cruise = True

            if terminated or truncated:
                break

        # Episode complete
        all_observations.extend(ep_observations)
        all_actions.extend(ep_actions)
        all_phases.extend(ep_phases)
        episode_lengths.append(len(ep_observations))

        if reached_cruise or max_altitude >= cruise_altitude_m * 0.9:
            successful_episodes += 1

        # Print progress
        success_rate = successful_episodes / (ep + 1) * 100
        print(f"  Episode {ep+1:3d}/{num_episodes}: "
              f"{len(ep_observations):4d} steps | "
              f"Max alt: {max_altitude:6.0f}m ({max_altitude/0.3048:5.0f}ft) | "
              f"Phase: {controller.phase_names[controller.phase]:15s} | "
              f"Success: {success_rate:.0f}%")

    env.close()

    # Convert to arrays
    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    phases = np.array(all_phases, dtype=np.int32)

    # Metadata
    metadata = {
        'num_episodes': num_episodes,
        'num_transitions': len(observations),
        'successful_episodes': successful_episodes,
        'success_rate': successful_episodes / num_episodes,
        'episode_lengths': np.array(episode_lengths),
        'mean_episode_length': np.mean(episode_lengths),
        'cruise_altitude_m': cruise_altitude_m,
        'cruise_altitude_ft': cruise_altitude_ft,
        'v_rotate': env.V_ROTATE,
        'v_climb': env.V_CLIMB,
        'v_cruise': env.V_CRUISE,
        'dt': dt,
        'obs_dim': observations.shape[1],
        'action_dim': actions.shape[1],
        'noise_std': noise_std if add_noise else 0.0,
        'generated_at': datetime.now().isoformat(),
        'controller_type': 'classical_full_mission',
    }

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"full_mission_demos_{timestamp}.npz"

    np.savez_compressed(
        filename,
        observations=observations,
        actions=actions,
        phases=phases,
        **metadata
    )

    print()
    print("="*70)
    print("  GENERATION COMPLETE")
    print("="*70)
    print(f"Total transitions: {len(observations):,}")
    print(f"Successful episodes: {successful_episodes}/{num_episodes} ({metadata['success_rate']*100:.1f}%)")
    print(f"Mean episode length: {metadata['mean_episode_length']:.1f} steps ({metadata['mean_episode_length']*dt:.1f}s)")
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Saved to: {filename}")

    print("\nPhase distribution:")
    phase_names = ['GROUND_ROLL', 'ROTATION', 'INITIAL_CLIMB', 'CLIMB', 'CRUISE']
    for i, name in enumerate(phase_names):
        count = np.sum(phases == i)
        pct = count / len(phases) * 100
        print(f"  {name:15s}: {count:6d} ({pct:5.1f}%)")

    print("\n" + "="*70)
    print("  NEXT STEPS")
    print("="*70)
    print(f"\n1. Train BC policy:")
    print(f"   python scripts/train_bc_full_mission.py --data {filename}")
    print(f"\n2. Fine-tune with PPO:")
    print(f"   python scripts/train_ppo_full_mission.py --bc-model <bc_model.zip>")
    print(f"\n3. Test NN policy:")
    print(f"   python scripts/run_ppo_policy_with_telemetry.py --model <model.zip>")

    return {
        'observations': observations,
        'actions': actions,
        'phases': phases,
        'metadata': metadata,
        'filename': filename,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate full mission expert demonstrations using classical controller'
    )
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes to generate (default: 50)')
    parser.add_argument('--max-steps', type=int, default=3000,
                        help='Maximum steps per episode (default: 3000 = 60s)')
    parser.add_argument('--dt', type=float, default=0.02,
                        help='Simulation timestep (default: 0.02)')
    parser.add_argument('--altitude', type=float, default=3000.0,
                        help='Cruise altitude in feet (default: 3000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: bc_data/)')
    parser.add_argument('--no-noise', action='store_true',
                        help='Disable action noise')
    parser.add_argument('--noise-std', type=float, default=0.01,
                        help='Action noise standard deviation (default: 0.01)')

    args = parser.parse_args()

    generate_full_mission_demos(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        dt=args.dt,
        cruise_altitude_ft=args.altitude,
        output_dir=args.output,
        add_noise=not args.no_noise,
        noise_std=args.noise_std,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
