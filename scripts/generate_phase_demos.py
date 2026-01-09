#!/usr/bin/env python3
"""
Generate Expert Demonstrations for All Curriculum Phases

Uses the classical takeoff controller to generate expert demonstrations
for each phase of the curriculum learning pipeline.

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from aida_sim.env.flight_env_cessna172 import Cessna172Env, StateIndex
from classical_takeoff_controller import TakeoffController, TakeoffPhase


# Phase configurations
PHASE_CONFIGS = {
    'ground_roll': {
        'task': 'ground_roll',
        'start_phase': TakeoffPhase.GROUND_ROLL,
        'success_check': lambda obs, ctrl: ctrl.phase != TakeoffPhase.GROUND_ROLL or np.sqrt(obs[StateIndex.U]**2 + obs[StateIndex.V]**2 + obs[StateIndex.W]**2) >= 28.0,
        'max_steps': 1500,
        'description': 'Accelerate from rest to V_rotate (28 m/s)',
    },
    'rotation': {
        'task': 'rotation',
        'start_phase': TakeoffPhase.ROTATION,
        'success_check': lambda obs, ctrl: ctrl.phase == TakeoffPhase.INITIAL_CLIMB or -obs[StateIndex.Z] > 2.0,
        'max_steps': 500,
        'description': 'Rotate and lift off (reach 2m altitude)',
    },
    'initial_climb': {
        'task': 'initial_climb',
        'start_phase': TakeoffPhase.INITIAL_CLIMB,
        'success_check': lambda obs, ctrl: -obs[StateIndex.Z] > 30.0,  # Reach 30m (~100ft)
        'max_steps': 1000,
        'description': 'Climb from liftoff to 100ft AGL',
    },
    'full_climb': {
        'task': 'full_climb',
        'start_phase': TakeoffPhase.CLIMB,
        'success_check': lambda obs, ctrl: -obs[StateIndex.Z] > 300.0,  # Reach 300m (~1000ft)
        'max_steps': 2000,
        'description': 'Climb from 500ft to 1000ft AGL',
    },
    'cruise': {
        'task': 'cruise',
        'start_phase': TakeoffPhase.CLIMB,  # Will transition to level flight
        'success_check': lambda obs, ctrl: False,  # Never terminate early - collect full episode data
        'max_steps': 500,  # Collect 10 seconds of cruise data per episode
        'description': 'Maintain level cruise flight',
    },
}


def generate_demonstrations(
    phase: str,
    num_episodes: int = 100,
    dt: float = 0.02,
    output_dir: str = None,
    add_noise: bool = True,
    noise_std: float = 0.02,
):
    """Generate expert demonstrations for a specific phase."""

    if phase not in PHASE_CONFIGS:
        print(f"Unknown phase: {phase}")
        print(f"Available phases: {list(PHASE_CONFIGS.keys())}")
        return None

    config = PHASE_CONFIGS[phase]

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "bc_data"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment for this phase
    env = Cessna172Env(task=config['task'], dt=dt, use_overlapping_init=True)
    controller = TakeoffController(v_rotate=env.V_ROTATE, runway_heading=env.runway_heading)

    print("="*60)
    print(f"  {phase.upper()} PHASE EXPERT DEMONSTRATION GENERATOR")
    print("="*60)
    print(f"Task: {config['task']}")
    print(f"Description: {config['description']}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps: {config['max_steps']} ({config['max_steps']*dt:.1f}s)")
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
        controller.phase = config['start_phase']

        ep_observations = []
        ep_actions = []
        ep_phases = []

        for step in range(config['max_steps']):
            action = controller.compute_action(obs)

            if add_noise:
                noise = np.random.normal(0, noise_std, size=action.shape)
                # Keep throttle at 1.0 during takeoff phases
                noise[0] = 0
                action_noisy = np.clip(action + noise, -1.0, 1.0)
            else:
                action_noisy = action

            ep_observations.append(obs.copy())
            ep_actions.append(action.copy())  # Store clean action
            ep_phases.append(controller.phase.value)

            obs, reward, terminated, truncated, info = env.step(action_noisy)

            # Check for success
            if config['success_check'](obs, controller):
                successful_episodes += 1
                break

            if terminated or truncated:
                break

        all_observations.extend(ep_observations)
        all_actions.extend(ep_actions)
        all_phases.extend(ep_phases)
        episode_lengths.append(len(ep_observations))

        if (ep + 1) % 20 == 0 or ep == 0:
            success_rate = successful_episodes / (ep + 1) * 100
            altitude = -obs[StateIndex.Z]
            airspeed = np.sqrt(obs[StateIndex.U]**2 + obs[StateIndex.V]**2 + obs[StateIndex.W]**2)
            pitch_deg = np.rad2deg(obs[StateIndex.THETA])
            print(f"  Episode {ep+1:3d}/{num_episodes}: "
                  f"{len(ep_observations):4d} steps, "
                  f"Alt={altitude:.1f}m, V={airspeed:.1f}m/s, Pitch={pitch_deg:.1f}°, "
                  f"Success: {success_rate:.1f}%")

    env.close()

    if len(all_observations) == 0:
        print("No data collected!")
        return None

    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    phases = np.array(all_phases, dtype=np.int32)

    # Action statistics
    print("\n" + "="*60)
    print("  ACTION STATISTICS (Expert)")
    print("="*60)
    action_names = ['throttle', 'aileron', 'elevator', 'rudder', 'flap', 'spoiler']
    for i, name in enumerate(action_names):
        print(f"  {name:10s}: mean={actions[:, i].mean():.4f}, std={actions[:, i].std():.4f}")

    metadata = {
        'phase': phase,
        'task': config['task'],
        'num_episodes': num_episodes,
        'num_transitions': len(observations),
        'successful_episodes': successful_episodes,
        'success_rate': successful_episodes / num_episodes,
        'episode_lengths': np.array(episode_lengths),
        'mean_episode_length': np.mean(episode_lengths),
        'v_rotate': env.V_ROTATE,
        'dt': dt,
        'obs_dim': observations.shape[1],
        'action_dim': actions.shape[1],
        'noise_std': noise_std if add_noise else 0.0,
        'generated_at': datetime.now().isoformat(),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{phase}_demos_{timestamp}.npz"

    np.savez_compressed(
        filename,
        observations=observations,
        actions=actions,
        phases=phases,
        **metadata
    )

    print()
    print("="*60)
    print("  GENERATION COMPLETE")
    print("="*60)
    print(f"Total transitions: {len(observations):,}")
    print(f"Successful episodes: {successful_episodes}/{num_episodes} ({metadata['success_rate']*100:.1f}%)")
    print(f"Mean episode length: {metadata['mean_episode_length']:.1f} steps")
    print(f"Saved to: {filename}")

    return {
        'observations': observations,
        'actions': actions,
        'phases': phases,
        'metadata': metadata,
        'filename': filename,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate expert demonstrations for curriculum phases')
    parser.add_argument('--phase', type=str, required=True,
                        choices=list(PHASE_CONFIGS.keys()),
                        help='Phase to generate demos for')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to generate')
    parser.add_argument('--dt', type=float, default=0.02,
                        help='Simulation timestep')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--no-noise', action='store_true',
                        help='Disable action noise')
    parser.add_argument('--noise-std', type=float, default=0.02,
                        help='Action noise standard deviation')

    args = parser.parse_args()

    generate_demonstrations(
        phase=args.phase,
        num_episodes=args.episodes,
        dt=args.dt,
        output_dir=args.output,
        add_noise=not args.no_noise,
        noise_std=args.noise_std,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
