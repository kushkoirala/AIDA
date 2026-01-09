#!/usr/bin/env python3
"""
Generate Expert Demonstrations for Behavioral Cloning

Uses the classical takeoff controller to generate expert demonstrations
that can be used to pre-train a policy network via behavioral cloning.

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from aida_sim.env.flight_env_cessna172 import Cessna172Env, StateIndex
from classical_takeoff_controller import TakeoffController, TakeoffPhase


def generate_demonstrations(
    num_episodes: int = 100,
    max_steps: int = 1500,
    dt: float = 0.02,
    output_dir: str = None,
    add_noise: bool = True,
    noise_std: float = 0.02,
):
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "expert_demos"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use ground_roll task for BC pretraining (Phase 1 curriculum)
    env = Cessna172Env(task='ground_roll', dt=dt)
    controller = TakeoffController(v_rotate=env.V_ROTATE, runway_heading=env.runway_heading)

    print("="*60)
    print("  EXPERT DEMONSTRATION GENERATOR")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Max steps: {max_steps} ({max_steps*dt:.1f}s)")
    print(f"V_rotate: {env.V_ROTATE:.1f} m/s ({env.V_ROTATE*1.944:.1f} KIAS)")
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
        controller.phase = TakeoffPhase.GROUND_ROLL

        ep_observations = []
        ep_actions = []
        ep_phases = []

        for step in range(max_steps):
            action = controller.compute_action(obs)

            if add_noise:
                noise = np.random.normal(0, noise_std, size=action.shape)
                if controller.phase == TakeoffPhase.GROUND_ROLL:
                    noise[0] = 0
                action_noisy = np.clip(action + noise, -1.0, 1.0)
            else:
                action_noisy = action

            ep_observations.append(obs.copy())
            ep_actions.append(action.copy())
            ep_phases.append(controller.phase.value)

            obs, reward, terminated, truncated, info = env.step(action_noisy)

            # Check for success (reaching rotation speed or higher phases)
            airspeed = np.sqrt(obs[StateIndex.U]**2 + obs[StateIndex.V]**2 + obs[StateIndex.W]**2)
            if controller.phase != TakeoffPhase.GROUND_ROLL or airspeed >= env.V_ROTATE:
                successful_episodes += 1
                break

            # Handle environment termination
            if terminated or truncated:
                break

        all_observations.extend(ep_observations)
        all_actions.extend(ep_actions)
        all_phases.extend(ep_phases)
        episode_lengths.append(len(ep_observations))

        if (ep + 1) % 10 == 0 or ep == 0:
            success_rate = successful_episodes / (ep + 1) * 100
            print(f"  Episode {ep+1:3d}/{num_episodes}: "
                  f"{len(ep_observations):4d} steps, "
                  f"Phase={controller.phase.name:15s}, "
                  f"Success rate: {success_rate:.1f}%")

    env.close()

    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    phases = np.array(all_phases, dtype=np.int32)

    metadata = {
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
    filename = output_dir / f"expert_demos_{timestamp}.npz"

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
    print(f"Total transitions: {len(observations)}")
    print(f"Successful episodes: {successful_episodes}/{num_episodes} ({metadata['success_rate']*100:.1f}%)")
    print(f"Mean episode length: {metadata['mean_episode_length']:.1f} steps")
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Saved to: {filename}")

    print("\nPhase distribution:")
    for phase in TakeoffPhase:
        count = np.sum(phases == phase.value)
        pct = count / len(phases) * 100
        print(f"  {phase.name:15s}: {count:6d} ({pct:5.1f}%)")

    return {
        'observations': observations,
        'actions': actions,
        'phases': phases,
        'metadata': metadata,
        'filename': filename,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate expert demonstrations')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to generate')
    parser.add_argument('--max-steps', type=int, default=1500,
                        help='Maximum steps per episode')
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
