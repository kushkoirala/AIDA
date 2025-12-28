#!/usr/bin/env python3
"""
Curriculum Learning Training Script for Cessna 172

Implements progressive phase-based training:
Phase 1: Ground Roll (200k steps)
Phase 2: Rotation (300k steps)
Phase 3: Initial Climb (400k steps)
Phase 4: Full Climb (500k steps)
Phase 5: Cruise (600k steps)

Based on Moving Window DAgger curriculum learning approach.

Author: Kushal Koirala (with Claude Code)
Date: December 27, 2024
"""

import os
import sys
import argparse
import subprocess
import atexit
from pathlib import Path
import numpy as np

# Add AIDA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from aida_sim.env.flight_env_cessna172 import Cessna172Env


# Global tensorboard process
_tensorboard_process = None


def start_tensorboard(logdir, port=6006):
    """Auto-start TensorBoard in background."""
    global _tensorboard_process

    try:
        # Kill any existing TensorBoard on this port
        subprocess.run(['pkill', '-f', f'tensorboard.*{port}'],
                      stderr=subprocess.DEVNULL, check=False)

        # Wait a moment for cleanup
        import time
        time.sleep(1)

        # Start TensorBoard
        print(f"\n🚀 Starting TensorBoard on http://localhost:{port}")
        print(f"   Logdir: {logdir}")

        _tensorboard_process = subprocess.Popen(
            ['tensorboard', '--logdir', logdir,
             '--port', str(port), '--host', '0.0.0.0',
             '--reload_interval', '10'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp  # Detach from parent process group
        )

        # Give it time to start
        time.sleep(2)

        print(f"✓ TensorBoard started (PID: {_tensorboard_process.pid})")
        print(f"  Open: http://localhost:{port}")

        # Register cleanup on exit
        def cleanup_tensorboard():
            if _tensorboard_process and _tensorboard_process.poll() is None:
                print("\n🛑 Shutting down TensorBoard...")
                _tensorboard_process.terminate()
                try:
                    _tensorboard_process.wait(timeout=5)
                except:
                    _tensorboard_process.kill()

        atexit.register(cleanup_tensorboard)
        return _tensorboard_process

    except Exception as e:
        print(f"⚠️  Could not start TensorBoard: {e}")
        print(f"   You can start it manually: tensorboard --logdir={logdir}")
        return None


def make_env(task, cruise_altitude_ft=3000.0, rank=0):
    """Create a single environment instance."""
    def _init():
        env = Cessna172Env(task=task, cruise_altitude_ft=cruise_altitude_ft)
        env = Monitor(env)
        return env
    return _init


def evaluate_phase(model, env, n_episodes=50, success_threshold=0.9):
    """
    Evaluate model on current phase.

    Returns:
        success_rate: Fraction of episodes that completed successfully
        mean_reward: Average episode reward
        mean_length: Average episode length
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATING PHASE")
    print(f"{'='*60}")

    success_count = 0
    rewards = []
    lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        rewards.append(episode_reward)
        lengths.append(episode_length)

        # Check if episode was successful based on reward
        # Success = high reward (agent completed the task well)
        if episode_reward > 50:  # Threshold for "good" episode
            success_count += 1

    success_rate = success_count / n_episodes
    mean_reward = np.mean(rewards)
    mean_length = np.mean(lengths)

    print(f"\nResults ({n_episodes} episodes):")
    print(f"  Success Rate:  {success_rate*100:.1f}% ({success_count}/{n_episodes})")
    print(f"  Mean Reward:   {mean_reward:.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean Length:   {mean_length:.1f} ± {np.std(lengths):.1f} steps")
    print(f"  Reward Range:  [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")

    passed = success_rate >= success_threshold
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\n{status}: Success rate {success_rate*100:.1f}% (threshold: {success_threshold*100:.1f}%)")
    print(f"{'='*60}\n")

    return success_rate, mean_reward, mean_length


def train_phase(phase_config, previous_model_path=None, n_envs=4, device='cuda'):
    """
    Train a single curriculum phase.

    Args:
        phase_config: Dict with 'name', 'task', 'timesteps', 'success_threshold'
        previous_model_path: Path to model from previous phase (for transfer learning)
        n_envs: Number of parallel environments
        device: 'cuda' or 'cpu'

    Returns:
        model_path: Path to saved model
        success_rate: Evaluation success rate
    """
    phase_name = phase_config['name']
    task = phase_config['task']
    timesteps = phase_config['timesteps']
    success_threshold = phase_config['success_threshold']

    print(f"\n{'='*60}")
    print(f"  PHASE {phase_config['phase_idx']}: {phase_name.upper()}")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Success Threshold: {success_threshold*100:.0f}%")
    print(f"Parallel Envs: {n_envs}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create checkpoint directory
    checkpoint_dir = f"checkpoints/cessna172_curriculum/{phase_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard logging
    tensorboard_log = f"{checkpoint_dir}/tensorboard"

    # Create environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(task, rank=i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(task)])

    # Evaluation environment (single instance)
    eval_env = Cessna172Env(task=task)
    eval_env = Monitor(eval_env)

    # Create or load model
    if previous_model_path and os.path.exists(previous_model_path):
        print(f"📥 Loading previous phase model: {previous_model_path}")
        model = PPO.load(previous_model_path, env=env, device=device)
        print(f"✓ Transfer learning from previous phase\n")
    else:
        print(f"🆕 Creating new PPO model")
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=512,          # Collect 512 steps per env before update
            batch_size=128,       # Minibatch size
            n_epochs=10,          # PPO update epochs
            gamma=0.99,           # Discount factor
            gae_lambda=0.95,      # GAE parameter
            clip_range=0.2,       # PPO clip range
            ent_coef=0.01,        # Entropy coefficient (exploration)
            vf_coef=0.5,          # Value function coefficient
            max_grad_norm=0.5,    # Gradient clipping
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])  # 3-layer policy/value networks
            ),
            tensorboard_log=tensorboard_log,
            device=device,
            verbose=1,
        )
        print(f"✓ Model initialized\n")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1000),
        save_path=checkpoint_dir,
        name_prefix=f'{phase_name}_ppo',
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=checkpoint_dir,
        eval_freq=max(10_000 // n_envs, 500),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Train
    print(f"🚀 Starting training for {timesteps:,} timesteps...\n")
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=phase_name,
        reset_num_timesteps=False if previous_model_path else True,
    )

    # Save final model
    final_model_path = f"{checkpoint_dir}/{phase_name}_ppo_final.zip"
    model.save(final_model_path)
    print(f"\n✓ Model saved: {final_model_path}")

    # Evaluate
    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION - {phase_name.upper()}")
    print(f"{'='*60}")

    success_rate, mean_reward, mean_length = evaluate_phase(
        model, eval_env, n_episodes=100, success_threshold=success_threshold
    )

    # Clean up
    env.close()
    eval_env.close()

    return final_model_path, success_rate


def main():
    parser = argparse.ArgumentParser(description='Cessna 172 Curriculum Learning Training')
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Training device')
    parser.add_argument('--start-phase', type=int, default=1, choices=[1,2,3,4,5], help='Starting phase (1-5)')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip evaluation between phases')
    parser.add_argument('--tensorboard-port', type=int, default=6006, help='TensorBoard port')
    args = parser.parse_args()

    # Curriculum phases
    curriculum = [
        {
            'phase_idx': 1,
            'name': 'phase1_ground_roll',
            'task': 'ground_roll',
            'timesteps': 200_000,
            'success_threshold': 0.90,
        },
        {
            'phase_idx': 2,
            'name': 'phase2_rotation',
            'task': 'rotation',
            'timesteps': 300_000,
            'success_threshold': 0.90,
        },
        {
            'phase_idx': 3,
            'name': 'phase3_initial_climb',
            'task': 'initial_climb',
            'timesteps': 400_000,
            'success_threshold': 0.85,
        },
        {
            'phase_idx': 4,
            'name': 'phase4_full_climb',
            'task': 'full_climb',
            'timesteps': 500_000,
            'success_threshold': 0.80,
        },
        {
            'phase_idx': 5,
            'name': 'phase5_cruise',
            'task': 'cruise',
            'timesteps': 600_000,
            'success_threshold': 0.80,
        },
    ]

    print("\n" + "="*60)
    print("  CESSNA 172 CURRICULUM LEARNING")
    print("="*60)
    print(f"\nCurriculum Plan:")
    for phase in curriculum:
        print(f"  Phase {phase['phase_idx']}: {phase['name']:30s} - {phase['timesteps']:>7,} steps (>={phase['success_threshold']*100:.0f}% success)")
    print(f"\nStarting from Phase {args.start_phase}")
    print(f"Parallel Environments: {args.n_envs}")
    print(f"Device: {args.device}")
    print("="*60 + "\n")

    # Start TensorBoard
    start_tensorboard("checkpoints/cessna172_curriculum", port=args.tensorboard_port)

    # Train each phase
    previous_model = None

    for phase in curriculum[args.start_phase - 1:]:
        try:
            model_path, success_rate = train_phase(
                phase,
                previous_model_path=previous_model,
                n_envs=args.n_envs,
                device=args.device
            )

            # Check if phase was mastered
            if not args.skip_evaluation and success_rate < phase['success_threshold']:
                print(f"\n⚠️  WARNING: Phase {phase['phase_idx']} did not reach success threshold!")
                print(f"   Success rate: {success_rate*100:.1f}% < {phase['success_threshold']*100:.0f}%")

                # Ask user whether to continue
                response = input(f"\nContinue to next phase anyway? (y/n): ")
                if response.lower() != 'y':
                    print(f"\n❌ Training stopped at Phase {phase['phase_idx']}")
                    print(f"   Retrain this phase with more timesteps or tune hyperparameters.")
                    return

            # Use this model for next phase (transfer learning)
            previous_model = model_path

            print(f"\n✅ Phase {phase['phase_idx']} complete!")
            print(f"   Model: {model_path}")
            print(f"   Success rate: {success_rate*100:.1f}%")

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Training interrupted by user at Phase {phase['phase_idx']}")
            print(f"   Last model: {previous_model}")
            return
        except Exception as e:
            print(f"\n\n❌ ERROR in Phase {phase['phase_idx']}: {e}")
            import traceback
            traceback.print_exc()
            return

    # All phases complete
    print("\n" + "="*60)
    print("  🎉 CURRICULUM TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal model: {previous_model}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate on full mission task:")
    print(f"     python scripts/test_cessna172_env.py --model {previous_model}")
    print(f"  2. Create visualization:")
    print(f"     python scripts/visualize_cessna172.py --model {previous_model}")
    print(f"  3. Compare performance to POH specs")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
