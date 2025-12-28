#!/usr/bin/env python3
"""
Continue Cessna 172 training from checkpoint

Resumes training from a saved checkpoint to complete the 1M timesteps.
"""
import argparse
import sys
import os
import subprocess
import atexit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aida_sim.env.flight_env_cessna172 import Cessna172Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import TensorBoard auto-start from train script
from train_cessna172_ppo import start_tensorboard


def make_env(task="full_mission", cruise_altitude_ft=3000.0, rank=0, seed=0):
    """Create environment"""
    def _init():
        env = Cessna172Env(task=task, cruise_altitude_ft=cruise_altitude_ft)
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total target timesteps (will train remaining)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("="*70)
    print("  CESSNA 172 - RESUME TRAINING")
    print("="*70)
    print(f"\nLoading checkpoint: {args.checkpoint}")

    # Load model
    model = PPO.load(args.checkpoint)
    print("✓ Checkpoint loaded")

    # Get current timesteps from model
    current_steps = model.num_timesteps
    remaining_steps = args.total_timesteps - current_steps

    print(f"\nTraining progress:")
    print(f"  Current timesteps: {current_steps:,}")
    print(f"  Target timesteps: {args.total_timesteps:,}")
    print(f"  Remaining: {remaining_steps:,} ({remaining_steps/args.total_timesteps*100:.1f}%)")

    if remaining_steps <= 0:
        print("\n✓ Training already complete!")
        return

    # Auto-start TensorBoard
    checkpoint_dir = "checkpoints/cessna172"
    tensorboard_logdir = f"{checkpoint_dir}/tensorboard"
    start_tensorboard(tensorboard_logdir, port=6006)

    # Create environments
    print(f"Creating {args.n_envs} parallel environments...")
    if args.n_envs == 1:
        env = DummyVecEnv([make_env("full_mission", 3000.0, 0, 42)])
    else:
        env = SubprocVecEnv([
            make_env("full_mission", 3000.0, i, 42 + i)
            for i in range(args.n_envs)
        ])

    eval_env = DummyVecEnv([make_env("full_mission", 3000.0, 999, 999)])

    print("✓ Environments created")

    # Update model environment
    model.set_env(env)

    # Setup callbacks
    checkpoint_dir = "checkpoints/cessna172"

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // args.n_envs,
        save_path=checkpoint_dir,
        name_prefix="cessna172_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=f"{checkpoint_dir}/eval",
        eval_freq=25_000 // args.n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    callbacks = [checkpoint_callback, eval_callback]

    print("\n" + "="*70)
    print("  RESUMING TRAINING")
    print("="*70)
    print(f"\nTraining for {remaining_steps:,} more timesteps...")
    print(f"Monitor in TensorBoard: tensorboard --logdir={checkpoint_dir}/tensorboard")
    print()

    # Continue training
    model.learn(
        total_timesteps=remaining_steps,
        callback=callbacks,
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=False,  # IMPORTANT: Don't reset timestep counter
    )

    # Save final model
    final_path = f"{checkpoint_dir}/cessna172_ppo_final.zip"
    model.save(final_path)

    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal model saved to: {final_path}")
    print(f"Total timesteps: {model.num_timesteps:,}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
