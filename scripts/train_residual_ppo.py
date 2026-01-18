#!/usr/bin/env python3
"""
Train Residual PPO for Cross-Country Flight

The expert controller flies the mission.
The NN learns small corrections (±15%) to improve performance.

Key insight: If NN outputs zeros, aircraft flies on pure expert control.
"""

import sys
from pathlib import Path
import argparse
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from residual_env import ResidualFlightEnv


def make_env(rank: int, seed: int = 0, residual_scale: float = 0.15, max_steps: int = 10000):
    """Create a single environment instance."""
    def _init():
        env = ResidualFlightEnv(
            dt=0.02,
            max_episode_steps=max_steps,  # Shorter episodes for faster training
            residual_scale=residual_scale,
            cruise_altitude_ft=5500.0,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description='Train Residual PPO')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--residual-scale', type=float, default=0.15,
                        help='λ: max residual correction (0.15 = ±15%)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='checkpoints/residual_ppo',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    args = parser.parse_args()

    print("=" * 70)
    print("  RESIDUAL PPO TRAINING")
    print("=" * 70)
    print(f"  Residual scale (λ): {args.residual_scale}")
    print(f"  This means: NN can adjust controls by ±{args.residual_scale*100:.0f}%")
    print(f"  If NN outputs 0: Pure expert control")
    print("=" * 70)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create vectorized environment
    # Use DummyVecEnv for GPU compatibility (SubprocVecEnv has issues with GPU)
    print(f"\nCreating {args.n_envs} environments (DummyVecEnv for GPU compatibility)...")
    env = DummyVecEnv([
        make_env(i, residual_scale=args.residual_scale)
        for i in range(args.n_envs)
    ])

    # Create eval environment
    eval_env = DummyVecEnv([make_env(100, residual_scale=args.residual_scale)])

    # Callbacks - eval less frequently since episodes are long
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir),
        log_path=str(checkpoint_dir),
        eval_freq=50000 // args.n_envs,  # Eval every 50k steps instead of 10k
        n_eval_episodes=2,  # Reduce from 5 to 2 episodes per eval
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // args.n_envs,
        save_path=str(checkpoint_dir),
        name_prefix='residual_ppo',
    )

    # Create or load model
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(checkpoint_dir / "logs"),
            policy_kwargs={
                "net_arch": [256, 256, 128],  # Smaller network for residuals
            }
        )

    print(f"\nPolicy architecture: {model.policy}")

    # Train
    print(f"\nTraining for {args.timesteps:,} timesteps...")
    print("The NN is learning corrections to the expert controller.")
    print("Tensorboard: tensorboard --logdir", checkpoint_dir / "logs")
    print()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_path = checkpoint_dir / "residual_ppo_final.zip"
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    # Cleanup
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
