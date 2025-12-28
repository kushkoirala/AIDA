#!/usr/bin/env python3
"""
PPO Training for Cessna 172 Autonomous Flight

Trains a PPO agent to:
1. Take off from runway
2. Climb to cruise altitude (3000-5000 ft)
3. Maintain level cruise flight

Uses GPU-accelerated physics and phase-based rewards.

Author: Kushal Koirala (with Claude Code)
Date: December 27, 2024
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import subprocess
import atexit
import signal

# Add AIDA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aida_sim.env.flight_env_cessna172 import Cessna172Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

# Global variable to track TensorBoard process
_tensorboard_process = None


def start_tensorboard(logdir, port=6006):
    """
    Auto-start TensorBoard in background.

    Args:
        logdir: Path to tensorboard logs
        port: Port number (default 6006)

    Returns:
        subprocess.Popen object or None if failed
    """
    global _tensorboard_process

    try:
        # Kill any existing TensorBoard on this port
        subprocess.run(['pkill', '-f', f'tensorboard.*{port}'],
                      stderr=subprocess.DEVNULL, check=False)

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

        print(f"✓ TensorBoard started (PID: {_tensorboard_process.pid})")
        print(f"   Open in browser: http://localhost:{port}\n")

        # Register cleanup on exit
        def cleanup_tensorboard():
            if _tensorboard_process and _tensorboard_process.poll() is None:
                print("\n🛑 Shutting down TensorBoard...")
                _tensorboard_process.terminate()
                try:
                    _tensorboard_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _tensorboard_process.kill()

        atexit.register(cleanup_tensorboard)

        return _tensorboard_process

    except Exception as e:
        print(f"⚠️  Could not start TensorBoard: {e}")
        print(f"   You can manually start it with:")
        print(f"   tensorboard --logdir={logdir} --port={port}")
        return None


def make_env(task="full_mission", cruise_altitude_ft=3000.0, rank=0, seed=0, use_monitor=True):
    """
    Create a single Cessna 172 environment.

    Args:
        task: Mission task type
        cruise_altitude_ft: Target cruise altitude
        rank: Environment rank for parallel training
        seed: Random seed
        use_monitor: Whether to wrap with Monitor (only for DummyVecEnv, not SyncVectorEnv)

    Returns:
        Callable that creates environment
    """
    def _init():
        env = Cessna172Env(task=task, cruise_altitude_ft=cruise_altitude_ft)
        env.reset(seed=seed + rank)
        if use_monitor:
            env = Monitor(env)
        return env
    return _init


def train_cessna172(
    task="full_mission",
    cruise_altitude_ft=3000.0,
    total_timesteps=1_000_000,
    n_envs=4,
    learning_rate=3e-4,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    device="cuda",
    checkpoint_dir="checkpoints/cessna172",
):
    """
    Train PPO agent for Cessna 172 autonomous flight.

    Args:
        task: Mission task ("takeoff", "climb", "cruise", "full_mission")
        cruise_altitude_ft: Target cruise altitude in feet
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: PPO learning rate
        batch_size: Mini-batch size for PPO updates
        n_epochs: Number of epochs for each PPO update
        gamma: Discount factor
        device: Device for training ("cuda" or "cpu")
        checkpoint_dir: Directory for saving checkpoints
    """

    print("="*70)
    print("  CESSNA 172 AUTONOMOUS FLIGHT - PPO TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Task: {task}")
    print(f"  Cruise altitude: {cruise_altitude_ft:.0f} ft ({cruise_altitude_ft * 0.3048:.0f} m)")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Checkpoint dir: {checkpoint_dir}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Auto-start TensorBoard
    tensorboard_logdir = f"{checkpoint_dir}/tensorboard"
    start_tensorboard(tensorboard_logdir, port=6006)

    # Create vectorized environments
    print(f"Creating {n_envs} parallel environments...")

    if n_envs == 1:
        # Single environment
        env = DummyVecEnv([make_env(task, cruise_altitude_ft, 0, 42, use_monitor=True)])
    else:
        # Multiple environments using SubprocVecEnv (true parallel execution)
        # This was working this morning with 4 environments
        env = SubprocVecEnv([
            make_env(task, cruise_altitude_ft, i, 42 + i, use_monitor=True)
            for i in range(n_envs)
        ])

    # Create evaluation environment (single) - use Monitor
    eval_env = DummyVecEnv([make_env(task, cruise_altitude_ft, 999, 999, use_monitor=True)])

    print("✓ Environments created successfully")

    # PPO policy network configuration
    policy_kwargs = dict(
        net_arch=[256, 256, 128],  # 3-layer MLP
        activation_fn=torch.nn.ReLU,
    )

    # Create PPO agent
    print("\nInitializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048 // n_envs,  # Steps per environment before update
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Entropy bonus for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=f"{checkpoint_dir}/tensorboard",
    )

    print("✓ PPO agent initialized")
    print(f"\nPolicy architecture:")
    print(f"  Input: 12D state vector")
    print(f"  Hidden layers: {policy_kwargs['net_arch']}")
    print(f"  Output: 4D action (throttle, aileron, elevator, rudder)")

    # Callbacks
    print("\nSetting up callbacks...")

    # Checkpoint callback - save every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=checkpoint_dir,
        name_prefix="cessna172_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Evaluation callback - evaluate every 25k steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=f"{checkpoint_dir}/eval",
        eval_freq=25_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    callbacks = [checkpoint_callback, eval_callback]

    print("✓ Callbacks configured")

    # Train
    print("\n" + "="*70)
    print("  STARTING TRAINING")
    print("="*70)
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    print("Monitor progress in TensorBoard:")
    print(f"  tensorboard --logdir={checkpoint_dir}/tensorboard")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,  # Log every 10 updates
        progress_bar=True,
    )

    # Save final model
    final_path = f"{checkpoint_dir}/cessna172_ppo_final.zip"
    model.save(final_path)

    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal model saved to: {final_path}")
    print(f"Best model saved to: {checkpoint_dir}/best_model.zip")
    print(f"\nCheckpoints saved every 50k steps in: {checkpoint_dir}/")

    # Test final policy
    print("\n" + "="*70)
    print("  TESTING FINAL POLICY")
    print("="*70)

    test_env = Cessna172Env(task=task, cruise_altitude_ft=cruise_altitude_ft)
    obs, info = test_env.reset(seed=12345)

    episode_reward = 0.0
    episode_length = 0
    max_altitude = 0.0

    print("\nRunning test episode...")
    for step in range(3000):  # Max 60 seconds
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        episode_reward += reward
        episode_length += 1
        max_altitude = max(max_altitude, info['altitude'])

        if terminated or truncated:
            break

    print(f"\nTest episode results:")
    print(f"  Episode length: {episode_length} steps ({episode_length * 0.02:.1f} s)")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Max altitude: {max_altitude:.1f} m ({max_altitude / 0.3048:.0f} ft)")
    print(f"  Mission phase: {info['mission_phase']}")
    print(f"  Termination: {info.get('termination_reason', 'time_limit')}")

    print("\n" + "="*70)
    print("  Next Steps:")
    print("="*70)
    print(f"\n1. Visualize trained policy:")
    print(f"   python scripts/visualize_cessna172.py --model {final_path}")
    print(f"\n2. Continue training:")
    print(f"   python scripts/train_cessna172_ppo.py --load {final_path}")
    print(f"\n3. View training metrics:")
    print(f"   tensorboard --logdir={checkpoint_dir}/tensorboard")
    print()

    env.close()
    eval_env.close()
    test_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO for Cessna 172 autonomous flight")

    parser.add_argument("--task", type=str, default="full_mission",
                        choices=["takeoff", "climb", "cruise", "full_mission"],
                        help="Mission task to train")
    parser.add_argument("--cruise-altitude", type=float, default=3000.0,
                        help="Target cruise altitude in feet AGL")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="PPO learning rate")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Mini-batch size for PPO updates")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of epochs per PPO update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for training")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/cessna172",
                        help="Directory for saving checkpoints")
    parser.add_argument("--load", type=str, default=None,
                        help="Load existing model and continue training")

    args = parser.parse_args()

    # Train model
    model = train_cessna172(
        task=args.task,
        cruise_altitude_ft=args.cruise_altitude,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
