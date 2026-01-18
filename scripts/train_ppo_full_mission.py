#!/usr/bin/env python3
"""
Train End-to-End PPO Policy for Full Mission Flight

NEURAL NETWORK TRAINING PIPELINE
================================
This trains a single neural network to fly the complete mission:
    Ground Roll → Rotation → Climb → Cruise

Training approach:
1. (Optional) Behavioral Cloning warm-start from expert demos
2. PPO reinforcement learning on full_mission task

The classical controller remains intact - this trains a SEPARATE
neural network that learns to fly autonomously.

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import os
import sys
import argparse
import subprocess
import atexit
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from aida_sim.env.flight_env_cessna172 import Cessna172Env


# TensorBoard process handle
_tensorboard_process = None


def start_tensorboard(logdir, port=6006):
    """Auto-start TensorBoard in background."""
    global _tensorboard_process

    try:
        subprocess.run(['pkill', '-f', f'tensorboard.*{port}'],
                      stderr=subprocess.DEVNULL, check=False)

        import time
        time.sleep(1)

        print(f"\n🚀 Starting TensorBoard on http://localhost:{port}")
        print(f"   Logdir: {logdir}")

        _tensorboard_process = subprocess.Popen(
            ['tensorboard', '--logdir', logdir,
             '--port', str(port), '--host', '0.0.0.0',
             '--reload_interval', '10'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp
        )

        time.sleep(2)
        print(f"✓ TensorBoard started (PID: {_tensorboard_process.pid})")

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
        return None


def make_env(cruise_altitude_ft=3000.0, rank=0):
    """Create a single full_mission environment."""
    def _init():
        env = Cessna172Env(
            task='full_mission',
            cruise_altitude_ft=cruise_altitude_ft,
        )
        env = Monitor(env)
        return env
    return _init


def pretrain_with_bc(model, bc_data_path, epochs=10, batch_size=64):
    """
    Pretrain policy with behavioral cloning from expert demonstrations.

    This gives the NN a warm-start so PPO doesn't have to learn from scratch.
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    print(f"\n{'='*60}")
    print(f"  BEHAVIORAL CLONING PRETRAINING")
    print(f"{'='*60}")
    print(f"Dataset: {bc_data_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    if not os.path.exists(bc_data_path):
        print(f"⚠️  BC dataset not found at {bc_data_path}")
        print(f"   Generate demos first: python scripts/generate_full_mission_demos.py")
        return model

    # Load data
    data = np.load(bc_data_path)
    observations = data['observations']
    actions = data['actions']

    print(f"✓ Loaded {len(observations):,} expert transitions")
    print(f"  Observation shape: {observations.shape}")
    print(f"  Action shape: {actions.shape}")

    # Handle action dimension mismatch if needed
    action_dim = model.action_space.shape[0]
    if actions.shape[1] < action_dim:
        print(f"  Padding actions from {actions.shape[1]} to {action_dim} dimensions")
        actions = np.concatenate([
            actions,
            np.zeros((len(actions), action_dim - actions.shape[1]), dtype=np.float32)
        ], axis=1)

    # Convert to tensors
    device = model.device
    obs_tensor = torch.FloatTensor(observations).to(device)
    act_tensor = torch.FloatTensor(actions).to(device)

    dataset = TensorDataset(obs_tensor, act_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train
    policy = model.policy
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    print(f"\n🚀 Starting BC pretraining...")
    policy.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for obs_batch, act_batch in dataloader:
            optimizer.zero_grad()

            action_dist = policy.get_distribution(obs_batch)
            predicted_actions = action_dist.distribution.mean

            loss = mse_loss(predicted_actions, act_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")

    policy.eval()

    # Set small exploration noise after BC
    with torch.no_grad():
        if hasattr(policy, 'log_std'):
            policy.log_std.fill_(-3.0)
        elif hasattr(policy.action_net, 'log_std'):
            policy.action_net.log_std.fill_(-3.0)

    print(f"\n✓ BC pretraining complete!")
    print(f"{'='*60}\n")

    return model


def train_full_mission(
    total_timesteps=2_000_000,
    n_envs=4,
    cruise_altitude_ft=3000.0,
    bc_data_path=None,
    bc_epochs=10,
    learning_rate=3e-4,
    device='cuda',
    checkpoint_dir='checkpoints/ppo_full_mission',
):
    """Train PPO on full_mission task."""

    print("\n" + "="*70)
    print("  END-TO-END NEURAL NETWORK TRAINING")
    print("  PPO on Full Mission (Takeoff → Climb → Cruise)")
    print("="*70)
    print("\n  NOTE: This trains a NEURAL NETWORK controller")
    print("        Classical controller: scripts/triangle_controller.py")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Cruise altitude: {cruise_altitude_ft:.0f} ft")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print(f"  BC pretraining: {'Yes' if bc_data_path else 'No'}")
    print(f"  Checkpoint dir: {checkpoint_dir}")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    tensorboard_log = f"{checkpoint_dir}/tensorboard"

    # Start TensorBoard
    start_tensorboard(tensorboard_log, port=6006)

    # Create environments
    print(f"\nCreating {n_envs} parallel environments...")
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(cruise_altitude_ft, i) for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(cruise_altitude_ft)])

    eval_env = DummyVecEnv([make_env(cruise_altitude_ft, 999)])
    print("✓ Environments created")

    # Create PPO model
    print("\nInitializing PPO agent...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=2048 // n_envs,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            log_std_init=-1.0,
        ),
        tensorboard_log=tensorboard_log,
        device=device,
        verbose=1,
    )
    print("✓ PPO agent initialized")
    print(f"  Policy: 256 → 256 → 128 (actor & critic)")
    print(f"  Input: 12D state | Output: 4D action")

    # Optional BC pretraining
    if bc_data_path:
        model = pretrain_with_bc(model, bc_data_path, epochs=bc_epochs)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1000),
        save_path=checkpoint_dir,
        name_prefix='ppo_full_mission',
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=f"{checkpoint_dir}/eval",
        eval_freq=max(25_000 // n_envs, 500),
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train
    print("\n" + "="*70)
    print("  STARTING TRAINING")
    print("="*70)
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    print(f"TensorBoard: http://localhost:6006")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name='ppo_full_mission',
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")

    # Save final model
    final_path = f"{checkpoint_dir}/ppo_full_mission_final.zip"
    model.save(final_path)

    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal model: {final_path}")
    print(f"Best model: {checkpoint_dir}/best_model.zip")

    print("\n" + "="*70)
    print("  NEXT STEPS")
    print("="*70)
    print(f"\n1. Test NN policy with telemetry:")
    print(f"   python scripts/run_ppo_policy_with_telemetry.py --model {final_path}")
    print(f"\n2. Compare with classical controller:")
    print(f"   python scripts/run_xc_sn65_khut.py")

    env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train end-to-end PPO for full mission flight'
    )
    parser.add_argument('--timesteps', type=int, default=2_000_000,
                        help='Total training timesteps (default: 2M)')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--altitude', type=float, default=3000.0,
                        help='Cruise altitude in feet')
    parser.add_argument('--bc-data', type=str, default=None,
                        help='Path to BC dataset for warm-start')
    parser.add_argument('--bc-epochs', type=int, default=10,
                        help='BC pretraining epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--checkpoint-dir', type=str,
                        default='checkpoints/ppo_full_mission',
                        help='Checkpoint directory')

    args = parser.parse_args()

    train_full_mission(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        cruise_altitude_ft=args.altitude,
        bc_data_path=args.bc_data,
        bc_epochs=args.bc_epochs,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
