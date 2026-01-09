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


def make_env(task, cruise_altitude_ft=3000.0, rank=0, use_overlapping_init=False):
    """Create a single environment instance.

    Args:
        task: The flight phase task
        cruise_altitude_ft: Cruise altitude target
        rank: Environment rank for parallel envs
        use_overlapping_init: Enable overlapping initial state distribution for curriculum handoff
    """
    def _init():
        env = Cessna172Env(
            task=task,
            cruise_altitude_ft=cruise_altitude_ft,
            use_overlapping_init=use_overlapping_init,
            overlap_range=0.2  # 20% variation in initial states
        )
        env = Monitor(env)
        return env
    return _init


def pretrain_with_bc(model, bc_data_path='bc_data/bc_dataset_corrected.npz', epochs=10, batch_size=64):
    """
    Pretrain policy network using behavioral cloning from expert demonstrations.

    This provides a warm-start for the policy, significantly accelerating RL training
    by starting from a reasonable policy rather than random initialization.

    Args:
        model: PPO model to pretrain
        bc_data_path: Path to behavioral cloning dataset (.npz)
        epochs: Number of BC training epochs
        batch_size: Minibatch size for BC training

    Returns:
        model: Pretrained model
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    print(f"\n{'='*60}")
    print(f"  BEHAVIORAL CLONING PRETRAINING")
    print(f"{'='*60}")
    print(f"Dataset: {bc_data_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")

    # Load BC dataset
    if not os.path.exists(bc_data_path):
        print(f"⚠️  BC dataset not found at {bc_data_path}")
        print(f"   Skipping pretraining, starting from scratch.")
        return model

    data = np.load(bc_data_path)
    observations = data['observations']
    actions = data['actions']

    print(f"✓ Loaded {len(observations):,} expert transitions")
    print(f"  Observation shape: {observations.shape}")
    print(f"  Action shape: {actions.shape}")

    # Handle action dimension mismatch
    # BC dataset has 4 actions (throttle, aileron, elevator, rudder)
    # Environment has 6 actions (+ flaps, spoiler)
    # Pad BC actions with zeros for flaps and spoiler
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

    # Create dataloader
    dataset = TensorDataset(obs_tensor, act_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get policy network and optimizer
    policy = model.policy
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    # Train BC
    print(f"\n🚀 Starting BC pretraining...")
    policy.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for obs_batch, act_batch in dataloader:
            optimizer.zero_grad()

            # Get action mean from policy (deterministic)
            # For continuous actions, get the mean of the distribution
            action_dist = policy.get_distribution(obs_batch)
            predicted_actions = action_dist.distribution.mean

            # Compute MSE loss between predicted and expert actions
            loss = mse_loss(predicted_actions, act_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")

    policy.eval()

    # CRITICAL: Set log_std to a small value after BC training
    # BC only trains the mean, but we need small exploration noise to preserve the learned behavior
    # Expert actions have std ~0.003-0.05, so we use log_std=-3 (std=0.05)
    with torch.no_grad():
        if hasattr(policy, 'log_std'):
            policy.log_std.fill_(-3.0)  # std = exp(-3) = 0.05
            print(f"  Set log_std to -3.0 (std=0.05) for fine control")
        elif hasattr(policy.action_net, 'log_std'):
            policy.action_net.log_std.fill_(-3.0)
            print(f"  Set log_std to -3.0 (std=0.05) for fine control")

    print(f"\n✓ BC pretraining complete!")
    print(f"  Final loss: {avg_loss:.6f}")
    print(f"{'='*60}\n")

    return model


def evaluate_phase(model, env, n_episodes=50, success_threshold=0.9, task='ground_roll'):
    """
    Evaluate model on current phase using task-specific success criteria.

    Success criteria by task:
    - ground_roll: Reached rotation speed (28 m/s) while staying on runway
    - rotation: Lifted off to 3m altitude
    - initial_climb: Reached 150m altitude
    - full_climb: Reached cruise altitude
    - cruise: Maintained altitude within tolerance

    Returns:
        success_rate: Fraction of episodes that completed successfully
        mean_reward: Average episode reward
        mean_length: Average episode length
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATING PHASE: {task.upper()}")
    print(f"{'='*60}")

    # Task-specific success thresholds
    SUCCESS_CRITERIA = {
        'ground_roll': {
            'min_airspeed': 26.0,      # ~93% of V_ROTATE (28 m/s)
            'max_lateral_dev': 10.0,    # Stay within 10m of centerline
            'min_reward': 5000,         # Minimum reward for "acceptable" run
        },
        'rotation': {
            'min_altitude': 2.5,        # Lifted off
            'min_reward': 3000,
        },
        'initial_climb': {
            'min_altitude': 140.0,      # ~460 ft (close to 500 ft target)
            'min_reward': 5000,
        },
        'full_climb': {
            'min_altitude': 800.0,      # Close to cruise altitude
            'min_reward': 5000,
        },
        'cruise': {
            'min_reward': 5000,         # Maintained stable flight
        },
    }

    criteria = SUCCESS_CRITERIA.get(task, {'min_reward': 5000})

    success_count = 0
    rewards = []
    lengths = []
    max_airspeeds = []
    max_altitudes = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        max_airspeed = 0
        max_altitude = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            # Track max values achieved
            airspeed = info.get('airspeed', 0)
            altitude = info.get('altitude', 0)
            max_airspeed = max(max_airspeed, airspeed)
            max_altitude = max(max_altitude, altitude)

        rewards.append(episode_reward)
        lengths.append(episode_length)
        max_airspeeds.append(max_airspeed)
        max_altitudes.append(max_altitude)

        # Task-specific success check
        success = False
        if task == 'ground_roll':
            # Success: reached rotation speed with decent reward
            success = (max_airspeed >= criteria['min_airspeed'] and
                      episode_reward >= criteria['min_reward'])
        elif task == 'rotation':
            success = (max_altitude >= criteria['min_altitude'] and
                      episode_reward >= criteria['min_reward'])
        elif task == 'initial_climb':
            success = (max_altitude >= criteria['min_altitude'] and
                      episode_reward >= criteria['min_reward'])
        elif task == 'full_climb':
            success = (max_altitude >= criteria['min_altitude'] and
                      episode_reward >= criteria['min_reward'])
        else:
            # Default: reward-based
            success = episode_reward >= criteria.get('min_reward', 5000)

        if success:
            success_count += 1

    success_rate = success_count / n_episodes
    mean_reward = np.mean(rewards)
    mean_length = np.mean(lengths)

    print(f"\nResults ({n_episodes} episodes):")
    print(f"  Success Rate:  {success_rate*100:.1f}% ({success_count}/{n_episodes})")
    print(f"  Mean Reward:   {mean_reward:.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean Length:   {mean_length:.1f} ± {np.std(lengths):.1f} steps")
    print(f"  Reward Range:  [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    print(f"  Max Airspeed:  {np.mean(max_airspeeds):.1f} ± {np.std(max_airspeeds):.1f} m/s")
    print(f"  Max Altitude:  {np.mean(max_altitudes):.1f} ± {np.std(max_altitudes):.1f} m")

    passed = success_rate >= success_threshold
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\n{status}: Success rate {success_rate*100:.1f}% (threshold: {success_threshold*100:.1f}%)")
    print(f"{'='*60}\n")

    return success_rate, mean_reward, mean_length


def train_phase(phase_config, previous_model_path=None, n_envs=4, device='cuda',
                use_overlapping_init=True, use_bc_pretrain=True, bc_epochs=10, bc_data_path=None):
    """
    Train a single curriculum phase.

    Args:
        phase_config: Dict with 'name', 'task', 'timesteps', 'success_threshold'
        previous_model_path: Path to model from previous phase (for transfer learning)
        n_envs: Number of parallel environments
        device: 'cuda' or 'cpu'
        use_overlapping_init: Enable overlapping initial state distribution
        use_bc_pretrain: Use behavioral cloning to pretrain policy
        bc_epochs: Number of BC pretraining epochs
        bc_data_path: Path to phase-specific BC dataset (if None, uses default)

    Returns:
        model_path: Path to saved model
        success_rate: Evaluation success rate
    """
    phase_name = phase_config['name']
    task = phase_config['task']
    timesteps = phase_config['timesteps']
    success_threshold = phase_config['success_threshold']
    phase_idx = phase_config['phase_idx']

    # Enable overlapping init for phases 2+ (they need to handle varied handoff states)
    enable_overlap = use_overlapping_init and phase_idx > 1

    print(f"\n{'='*60}")
    print(f"  PHASE {phase_idx}: {phase_name.upper()}")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Success Threshold: {success_threshold*100:.0f}%")
    print(f"Parallel Envs: {n_envs}")
    print(f"Device: {device}")
    print(f"Overlapping Init: {'ENABLED' if enable_overlap else 'disabled'}")
    print(f"{'='*60}\n")

    # Create checkpoint directory
    checkpoint_dir = f"checkpoints/cessna172_curriculum/{phase_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard logging
    tensorboard_log = f"{checkpoint_dir}/tensorboard"

    # Create environments with overlapping init for phases 2+
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(task, rank=i, use_overlapping_init=enable_overlap)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(task, use_overlapping_init=enable_overlap)])

    # Evaluation environment (use standard init for fair comparison)
    eval_env = Cessna172Env(task=task, use_overlapping_init=False)
    eval_env = Monitor(eval_env)

    # Create or load model
    # Transfer learning: load previous phase model if available
    # BC pretraining: fine-tune on phase-specific expert data
    # These are ADDITIVE - transfer first, then BC on top

    if previous_model_path and os.path.exists(previous_model_path):
        print(f"📥 Loading previous phase model: {previous_model_path}")
        model = PPO.load(previous_model_path, env=env, device=device)
        print(f"✓ Transfer learning from previous phase\n")
    else:
        print(f"🆕 Creating new PPO model")
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=1e-4,   # Lower LR to preserve BC-initialized weights
            n_steps=512,          # Collect 512 steps per env before update
            batch_size=128,       # Minibatch size
            n_epochs=10,          # PPO update epochs
            gamma=0.99,           # Discount factor
            gae_lambda=0.95,      # GAE parameter
            clip_range=0.1,       # Smaller clip range for more conservative updates
            ent_coef=0.001,       # Very low entropy - BC gives good init, don't explore wildly
            vf_coef=0.5,          # Value function coefficient
            max_grad_norm=0.5,    # Gradient clipping
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),  # 3-layer policy/value networks
                log_std_init=-2.0,  # Start with std=0.135 (exp(-2)) - CRITICAL for fine control after BC!
            ),
            tensorboard_log=tensorboard_log,
            device=device,
            verbose=1,
        )
        print(f"✓ Model initialized\n")

    # Apply BC pretraining ON TOP of loaded/fresh model
    # This fine-tunes the policy to learn phase-specific behavior
    if use_bc_pretrain and bc_data_path:
        model = pretrain_with_bc(model, bc_data_path=bc_data_path, epochs=bc_epochs)

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
        model, eval_env, n_episodes=100, success_threshold=success_threshold, task=task
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
    parser.add_argument('--no-overlap', action='store_true',
                       help='Disable overlapping initial state distribution (default: enabled)')
    parser.add_argument('--auto-continue', action='store_true',
                       help='Automatically continue to next phase even if success threshold not met')
    parser.add_argument('--no-bc', action='store_true',
                       help='Disable behavioral cloning pretraining (default: BC enabled)')
    parser.add_argument('--bc-epochs', type=int, default=10,
                       help='Number of BC pretraining epochs (default: 10)')
    args = parser.parse_args()

    use_overlapping_init = not args.no_overlap
    use_bc_pretrain = not args.no_bc

    # Curriculum phases with BC data paths
    curriculum = [
        {
            'phase_idx': 1,
            'name': 'phase1_ground_roll',
            'task': 'ground_roll',
            'timesteps': 200_000,
            'success_threshold': 0.90,
            'bc_data': 'bc_data/bc_dataset_corrected.npz',
        },
        {
            'phase_idx': 2,
            'name': 'phase2_rotation',
            'task': 'rotation',
            'timesteps': 300_000,
            'success_threshold': 0.90,
            'bc_data': 'bc_data/rotation_demos_20260104_131747.npz',
        },
        {
            'phase_idx': 3,
            'name': 'phase3_initial_climb',
            'task': 'initial_climb',
            'timesteps': 400_000,
            'success_threshold': 0.85,
            'bc_data': 'bc_data/initial_climb_demos_20260104_161049.npz',
        },
        {
            'phase_idx': 4,
            'name': 'phase4_full_climb',
            'task': 'full_climb',
            'timesteps': 500_000,
            'success_threshold': 0.80,
            'bc_data': 'bc_data/full_climb_demos_20260104_172655.npz',
        },
        {
            'phase_idx': 5,
            'name': 'phase5_cruise',
            'task': 'cruise',
            'timesteps': 600_000,
            'success_threshold': 0.80,
            'bc_data': 'bc_data/cruise_demos_20260104_173201.npz',
        },
    ]

    print("\n" + "="*60)
    print("  CESSNA 172 CURRICULUM LEARNING")
    print("  (with Overlapping Training Regions)")
    print("="*60)
    print(f"\nCurriculum Plan:")
    for phase in curriculum:
        print(f"  Phase {phase['phase_idx']}: {phase['name']:30s} - {phase['timesteps']:>7,} steps (>={phase['success_threshold']*100:.0f}% success)")
    print(f"\nStarting from Phase {args.start_phase}")
    print(f"Parallel Environments: {args.n_envs}")
    print(f"Device: {args.device}")
    print(f"Overlapping Init: {'ENABLED' if use_overlapping_init else 'disabled'}")
    print(f"BC Pretraining: {'ENABLED ({} epochs)'.format(args.bc_epochs) if use_bc_pretrain else 'disabled'}")
    print("="*60 + "\n")

    # Start TensorBoard
    start_tensorboard("checkpoints/cessna172_curriculum", port=args.tensorboard_port)

    # Train each phase
    # If starting from a later phase, try to load the previous phase's model
    previous_model = None
    if args.start_phase > 1:
        # Find previous phase's model (prefer best_model.zip, fallback to final)
        prev_phase_idx = args.start_phase - 1
        prev_phase_name = curriculum[prev_phase_idx - 1]['name']
        prev_dir = f"checkpoints/cessna172_curriculum/{prev_phase_name}"

        # Try best_model first, then final
        import os
        best_model_path = f"{prev_dir}/best_model.zip"
        final_model_path = f"{prev_dir}/{prev_phase_name}_ppo_final.zip"

        if os.path.exists(best_model_path):
            previous_model = best_model_path
            print(f"📥 Found previous phase best model: {best_model_path}")
        elif os.path.exists(final_model_path):
            previous_model = final_model_path
            print(f"📥 Found previous phase final model: {final_model_path}")
        else:
            print(f"⚠️  No previous phase model found in {prev_dir}")
            print(f"   Training will start from scratch")

    for phase in curriculum[args.start_phase - 1:]:
        try:
            model_path, success_rate = train_phase(
                phase,
                previous_model_path=previous_model,
                n_envs=args.n_envs,
                device=args.device,
                use_overlapping_init=use_overlapping_init,
                use_bc_pretrain=use_bc_pretrain,
                bc_epochs=args.bc_epochs,
                bc_data_path=phase.get('bc_data')
            )

            # Check if phase was mastered
            if not args.skip_evaluation and success_rate < phase['success_threshold']:
                print(f"\n⚠️  WARNING: Phase {phase['phase_idx']} did not reach success threshold!")
                print(f"   Success rate: {success_rate*100:.1f}% < {phase['success_threshold']*100:.0f}%")

                if args.auto_continue:
                    print(f"   --auto-continue enabled, proceeding to next phase...")
                else:
                    # Ask user whether to continue (only in interactive mode)
                    try:
                        response = input(f"\nContinue to next phase anyway? (y/n): ")
                        if response.lower() != 'y':
                            print(f"\n❌ Training stopped at Phase {phase['phase_idx']}")
                            print(f"   Retrain this phase with more timesteps or tune hyperparameters.")
                            return
                    except EOFError:
                        # Non-interactive mode (background execution), continue automatically
                        print(f"   Non-interactive mode detected, proceeding to next phase...")

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
