#!/usr/bin/env python3
"""
Train Neural Network for Teardrop Approach

TRAINING PIPELINE
=================
Phase 1: Behavioral Cloning (BC) on Triangle Approach
    - Generate expert demos using classical triangle_controller
    - NN learns: flight dynamics, approach, glideslope, landing
    - Train on RWY 31 approaches

Phase 2: PPO Fine-tuning on Teardrop Task (RWY 13)
    - Start aircraft approaching KHUT from wrong direction
    - Aircraft must land on RWY 13 (opposite of RWY 31)
    - RL must DISCOVER the teardrop/course reversal maneuver
    - No expert demos for teardrop - NN figures it out!

SUCCESS CRITERIA:
    NN learns to execute teardrop approach and land on RWY 13,
    even though it was only trained on RWY 31 triangle approaches.

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import os
import sys
import argparse
import subprocess
import atexit
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

# Import environments
from aida_sim.env.teardrop_approach_env import TeardropApproachEnv

# Import triangle controller for demo generation
sys.path.insert(0, str(Path(__file__).parent))
from triangle_controller import TriangleInterceptController, XCPhase


# TensorBoard process
_tensorboard_process = None


def start_tensorboard(logdir, port=6006):
    """Start TensorBoard in background."""
    global _tensorboard_process
    try:
        subprocess.run(['pkill', '-f', f'tensorboard.*{port}'],
                      stderr=subprocess.DEVNULL, check=False)
        import time
        time.sleep(1)

        print(f"\n🚀 Starting TensorBoard on http://localhost:{port}")
        _tensorboard_process = subprocess.Popen(
            ['tensorboard', '--logdir', logdir, '--port', str(port),
             '--host', '0.0.0.0', '--reload_interval', '10'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp
        )
        time.sleep(2)
        print(f"✓ TensorBoard started (PID: {_tensorboard_process.pid})")

        def cleanup():
            if _tensorboard_process and _tensorboard_process.poll() is None:
                _tensorboard_process.terminate()
        atexit.register(cleanup)

    except Exception as e:
        print(f"⚠️  Could not start TensorBoard: {e}")


class LandingSuccessCallback(BaseCallback):
    """Track landing success rate during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.landing_successes = 0
        self.landing_attempts = 0

    def _on_step(self):
        # Check for episode end in infos
        for info in self.locals.get('infos', []):
            if 'termination_reason' in info:
                self.landing_attempts += 1
                if info['termination_reason'] == 'landed_success':
                    self.landing_successes += 1

                if self.landing_attempts % 100 == 0:
                    success_rate = self.landing_successes / self.landing_attempts * 100
                    print(f"  Landing success rate: {self.landing_successes}/{self.landing_attempts} ({success_rate:.1f}%)")

        return True


def generate_triangle_demos(num_episodes=50, output_path=None):
    """
    Generate expert demonstrations using triangle_controller.

    These demos train the NN on basic flight and RWY 31 approaches.
    """
    from flight_dynamics import FlightSimulator, StateIndex, STATE_DIM, CONTROL_DIM
    from aircraft_database import get_aircraft

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent.parent / "bc_data" / f"triangle_approach_demos_{timestamp}.npz"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("  GENERATING TRIANGLE APPROACH DEMOS (Classical Controller)")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Output: {output_path}")

    # Setup simulator
    aircraft = get_aircraft('cessna172')
    from aida_sim.env.flight_env_cessna172 import Cessna172Env
    env = Cessna172Env.__new__(Cessna172Env)
    params = env._config_to_params.__func__(env, aircraft)

    dt = 0.02
    sim = FlightSimulator(n_instances=1, params=params, dt=dt)
    controller = TriangleInterceptController()

    all_observations = []
    all_actions = []
    all_phases = []
    successful_episodes = 0

    for ep in range(num_episodes):
        # Reset to SN65 starting position
        initial_state = np.zeros((1, STATE_DIM), dtype=np.float32)
        initial_state[0, StateIndex.X] = -500.0  # On runway
        initial_state[0, StateIndex.Y] = 0.0
        initial_state[0, StateIndex.Z] = -0.5  # On ground
        initial_state[0, StateIndex.U] = 0.0  # At rest
        initial_state[0, StateIndex.PSI] = 0.0  # Runway heading

        sim.reset(initial_state=initial_state)
        controller.reset()

        ep_obs = []
        ep_actions = []
        ep_phases = []

        sim_time = 0.0
        max_steps = 15000  # 5 minutes

        for step in range(max_steps):
            state = sim.get_states()
            state = state.get()[0] if hasattr(state, 'get') else state[0]

            # Get expert action
            action = controller.compute_action(state, sim_time)

            # Store (only 4 control channels for NN)
            ep_obs.append(state.copy())
            ep_actions.append(action[:4].copy())  # throttle, aileron, elevator, rudder
            ep_phases.append(controller.phase.value)

            # Apply action
            controls = np.zeros((1, CONTROL_DIM), dtype=np.float32)
            controls[0, :len(action)] = action
            sim.set_controls(controls)
            sim.step()

            sim_time += dt

            # Check if landed
            if controller.phase == XCPhase.LANDED:
                successful_episodes += 1
                break

            # Check for problems
            altitude = -state[StateIndex.Z]
            if altitude < -10:  # Crashed
                break

        all_observations.extend(ep_obs)
        all_actions.extend(ep_actions)
        all_phases.extend(ep_phases)

        print(f"  Episode {ep+1}/{num_episodes}: {len(ep_obs)} steps, Phase={controller.phase.name}")

    # Save
    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    phases = np.array(all_phases, dtype=np.int32)

    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        phases=phases,
        num_episodes=num_episodes,
        successful_episodes=successful_episodes,
        dt=dt,
    )

    print(f"\n✓ Generated {len(observations):,} transitions from {successful_episodes}/{num_episodes} successful flights")
    print(f"  Saved to: {output_path}")

    return output_path


def pretrain_bc(model, bc_data_path, epochs=20, batch_size=128):
    """Pretrain policy with behavioral cloning."""
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    print(f"\n{'='*60}")
    print(f"  BEHAVIORAL CLONING - Learning from Triangle Approach")
    print(f"{'='*60}")

    if not os.path.exists(bc_data_path):
        print(f"⚠️  BC data not found: {bc_data_path}")
        return model

    data = np.load(bc_data_path)
    observations = data['observations']
    actions = data['actions']

    print(f"Loaded {len(observations):,} expert transitions")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")

    # Handle observation dimension (might need to pad for goal info)
    obs_dim = model.observation_space.shape[0]
    if observations.shape[1] < obs_dim:
        # Pad with zeros for goal-relative features (will be ignored by BC)
        padding = np.zeros((len(observations), obs_dim - observations.shape[1]), dtype=np.float32)
        observations = np.concatenate([observations, padding], axis=1)

    # Handle action dimension
    action_dim = model.action_space.shape[0]
    if actions.shape[1] < action_dim:
        padding = np.zeros((len(actions), action_dim - actions.shape[1]), dtype=np.float32)
        actions = np.concatenate([actions, padding], axis=1)

    device = model.device
    obs_tensor = torch.FloatTensor(observations).to(device)
    act_tensor = torch.FloatTensor(actions).to(device)

    dataset = TensorDataset(obs_tensor, act_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = model.policy
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    policy.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for obs_batch, act_batch in dataloader:
            optimizer.zero_grad()
            action_dist = policy.get_distribution(obs_batch)
            predicted = action_dist.distribution.mean
            loss = mse_loss(predicted, act_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        print(f"  Epoch {epoch+1}/{epochs}: Loss = {total_loss/n_batches:.6f}")

    policy.eval()

    # Set small exploration noise
    with torch.no_grad():
        if hasattr(policy, 'log_std'):
            policy.log_std.fill_(-2.0)

    print(f"✓ BC pretraining complete!")
    return model


def make_teardrop_env(start_mode="cruise", rank=0):
    """Create teardrop approach environment."""
    def _init():
        env = TeardropApproachEnv(start_mode=start_mode)
        env = Monitor(env)
        return env
    return _init


def train_teardrop(
    bc_data_path=None,
    bc_epochs=20,
    ppo_timesteps=2_000_000,
    n_envs=4,
    start_mode="cruise",
    device='cuda',
    checkpoint_dir='checkpoints/teardrop_approach',
):
    """Train PPO for teardrop approach."""

    print("\n" + "="*70)
    print("  TEARDROP APPROACH TRAINING")
    print("="*70)
    print("\n  Strategy:")
    print("    1. BC warm-start on triangle approach demos (RWY 31)")
    print("    2. PPO fine-tune to discover teardrop for RWY 13")
    print("="*70)
    print(f"\nBC data: {bc_data_path}")
    print(f"PPO timesteps: {ppo_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Start mode: {start_mode}")
    print(f"Device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    tensorboard_log = f"{checkpoint_dir}/tensorboard"
    start_tensorboard(tensorboard_log)

    # Create environments
    print(f"\nCreating {n_envs} teardrop approach environments...")
    if n_envs > 1:
        env = SubprocVecEnv([make_teardrop_env(start_mode, i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_teardrop_env(start_mode)])

    eval_env = DummyVecEnv([make_teardrop_env(start_mode, 999)])

    # Create PPO model
    print("\nInitializing PPO...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
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

    # BC pretraining
    if bc_data_path and os.path.exists(bc_data_path):
        model = pretrain_bc(model, bc_data_path, epochs=bc_epochs)
    else:
        print("\n⚠️  No BC data - training from scratch (will be harder!)")

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1000),
        save_path=checkpoint_dir,
        name_prefix='teardrop_ppo',
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=f"{checkpoint_dir}/eval",
        eval_freq=max(25_000 // n_envs, 500),
        n_eval_episodes=10,
        deterministic=True,
    )

    landing_cb = LandingSuccessCallback()

    # Train
    print("\n" + "="*70)
    print("  PPO TRAINING - Discovering Teardrop Maneuver")
    print("="*70)
    print(f"\nTraining for {ppo_timesteps:,} timesteps...")
    print("The NN must figure out how to reverse course and land on RWY 13!")
    print(f"\nTensorBoard: http://localhost:6006")

    try:
        model.learn(
            total_timesteps=ppo_timesteps,
            callback=[checkpoint_cb, eval_cb, landing_cb],
            tb_log_name='teardrop',
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")

    # Save final
    final_path = f"{checkpoint_dir}/teardrop_ppo_final.zip"
    model.save(final_path)

    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal model: {final_path}")
    print(f"Best model: {checkpoint_dir}/best_model.zip")

    if landing_cb.landing_attempts > 0:
        success_rate = landing_cb.landing_successes / landing_cb.landing_attempts * 100
        print(f"\nFinal landing success rate: {success_rate:.1f}%")

    print("\n" + "="*70)
    print("  NEXT STEPS")
    print("="*70)
    print(f"\n1. Test teardrop approach:")
    print(f"   python scripts/run_teardrop_with_telemetry.py --model {final_path}")
    print(f"\n2. Compare with classical triangle approach:")
    print(f"   python scripts/run_xc_sn65_khut.py")

    env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description='Train teardrop approach')

    parser.add_argument('--generate-demos', action='store_true',
                        help='Generate triangle approach demos first')
    parser.add_argument('--demo-episodes', type=int, default=50,
                        help='Number of demo episodes to generate')
    parser.add_argument('--bc-data', type=str, default=None,
                        help='Path to BC data (if not generating)')
    parser.add_argument('--bc-epochs', type=int, default=20,
                        help='BC pretraining epochs')
    parser.add_argument('--timesteps', type=int, default=2_000_000,
                        help='PPO training timesteps')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Parallel environments')
    parser.add_argument('--start-mode', type=str, default='cruise',
                        choices=['cruise', 'approach'],
                        help='Where to start aircraft')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--checkpoint-dir', type=str,
                        default='checkpoints/teardrop_approach')

    args = parser.parse_args()

    # Generate demos if requested
    bc_data = args.bc_data
    if args.generate_demos or bc_data is None:
        print("\nGenerating triangle approach demos...")
        bc_data = generate_triangle_demos(num_episodes=args.demo_episodes)

    # Train
    train_teardrop(
        bc_data_path=bc_data,
        bc_epochs=args.bc_epochs,
        ppo_timesteps=args.timesteps,
        n_envs=args.n_envs,
        start_mode=args.start_mode,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
