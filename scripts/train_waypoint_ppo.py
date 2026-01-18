#!/usr/bin/env python3
"""
GPU-Accelerated PPO Training for Waypoint-Following Pilot

Optimizations:
- Vectorized environments (parallel simulation on CPU)
- GPU for neural network forward/backward passes
- Large batch sizes for GPU efficiency
- Multi-worker data collection

Hardware target: RTX 4060 (8GB) + 64GB RAM + Multi-core CPU

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from datetime import datetime
import argparse
from collections import deque
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))

from flight_dynamics import FlightSimulator, StateIndex
from triangle_controller import TriangleInterceptController, XCPhase
from generate_waypoint_demos import WaypointDemoGenerator
from train_waypoint_bc import WaypointPilotNN

# Constants
M_TO_FT = 3.28084
FT_TO_M = 0.3048


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class WaypointPilotActor(nn.Module):
    """Actor network for PPO - matches BC architecture exactly for weight transfer."""
    def __init__(self, input_dim=18, action_dim=4, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Match BC architecture exactly: Linear->LayerNorm->ReLU pattern
        # BC has: 18->256->256->128->4
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # actor.0
            nn.LayerNorm(hidden_dim),              # actor.1
            nn.ReLU(),                             # actor.2
            nn.Linear(hidden_dim, hidden_dim),     # actor.3
            nn.LayerNorm(hidden_dim),              # actor.4
            nn.ReLU(),                             # actor.5
            nn.Linear(hidden_dim, hidden_dim // 2),  # actor.6 (256->128)
            nn.LayerNorm(hidden_dim // 2),         # actor.7
            nn.ReLU(),                             # actor.8
            nn.Linear(hidden_dim // 2, action_dim),  # actor.9
            nn.Tanh(),                             # actor.10
        )
        
        # Log std for stochastic policy
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, x):
        mean = self.actor(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_action(self, x, deterministic=False):
        mean, std = self.forward(x)
        if deterministic:
            return mean, None, None
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist.entropy().sum(dim=-1)

    def evaluate_actions(self, x, actions):
        mean, std = self.forward(x)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class WaypointPilotCritic(nn.Module):
    """Value function for PPO - GPU accelerated."""
    def __init__(self, input_dim=18, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class VectorizedWaypointEnv:
    """
    Vectorized environment - runs N parallel simulations.
    Uses FlightSimulator's native batch support for efficiency.
    """
    def __init__(self, n_envs=32, dt=0.02, max_steps=3000):
        self.n_envs = n_envs
        self.dt = dt
        self.max_steps = max_steps

        # Single simulator handles all environments in parallel
        self.sim = FlightSimulator(n_instances=n_envs, dt=dt, use_gpu=False)

        # Per-environment state
        self.controllers = [
            TriangleInterceptController(cruise_altitude_ft=5500.0, pattern_altitude_ft=1500.0)
            for _ in range(n_envs)
        ]
        self.wp_generators = [
            WaypointDemoGenerator(ctrl) for ctrl in self.controllers
        ]
        self.step_counts = np.zeros(n_envs, dtype=np.int32)
        self.prev_distances = np.zeros(n_envs, dtype=np.float32)
        self.prev_actions = np.zeros((n_envs, 4), dtype=np.float32)

    def reset(self, env_ids=None):
        """Reset specified environments (or all if None)."""
        if env_ids is None:
            env_ids = np.arange(self.n_envs)

        # Reset state for specified envs
        initial_states = np.zeros((self.n_envs, 12), dtype=np.float32)
        current_states = self.sim.get_states()

        for i in env_ids:
            initial_states[i, 0] = -400.0  # X
            initial_states[i, 3] = 5.0     # U (taxi speed)
            self.step_counts[i] = 0
            self.prev_distances[i] = 0
            self.prev_actions[i] = 0

            # Reset controller
            self.controllers[i] = TriangleInterceptController(
                cruise_altitude_ft=5500.0, pattern_altitude_ft=1500.0
            )
            self.wp_generators[i] = WaypointDemoGenerator(self.controllers[i])

        # Keep non-reset environments at their current state
        for i in range(self.n_envs):
            if i not in env_ids:
                initial_states[i] = current_states[i]

        self.sim.reset(initial_states)

        return self._get_obs()

    def step(self, actions):
        """
        Step all environments in parallel.
        actions: (n_envs, 4) array of [throttle, aileron, elevator, rudder]
        """
        # Convert actions to full control vectors
        full_actions = np.zeros((self.n_envs, 7), dtype=np.float32)
        full_actions[:, 0] = np.clip((actions[:, 0] + 1) / 2, 0, 1)  # throttle
        full_actions[:, 1:4] = np.clip(actions[:, 1:4], -1, 1)  # aileron, elevator, rudder

        # Get classical controller actions for flaps/spoilers/brakes
        states = self.sim.get_states()
        for i in range(self.n_envs):
            classical = self.controllers[i].compute_action(states[i], self.step_counts[i] * self.dt)
            full_actions[i, 4:7] = classical[4:7]

        # Step simulation (vectorized!)
        self.sim.set_controls(full_actions)
        self.sim.step()
        self.step_counts += 1

        # Get new states
        new_states = self.sim.get_states()

        # Compute rewards and dones
        rewards, infos = self._compute_rewards(new_states, actions)
        dones = self._check_dones(new_states)

        # Store previous actions
        self.prev_actions = actions.copy()

        # Get observations
        obs, waypoints = self._get_obs()

        return obs, waypoints, rewards, dones, infos

    def _get_obs(self):
        """Get observations and waypoints for all environments."""
        states = self.sim.get_states()
        waypoints = np.zeros((self.n_envs, 6), dtype=np.float32)

        for i in range(self.n_envs):
            waypoints[i] = self.wp_generators[i].get_current_waypoint(
                self.controllers[i].phase, states[i]
            )

        return states, waypoints

    def _compute_rewards(self, states, actions):
        """Compute rewards for all environments (vectorized where possible)."""
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        infos = [{} for _ in range(self.n_envs)]

        _, waypoints = self._get_obs()

        for i in range(self.n_envs):
            state = states[i]
            waypoint = waypoints[i]
            action = actions[i]

            x, y, z = state[0], state[1], state[2]
            u, v, w = state[3], state[4], state[5]
            phi, theta, psi = state[6], state[7], state[8]

            altitude = -z
            airspeed = np.sqrt(u**2 + v**2 + w**2)

            wp_x, wp_y, wp_z = waypoint[0], waypoint[1], waypoint[2]
            target_speed = waypoint[3]
            distance = waypoint[5]

            reward = 0.0

            # 1. Distance reduction (progress reward)
            if self.prev_distances[i] > 0:
                distance_delta = self.prev_distances[i] - distance
                reward += 0.1 * distance_delta
            self.prev_distances[i] = distance

            # 2. Altitude tracking
            target_alt = -wp_z if wp_z != 0 else 1676.0
            alt_error = abs(altitude - target_alt) / 100.0
            reward -= 0.02 * min(alt_error, 5.0)

            # 3. Speed tracking
            speed_error = abs(airspeed - target_speed) / 10.0
            reward -= 0.01 * min(speed_error, 3.0)

            # 4. Heading alignment
            dx, dy = wp_x - x, wp_y - y
            desired_heading = np.arctan2(dy, dx)
            heading_error = abs(self._angle_diff(psi, desired_heading))
            reward -= 0.02 * heading_error / np.pi

            # 5. Control smoothness
            action_delta = np.sum(np.abs(action - self.prev_actions[i]))
            reward -= 0.01 * action_delta

            # 6. Survival bonus
            reward += 0.01

            # 7. Landing bonus
            if self.controllers[i].phase == XCPhase.LANDED:
                reward += 100.0

            # 8. Crash penalty
            if altitude < -5 and self.step_counts[i] > 100:
                reward -= 50.0

            # 9. Attitude penalties (keep aircraft stable)
            if abs(phi) > np.deg2rad(45):
                reward -= 0.1 * (abs(phi) - np.deg2rad(45))
            if abs(theta) > np.deg2rad(30):
                reward -= 0.1 * (abs(theta) - np.deg2rad(30))

            rewards[i] = reward
            infos[i] = {'distance': distance, 'altitude': altitude, 'speed': airspeed}

        return rewards, infos

    def _angle_diff(self, a, b):
        diff = a - b
        while diff > np.pi: diff -= 2 * np.pi
        while diff < -np.pi: diff += 2 * np.pi
        return diff

    def _check_dones(self, states):
        """Check termination for all environments."""
        dones = np.zeros(self.n_envs, dtype=bool)

        for i in range(self.n_envs):
            altitude = -states[i, 2]

            if self.step_counts[i] >= self.max_steps:
                dones[i] = True
            elif altitude < -10 and self.step_counts[i] > 100:
                dones[i] = True
            elif self.controllers[i].phase == XCPhase.LANDED:
                dones[i] = True
            elif altitude > 3000:
                dones[i] = True

        return dones


class PPOTrainerGPU:
    """GPU-accelerated PPO trainer with KL divergence regularization against BC reference.

    Key feature: Maintains a frozen copy of the BC policy and penalizes divergence
    from it, preventing the policy from forgetting expert behavior during RL fine-tuning.
    Reference: https://arxiv.org/abs/2512.16911 (Posterior BC), verl PPO implementation
    """

    def __init__(
        self,
        actor: WaypointPilotActor,
        critic: WaypointPilotCritic,
        reference_actor: WaypointPilotActor,  # Frozen BC reference policy
        norm_stats: dict,
        device: torch.device,
        lr_actor: float = 1e-5,  # Lower LR for more stable fine-tuning
        lr_critic: float = 3e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.1,  # Standard PPO clipping
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        kl_coef: float = 0.1,  # KL divergence penalty coefficient
        kl_target: float = 0.01,  # Target KL for adaptive coefficient
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 512,  # Larger batch for GPU
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device

        # Store frozen BC reference policy (never updated)
        self.reference_actor = reference_actor.to(device)
        self.reference_actor.eval()
        for param in self.reference_actor.parameters():
            param.requires_grad = False

        self.norm_stats = {
            k: torch.FloatTensor(v).to(device) if isinstance(v, np.ndarray) else v
            for k, v in norm_stats.items()
        }

        self.actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.kl_coef = kl_coef
        self.kl_target = kl_target
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def compute_kl_divergence(self, obs, actions):
        """Compute KL divergence between current policy and BC reference.

        KL(current || reference) = E[log(current/reference)]
        For Gaussian policies: KL = sum((mu_curr - mu_ref)^2 / (2*var_ref) + log(std_ref/std_curr) + var_curr/(2*var_ref) - 0.5)
        """
        # Get distributions from both policies
        current_mean, current_std = self.actor(obs)
        with torch.no_grad():
            ref_mean, ref_std = self.reference_actor(obs)

        # KL divergence for multivariate Gaussian (per-dimension, then sum)
        var_curr = current_std ** 2
        var_ref = ref_std ** 2

        kl = 0.5 * (
            (current_mean - ref_mean) ** 2 / var_ref +
            var_curr / var_ref -
            1.0 +
            2 * (torch.log(ref_std) - torch.log(current_std))
        ).sum(dim=-1)

        return kl.mean()

    def normalize_obs_batch(self, states, waypoints):
        """Normalize observations (vectorized, GPU)."""
        # Convert to tensors on GPU
        states_t = torch.FloatTensor(states).to(self.device)
        waypoints_t = torch.FloatTensor(waypoints).to(self.device)

        # Normalize states
        obs_mean = self.norm_stats['obs_mean']
        obs_std = self.norm_stats['obs_std']
        states_norm = (states_t - obs_mean) / obs_std

        # Normalize waypoints
        wp_norm = torch.zeros_like(waypoints_t)
        wp_norm[:, 0] = (waypoints_t[:, 0] - self.norm_stats['wp_x_mean']) / self.norm_stats['wp_x_std']
        wp_norm[:, 1] = (waypoints_t[:, 1] - self.norm_stats['wp_y_mean']) / self.norm_stats['wp_y_std']
        wp_norm[:, 2] = (waypoints_t[:, 2] - self.norm_stats['wp_z_mean']) / self.norm_stats['wp_z_std']
        wp_norm[:, 3] = (waypoints_t[:, 3] - self.norm_stats['wp_spd_mean']) / self.norm_stats['wp_spd_std']
        wp_norm[:, 4] = waypoints_t[:, 4]  # categorical
        wp_norm[:, 5] = (waypoints_t[:, 5] - self.norm_stats['wp_dist_mean']) / self.norm_stats['wp_dist_std']

        return torch.cat([states_norm, wp_norm], dim=1)

    def collect_rollout(self, env: VectorizedWaypointEnv, n_steps=256):
        """Collect experience from vectorized environment."""
        n_envs = env.n_envs

        # Storage
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        states, waypoints = env.reset()

        for step in range(n_steps):
            # Normalize and get actions (GPU)
            obs = self.normalize_obs_batch(states, waypoints)

            with torch.no_grad():
                action, log_prob, _ = self.actor.get_action(obs)
                value = self.critic(obs)

            action_np = action.cpu().numpy()

            # Step environment (CPU)
            next_states, next_waypoints, reward, done, info = env.step(action_np)

            # Store
            observations.append(obs.cpu().numpy())
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            values.append(value.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())

            # Reset done environments
            done_ids = np.where(done)[0]
            if len(done_ids) > 0:
                env.reset(done_ids)

            states, waypoints = next_states, next_waypoints

        # Get bootstrap value
        obs = self.normalize_obs_batch(states, waypoints)
        with torch.no_grad():
            last_value = self.critic(obs).cpu().numpy()

        # Flatten: (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        return {
            'observations': np.array(observations).reshape(-1, 18),
            'actions': np.array(actions).reshape(-1, 4),
            'rewards': np.array(rewards).flatten(),
            'dones': np.array(dones).flatten(),
            'values': np.array(values).flatten(),
            'log_probs': np.array(log_probs).flatten(),
            'last_value': last_value,
            'n_envs': n_envs,
            'n_steps': n_steps,
        }

    def compute_gae(self, rewards, values, dones, last_value, n_envs, n_steps):
        """Compute GAE (vectorized across environments)."""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        # Reshape for per-env computation
        rewards = rewards.reshape(n_steps, n_envs)
        values = values.reshape(n_steps, n_envs)
        dones = dones.reshape(n_steps, n_envs)

        gae = np.zeros(n_envs)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae

            advantages[t * n_envs:(t + 1) * n_envs] = gae
            returns[t * n_envs:(t + 1) * n_envs] = gae + values[t]

        return advantages.flatten(), returns.flatten()

    def update(self, rollout):
        """Perform PPO update (GPU accelerated)."""
        # Move data to GPU
        observations = torch.FloatTensor(rollout['observations']).to(self.device)
        actions = torch.FloatTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)

        advantages, returns = self.compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'],
            rollout['last_value'], rollout['n_envs'], rollout['n_steps']
        )

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = len(observations)
        indices = np.arange(n_samples)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]

                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Actor update
                new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                # KL divergence penalty against BC reference policy
                kl_loss = self.compute_kl_divergence(batch_obs, batch_actions)

                # Total actor loss: PPO objective + entropy bonus + KL penalty
                total_loss = actor_loss + self.entropy_coef * entropy_loss + self.kl_coef * kl_loss

                self.actor_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                values = self.critic(batch_obs)
                critic_loss = nn.MSELoss()(values, batch_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl_loss.item()
                n_updates += 1

        # Adaptive KL coefficient (increase if KL too high, decrease if too low)
        avg_kl = total_kl / n_updates
        if avg_kl > self.kl_target * 1.5:
            self.kl_coef *= 1.5
        elif avg_kl < self.kl_target / 1.5:
            self.kl_coef /= 1.5
        self.kl_coef = max(0.01, min(1.0, self.kl_coef))  # Clamp to reasonable range

        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'kl_divergence': avg_kl,
            'kl_coef': self.kl_coef,
        }


def load_bc_model(checkpoint_path):
    """Load BC model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    bc_model = WaypointPilotNN(
        input_dim=checkpoint['input_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    bc_model.load_state_dict(checkpoint['model_state_dict'])

    norm_stats = {
        'obs_mean': checkpoint['obs_mean'],
        'obs_std': checkpoint['obs_std'],
        'wp_x_mean': float(checkpoint['wp_xy_mean'][0]),
        'wp_x_std': float(checkpoint['wp_xy_std'][0]),
        'wp_y_mean': float(checkpoint['wp_xy_mean'][1]),
        'wp_y_std': float(checkpoint['wp_xy_std'][1]),
        'wp_z_mean': float(checkpoint['wp_xy_mean'][2]),
        'wp_z_std': float(checkpoint['wp_xy_std'][2]),
        'wp_spd_mean': float(checkpoint['wp_spd_mean']),
        'wp_spd_std': float(checkpoint['wp_spd_std']),
        'wp_dist_mean': float(checkpoint['wp_dist_mean']),
        'wp_dist_std': float(checkpoint['wp_dist_std']),
    }

    return bc_model, norm_stats, checkpoint



def transfer_bc_to_actor(bc_model, actor):
    """Transfer BC weights to PPO actor (architectures must match exactly)."""
    bc_state = bc_model.state_dict()
    actor_state = actor.state_dict()
    
    # BC and PPO actor now have identical architecture
    # Just copy the actor weights directly
    transferred = 0
    for key in actor_state.keys():
        if key.startswith("actor.") and key in bc_state:
            actor_state[key] = bc_state[key].clone()
            transferred += 1
        elif key == "log_std" and key in bc_state:
            actor_state[key] = bc_state[key].clone()
            transferred += 1
    
    actor.load_state_dict(actor_state)
    print(f"BC weights transferred to PPO actor ({transferred} tensors)")


def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated PPO Training')
    parser.add_argument('--bc-checkpoint', type=str,
                        default='checkpoints/waypoint_pilot/waypoint_pilot_best.pt')
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--n-envs', type=int, default=32, help='Parallel environments')
    parser.add_argument('--steps-per-env', type=int, default=256, help='Steps per env per iteration')
    parser.add_argument('--batch-size', type=int, default=512, help='GPU batch size')
    parser.add_argument('--output-dir', type=str, default='checkpoints/waypoint_pilot_ppo')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 70)
    print("  GPU-ACCELERATED PPO TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Steps/env/iter: {args.steps_per_env}")
    print(f"Total steps/iter: {args.n_envs * args.steps_per_env}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.iterations}")
    print("=" * 70)

    # Load BC model
    print("\nLoading BC model...")
    bc_model, norm_stats, bc_checkpoint = load_bc_model(args.bc_checkpoint)

    # Create networks
    actor = WaypointPilotActor(
        input_dim=bc_checkpoint['input_dim'],
        action_dim=bc_checkpoint['action_dim'],
        hidden_dim=bc_checkpoint['hidden_dim']
    )
    critic = WaypointPilotCritic(
        input_dim=bc_checkpoint['input_dim'],
        hidden_dim=bc_checkpoint['hidden_dim']
    )

    # Create frozen reference actor (will hold BC weights)
    reference_actor = WaypointPilotActor(
        input_dim=bc_checkpoint['input_dim'],
        action_dim=bc_checkpoint['action_dim'],
        hidden_dim=bc_checkpoint['hidden_dim']
    )

    # Transfer BC weights to both actor and reference
    transfer_bc_to_actor(bc_model, actor)
    transfer_bc_to_actor(bc_model, reference_actor)
    print("BC weights transferred to reference actor (frozen)")

    # Create vectorized environment
    print(f"\nCreating {args.n_envs} parallel environments...")
    env = VectorizedWaypointEnv(n_envs=args.n_envs, dt=0.02, max_steps=3000)

    # Create trainer with KL regularization against BC reference
    trainer = PPOTrainerGPU(
        actor=actor,
        critic=critic,
        reference_actor=reference_actor,  # Frozen BC reference for KL penalty
        norm_stats=norm_stats,
        device=device,
        lr_actor=3e-5,
        lr_critic=1e-4,
        kl_coef=0.1,  # KL penalty coefficient
        kl_target=0.01,  # Target KL divergence
        batch_size=args.batch_size,
    )

    print("\nStarting training...")
    print("-" * 70)

    best_reward = float('-inf')
    reward_history = deque(maxlen=20)
    start_time = time.time()

    for iteration in range(args.iterations):
        iter_start = time.time()

        # Collect rollout
        rollout = trainer.collect_rollout(env, n_steps=args.steps_per_env)

        # Compute episode rewards
        ep_rewards = []
        ep_reward = 0
        for r, d in zip(rollout['rewards'], rollout['dones']):
            ep_reward += r
            if d:
                ep_rewards.append(ep_reward)
                ep_reward = 0

        mean_reward = np.mean(ep_rewards) if ep_rewards else rollout['rewards'].sum() / args.n_envs
        reward_history.append(mean_reward)

        # PPO update
        losses = trainer.update(rollout)

        iter_time = time.time() - iter_start
        steps_per_sec = args.n_envs * args.steps_per_env / iter_time

        if iteration % 5 == 0:
            avg_reward = np.mean(reward_history)
            kl_info = f"KL: {losses['kl_divergence']:.4f} (coef={losses['kl_coef']:.3f})" if 'kl_divergence' in losses else ""
            print(f"Iter {iteration:4d} | "
                  f"Reward: {mean_reward:8.1f} (avg: {avg_reward:7.1f}) | "
                  f"Actor: {losses['actor_loss']:.4f} | "
                  f"Critic: {losses['critic_loss']:.4f} | "
                  f"{kl_info} | "
                  f"{steps_per_sec:.0f} steps/s")

        # Save best
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                'iteration': iteration,
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'best_reward': best_reward,
                'input_dim': bc_checkpoint['input_dim'],
                'action_dim': bc_checkpoint['action_dim'],
                'hidden_dim': bc_checkpoint['hidden_dim'],
                'obs_mean': bc_checkpoint['obs_mean'],
                'obs_std': bc_checkpoint['obs_std'],
                'wp_xy_mean': bc_checkpoint['wp_xy_mean'],
                'wp_xy_std': bc_checkpoint['wp_xy_std'],
                'wp_spd_mean': bc_checkpoint['wp_spd_mean'],
                'wp_spd_std': bc_checkpoint['wp_spd_std'],
                'wp_dist_mean': bc_checkpoint['wp_dist_mean'],
                'wp_dist_std': bc_checkpoint['wp_dist_std'],
            }, output_dir / 'waypoint_pilot_ppo_best.pt')
            print(f"  -> New best! (reward={best_reward:.1f})")

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"  TRAINING COMPLETE in {total_time/60:.1f} min")
    print(f"  Best reward: {best_reward:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
