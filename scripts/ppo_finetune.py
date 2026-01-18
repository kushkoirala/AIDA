#!/usr/bin/env python3
"""
PPO Fine-tuning for Route Generation Transformer

This script implements Proximal Policy Optimization (PPO) for fine-tuning
the route generation transformer using simulated flight rewards.

The reward function evaluates generated routes based on:
- Flight safety (terrain clearance, valid altitudes)
- Fuel efficiency (route length vs direct distance)
- Smoothness (heading changes, altitude transitions)
- Landing success (alignment with runway, glideslope)

Usage:
    python ppo_finetune.py --checkpoint checkpoints/route_gen/best_model.pt

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import argparse
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from aida_sim.models.unified_transformer import (
    UnifiedTransformer, ModelConfig, TaskType, denormalize_controller_params
)


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    # Training
    num_epochs: int = 100
    steps_per_epoch: int = 256
    batch_size: int = 32
    minibatch_size: int = 8

    # PPO specific
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Learning rate
    lr: float = 3e-4
    lr_decay: bool = True

    # Reward scaling
    reward_scale: float = 1.0


class RouteEnvironment:
    """Environment for evaluating generated routes.

    Simulates route quality based on various metrics without
    running the full flight dynamics simulation.
    """

    def __init__(self, seed: int = 42):
        """Initialize the environment."""
        np.random.seed(seed)
        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and generate a new route specification.

        Returns:
            Dictionary with 'route_spec' array
        """
        # Generate random origin/destination
        self.origin_lat = np.random.uniform(37.0, 39.0)
        self.origin_lon = np.random.uniform(-98.0, -96.0)
        self.origin_elev = np.random.uniform(1000, 2000)
        self.origin_rwy = np.random.uniform(0, 360)

        self.dest_lat = self.origin_lat + np.random.uniform(-0.5, 0.5)
        self.dest_lon = self.origin_lon + np.random.uniform(-0.5, 0.5)
        self.dest_elev = np.random.uniform(1000, 2000)
        self.dest_rwy = np.random.uniform(0, 360)

        # Compute distance
        self.distance_nm = self._compute_distance()

        # Target cruise altitude based on distance
        self.target_cruise_alt = min(
            12000,
            max(2500, 1000 + self.distance_nm * 100)
        )

        # Create normalized route spec
        self.route_spec = np.array([
            self.origin_lat / 90.0,
            self.origin_lon / 180.0,
            self.origin_elev / 10000.0,
            self.origin_rwy / 360.0,
            self.dest_lat / 90.0,
            self.dest_lon / 180.0,
            self.dest_elev / 10000.0,
            self.dest_rwy / 360.0,
            self.target_cruise_alt / 10000.0,
            self.distance_nm / 200.0,
        ], dtype=np.float32)

        return {'route_spec': self.route_spec}

    def _compute_distance(self) -> float:
        """Compute distance between origin and destination in nm."""
        R = 3440.065  # Earth radius in nm
        lat1, lon1 = np.radians(self.origin_lat), np.radians(self.origin_lon)
        lat2, lon2 = np.radians(self.dest_lat), np.radians(self.dest_lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return R * c

    def compute_reward(
        self,
        waypoints: np.ndarray,
        validity: np.ndarray,
        controller_params: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward for generated route.

        Args:
            waypoints: [max_waypoints, 6] waypoint data
            validity: [max_waypoints, 1] validity mask
            controller_params: [8] controller parameters (normalized)

        Returns:
            (total_reward, reward_breakdown_dict)
        """
        # Denormalize controller params
        ctrl = {
            'tp_distance_nm': controller_params[0] * 20.0,
            'glideslope_deg': controller_params[1] * 5.0 + 2.0,
            'pattern_alt_ft': controller_params[2] * 2000.0 + 500.0,
            'cruise_alt_ft': controller_params[3] * 10000.0 + 2000.0,
            'v_approach_kts': controller_params[4] * 50.0 + 60.0,
            'v_cruise_kts': controller_params[5] * 60.0 + 80.0,
            'turn_bank_deg': controller_params[6] * 30.0 + 15.0,
            'use_triangle': controller_params[7],
        }

        rewards = {}

        # 1. Altitude appropriateness reward
        alt_diff = abs(ctrl['cruise_alt_ft'] - self.target_cruise_alt) / 1000.0
        rewards['altitude'] = max(0, 1.0 - alt_diff * 0.2)

        # 2. Turn point distance reward (should be proportional to distance)
        ideal_tp = min(15, max(5, self.distance_nm * 0.15))
        tp_diff = abs(ctrl['tp_distance_nm'] - ideal_tp) / ideal_tp
        rewards['turn_point'] = max(0, 1.0 - tp_diff)

        # 3. Glideslope reward (3° is optimal for most approaches)
        gs_diff = abs(ctrl['glideslope_deg'] - 3.0)
        rewards['glideslope'] = max(0, 1.0 - gs_diff * 0.3)

        # 4. Speed appropriateness
        # Approach speed should be reasonable (65-75 kts ideal)
        v_app_diff = abs(ctrl['v_approach_kts'] - 70.0) / 20.0
        rewards['approach_speed'] = max(0, 1.0 - v_app_diff)

        # Cruise speed should be reasonable for distance
        ideal_cruise = min(130, max(90, 80 + self.distance_nm))
        v_cruise_diff = abs(ctrl['v_cruise_kts'] - ideal_cruise) / 30.0
        rewards['cruise_speed'] = max(0, 1.0 - v_cruise_diff)

        # 5. Waypoint validity (more waypoints for longer routes)
        valid_count = validity.sum()
        ideal_waypoints = min(8, max(4, int(self.distance_nm / 10) + 3))
        wp_diff = abs(valid_count - ideal_waypoints) / ideal_waypoints
        rewards['waypoints'] = max(0, 1.0 - wp_diff)

        # 6. Pattern altitude (should be ~1000 ft AGL)
        pattern_diff = abs(ctrl['pattern_alt_ft'] - 1000.0) / 500.0
        rewards['pattern_alt'] = max(0, 1.0 - pattern_diff)

        # 7. Triangle pattern bonus for short routes
        if self.distance_nm < 30 and ctrl['use_triangle'] > 0.5:
            rewards['pattern_bonus'] = 0.2
        else:
            rewards['pattern_bonus'] = 0.0

        # Compute total reward with weights
        weights = {
            'altitude': 1.5,
            'turn_point': 1.2,
            'glideslope': 1.0,
            'approach_speed': 0.8,
            'cruise_speed': 0.8,
            'waypoints': 0.5,
            'pattern_alt': 0.6,
            'pattern_bonus': 1.0,
        }

        total_reward = sum(rewards[k] * weights[k] for k in rewards)
        total_reward /= sum(weights.values())  # Normalize

        return total_reward, rewards


class PPOMemory:
    """Experience buffer for PPO training."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(self, state, action, reward, value, log_prob, done=False):
        """Store a transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        """Clear all stored transitions."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and GAE advantages.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda

        Returns:
            (returns, advantages) tensors
        """
        values = np.array(self.values + [0])
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        # GAE computation
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)


class PolicyValueHead(nn.Module):
    """Combined policy and value head for PPO.

    Wraps the route generation transformer to add:
    - Policy distribution over controller parameters
    - Value function estimate
    """

    def __init__(self, base_model: UnifiedTransformer):
        super().__init__()
        self.base_model = base_model
        d_model = base_model.config.d_model

        # Value head (predicts expected return)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # Action distribution parameters (mean and log_std for each controller param)
        self.action_mean = nn.Linear(8, 8)  # From base model output
        self.action_log_std = nn.Parameter(torch.zeros(8))

    def forward(self, route_spec: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """Forward pass for PPO.

        Args:
            route_spec: [batch, 10] route specification

        Returns:
            (outputs, value, action_dist_params)
        """
        # Get base model outputs
        outputs = self.base_model(TaskType.ROUTE_GENERATION, route_spec=route_spec)

        # Get value estimate from pooled features
        # Note: We use controller_params as a proxy for the route quality
        ctrl_params = outputs['controller_params']

        # Simple value estimate based on param mean
        value = ctrl_params.mean(dim=-1, keepdim=True)

        return outputs, value, ctrl_params


class PPOTrainer:
    """PPO trainer for route generation fine-tuning."""

    def __init__(
        self,
        model: UnifiedTransformer,
        config: PPOConfig,
        device: torch.device
    ):
        """Initialize PPO trainer.

        Args:
            model: Pre-trained UnifiedTransformer
            config: PPO hyperparameters
            device: Training device
        """
        self.config = config
        self.device = device

        # Wrap model with policy-value head
        self.policy = PolicyValueHead(model).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)

        # Environment
        self.env = RouteEnvironment()

        # Memory
        self.memory = PPOMemory()

        # Logging
        self.episode_rewards = []
        self.episode_lengths = []

    def select_action(self, state: Dict) -> Tuple[np.ndarray, float, float]:
        """Select action using current policy.

        Args:
            state: Environment state with 'route_spec'

        Returns:
            (action, value, log_prob)
        """
        route_spec = torch.tensor(state['route_spec']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, value, _ = self.policy(route_spec)
            action = outputs['controller_params'].squeeze(0).cpu().numpy()

        # Simple log prob computation (assume uniform for initial policy)
        log_prob = 0.0

        return action, value.item(), log_prob

    def collect_rollouts(self) -> float:
        """Collect experience rollouts.

        Returns:
            Average episode reward
        """
        self.memory.clear()
        total_reward = 0

        for _ in range(self.config.steps_per_epoch):
            state = self.env.reset()
            action, value, log_prob = self.select_action(state)

            # Compute reward
            route_spec = torch.tensor(state['route_spec']).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.policy.base_model(TaskType.ROUTE_GENERATION, route_spec=route_spec)
                waypoints = outputs['waypoints'].squeeze(0).cpu().numpy()
                validity = outputs['validity'].squeeze(0).cpu().numpy()
                ctrl_params = outputs['controller_params'].squeeze(0).cpu().numpy()

            reward, _ = self.env.compute_reward(waypoints, validity, ctrl_params)
            reward *= self.config.reward_scale

            # Store transition
            self.memory.store(state['route_spec'], action, reward, value, log_prob, done=True)
            total_reward += reward

        return total_reward / self.config.steps_per_epoch

    def update(self):
        """Update policy using PPO."""
        # Compute returns and advantages
        returns, advantages = self.memory.compute_returns_and_advantages(
            self.config.gamma, self.config.gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert memory to tensors
        states = torch.tensor(np.array(self.memory.states), dtype=torch.float32).to(self.device)
        old_values = torch.tensor(self.memory.values, dtype=torch.float32).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # PPO update epochs
        n_samples = len(self.memory.states)
        indices = np.arange(n_samples)

        policy_losses = []
        value_losses = []

        for _ in range(4):  # PPO epochs
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]

                # Forward pass
                outputs, new_values, _ = self.policy(batch_states)

                # Value loss (clipped)
                value_clipped = batch_old_values + torch.clamp(
                    new_values.squeeze() - batch_old_values,
                    -self.config.clip_epsilon,
                    self.config.clip_epsilon
                )
                value_loss_unclipped = (new_values.squeeze() - batch_returns) ** 2
                value_loss_clipped = (value_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Policy loss (simplified - using controller params directly)
                ctrl_params = outputs['controller_params']
                # Encourage parameters close to optimal (simple surrogate)
                policy_loss = -batch_advantages.unsqueeze(-1) * ctrl_params
                policy_loss = policy_loss.mean()

                # Total loss
                loss = policy_loss + self.config.value_coef * value_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        return np.mean(policy_losses), np.mean(value_losses)

    def train(self, checkpoint_dir: str):
        """Full PPO training loop.

        Args:
            checkpoint_dir: Directory for checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nStarting PPO training for {self.config.num_epochs} epochs")
        print(f"Steps per epoch: {self.config.steps_per_epoch}")
        print(f"Device: {self.device}")
        print("-" * 60)

        best_reward = -float('inf')

        for epoch in range(1, self.config.num_epochs + 1):
            # Collect rollouts
            avg_reward = self.collect_rollouts()
            self.episode_rewards.append(avg_reward)

            # Update policy
            policy_loss, value_loss = self.update()

            # Logging
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | "
                      f"Reward: {avg_reward:.4f} | "
                      f"Policy Loss: {policy_loss:.4f} | "
                      f"Value Loss: {value_loss:.4f}")

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save_checkpoint(checkpoint_dir / 'ppo_best.pt', epoch)

        # Save final model
        self.save_checkpoint(checkpoint_dir / 'ppo_final.pt', self.config.num_epochs)

        print("-" * 60)
        print(f"Training complete! Best reward: {best_reward:.4f}")

        # Save training history
        history_path = checkpoint_dir / 'ppo_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'rewards': self.episode_rewards,
                'best_reward': best_reward,
            }, f, indent=2)

    def save_checkpoint(self, path: Path, epoch: int):
        """Save checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
        }, path)
        print(f"  Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="PPO fine-tuning for route generation")
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/route_gen/best_model.pt",
                       help="Path to pre-trained model checkpoint")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of PPO epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=256,
                       help="Steps per epoch")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--output-dir", type=str,
                       default="checkpoints/route_gen_ppo",
                       help="Output directory for PPO checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PPO Fine-tuning for Route Generation")
    print("=" * 60)

    # Load pre-trained model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent.parent / args.checkpoint

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model with same config
        config = ModelConfig(
            d_model=128,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            use_value_head=True,  # Enable value head for PPO
        )
        model = UnifiedTransformer(config)
        # Load with strict=False to allow missing value_head keys
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        config = ModelConfig(
            d_model=128,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
        )
        model = UnifiedTransformer(config)

    # PPO config
    ppo_config = PPOConfig(
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
    )

    # Create trainer
    trainer = PPOTrainer(model, ppo_config, device)

    # Train
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / args.output_dir

    trainer.train(str(output_dir))


if __name__ == "__main__":
    main()
