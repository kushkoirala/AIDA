"""
PPO (Proximal Policy Optimization) Trainer for AIDA Flight Control

Uses Monte Carlo rollouts to estimate advantages and trains a neural network
policy to fly the aircraft within the flight envelope.

Based on: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from collections import deque
import time
import os

# Import our RL environment
from aida_sim.env.flight_env_rl import FlightEnvRL


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    Actor outputs mean and std for continuous actions.
    Critic estimates state value.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )
        
        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    
    def forward(self, obs):
        features = self.shared(obs)
        return features
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        features = self.forward(obs)
        action_mean = self.actor_mean(features)
        
        if deterministic:
            return action_mean, None, None
        
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, dist.entropy().sum(dim=-1)
    
    def get_value(self, obs):
        """Estimate state value."""
        features = self.forward(obs)
        return self.critic(features)
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions for PPO update."""
        features = self.forward(obs)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(features)
        
        return log_probs, entropy, values


class RolloutBuffer:
    """
    Buffer to store Monte Carlo rollout data for PPO training.
    """
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, obs, action, reward, done, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute Monte Carlo returns and GAE advantages.
        """
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])
        
        # GAE (Generalized Advantage Estimation)
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        
        return returns, advantages
    
    def get_batches(self, batch_size, returns, advantages, device):
        """Generate random minibatches for training."""
        n_samples = len(self.observations)
        indices = np.random.permutation(n_samples)
        
        obs = torch.FloatTensor(np.array(self.observations)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                obs[batch_indices],
                actions[batch_indices],
                old_log_probs[batch_indices],
                returns_t[batch_indices],
                advantages_t[batch_indices],
            )


class PPOTrainer:
    """
    PPO Trainer for flight control.
    """
    def __init__(
        self,
        env_fn,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        n_steps=2048,
        device="cpu",
    ):
        self.env = env_fn()
        self.device = torch.device(device)
        
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.policy = ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        
        self.buffer = RolloutBuffer()
        
        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def collect_rollouts(self):
        """
        Collect Monte Carlo rollouts using current policy.
        """
        self.buffer.clear()
        obs, _ = self.env.reset()
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, _ = self.policy.get_action(obs_tensor)
                value = self.policy.get_value(obs_tensor)
            
            action_np = action.cpu().numpy().squeeze()
            log_prob_np = log_prob.cpu().numpy().item()
            value_np = value.cpu().numpy().item()
            
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            self.buffer.add(obs, action_np, reward, done, log_prob_np, value_np)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
        
        # Get last value for GAE computation
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            last_value = self.policy.get_value(obs_tensor).cpu().numpy().item()
        
        return last_value
    
    def update(self):
        """
        Perform PPO update on collected rollouts.
        """
        # Get last value and compute returns/advantages
        last_value = self.buffer.values[-1] if self.buffer.values else 0
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(
                self.batch_size, returns, advantages, self.device
            ):
                obs, actions, old_log_probs, batch_returns, batch_advantages = batch
                
                # Evaluate current policy on batch
                log_probs, entropy, values = self.policy.evaluate_actions(obs, actions)
                values = values.squeeze()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": -np.mean(entropy_losses),
        }
    
    def train(self, total_timesteps, log_interval=10, save_interval=50):
        """
        Main training loop.
        """
        n_iterations = total_timesteps // self.n_steps
        
        print(f"Starting PPO Training")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Rollout steps: {self.n_steps}")
        print(f"  Iterations: {n_iterations}")
        print(f"  Device: {self.device}")
        print()
        
        start_time = time.time()
        timesteps = 0
        
        for iteration in range(1, n_iterations + 1):
            # Collect rollouts
            self.collect_rollouts()
            timesteps += self.n_steps
            
            # Update policy
            metrics = self.update()
            
            # Logging
            if iteration % log_interval == 0:
                elapsed = time.time() - start_time
                fps = timesteps / elapsed
                
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                
                print(f"Iteration {iteration}/{n_iterations}")
                print(f"  Timesteps: {timesteps}, FPS: {fps:.0f}")
                print(f"  Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.0f}")
                print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {metrics['value_loss']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
                print()
            
            # Save checkpoint
            if iteration % save_interval == 0:
                self.save(f"checkpoints/ppo_flight_{iteration}.pt")
        
        print(f"Training complete! Total time: {time.time() - start_time:.1f}s")
        return self.policy
    
    def save(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"  Saved checkpoint: {path}")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint: {path}")
    
    def evaluate(self, n_episodes=10, render=False):
        """
        Evaluate trained policy.
        """
        rewards = []
        lengths = []
        termination_reasons = {}
        
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _ = self.policy.get_action(obs_tensor, deterministic=True)
                action_np = action.cpu().numpy().squeeze()
                
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    reason = info.get("termination_reason", "timeout")
                    termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
                    break
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
        
        print(f"\nEvaluation Results ({n_episodes} episodes):")
        print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Mean Length: {np.mean(lengths):.0f} ± {np.std(lengths):.0f}")
        print(f"  Termination Reasons: {termination_reasons}")
        
        return rewards, lengths


def main():
    """Run PPO training."""
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO flight controller")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--task", type=str, default="cruise", choices=["cruise", "takeoff"],
                        help="Training task type")
    args = parser.parse_args()
    
    # Create trainer
    trainer = PPOTrainer(
        env_fn=lambda: FlightEnvRL(task=args.task),
        hidden_dim=256,
        lr=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        device=args.device,
    )
    
    if args.checkpoint:
        trainer.load(args.checkpoint)
    
    if args.eval_only:
        trainer.evaluate(n_episodes=20)
    else:
        trainer.train(total_timesteps=args.timesteps, log_interval=5, save_interval=25)
        trainer.evaluate(n_episodes=20)
        trainer.save("checkpoints/ppo_flight_final.pt")


if __name__ == "__main__":
    main()
