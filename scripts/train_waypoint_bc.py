#!/usr/bin/env python3
"""
Train Waypoint-Following NN via Behavioral Cloning

This trains the NN to follow waypoint commands using expert demonstrations.
The NN learns: Given (aircraft_state, waypoint_command) -> controls

Input (18D):
    - Aircraft state (12D): [x, y, z, u, v, w, phi, theta, psi, p, q, r]
    - Waypoint command (6D): [target_x, target_y, target_alt, target_speed, wp_type, dist_to_wp]

Output (4D):
    - Controls: [throttle, aileron, elevator, rudder]

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


class WaypointFlightDataset(Dataset):
    """Dataset for waypoint-following behavioral cloning."""

    def __init__(self, observations, waypoints, actions):
        """
        Args:
            observations: (N, 12) aircraft states
            waypoints: (N, 6) waypoint commands
            actions: (N, 4) expert control outputs
        """
        # Combine observations and waypoints into single input
        combined = np.concatenate([observations, waypoints], axis=1)
        self.inputs = torch.FloatTensor(combined)
        self.actions = torch.FloatTensor(actions)

        print(f"Dataset loaded: {len(self.inputs)} samples")
        print(f"  Input shape: {self.inputs.shape} (state + waypoint)")
        print(f"  Action shape: {self.actions.shape}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]


class WaypointPilotNN(nn.Module):
    """
    Neural network that maps (aircraft_state, waypoint) -> controls.

    This is the "pilot" - it knows HOW to fly to a given waypoint,
    but doesn't decide WHERE to go (that's the mission planner's job).
    """

    def __init__(self, input_dim=18, action_dim=4, hidden_dim=256):
        super().__init__()

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Critic network (for future PPO fine-tuning)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Log std for stochastic policy (for PPO)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

        # Final layer smaller init for stable outputs
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        """Forward pass for BC training."""
        return self.actor(x)

    def get_action(self, x, deterministic=True):
        """
        Get action (compatible with PPO interface).

        Returns:
            action: (batch, action_dim)
            log_prob: (batch,)
            value: (batch,)
        """
        mean = self.actor(x)
        value = self.critic(x).squeeze(-1)

        if deterministic:
            return mean, torch.zeros_like(value), value
        else:
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, value


def normalize_features(data, mean=None, std=None):
    """Normalize features to zero mean and unit variance."""
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
        std[std < 1e-8] = 1.0  # Prevent division by zero
    return (data - mean) / std, mean, std


def train_waypoint_bc(
    dataset_path,
    output_dir,
    epochs=50,
    batch_size=256,
    learning_rate=3e-4,
    val_split=0.1,
    device="cuda",
    hidden_dim=256,
):
    """
    Train waypoint-following NN via behavioral cloning.
    """
    print("\n" + "=" * 70)
    print("  TRAINING WAYPOINT-FOLLOWING NN (Behavioral Cloning)")
    print("=" * 70)

    # Setup device
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    data = np.load(dataset_path)

    observations = data['observations']  # (N, 12)
    waypoints = data['waypoints']  # (N, 6)
    actions = data['actions']  # (N, 4)

    print(f"Raw data shapes:")
    print(f"  Observations: {observations.shape}")
    print(f"  Waypoints: {waypoints.shape}")
    print(f"  Actions: {actions.shape}")

    # Normalize observations (not waypoints - they need to stay interpretable)
    obs_normalized, obs_mean, obs_std = normalize_features(observations)

    # For waypoints, normalize distance and coordinates but keep type as-is
    wp_normalized = waypoints.copy()
    # Normalize x, y, alt (first 3 columns)
    wp_normalized[:, :3], wp_xy_mean, wp_xy_std = normalize_features(waypoints[:, :3])
    # Normalize speed (column 3)
    speed_normalized, wp_spd_mean, wp_spd_std = normalize_features(waypoints[:, 3:4])
    wp_normalized[:, 3] = speed_normalized.flatten()
    # Keep waypoint type (4) as-is (it's categorical)
    # Normalize distance (column 5)
    dist_normalized, wp_dist_mean, wp_dist_std = normalize_features(waypoints[:, 5:6])
    wp_normalized[:, 5] = dist_normalized.flatten()

    print(f"\nNormalization stats:")
    print(f"  Obs mean: [{obs_mean[:3].round(2)}...]")
    print(f"  Obs std: [{obs_std[:3].round(2)}...]")

    # Split data
    n_samples = len(observations)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = WaypointFlightDataset(
        obs_normalized[train_idx],
        wp_normalized[train_idx],
        actions[train_idx],
    )
    val_dataset = WaypointFlightDataset(
        obs_normalized[val_idx],
        wp_normalized[val_idx],
        actions[val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nTrain samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    # Create model
    input_dim = 18  # 12 (state) + 6 (waypoint)
    action_dim = 4
    model = WaypointPilotNN(input_dim, action_dim, hidden_dim).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: WaypointPilotNN")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Output dim: {action_dim}")
    print(f"  Parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"\nTraining for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print()

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward
            predictions = model(inputs)
            loss = nn.functional.mse_loss(predictions, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(inputs)
                loss = nn.functional.mse_loss(predictions, targets)
                val_loss += loss.item()
                n_val_batches += 1
        val_loss /= n_val_batches
        val_losses.append(val_loss)

        scheduler.step()

        # Log
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={lr:.2e}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / f"waypoint_pilot_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'input_dim': input_dim,
                'action_dim': action_dim,
                'hidden_dim': hidden_dim,
                # Normalization stats for inference
                'obs_mean': obs_mean,
                'obs_std': obs_std,
                'wp_xy_mean': wp_xy_mean,
                'wp_xy_std': wp_xy_std,
                'wp_spd_mean': wp_spd_mean,
                'wp_spd_std': wp_spd_std,
                'wp_dist_mean': wp_dist_mean,
                'wp_dist_std': wp_dist_std,
            }, checkpoint_path)
            print(f"  -> New best model saved (val_loss={val_loss:.6f})")

    # Save final
    final_path = output_dir / f"waypoint_pilot_{timestamp}.pt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_losses[-1],
        'input_dim': input_dim,
        'action_dim': action_dim,
        'hidden_dim': hidden_dim,
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'wp_xy_mean': wp_xy_mean,
        'wp_xy_std': wp_xy_std,
        'wp_spd_mean': wp_spd_mean,
        'wp_spd_std': wp_spd_std,
        'wp_dist_mean': wp_dist_mean,
        'wp_dist_std': wp_dist_std,
    }, final_path)

    print("\n" + "=" * 70)
    print("  Training Complete!")
    print("=" * 70)
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Best model: {checkpoint_path}")
    print(f"  Final model: {final_path}")

    # Plot training curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, epochs + 1), train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(range(1, epochs + 1), val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Waypoint Pilot BC Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plot_path = output_dir / f"waypoint_pilot_training_{timestamp}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining plot: {plot_path}")

    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train waypoint-following NN')
    parser.add_argument('--dataset', type=str, required=True, help='Path to waypoint demos .npz')
    parser.add_argument('--output-dir', type=str, default='checkpoints/waypoint_pilot',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'])

    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Error: Dataset not found at {args.dataset}")
        print("\nGenerate demos first:")
        print("  python scripts/generate_waypoint_demos.py --episodes 50")
        return

    train_waypoint_bc(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )


if __name__ == "__main__":
    main()
