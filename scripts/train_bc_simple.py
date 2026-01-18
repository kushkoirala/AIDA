#!/usr/bin/env python3
"""
Simple BC training script (no matplotlib dependency)
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


class WaypointFlightDataset(Dataset):
    def __init__(self, observations, waypoints, actions):
        combined = np.concatenate([observations, waypoints], axis=1)
        self.inputs = torch.FloatTensor(combined)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]


class WaypointPilotNN(nn.Module):
    def __init__(self, input_dim=18, action_dim=4, hidden_dim=256):
        super().__init__()
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
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        return self.actor(x)


def normalize_features(data, mean=None, std=None):
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
        std[std < 1e-8] = 1.0
    return (data - mean) / std, mean, std


def train_waypoint_bc(dataset_path, output_dir, epochs=50, batch_size=256,
                      learning_rate=3e-4, val_split=0.1, device="cuda", hidden_dim=256):
    print("\n" + "=" * 70)
    print("  TRAINING WAYPOINT-FOLLOWING NN (Behavioral Cloning)")
    print("=" * 70)

    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    print(f"\nLoading dataset: {dataset_path}")
    data = np.load(dataset_path)

    observations = data['observations']
    waypoints = data['waypoints']
    actions = data['actions']

    print(f"Raw data shapes:")
    print(f"  Observations: {observations.shape}")
    print(f"  Waypoints: {waypoints.shape}")
    print(f"  Actions: {actions.shape}")

    obs_normalized, obs_mean, obs_std = normalize_features(observations)

    # Determine waypoint dimension (6 for old format, 7 for new format with bearing_error)
    wp_dim = waypoints.shape[1]
    print(f"  Waypoint dim: {wp_dim}")

    wp_normalized = waypoints.copy()
    wp_normalized[:, :3], wp_xy_mean, wp_xy_std = normalize_features(waypoints[:, :3])
    speed_normalized, wp_spd_mean, wp_spd_std = normalize_features(waypoints[:, 3:4])
    wp_normalized[:, 3] = speed_normalized.flatten()
    dist_normalized, wp_dist_mean, wp_dist_std = normalize_features(waypoints[:, 5:6])
    wp_normalized[:, 5] = dist_normalized.flatten()

    # Normalize bearing_error if present (7D waypoint format)
    if wp_dim >= 7:
        bearing_normalized, wp_bearing_mean, wp_bearing_std = normalize_features(waypoints[:, 6:7])
        wp_normalized[:, 6] = bearing_normalized.flatten()
    else:
        wp_bearing_mean = np.array([0.0])
        wp_bearing_std = np.array([1.0])

    n_samples = len(observations)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = WaypointFlightDataset(obs_normalized[train_idx], wp_normalized[train_idx], actions[train_idx])
    val_dataset = WaypointFlightDataset(obs_normalized[val_idx], wp_normalized[val_idx], actions[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nTrain samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    input_dim = 12 + wp_dim  # 12 (state) + wp_dim (waypoint)
    action_dim = 4
    print(f"Input dim: {input_dim} (12 state + {wp_dim} waypoint)")
    model = WaypointPilotNN(input_dim, action_dim, hidden_dim).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: WaypointPilotNN ({n_params:,} params)")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_val_loss = float('inf')

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = nn.functional.mse_loss(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches

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

        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        improved = ""
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
                'wp_dim': wp_dim,
                'obs_mean': obs_mean,
                'obs_std': obs_std,
                'wp_xy_mean': wp_xy_mean,
                'wp_xy_std': wp_xy_std,
                'wp_spd_mean': wp_spd_mean,
                'wp_spd_std': wp_spd_std,
                'wp_dist_mean': wp_dist_mean,
                'wp_dist_std': wp_dist_std,
                'wp_bearing_mean': wp_bearing_mean,
                'wp_bearing_std': wp_bearing_std,
            }, checkpoint_path)
            improved = " *"

        print(f"Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}, lr={lr:.2e}{improved}")

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_dir}/waypoint_pilot_best.pt")

    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='checkpoints/waypoint_pilot')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train_waypoint_bc(
        dataset_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )


if __name__ == "__main__":
    main()
