"""
Behavior Cloning Policy Training

Train a neural network policy to imitate expert flight demonstrations
from the GPU-generated dataset. This pre-trained policy can then be used
to warm-start PPO training.

Author: Kushal Koirala
Date: December 2024

Usage:
    python train_bc_policy.py --dataset checkpoints/bc_dataset_udaan.npz --epochs 20
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the same ActorCritic architecture used in PPO
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_ppo_flight import ActorCritic


class FlightDataset(Dataset):
    """Dataset for behavior cloning from expert trajectories"""

    def __init__(self, observations, actions):
        """
        Args:
            observations: (N, obs_dim) numpy array
            actions: (N, action_dim) numpy array
        """
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)

        print(f"Dataset loaded: {len(self.observations)} samples")
        print(f"  Observation shape: {self.observations.shape}")
        print(f"  Action shape: {self.actions.shape}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def normalize_actions(actions, old_range=(-1, 1), new_range=(-1, 1)):
    """
    Normalize actions if needed. The BC dataset uses raw control values,
    but PPO policy outputs actions in [-1, 1].

    Args:
        actions: (N, action_dim) array
        old_range: Original action range
        new_range: Target action range

    Returns:
        Normalized actions
    """
    # If already in correct range, return as-is
    return actions


def evaluate_policy(policy, dataloader, device):
    """
    Evaluate BC policy on validation set.

    Args:
        policy: ActorCritic network
        dataloader: Validation DataLoader
        device: torch.device

    Returns:
        Dictionary with evaluation metrics
    """
    policy.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for obs_batch, action_batch in dataloader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)

            # Get policy prediction (deterministic)
            action_pred, _, _ = policy.get_action(obs_batch, deterministic=True)

            # MSE loss
            loss = nn.functional.mse_loss(action_pred, action_batch)

            total_loss += loss.item() * len(obs_batch)
            total_samples += len(obs_batch)

    mean_loss = total_loss / total_samples
    return {"mse_loss": mean_loss}


def train_bc_policy(
    dataset_path,
    output_path,
    epochs=20,
    batch_size=256,
    learning_rate=1e-3,
    val_split=0.1,
    device="cpu",
    hidden_dim=256,
):
    """
    Train behavior cloning policy.

    Args:
        dataset_path: Path to .npz dataset file
        output_path: Path to save trained policy
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Adam learning rate
        val_split: Fraction of data for validation
        device: Device to train on ('cpu', 'cuda', 'mps')
        hidden_dim: Hidden layer dimension
    """
    print("\n" + "="*60)
    print("  BEHAVIOR CLONING POLICY TRAINING")
    print("="*60)

    # Set device
    if device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\nDevice: {device}")

    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    data = np.load(dataset_path)

    observations = data['observations']
    actions = data['actions']

    print(f"Dataset shape:")
    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")

    # Normalize actions to [-1, 1] if needed
    # BC dataset actions are already in correct format from classical controller

    # Split into train/val
    n_samples = len(observations)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = FlightDataset(observations[train_indices], actions[train_indices])
    val_dataset = FlightDataset(observations[val_indices], actions[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create policy network (same architecture as PPO)
    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]

    policy = ActorCritic(obs_dim, action_dim, hidden_dim=hidden_dim).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\nPolicy architecture:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Total parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, 
    )

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    # Training loop
    for epoch in range(1, epochs + 1):
        policy.train()
        epoch_loss = 0.0
        n_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for obs_batch, action_batch in progress_bar:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)

            # Forward pass (deterministic action prediction)
            action_pred, _, _ = policy.get_action(obs_batch, deterministic=True)

            # MSE loss between predicted and expert actions
            loss = nn.functional.mse_loss(action_pred, action_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

            optimizer.step()

            # Track loss
            epoch_loss += loss.item()
            n_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})

        # Average training loss
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation
        val_metrics = evaluate_policy(policy, val_loader, device)
        val_loss = val_metrics['mse_loss']
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save checkpoint
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'hidden_dim': hidden_dim,
            }, output_path)

            print(f"  ✓ New best model saved (val loss: {val_loss:.6f})")

    print("\n" + "="*60)
    print("  Training Complete!")
    print("="*60)
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Model saved to: {output_path}")

    # Plot training curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, epochs + 1), train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(range(1, epochs + 1), val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Behavior Cloning Training Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plot_path = output_path.replace('.pt', '_training_curve.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curve saved to: {plot_path}")

    return policy, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Train BC policy from expert demonstrations")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--output", type=str, default="checkpoints/bc_policy.pt", help="Output path for trained policy")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")

    args = parser.parse_args()

    # Verify dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
        print("\nGenerate dataset first:")
        print(f"  python scripts/generate_bc_dataset_gpu.py --output {args.dataset}")
        return

    # Train BC policy
    policy, train_losses, val_losses = train_bc_policy(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        device=args.device,
        hidden_dim=args.hidden_dim,
    )

    print("\n" + "="*60)
    print("  Next Steps:")
    print("="*60)
    print("  Use BC policy to warm-start PPO training:")
    print(f"    python scripts/train_ppo_flight.py \\")
    print(f"      --policy-init {args.output} \\")
    print(f"      --task takeoff \\")
    print(f"      --timesteps 500000 \\")
    print(f"      --device {args.device}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
