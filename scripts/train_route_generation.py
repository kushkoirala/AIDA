#!/usr/bin/env python3
"""
Train Unified Transformer on Route Generation Task

This script trains the UnifiedTransformer's RouteGenerationHead to predict
waypoints and controller parameters given route specifications.

Usage:
    python train_route_generation.py --data route_training_data.npz --epochs 100

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from aida_sim.models.unified_transformer import (
    UnifiedTransformer, ModelConfig, TaskType, RouteGenerationLoss
)


class RouteDataset(Dataset):
    """PyTorch Dataset for route generation training data."""

    def __init__(self, npz_path: str):
        """Load dataset from npz file.

        Args:
            npz_path: Path to the .npz file created by generate_route_training_data.py
        """
        data = np.load(npz_path)
        self.route_specs = torch.from_numpy(data['route_specs']).float()
        self.waypoints = torch.from_numpy(data['waypoints']).float()
        self.validity = torch.from_numpy(data['validity']).float()
        self.phases = torch.from_numpy(data['phases']).long()
        self.controller_params = torch.from_numpy(data['controller_params']).float()

        print(f"Loaded {len(self)} routes from {npz_path}")

    def __len__(self):
        return len(self.route_specs)

    def __getitem__(self, idx):
        return {
            'route_spec': self.route_specs[idx],
            'waypoints': self.waypoints[idx],
            'validity': self.validity[idx],
            'phase': self.phases[idx],
            'controller_params': self.controller_params[idx],
        }


class RouteGenerationTrainer:
    """Trainer for route generation task."""

    def __init__(self, model: UnifiedTransformer, device: torch.device,
                 lr: float = 1e-4, weight_decay: float = 1e-5):
        """Initialize trainer.

        Args:
            model: UnifiedTransformer model
            device: Device to train on (cpu/cuda)
            lr: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.loss_fn = RouteGenerationLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary of average losses
        """
        self.model.train()
        total_losses = {
            'loss': 0.0,
            'waypoint_loss': 0.0,
            'validity_loss': 0.0,
            'phase_loss': 0.0,
            'controller_loss': 0.0,
        }
        num_batches = 0

        for batch in dataloader:
            # Move to device
            route_spec = batch['route_spec'].to(self.device)
            targets = {
                'waypoints': batch['waypoints'].to(self.device),
                'validity': batch['validity'].to(self.device),
                'phase': batch['phase'].to(self.device),
                'controller_params': batch['controller_params'].to(self.device),
            }

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(TaskType.ROUTE_GENERATION, route_spec=route_spec)

            # Compute loss
            losses = self.loss_fn(outputs, targets)
            loss = losses['loss']

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Validate on held-out data.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of average losses
        """
        self.model.eval()
        total_losses = {
            'loss': 0.0,
            'waypoint_loss': 0.0,
            'validity_loss': 0.0,
            'phase_loss': 0.0,
            'controller_loss': 0.0,
        }
        num_batches = 0

        for batch in dataloader:
            route_spec = batch['route_spec'].to(self.device)
            targets = {
                'waypoints': batch['waypoints'].to(self.device),
                'validity': batch['validity'].to(self.device),
                'phase': batch['phase'].to(self.device),
                'controller_params': batch['controller_params'].to(self.device),
            }

            outputs = self.model(TaskType.ROUTE_GENERATION, route_spec=route_spec)
            losses = self.loss_fn(outputs, targets)

            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, checkpoint_dir: str, log_interval: int = 10):
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            log_interval: How often to log progress
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nStarting training for {epochs} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Device: {self.device}")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            # Train
            train_losses = self.train_epoch(train_loader)
            self.train_losses.append(train_losses['loss'])

            # Validate
            val_losses = self.validate(val_loader)
            self.val_losses.append(val_losses['loss'])

            # Update scheduler
            self.scheduler.step()

            # Log progress
            if epoch % log_interval == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} | "
                      f"Train Loss: {train_losses['loss']:.4f} | "
                      f"Val Loss: {val_losses['loss']:.4f} | "
                      f"LR: {lr:.2e}")
                print(f"         | "
                      f"WP: {train_losses['waypoint_loss']:.4f} | "
                      f"Val: {train_losses['validity_loss']:.4f} | "
                      f"Phase: {train_losses['phase_loss']:.4f} | "
                      f"Ctrl: {train_losses['controller_loss']:.4f}")

            # Save best model
            if val_losses['loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['loss']
                self.save_checkpoint(
                    checkpoint_dir / 'best_model.pt',
                    epoch, val_losses
                )

            # Periodic checkpoint
            if epoch % 50 == 0:
                self.save_checkpoint(
                    checkpoint_dir / f'checkpoint_epoch{epoch}.pt',
                    epoch, val_losses
                )

        # Save final model
        self.save_checkpoint(
            checkpoint_dir / 'final_model.pt',
            epochs, val_losses
        )

        print("-" * 60)
        print(f"Training complete! Best val loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, path: Path, epoch: int, val_losses: dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_losses': val_losses,
            'train_losses': self.train_losses,
            'val_losses_history': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }, path)
        print(f"  Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train route generation transformer")
    parser.add_argument("--data", type=str, default="route_training_data.npz",
                       help="Path to training data")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--d-model", type=int, default=256,
                       help="Transformer dimension")
    parser.add_argument("--nhead", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4,
                       help="Number of transformer layers")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/route_gen",
                       help="Directory for checkpoints")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Route Generation Transformer Training")
    print("=" * 60)

    # Load data
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / args.data

    dataset = RouteDataset(str(data_path))

    # Split into train/val
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Create model
    config = ModelConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        use_value_head=False,  # Not needed for supervised learning
    )
    model = UnifiedTransformer(config)

    print(f"\nModel configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  nhead: {config.nhead}")
    print(f"  layers: {config.num_encoder_layers}")
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = RouteGenerationTrainer(
        model, device,
        lr=args.lr,
        weight_decay=1e-5
    )

    # Train
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path(__file__).parent.parent / args.checkpoint_dir

    trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        checkpoint_dir=str(checkpoint_dir),
        log_interval=10
    )

    # Save training config
    config_path = checkpoint_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'val_split': args.val_split,
            'seed': args.seed,
            'device': str(device),
            'model_params': sum(p.numel() for p in model.parameters()),
            'trained_at': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nSaved training config to {config_path}")


if __name__ == "__main__":
    main()
