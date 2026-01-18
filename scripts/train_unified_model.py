#!/usr/bin/env python3
"""
Training Script for AIDA Unified Transformer

Supports multiple training modes:
1. NLP Command Parsing - Train on text -> command pairs
2. Behavioral Cloning - Train on expert demonstrations
3. Trajectory Prediction - Train on state -> action sequences
4. PPO Fine-tuning - RL fine-tuning after BC pretraining

Usage:
    # Train NLP command parser
    python train_unified_model.py --mode nlp --epochs 50

    # Train behavioral cloning from expert data
    python train_unified_model.py --mode bc --data expert_demos.npz --epochs 100

    # Train trajectory predictor
    python train_unified_model.py --mode trajectory --data flights.npz --epochs 100

    # PPO fine-tuning (after BC pretraining)
    python train_unified_model.py --mode ppo --checkpoint bc_model.pt --epochs 1000

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from aida_sim.models import (
    UnifiedTransformer,
    ModelConfig,
    TaskType,
    SimpleTokenizer,
    CommandLoss,
    TrajectoryLoss,
    ImitationLoss,
)


# ============================================================================
# Datasets
# ============================================================================

class NLPCommandDataset(Dataset):
    """Dataset for NLP command parsing."""

    COMMAND_TYPES = [
        'SET_HEADING', 'SET_ALTITUDE', 'SET_AIRSPEED', 'SET_VS',
        'TURN', 'DIRECT_TO', 'SET_FLAPS', 'HOLD_CURRENT', 'GO_AROUND', 'UNKNOWN'
    ]

    def __init__(self, data_path: Optional[str] = None, tokenizer: SimpleTokenizer = None):
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.samples = []

        if data_path and Path(data_path).exists():
            self._load_data(data_path)
        else:
            # Generate synthetic training data
            self._generate_synthetic_data()

    def _load_data(self, path: str):
        """Load training data from JSON file."""
        with open(path) as f:
            data = json.load(f)
        for item in data:
            self.samples.append({
                'text': item['text'],
                'command_type': self.COMMAND_TYPES.index(item['command']),
                'heading': item.get('heading', 0.0),
                'altitude': item.get('altitude', 0.0),
                'airspeed': item.get('airspeed', 0.0),
            })

    def _generate_synthetic_data(self, n_samples: int = 5000):
        """Generate synthetic training data from templates."""
        import random

        templates = {
            'SET_HEADING': [
                "turn heading {heading}",
                "heading {heading}",
                "fly heading {heading}",
                "steer {heading}",
                "turn to {heading} degrees",
            ],
            'SET_ALTITUDE': [
                "climb to {altitude} feet",
                "descend to {altitude}",
                "altitude {altitude}",
                "maintain {altitude} feet",
                "go to {altitude} ft",
            ],
            'SET_AIRSPEED': [
                "speed {airspeed} knots",
                "airspeed {airspeed}",
                "slow to {airspeed} kts",
                "accelerate to {airspeed}",
            ],
            'TURN': [
                "turn left {turn} degrees",
                "turn right {turn} degrees",
                "{turn} degrees left",
                "{turn} degrees right",
            ],
            'SET_VS': [
                "climb at {vs} fpm",
                "descend at {vs} feet per minute",
                "vertical speed {vs}",
            ],
            'GO_AROUND': [
                "go around",
                "missed approach",
                "execute go around",
            ],
            'HOLD_CURRENT': [
                "hold current",
                "maintain",
                "keep current heading",
                "hold altitude",
            ],
        }

        for _ in range(n_samples):
            cmd_type = random.choice(list(templates.keys()))
            template = random.choice(templates[cmd_type])

            heading = random.randint(0, 359)
            altitude = random.choice([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
            airspeed = random.randint(60, 130)
            turn = random.randint(10, 90)
            vs = random.randint(300, 1000)

            text = template.format(
                heading=heading,
                altitude=altitude,
                airspeed=airspeed,
                turn=turn,
                vs=vs,
            )

            self.samples.append({
                'text': text,
                'command_type': self.COMMAND_TYPES.index(cmd_type),
                'heading': float(heading) if 'heading' in template else 0.0,
                'altitude': float(altitude) if 'altitude' in template else 0.0,
                'airspeed': float(airspeed) if 'airspeed' in template else 0.0,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = self.tokenizer.encode(sample['text'])

        return {
            'tokens': tokens,
            'command_type': torch.tensor(sample['command_type'], dtype=torch.long),
            'heading': torch.tensor(sample['heading'], dtype=torch.float32),
            'altitude': torch.tensor(sample['altitude'], dtype=torch.float32),
            'airspeed': torch.tensor(sample['airspeed'], dtype=torch.float32),
        }


class ExpertDemoDataset(Dataset):
    """Dataset for behavioral cloning from expert demonstrations."""

    def __init__(self, data_path: str):
        """
        Load expert demonstration data.

        Expected format: .npz with:
        - states: [N, state_dim] or [N, seq_len, state_dim]
        - actions: [N, action_dim] or [N, seq_len, action_dim]
        """
        data = np.load(data_path)
        self.states = torch.from_numpy(data['states']).float()
        self.actions = torch.from_numpy(data['actions']).float()

        # Ensure 2D for BC (single timestep)
        if self.states.dim() == 3:
            # Flatten sequences
            N, T, D = self.states.shape
            self.states = self.states.reshape(N * T, D)
            self.actions = self.actions.reshape(N * T, -1)

        print(f"Loaded {len(self.states)} expert demonstration samples")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
        }


class TrajectoryDataset(Dataset):
    """Dataset for trajectory sequence prediction."""

    def __init__(self, data_path: str, seq_len: int = 64):
        """
        Load trajectory data.

        Expected format: .npz with:
        - states: [N, T, state_dim] flight state sequences
        - actions: [N, T, action_dim] corresponding actions
        """
        data = np.load(data_path)
        self.states = torch.from_numpy(data['states']).float()
        self.actions = torch.from_numpy(data['actions']).float()
        self.seq_len = seq_len

        # Truncate or pad to seq_len
        if self.states.size(1) > seq_len:
            self.states = self.states[:, :seq_len, :]
            self.actions = self.actions[:, :seq_len, :]

        print(f"Loaded {len(self.states)} trajectory sequences")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'states': self.states[idx],
            'actions': self.actions[idx],
        }


# ============================================================================
# Training Functions
# ============================================================================

def train_nlp(
    model: UnifiedTransformer,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: CommandLoss,
    device: torch.device,
    epoch: int,
) -> float:
    """Train one epoch of NLP command parsing."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        tokens = batch['tokens'].to(device)
        targets = {
            'command_type': batch['command_type'].to(device),
            'heading': batch['heading'].to(device),
            'altitude': batch['altitude'].to(device),
        }

        optimizer.zero_grad()
        outputs = model(TaskType.NLP_COMMAND, text_tokens=tokens)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)


def train_bc(
    model: UnifiedTransformer,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: ImitationLoss,
    device: torch.device,
    epoch: int,
) -> float:
    """Train one epoch of behavioral cloning."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        states = batch['state'].to(device)
        expert_actions = batch['action'].to(device)

        optimizer.zero_grad()
        outputs = model(TaskType.IMITATION, states=states)
        loss = criterion(outputs['actions'], expert_actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)


def train_trajectory(
    model: UnifiedTransformer,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: TrajectoryLoss,
    device: torch.device,
    epoch: int,
) -> float:
    """Train one epoch of trajectory prediction."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        states = batch['states'].to(device)
        target_actions = batch['actions'].to(device)

        optimizer.zero_grad()
        outputs = model(TaskType.TRAJECTORY, states=states, target_actions=target_actions)
        loss = criterion(outputs['actions'], target_actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)


def evaluate_nlp(
    model: UnifiedTransformer,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate NLP command parsing accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['command_type'].to(device)

            outputs = model(TaskType.NLP_COMMAND, text_tokens=tokens)
            preds = outputs['command_logits'].argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train AIDA Unified Transformer")
    parser.add_argument("--mode", type=str, choices=['nlp', 'bc', 'trajectory', 'ppo'],
                        default='nlp', help="Training mode")
    parser.add_argument("--data", type=str, default=None, help="Path to training data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model config
    config = ModelConfig(
        d_model=args.d_model,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers // 2,
    )

    # Create model
    model = UnifiedTransformer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training mode specific setup
    if args.mode == 'nlp':
        print("\n=== Training NLP Command Parser ===")
        dataset = NLPCommandDataset(args.data)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        criterion = CommandLoss(config)

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            train_loss = train_nlp(model, train_loader, optimizer, criterion, device, epoch)
            accuracy = evaluate_nlp(model, val_loader, device)
            scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Val Accuracy: {accuracy:.2%}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                ckpt_path = output_dir / f"nlp_epoch_{epoch + 1}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

    elif args.mode == 'bc':
        print("\n=== Training Behavioral Cloning ===")
        if not args.data:
            print("ERROR: --data required for BC training")
            return

        dataset = ExpertDemoDataset(args.data)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        criterion = ImitationLoss()

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            train_loss = train_bc(model, train_loader, optimizer, criterion, device, epoch)
            scheduler.step()

            print(f"Train Loss: {train_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                ckpt_path = output_dir / f"bc_epoch_{epoch + 1}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

    elif args.mode == 'trajectory':
        print("\n=== Training Trajectory Predictor ===")
        if not args.data:
            print("ERROR: --data required for trajectory training")
            return

        dataset = TrajectoryDataset(args.data, seq_len=config.trajectory_seq_len)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        criterion = TrajectoryLoss()

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            train_loss = train_trajectory(model, train_loader, optimizer, criterion, device, epoch)
            scheduler.step()

            print(f"Train Loss: {train_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                ckpt_path = output_dir / f"trajectory_epoch_{epoch + 1}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

    elif args.mode == 'ppo':
        print("\n=== PPO Fine-tuning ===")
        print("PPO training requires environment interaction.")
        print("See aida_sim/training/ppo_trainer.py for full implementation.")
        # PPO would require the flight simulator environment
        # This is a placeholder - full PPO implementation would be separate

    # Save final model
    final_path = output_dir / f"{args.mode}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
