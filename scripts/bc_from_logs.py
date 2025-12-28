#!/usr/bin/env python
"""
Behavior cloning from telemetry logs (JSONL) to warm-start PPO.

Each JSONL line from run_sim_with_telemetry.py --log-dir should contain:
- "obs": observation array (env._obs output)
- "action": continuous action array in [-1,1]
Other fields are ignored for BC.
"""
import argparse
import glob
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.train_ppo_flight import ActorCritic


def load_dataset(paths):
    obs_list, act_list = [], []
    for path in paths:
        with open(path, "r") as fp:
            for line in fp:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "obs" not in rec or "action" not in rec:
                    continue
                obs = np.asarray(rec["obs"], dtype=np.float32)
                act = np.asarray(rec["action"], dtype=np.float32)
                if obs.ndim != 1 or act.ndim != 1:
                    continue
                obs_list.append(obs)
                act_list.append(act)
    if not obs_list:
        raise RuntimeError("No (obs, action) pairs found in provided logs.")
    return np.stack(obs_list), np.stack(act_list)


def train_bc(obs, actions, hidden_dim=256, epochs=10, batch_size=256, device="cpu", lr=1e-3):
    device = torch.device(device)
    obs_dim = obs.shape[1]
    act_dim = actions.shape[1]
    policy = ActorCritic(obs_dim, act_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()

    dataset = list(range(len(obs)))
    for epoch in range(1, epochs + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        for start in range(0, len(dataset), batch_size):
            idx = dataset[start : start + batch_size]
            obs_batch = torch.FloatTensor(obs[idx]).to(device)
            act_batch = torch.FloatTensor(actions[idx]).to(device)
            pred, _, _ = policy.get_action(obs_batch, deterministic=True)
            loss = mse(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        avg_loss = total_loss / len(dataset)
        print(f"[bc] epoch {epoch}/{epochs} loss={avg_loss:.6f}")
    return policy


def main():
    parser = argparse.ArgumentParser(description="Behavior clone from telemetry logs.")
    parser.add_argument("--logs", type=str, required=True, help="Glob for JSONL log files (e.g., logs/*.jsonl)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", type=str, default="checkpoints/bc_init.pt")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.logs))
    if not paths:
        raise SystemExit(f"No log files matched pattern: {args.logs}")
    print(f"[bc] loading {len(paths)} log file(s)")
    obs, actions = load_dataset(paths)
    print(f"[bc] dataset size: {len(obs)} samples, obs_dim={obs.shape[1]}, act_dim={actions.shape[1]}")
    policy = train_bc(obs, actions, epochs=args.epochs, batch_size=args.batch_size, device=args.device, lr=args.lr)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({"policy_state_dict": policy.state_dict()}, args.save)
    print(f"[bc] saved policy weights to {args.save}")


if __name__ == "__main__":
    main()
