"""
Behavior Cloning pretrain for FlightEnvRL using a scripted expert dataset.

Usage:
    PYTHONPATH=. python scripts/pretrain_bc.py --data checkpoints/expert_bc_takeoff_to_cruise.npz --task takeoff_and_cruise --curriculum-level 2 --out checkpoints/ppo_bc_init.zip
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from aida_sim.env.flight_env_rl import FlightEnvRL


def make_env(task: str, curriculum_level: int | None):
    def _fn():
        env = FlightEnvRL(task=task)
        if curriculum_level is not None:
            env.set_curriculum_level(curriculum_level)
        return env
    return _fn


def bc_pretrain(data_path: str, task: str, curriculum_level: int | None, out_path: str, epochs: int = 5, batch_size: int = 256, lr: float = 3e-4):
    data = np.load(data_path)
    obs = torch.as_tensor(data["observations"], dtype=torch.float32)
    acts = torch.as_tensor(data["actions"], dtype=torch.float32)
    ds = TensorDataset(obs, acts)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    env = DummyVecEnv([make_env(task, curriculum_level)])
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=lr, n_steps=128, batch_size=128, ent_coef=0.0, vf_coef=0.0)
    optim = model.policy.optimizer

    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for b_obs, b_act in loader:
            optim.zero_grad()
            dist = model.policy.get_distribution(b_obs)
            pred_mean = dist.distribution.mean
            loss = torch.nn.functional.mse_loss(pred_mean, b_act)
            loss.backward()
            optim.step()
            total_loss += loss.item() * b_obs.size(0)
            total += b_obs.size(0)
        print(f"[BC] epoch {epoch+1}/{epochs} loss={total_loss/total:.6f}")

    model.save(out_path)
    print(f"Saved BC-initialized PPO policy to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to npz with observations/actions")
    parser.add_argument("--task", type=str, default="takeoff_and_cruise")
    parser.add_argument("--curriculum-level", type=int, default=2)
    parser.add_argument("--out", type=str, default="checkpoints/ppo_bc_init.zip")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    bc_pretrain(args.data, args.task, args.curriculum_level, args.out, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
