"""
Behavior Cloning + PPO fine-tuning with vectorized envs (stable-baselines3).

1) Collect scripted trajectories using ScriptedPolicy.
2) Pretrain PPO policy via BC on the collected data.
3) Continue training with PPO.

Usage:
    PYTHONPATH=. python scripts/train_bc_ppo_vec.py --task takeoff_and_cruise --timesteps 400000 --n-envs 8 --device cpu
"""
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from aida_sim.env.flight_env_rl import FlightEnvRL
from scripts.scripted_policy import ScriptedPolicy


def collect_scripted_rollouts(env, policy, n_steps=50000):
    obs, _ = env.reset()
    observations = []
    actions = []
    dones = []
    infos = []
    for _ in range(n_steps):
        act = policy.act(obs)
        step_out = env.step(act)
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_out
        observations.append(obs)
        actions.append(act)
        dones.append(done)
        infos.append(info)
        obs = next_obs
        if done:
            obs, _ = env.reset()
    return np.array(observations), np.array(actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="takeoff_and_cruise",
                        choices=["cruise", "takeoff", "takeoff_and_cruise"])
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bc-steps", type=int, default=50_000, help="Scripted steps to collect for BC.")
    parser.add_argument("--bc-epochs", type=int, default=5, help="BC epochs on collected data.")
    parser.add_argument("--save-path", type=str, default="checkpoints/ppo_bc_ppo_vec_final.zip")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    # Single env for scripted rollouts
    env_single = Monitor(FlightEnvRL(task=args.task))
    scripted = ScriptedPolicy()
    obs_bc, act_bc = collect_scripted_rollouts(env_single, scripted, n_steps=args.bc_steps)

    # Vectorized env for PPO
    vec_env = make_vec_env(lambda: Monitor(FlightEnvRL(task=args.task)),
                           n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

    model = PPO("MlpPolicy", vec_env,
                n_steps=256,
                batch_size=512,
                n_epochs=8,
                learning_rate=2e-4,
                clip_range=0.15,
                gae_lambda=0.95,
                gamma=0.99,
                max_grad_norm=0.3,
                device=args.device,
                verbose=1,
                seed=0)

    # Behavior cloning pretrain
    model.policy.optimizer.zero_grad()
    bc_batch_size = 1024
    bc_obs = torch.tensor(obs_bc, dtype=torch.float32, device=args.device)
    bc_act = torch.tensor(act_bc, dtype=torch.float32, device=args.device)
    for _ in range(args.bc_epochs):
        idx = torch.randperm(len(bc_obs))
        for start in range(0, len(bc_obs), bc_batch_size):
            end = start + bc_batch_size
            batch_idx = idx[start:end]
            obs_batch = bc_obs[batch_idx]
            act_batch = bc_act[batch_idx]
            dist = model.policy.get_distribution(obs_batch)
            log_prob = dist.log_prob(act_batch).sum(-1)
            loss = -log_prob.mean()
            loss.backward()
            model.policy.optimizer.step()
            model.policy.optimizer.zero_grad()

    # PPO fine-tuning
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)

    eval_env = Monitor(FlightEnvRL(task=args.task))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
    print(f"Evaluation over {args.eval_episodes} eps: mean_reward={mean_reward:.2f} ± {std_reward:.2f}")


if __name__ == "__main__":
    main()
