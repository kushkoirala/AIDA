"""
Vectorized PPO training for FlightEnvRL using stable-baselines3.

Usage example (CPU, 8 envs):
    PYTHONPATH=. python scripts/train_sb3_vec.py --task takeoff_and_cruise --timesteps 600000 --n-envs 8 --device cpu

For CUDA (if available):
    PYTHONPATH=. python scripts/train_sb3_vec.py --task takeoff_and_cruise --timesteps 600000 --n-envs 16 --device cuda
"""
import argparse
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from aida_sim.env.flight_env_rl import FlightEnvRL
from scripts.scripted_policy import ScriptedPolicy
from gymnasium import Wrapper


class CurriculumCallback(BaseCallback):
    """Promote curriculum levels automatically based on success rate."""
    def __init__(self, check_freq: int = 5000, verbose: int = 1, max_level: int = 2):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.current_level = 0
        self.success_buffer = []
        self.max_level = max_level

    def _on_step(self) -> bool:
        # Collect successes from vectorized infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "is_success" in info:
                self.success_buffer.append(float(info["is_success"]))
        if len(self.success_buffer) > 500:
            self.success_buffer = self.success_buffer[-500:]

        if self.n_calls % self.check_freq == 0 and self.success_buffer:
            mean_success = float(np.mean(self.success_buffer))
            if self.verbose > 0:
                print(f"[Curriculum] Level={self.current_level} mean_success={mean_success:.2f}")
            if self.current_level == 0 and mean_success > 0.50:
                self._promote(1)
            elif self.current_level == 1 and mean_success > 0.50:
                self._promote(2)
        return True

    def _promote(self, new_level: int):
        if new_level > self.max_level:
            return
        self.current_level = new_level
        self.success_buffer = []
        if self.verbose > 0:
            print(f"*** PROMOTING CURRICULUM TO LEVEL {new_level} ***")
        # propagate to all subproc envs
        self.training_env.env_method("set_curriculum_level", new_level)


class WarmupWrapper(Wrapper):
    """Apply scripted policy actions for the first N steps of each episode."""
    def __init__(self, env, warmup_steps: int = 0):
        super().__init__(env)
        self.warmup_steps = warmup_steps
        self._step_count = 0
        self._policy = ScriptedPolicy(
            mission_alt=float(env.mission_altitude),
            cruise_speed=float(env.cruise_speed),
            runway_heading=float(env.runway_heading),
        )
        self._last_obs = None

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = super().reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        # During warmup, ignore PPO action and use scripted policy
        if self._step_count < self.warmup_steps:
            scripted_action = self._policy.act(self._last_obs, None)
            action = scripted_action
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._last_obs = obs
        return obs, reward, terminated, truncated, info


def behavior_cloning_pretrain(model: PPO, steps: int = 5000, epochs: int = 5, curriculum_level: int = 0):
    """
    Collect expert data from the scripted policy and run a supervised pretrain on the policy network.
    """
    if steps <= 0 or epochs <= 0:
        return

    # Single-env VecNormalize to match training obs scaling
    base_env = FlightEnvRL(task="takeoff_and_cruise")
    base_env.set_curriculum_level(curriculum_level)
    expert_policy = ScriptedPolicy(
        mission_alt=float(base_env.mission_altitude),
        cruise_speed=float(base_env.cruise_speed),
        runway_heading=float(base_env.runway_heading),
    )
    vec_env = VecNormalize(DummyVecEnv([lambda: base_env]), norm_obs=True, norm_reward=False, clip_obs=10.0)

    observations = []
    actions = []
    reset_out = vec_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    for _ in range(steps):
        act = expert_policy.act(obs[0])
        step_out = vec_env.step([act])
        if len(step_out) == 4:
            obs, _, dones, _ = step_out
        else:
            obs, _, dones, _, _ = step_out
        observations.append(obs.copy())
        actions.append(np.array([act], dtype=np.float32))
        if dones.any():
            reset_out = vec_env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    obs_tensor = th.tensor(np.concatenate(observations, axis=0), dtype=th.float32)
    act_tensor = th.tensor(np.concatenate(actions, axis=0), dtype=th.float32)
    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model.policy.train()
    optimizer = th.optim.Adam(model.policy.parameters(), lr=3e-4)
    print(f"[BC] Pretraining on {len(dataset)} samples for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(model.device)
            batch_act = batch_act.to(model.device)
            optimizer.zero_grad()
            dist = model.policy.get_distribution(batch_obs)
            pred = dist.mode()
            loss = F.mse_loss(pred, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[BC] Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")
    model.policy.eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="takeoff_and_cruise",
                        choices=["cruise", "takeoff", "takeoff_and_cruise"])
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SB3 checkpoint .zip to resume from")
    parser.add_argument("--save-path", type=str, default="checkpoints/ppo_sb3_vec_final.zip")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--curriculum-level", type=int, default=None,
                        help="0=hold,1=capture,2=runway (applies to envs that support it)")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Number of initial steps to apply scripted policy before handing off to PPO (applied via env wrapper)")
    parser.add_argument("--pretrain-steps", type=int, default=0,
                        help="Number of expert steps for behavior cloning before PPO (0 to disable)")
    parser.add_argument("--pretrain-epochs", type=int, default=0,
                        help="Behavior cloning epochs")
    args = parser.parse_args()

    def _make_env():
        env = FlightEnvRL(task=args.task)
        if args.curriculum_level is not None:
            env.set_curriculum_level(args.curriculum_level)
        return env

    def _wrap_env():
        e = _make_env()
        if args.warmup_steps > 0:
            # Wrap to apply scripted policy for initial steps
            e = WarmupWrapper(e, warmup_steps=args.warmup_steps)
        return e

    env = make_vec_env(
        _wrap_env,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    curriculum_cb = None
    if args.curriculum_level is None:
        curriculum_cb = CurriculumCallback(check_freq=5000, max_level=2)

    ppo_kwargs = dict(
        n_steps=512,
        batch_size=1024,
        n_epochs=8,
        gae_lambda=0.95,
        gamma=0.995,
        learning_rate=2e-4,
        clip_range=0.15,
        max_grad_norm=0.4,
        ent_coef=0.001,
        verbose=1,
        device=args.device,
        seed=args.seed,
    )

    if args.checkpoint:
        model = PPO.load(args.checkpoint, env=env, device=args.device)
        if not isinstance(model.get_env(), VecNormalize):
            model.set_env(env)
    else:
        model = PPO("MlpPolicy", env, **ppo_kwargs)

    # Optional BC pretrain using scripted expert
    if args.pretrain_steps > 0 and args.pretrain_epochs > 0:
        behavior_cloning_pretrain(model, steps=args.pretrain_steps, epochs=args.pretrain_epochs,
                                  curriculum_level=args.curriculum_level or 0)

    if curriculum_cb:
        model.learn(total_timesteps=args.timesteps, callback=curriculum_cb)
    else:
        model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)

    eval_env = make_vec_env(_make_env, n_envs=1, vec_env_cls=SubprocVecEnv)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.0)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
    print(f"Evaluation over {args.eval_episodes} eps: mean_reward={mean_reward:.2f} ± {std_reward:.2f}")


if __name__ == "__main__":
    main()
