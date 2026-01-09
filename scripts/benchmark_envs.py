#!/usr/bin/env python3
"""
Benchmark different numbers of parallel environments to find optimal GPU utilization.
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from aida_sim.env.flight_env_cessna172 import Cessna172Env


def make_env(task='ground_roll', rank=0):
    def _init():
        env = Cessna172Env(task=task)
        env = Monitor(env)
        return env
    return _init


def benchmark_envs(n_envs_list, timesteps=10000):
    """Benchmark different numbers of environments."""
    results = []

    for n_envs in n_envs_list:
        print(f"\n{'='*50}")
        print(f"  Testing {n_envs} environments")
        print(f"{'='*50}")

        # Create environments
        if n_envs > 1:
            env = SubprocVecEnv([make_env(rank=i) for i in range(n_envs)])
        else:
            env = DummyVecEnv([make_env()])

        # Create model
        model = PPO(
            'MlpPolicy',
            env,
            n_steps=512,
            batch_size=128,
            verbose=0,
            device='cuda',
        )

        # Warmup
        model.learn(total_timesteps=2048)

        # Benchmark
        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        elapsed = time.time() - start_time

        fps = timesteps / elapsed
        results.append((n_envs, fps, elapsed))

        print(f"  Timesteps: {timesteps:,}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  FPS: {fps:.1f}")

        env.close()
        del model

    return results


if __name__ == "__main__":
    print("GPU Environment Benchmark")
    print("="*50)

    # Test different env counts
    n_envs_list = [1, 2, 4, 8, 16]

    results = benchmark_envs(n_envs_list, timesteps=20000)

    print("\n" + "="*50)
    print("  RESULTS SUMMARY")
    print("="*50)
    print(f"{'Envs':<8} {'FPS':<12} {'Time (s)':<12} {'Speedup':<10}")
    print("-"*50)

    base_fps = results[0][1]
    for n_envs, fps, elapsed in results:
        speedup = fps / base_fps
        print(f"{n_envs:<8} {fps:<12.1f} {elapsed:<12.2f} {speedup:<10.2f}x")

    # Find best
    best = max(results, key=lambda x: x[1])
    print(f"\nOptimal: {best[0]} environments ({best[1]:.1f} FPS)")
