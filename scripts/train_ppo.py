"""Stub for PPO training harness using Stable Baselines3.
Fill in after env wiring.
"""

import gymnasium as gym
import aida_sim.env  # ensures env is registered with Gymnasium
from stable_baselines3 import PPO


def main():
    env = gym.make("AIDA-Flight-v0")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)


if __name__ == "__main__":
    main()
