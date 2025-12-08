"""Stub for deterministic rollout / manual control testing."""

import gymnasium as gym
import aida_sim.env  # ensures env is registered with Gymnasium


def main():
    env = gym.make("AIDA-Flight-v0")
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if truncated:
            break


if __name__ == "__main__":
    main()
