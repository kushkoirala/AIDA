"""
Generate an expert dataset (obs, actions) using the scripted policy for BC warm-start.

Usage:
    PYTHONPATH=. python scripts/generate_expert_bc.py --episodes 100 --task takeoff_and_cruise --curriculum-level 2 --out checkpoints/expert_bc_takeoff_to_cruise.npz
"""
import argparse
import numpy as np

from aida_sim.env.flight_env_rl import FlightEnvRL
from scripts.scripted_policy import ScriptedPolicy


def collect(episodes: int, task: str, curriculum_level: int | None, out_path: str):
    env = FlightEnvRL(task=task)
    if curriculum_level is not None:
        env.set_curriculum_level(curriculum_level)
    policy = ScriptedPolicy(mission_alt=float(env.mission_altitude), cruise_speed=float(env.cruise_speed), runway_heading=float(env.runway_heading))

    obs_buf = []
    act_buf = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            act = policy.act(obs, info)
            obs_buf.append(obs.copy())
            act_buf.append(act.copy())
            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
    np.savez(out_path, observations=np.array(obs_buf, dtype=np.float32), actions=np.array(act_buf, dtype=np.float32))
    print(f"Saved expert dataset to {out_path} with {len(obs_buf)} steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--task", type=str, default="takeoff_and_cruise")
    parser.add_argument("--curriculum-level", type=int, default=2, help="0=hold,1=capture,2=runway")
    parser.add_argument("--out", type=str, default="checkpoints/expert_bc_takeoff_to_cruise.npz")
    args = parser.parse_args()
    collect(args.episodes, args.task, args.curriculum_level, args.out)


if __name__ == "__main__":
    main()
