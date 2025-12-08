"""Quick smoke test: env rollout + STL load check.

Usage (recommended conda env):
  conda run -n aida python scripts/smoke_test.py

What it does:
- Imports and instantiates the Gym env (AIDA-Flight-v0) and runs 200 random steps.
- Tries to load the STL mesh via pybullet as a collision shape to confirm it parses.
"""

import os
import sys

# Ensure project root is on sys.path when run from scripts/ or elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gymnasium as gym
import aida_sim.env  # registers env id


def test_env_rollout(steps: int = 200):
    env = gym.make("AIDA-Flight-v0")
    obs, info = env.reset()
    for _ in range(steps):
        obs, rew, done, trunc, info = env.step(env.action_space.sample())
        if done or trunc:
            break
    print("[ok] env rollout completed")


def test_stl_load(mesh_path: str):
    try:
        import pybullet as p
    except Exception as e:  # pragma: no cover - optional dep
        print(f"[warn] pybullet not available: {e}")
        return

    if not os.path.exists(mesh_path):
        print(f"[warn] mesh not found: {mesh_path}")
        return

    cid = None
    try:
        p.connect(p.DIRECT)
        cid = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[1, 1, 1])
        if cid < 0:
            print(f"[fail] pybullet could not create collision shape from {mesh_path}")
        else:
            print(f"[ok] pybullet loaded mesh as collision shape (id={cid})")
    except Exception as e:  # pragma: no cover
        print(f"[fail] pybullet mesh load error: {e}")
    finally:
        try:
            p.disconnect()
        except Exception:
            pass


def main():
    test_env_rollout()
    # Prefer collision mesh if present, else fallback to the placeholder STL.
    mesh_candidates = [
        "aida_sim/assets/Udaan_collision.obj",
        "aida_sim/assets/Udaan_collision.stl",
        "aida_sim/assets/Udaan-Product4.stl",
    ]
    for m in mesh_candidates:
        if os.path.exists(m):
            test_stl_load(m)
            break
    else:
        print("[warn] no mesh found to test in aida_sim/assets/")


if __name__ == "__main__":
    sys.exit(main())
