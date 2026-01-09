#!/usr/bin/env python3
import argparse, asyncio, threading, time, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from stable_baselines3 import PPO
from aida_sim.env.flight_env_cessna172 import Cessna172Env
from aida_sim.io.telemetry import telemetry_server

shared_state = {
    "position": [0, 0, 0], "quaternion": [1, 0, 0, 0], "velocity": [0, 0, 0],
    "rates": [0, 0, 0], "surfaces": [0, 0, 0], "throttle": 0.0,
    "soc": 1.0, "voltage": 12.0, "load_factor": 1.0,
    "units": "metric", "model": "cessna172", "phase": "ground_roll",
}

def euler_to_quat(roll, pitch, yaw):
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    return [cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy]

def update_telemetry(obs, action, info, phase):
    x, y, z, u, v, w = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
    phi, theta, psi = obs[6], obs[7], obs[8]
    shared_state["position"] = [float(x + 500), float(y), max(0.0, float(-z))]
    shared_state["quaternion"] = [float(q) for q in euler_to_quat(phi, theta, psi)]
    shared_state["velocity"] = [float(u), float(v), float(w)]
    shared_state["rates"] = [float(obs[9]), float(obs[10]), float(obs[11])]
    shared_state["throttle"] = float(np.clip((action[0] + 1) / 2, 0, 1))
    shared_state["surfaces"] = [float(np.clip(action[1], -1, 1)), float(np.clip(action[2], -1, 1)), float(np.clip(action[3], -1, 1))]
    shared_state["roll_deg"] = float(np.rad2deg(phi))
    shared_state["pitch_deg"] = float(np.rad2deg(theta))
    shared_state["heading_deg"] = float(np.rad2deg(psi)) % 360.0
    shared_state["airspeed_kts"] = float(math.sqrt(u**2 + v**2 + w**2) * 1.94384)
    shared_state["altitude_ft"] = max(0.0, float(-z * 3.28084))
    shared_state["phase"] = phase

def get_phase(obs, info):
    # Get state from info (more reliable) or obs
    alt = info.get("altitude", -obs[2])  # meters AGL
    spd = info.get("airspeed", np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2))  # m/s
    
    # Phase transitions based on altitude and airspeed
    # V_ROTATE = 28 m/s (55 KIAS), V_CLIMB = 35 m/s (68 KIAS)
    
    # Ground Roll: On ground, accelerating to rotation speed
    if alt < 3.0:
        if spd < 28.0:
            return "ground_roll"  # Still accelerating
        else:
            return "rotation"  # At rotation speed, time to rotate
    
    # After liftoff (alt >= 3m)
    if alt < 152.4:  # Below 500 ft
        return "initial_climb"
    else:
        return "full_climb"

def run_sim(models, dt=0.02, max_steps=3000, http_port=8000):
    sep = "=" * 60
    print(sep)
    print("  CESSNA 172 CONTINUOUS TAKEOFF")
    print(sep)
    print("Loaded phases:", list(models.keys()))
    env = Cessna172Env(task="ground_roll", cruise_altitude_ft=3000.0, dt=dt, max_episode_steps=max_steps)
    print("Open browser to http://localhost:" + str(http_port) + " to view telemetry")
    try:
        ep = 0
        while True:
            ep += 1
            print("--- TAKEOFF", ep, "---")
            obs, info = env.reset()
            done, steps, phase = False, 0, "ground_roll"
            while not done and steps < max_steps:
                new_phase = get_phase(obs, info)
                if new_phase != phase and new_phase in models:
                    print(">>> " + phase + " -> " + new_phase + " at Step", steps)
                    phase = new_phase
                action, _ = models.get(phase, models["ground_roll"]).predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                update_telemetry(obs, action, info, phase)
                if steps % 100 == 0:
                    print("  [" + phase + "] Step", steps, ": Speed=", round(info["airspeed"]*1.944,1), "KIAS, Alt=", round(info["altitude"]*3.281), "ft")
                steps += 1
                time.sleep(dt)
            print("Completed:", steps, "steps,", info.get("termination_reason", "max_steps"))
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("Stopped")
    env.close()

async def main_async(models, dt, max_steps, ws_port, http_port):
    server = asyncio.create_task(telemetry_server(lambda: shared_state, host="0.0.0.0", port=ws_port))
    await asyncio.sleep(1)
    threading.Thread(target=run_sim, args=(models, dt, max_steps, http_port), daemon=True).start()
    try: await server
    except KeyboardInterrupt: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--http-port", type=int, default=8000)
    args = parser.parse_args()
    base = Path("/home/AIDA/checkpoints/cessna172_curriculum")
    models = {}
    for p, d in [("ground_roll", "phase1_ground_roll"), ("rotation", "phase2_rotation"), 
                 ("initial_climb", "phase3_initial_climb"), ("full_climb", "phase4_full_climb")]:
        path = base / d / "best_model.zip"
        if path.exists():
            models[p] = PPO.load(str(path))
            print("Loaded", p)
    if not models:
        print("No models found!")
        return 1
    asyncio.run(main_async(models, args.dt, args.max_steps, args.ws_port, args.http_port))
    return 0

if __name__ == "__main__":
    sys.exit(main())