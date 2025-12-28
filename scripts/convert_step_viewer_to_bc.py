#!/usr/bin/env python
"""
Convert StepViewer JSONL logs into (obs, action) pairs usable by bc_from_logs.py.

Assumptions:
- Log contains controls.{throttle,pitch,roll,yaw} and attitude_deg.{roll,pitch,heading}.
- We only target the takeoff task (throttle + elevator stick). Other action dims are zeroed.
- Observations are synthesized near the runway start using FlightEnvRL's normalization.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from aida_sim.dynamics.state import VehicleState
from aida_sim.env.flight_env_rl import FlightEnvRL


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll/pitch/yaw (rad) to quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)


def synth_obs(env: FlightEnvRL, attitude_deg: dict, airspeed: float = 0.0) -> np.ndarray:
    """Synthesize a normalized observation near runway start using logged attitude and airspeed."""
    pos = env.runway_start.copy()
    pos[2] = env.ground_height
    roll = np.deg2rad(attitude_deg.get("roll", 0.0))
    pitch = np.deg2rad(attitude_deg.get("pitch", 0.0))
    yaw = np.deg2rad(attitude_deg.get("heading", 0.0))
    quat = euler_to_quat(roll, pitch, yaw)
    vel_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
    vel = vel_dir * float(max(0.0, airspeed))
    vs = VehicleState(
        position=pos.astype(np.float32),
        velocity=vel,
        orientation=quat,
        body_rates=np.zeros(3, dtype=np.float32),
        surfaces=np.zeros(3, dtype=np.float32),
        soc=1.0,
        voltage=getattr(env, "battery", None).voltage if hasattr(env, "battery") else 17.0,
        load_factor=1.0,
    )
    return env._obs(vs)


def map_actions(controls: dict) -> np.ndarray:
    """
    Map StepViewer controls to the FlightEnvRL action vector:
    [throttle, trim_elev_delta, trim_rudder_delta, elevator, aileron, rudder]
    """
    throttle = float(np.clip(controls.get("throttle", 0.0), 0.0, 1.0))

    # StepViewer currently sends control surface deflections in radians.
    # Normalize them by the env surface limit (20 deg ≈ 0.349 rad).
    surf_limit = 0.3490658503988659  # rad
    raw_pitch = float(controls.get("pitch", 0.0))
    raw_roll = float(controls.get("roll", 0.0))
    raw_yaw = float(controls.get("yaw", 0.0))

    max_abs = max(abs(raw_pitch), abs(raw_roll), abs(raw_yaw))
    if max_abs <= 0.35:  # treat as radians
        pitch_cmd = raw_pitch / surf_limit
        roll_cmd = raw_roll / surf_limit
        yaw_cmd = raw_yaw / surf_limit
    else:  # already normalized sticks
        pitch_cmd = raw_pitch
        roll_cmd = raw_roll
        yaw_cmd = raw_yaw

    pitch_cmd = float(np.clip(pitch_cmd, -1.0, 1.0))
    roll_cmd = float(np.clip(roll_cmd, -1.0, 1.0))
    yaw_cmd = float(np.clip(yaw_cmd, -1.0, 1.0))
    action = np.array(
        [
            throttle * 2.0 - 1.0,  # throttle in [-1, 1]
            0.0,                   # elevator trim delta (not logged)
            0.0,                   # rudder trim delta (not logged)
            pitch_cmd,             # elevator stick
            roll_cmd,              # aileron stick
            yaw_cmd,               # rudder stick
        ],
        dtype=np.float32,
    )
    return action


def convert_file(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """Convert a single JSONL file. Returns (read_lines, written_lines)."""
    env = FlightEnvRL(task="takeoff")
    # Zero trims so observations don't bake in prior trim offsets.
    env.trim_elevator = 0.0
    env.trim_rudder = 0.0
    env.trim_memory["elevator"] = 0.0
    env.trim_memory["rudder"] = 0.0

    read_lines = 0
    written_lines = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r") as src, output_path.open("w") as dst:
        for line in src:
            read_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            controls = rec.get("controls")
            attitude = rec.get("attitude_deg")
            if not controls or not attitude:
                continue
            obs = synth_obs(env, attitude, airspeed=rec.get("telemetry", {}).get("airspeed", 0.0))
            action = map_actions(controls)
            out_rec = {
                "obs": obs.tolist(),
                "action": action.tolist(),
                "ts_ms": rec.get("ts_ms"),
                "flightMode": rec.get("flightMode"),
            }
            dst.write(json.dumps(out_rec) + "\n")
            written_lines += 1
    return read_lines, written_lines


def main():
    ap = argparse.ArgumentParser(description="Convert StepViewer logs to BC-ready JSONL (obs/action).")
    ap.add_argument("--input", required=True, type=Path, help="Input StepViewer JSONL file")
    ap.add_argument("--output", required=True, type=Path, help="Output JSONL path for BC")
    args = ap.parse_args()

    read_lines, written_lines = convert_file(args.input, args.output)
    print(f"[convert] read {read_lines} lines, wrote {written_lines} BC samples to {args.output}")


if __name__ == "__main__":
    main()
