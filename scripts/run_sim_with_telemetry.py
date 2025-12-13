
import argparse
import asyncio
import threading
import time
import numpy as np
import torch
import gymnasium as gym
import aida_sim.env  # Register env so Gym IDs are available

from aida_sim.io.telemetry import telemetry_server
from aida_sim.env.flight_env_rl import FlightEnvRL
from scripts.train_ppo_flight import ActorCritic
from stable_baselines3 import PPO as SB3PPO
from queue import Queue, Empty
from aida_sim.dynamics.integrator import _quat_to_rotmat
from aida_sim.dynamics.forces import _calc_alpha_beta

# Shared state for telemetry
TELEMETRY_UNITS = "metric"  # or "imperial"

shared_state = {
    "position": [0, 0, 0],
    "quaternion": [1, 0, 0, 0],
    "velocity": [0, 0, 0],
    "rates": [0, 0, 0],
    "surfaces": [0, 0, 0],
    "throttle": 0.0,
    "soc": 1.0,
    "voltage": 12.0,
    "load_factor": 1.0,
    "alpha_deg": 0.0,
    "beta_deg": 0.0,
    "units": TELEMETRY_UNITS,
}


def _normalize_throttle(action_value: float) -> float:
    """Map PPO throttle action (-1..1) onto 0..1 percent for telemetry."""
    throttle_cmd = 0.5 * (np.clip(action_value, -1.0, 1.0) + 1.0)
    return float(np.clip(throttle_cmd, 0.0, 1.0))


def _load_policy(env: FlightEnvRL, checkpoint: str, device: torch.device):
    if checkpoint.endswith(".zip"):
        # SB3 checkpoint
        model = SB3PPO.load(checkpoint, device=device)
        expected_obs_dim = env.observation_space.shape[0]
        print(f"[telemetry] Loaded SB3 checkpoint {checkpoint}")
        return model, expected_obs_dim
    # Legacy torch ActorCritic checkpoint
    ckpt = torch.load(checkpoint, map_location=device)
    # Infer expected obs dim from checkpoint weights
    expected_obs_dim = ckpt["policy_state_dict"]["shared.0.weight"].shape[1]
    policy = ActorCritic(
        expected_obs_dim,
        env.action_space.shape[0],
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    print(f"[telemetry] Loaded policy checkpoint {checkpoint} (expects obs dim {expected_obs_dim})")
    return policy, expected_obs_dim


def physics_loop(task: str, checkpoint: str, device: str,
                 client_event: threading.Event,
                 control_queue: Queue,
                 wait_for_client: bool,
                 reset_on_connect: bool,
                 curriculum_level: int):
    """Runs the simulation physics in a separate thread and updates telemetry."""
    env = FlightEnvRL(task=task)
    env.set_curriculum_level(curriculum_level)
    obs, info = env.reset()
    torch_device = torch.device(device)
    policy = None
    expected_obs_dim = env.observation_space.shape[0]
    if checkpoint:
        policy, expected_obs_dim = _load_policy(env, checkpoint, torch_device)

    print(f"[telemetry] Physics loop started for task={task}, checkpoint={checkpoint}.")

    step_counter = 0
    while True:
        if wait_for_client and not client_event.is_set():
            time.sleep(0.05)
            continue

        # Handle control messages
        try:
            while True:
                msg = control_queue.get_nowait()
                if msg == "reset":
                    obs, info = env.reset()
                    step_counter = 0
                elif msg == "disconnect":
                    if wait_for_client:
                        client_event.clear()
                control_queue.task_done()
        except Empty:
            pass
        if policy is not None:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(torch_device)
            if obs_tensor.shape[1] != expected_obs_dim:
                # Slice or pad to match the checkpoint's expected obs dim
                if obs_tensor.shape[1] > expected_obs_dim:
                    obs_tensor = obs_tensor[:, :expected_obs_dim]
                else:
                    pad = expected_obs_dim - obs_tensor.shape[1]
                    obs_tensor = torch.nn.functional.pad(obs_tensor, (0, pad), "constant", 0.0)
            if isinstance(policy, SB3PPO):
                action, _ = policy.predict(obs, deterministic=True)
            else:
                with torch.no_grad():
                    action_tensor, _, _ = policy.get_action(obs_tensor, deterministic=True)
                action = action_tensor.cpu().numpy().squeeze().astype(np.float32)
        else:
            # fallback to idle throttle if no policy provided
            action = np.zeros(env.action_space.shape, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        sim_state = env.state

        shared_state["position"] = sim_state.position.tolist()
        shared_state["quaternion"] = sim_state.orientation.tolist()
        shared_state["velocity"] = sim_state.velocity.tolist()
        shared_state["rates"] = sim_state.body_rates.tolist()
        shared_state["surfaces"] = sim_state.surfaces.tolist()
        shared_state["throttle"] = _normalize_throttle(action[0])
        shared_state["soc"] = float(sim_state.soc)
        shared_state["voltage"] = float(sim_state.voltage)
        shared_state["load_factor"] = float(sim_state.load_factor)
        # Match flight dynamics sign convention for AoA/Beta
        R_bw = _quat_to_rotmat(sim_state.orientation)
        vel_body = R_bw.T @ sim_state.velocity
        alpha, beta, airspeed = _calc_alpha_beta(vel_body)
        # In our sim frame, positive body-Z is up; flip alpha to report conventional nose-up positive
        shared_state["alpha_deg"] = -np.degrees(alpha)
        shared_state["beta_deg"] = np.degrees(beta)

        if step_counter % 200 == 0:
            print(
                f"[telemetry] step={step_counter} pos={shared_state['position']} "
                f"vel={shared_state['velocity']} thr={shared_state['throttle']*100:.1f}%"
            )

        if terminated or truncated:
            reason = info.get("termination_reason", "timeout" if truncated else "unknown")
            print(f"[telemetry] Episode ended: {reason}. Resetting...")
            obs, info = env.reset()
            step_counter = 0
        else:
            step_counter += 1

        time.sleep(env.dt)


def get_state():
    return shared_state


async def main():
    parser = argparse.ArgumentParser(description="Run PPO evaluation and stream telemetry.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PPO checkpoint (.pt)")
    parser.add_argument("--task", type=str, default="takeoff_and_cruise", choices=["takeoff", "cruise", "takeoff_and_cruise"], help="FlightEnvRL task")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for policy evaluation")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Telemetry server host")
    parser.add_argument("--port", type=int, default=8765, help="Telemetry server port")
    parser.add_argument("--interval", type=float, default=0.02, help="Telemetry publish period")
    parser.add_argument("--wait-for-client", action="store_true", help="Hold physics until a websocket client connects")
    parser.add_argument("--reset-on-connect", action="store_true", help="Reset the simulation whenever a new client connects")
    parser.add_argument("--curriculum-level", type=int, default=2, choices=[0, 1, 2], help="Curriculum level to set before running")
    args = parser.parse_args()

    client_event = threading.Event()
    control_queue: Queue[str] = Queue()

    physics_thread = threading.Thread(
        target=physics_loop,
        args=(args.task, args.checkpoint, args.device, client_event, control_queue,
              args.wait_for_client, args.reset_on_connect, args.curriculum_level),
        daemon=True
    )
    physics_thread.start()

    def on_connect():
        client_event.set()
        if args.reset_on_connect:
            control_queue.put("reset")

    def on_disconnect():
        control_queue.put("disconnect")

    print(f"[telemetry] Starting websocket server at ws://{args.host}:{args.port}")
    await telemetry_server(
        get_state,
        host=args.host,
        port=args.port,
        interval=args.interval,
        on_client_connect=on_connect,
        on_client_disconnect=on_disconnect,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping...")
