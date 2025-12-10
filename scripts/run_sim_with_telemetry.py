
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
from queue import Queue, Empty

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
    "units": TELEMETRY_UNITS,
}


def _normalize_throttle(action_value: float) -> float:
    """Map PPO throttle action (-1..1) onto 0..1 percent for telemetry."""
    throttle_cmd = 0.5 * (np.clip(action_value, -1.0, 1.0) + 1.0)
    return float(np.clip(throttle_cmd, 0.0, 1.0))


def _load_policy(env: FlightEnvRL, checkpoint: str, device: torch.device):
    policy = ActorCritic(
        env.observation_space.shape[0],
        env.action_space.shape[0],
    )
    ckpt = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    print(f"[telemetry] Loaded policy checkpoint {checkpoint}")
    return policy


def physics_loop(task: str, checkpoint: str, device: str,
                 client_event: threading.Event,
                 control_queue: Queue,
                 wait_for_client: bool,
                 reset_on_connect: bool):
    """Runs the simulation physics in a separate thread and updates telemetry."""
    env = FlightEnvRL(task=task)
    obs, info = env.reset()
    torch_device = torch.device(device)
    policy = None
    if checkpoint:
        policy = _load_policy(env, checkpoint, torch_device)

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
    parser.add_argument("--task", type=str, default="takeoff", choices=["takeoff", "cruise"], help="FlightEnvRL task")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for policy evaluation")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Telemetry server host")
    parser.add_argument("--port", type=int, default=8765, help="Telemetry server port")
    parser.add_argument("--interval", type=float, default=0.02, help="Telemetry publish period")
    parser.add_argument("--wait-for-client", action="store_true", help="Hold physics until a websocket client connects")
    parser.add_argument("--reset-on-connect", action="store_true", help="Reset the simulation whenever a new client connects")
    args = parser.parse_args()

    client_event = threading.Event()
    control_queue: Queue[str] = Queue()

    physics_thread = threading.Thread(
        target=physics_loop,
        args=(args.task, args.checkpoint, args.device, client_event, control_queue,
              args.wait_for_client, args.reset_on_connect),
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
