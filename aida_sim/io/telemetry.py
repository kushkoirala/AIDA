import asyncio
import json
import time
from typing import Callable, Awaitable

import numpy as np
import websockets

# Minimal telemetry broadcaster over WebSocket. Adds heartbeat and sim_time if missing.
async def telemetry_server(
    state_fn: Callable[[], dict],
    host: str = "0.0.0.0",
    port: int = 8765,
    interval: float = 0.02,
):
    async def handler(websocket):
        heartbeat = 0
        start = time.monotonic()
        while True:
            payload = dict(state_fn())
            heartbeat += 1
            payload.setdefault("heartbeat", heartbeat)
            payload.setdefault("sim_time", time.monotonic() - start)
            payload.setdefault("mode", "server")
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(interval)

    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # run forever


# Example state_fn for wiring later.
def example_state():
    return {
        "position": [0, 0, 0],
        "quaternion": [1, 0, 0, 0],
        "velocity": [0, 0, 0],
        "rates": [0, 0, 0],
        "surfaces": [0, 0, 0],
        "throttle": 0.0,
        "soc": 1.0,
        "voltage": 12.0,
        "load_factor": 1.0,
    }


async def _spin_demo_state(interval: float = 0.02):
    """Generator-like coroutine that yields a slowly spinning pose."""
    t = 0.0
    heartbeat = 0
    while True:
        # Small circle in X/Y, gentle descent/ascent in Z with a slow yaw spin.
        pos = [0.5 * np.cos(t), 0.5 * np.sin(t), -0.2 * np.sin(0.3 * t)]
        # Yaw spin converted to quaternion (Three.js expects w,x,y,z order).
        yaw = 0.3 * t
        q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        state = {
            "position": pos,
            "quaternion": q,
            "velocity": [0.0, 0.0, 0.0],
            "rates": [0.0, 0.0, 0.3],
            "surfaces": [0.0, 0.0, 0.0],
            "throttle": 0.0,
            "soc": 1.0,
            "voltage": 12.0,
            "load_factor": 1.0,
            "sim_time": t,
            "heartbeat": heartbeat,
            "mode": "demo",
        }
        yield state
        heartbeat += 1
        t += interval
        await asyncio.sleep(interval)


async def run_spin_demo(host: str = "0.0.0.0", port: int = 8765, interval: float = 0.02):
    """Run a self-contained demo telemetry loop for the viewer."""
    async def handler(websocket):
        t = 0.0
        heartbeat = 0
        while True:
            pos = [0.5 * np.cos(t), 0.5 * np.sin(t), -0.2 * np.sin(0.3 * t)]
            yaw = 0.3 * t
            q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
            payload = {
                "position": pos,
                "quaternion": q,
                "velocity": [0.0, 0.0, 0.0],
                "rates": [0.0, 0.0, 0.3],
                "surfaces": [0.0, 0.0, 0.0],
                "throttle": 0.0,
                "soc": 1.0,
                "voltage": 12.0,
                "load_factor": 1.0,
                "sim_time": t,
                "heartbeat": heartbeat,
                "mode": "demo",
            }
            await websocket.send(json.dumps(payload))
            heartbeat += 1
            t += interval
            await asyncio.sleep(interval)

    async with websockets.serve(handler, host, port):
        await asyncio.Future()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Telemetry server for the viewer")
    parser.add_argument("--host", default="0.0.0.0", help="Listen address")
    parser.add_argument("--port", type=int, default=8765, help="Listen port")
    parser.add_argument("--mode", choices=["demo", "noop"], default="demo", help="Demo sends a spinning pose; noop uses example_state")
    parser.add_argument("--interval", type=float, default=0.02, help="Telemetry update period in seconds")
    args = parser.parse_args()

    if args.mode == "demo":
        asyncio.run(run_spin_demo(host=args.host, port=args.port, interval=args.interval))
    else:
        asyncio.run(telemetry_server(example_state, host=args.host, port=args.port, interval=args.interval))


if __name__ == "__main__":
    main()
