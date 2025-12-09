import asyncio
import json
from typing import Callable, Awaitable
import numpy as np
import websockets

# Minimal telemetry broadcaster over WebSocket.
async def telemetry_server(state_fn: Callable[[], dict], host: str = "0.0.0.0", port: int = 8765):
    async def handler(websocket):
        while True:
            payload = state_fn()
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(0.02)

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
        }
        yield state
        t += interval
        await asyncio.sleep(interval)


async def run_spin_demo(host: str = "0.0.0.0", port: int = 8765, interval: float = 0.02):
    """Run a self-contained demo telemetry loop for the viewer."""
    spinner = _spin_demo_state(interval)

    def next_state():
        return spinner.asend(None)

    async def handler(websocket):
        # Warm one step to prime generator.
        await spinner.asend(None)
        while True:
            payload = await spinner.asend(None)
            await websocket.send(json.dumps(payload))

    async with websockets.serve(handler, host, port):
        await asyncio.Future()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Telemetry server for the viewer")
    parser.add_argument("--host", default="0.0.0.0", help="Listen address")
    parser.add_argument("--port", type=int, default=8765, help="Listen port")
    parser.add_argument("--mode", choices=["demo", "noop"], default="demo", help="Demo sends a spinning pose; noop uses example_state")
    args = parser.parse_args()

    if args.mode == "demo":
        asyncio.run(run_spin_demo(host=args.host, port=args.port))
    else:
        asyncio.run(telemetry_server(example_state, host=args.host, port=args.port))


if __name__ == "__main__":
    main()
