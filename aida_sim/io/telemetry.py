import asyncio
import json
from typing import Callable, Awaitable
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
