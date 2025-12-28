#!/usr/bin/env python
"""
WebSocket logger for StepViewer/DigitalTwin telemetry.

StepViewer will push JSON payloads of controls/attitude/aero metrics to this server.
Each message is appended to a JSONL file in the specified log directory.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets


async def logger_server(host: str, port: int, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"step_viewer_{int(time.time())}.jsonl"
    log_fp = log_path.open("a", buffering=1)
    print(f"[logger] writing to {log_path}")

    async def handler(websocket):
        client = websocket.remote_address
        print(f"[logger] client connected: {client}")
        async for msg in websocket:
            try:
                parsed = json.loads(msg)
            except json.JSONDecodeError:
                continue
            parsed.setdefault("ingest_ts", time.time())
            log_fp.write(json.dumps(parsed) + "\n")
        print(f"[logger] client disconnected: {client}")

    async with websockets.serve(handler, host, port):
        print(f"[logger] listening on ws://{host}:{port}")
        await asyncio.Future()  # run forever


def main():
    ap = argparse.ArgumentParser(description="StepViewer telemetry logger (WebSocket -> JSONL).")
    ap.add_argument("--host", default="127.0.0.1", help="Listen address")
    ap.add_argument("--port", type=int, default=8787, help="Listen port for WS logging")
    ap.add_argument("--log-dir", type=Path, default=Path("logs/step_viewer"), help="Directory to write JSONL logs")
    args = ap.parse_args()

    try:
        asyncio.run(logger_server(args.host, args.port, args.log_dir))
    except KeyboardInterrupt:
        print("[logger] stopping...")


if __name__ == "__main__":
    main()
