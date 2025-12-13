# Unreal Viewer Plan (Telemetry-Driven)

This is a minimal plan to replace the current web viewer with an Unreal-based one while reusing the existing telemetry stream on `ws://localhost:8765`.

## Telemetry contract
- Transport: WebSocket JSON (20 ms period by default).
- Message shape (all numeric are floats unless noted):
  ```json
  {
    "position": [x, y, z],            // meters, X fwd, Y right, Z up
    "quaternion": [w, x, y, z],       // right-handed, matches sim frame
    "velocity": [vx, vy, vz],         // m/s, world frame
    "rates": [p, q, r],               // rad/s, body frame
    "surfaces": [elev, ail, rud],     // radians
    "throttle": 0.0..1.0,
    "soc": 0.0..1.0,
    "voltage": volts,
    "load_factor": g,
    "heartbeat": int,                 // monotonically increasing
    "sim_time": seconds,
    "mode": "server" | "demo"         // optional
  }
  ```
- Server lives in `aida_sim/io/telemetry.py` and is started by `scripts/run_sim_with_telemetry.py`.
  - Run it locally: `PYTHONPATH=. python scripts/run_sim_with_telemetry.py --task takeoff_and_cruise --checkpoint <ppo_checkpoint>`
  - Or use the built-in demo spinner: `PYTHONPATH=. python -m aida_sim.io.telemetry --mode demo`

## Unreal integration (minimal)
1) Create an Unreal C++ Actor `ATelemetryReceiver` (or a plugin) that:
   - On BeginPlay, connects to `ws://127.0.0.1:8765` using the WebSockets plugin (`FWebSocketsModule`).
   - Registers an `OnMessage` handler; parse JSON into a struct matching the schema above.
   - Stores latest state in a thread-safe buffer (FCriticalSection or atomics).
2) Create a Pawn/Actor for the aircraft:
   - Mesh imported from existing GLTF/FBX; set X forward, Y right, Z up.
   - Control surfaces as child components (bones or separate meshes) for elevator/aileron/rudder; animate by setting relative rotation from `surfaces` (convert rad→deg).
   - Tick: read latest telemetry, set actor location/rotation from `position` and `quaternion`, apply surface angles, throttle-driven prop blur if desired.
3) Camera:
   - Add a chase camera spring arm; optionally a free-fly camera.
4) HUD:
   - Lightweight UMG widget showing basic fields (altitude, speed, AoA if you compute it, heartbeat).
5) Debug first with the demo feed to avoid sim dependence; then point to the live server.

## Asset prep
- Export the current aircraft + runway as FBX/GLTF from your DCC tool.
- Align axes to X-forward, Y-right, Z-up before import.
- Keep the origin at the aircraft CG for stable placement.

## Build/run checklist
- Enable WebSockets plugin in the UE project.
- Add command-line/selectable host/port (default 127.0.0.1:8765).
- Handle reconnects (if the sim restarts) and ignore malformed frames.

## Stretch goals (optional)
- Replay/record: write incoming telemetry to a JSONL file and play back offline.
- Multiple views: tower cam, onboard cam, orbit cam with hotkeys.
- Effects: simple nav/strobe lights, prop blur based on throttle.
