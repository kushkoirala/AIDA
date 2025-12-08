# Viewer wiring (React + Three.js)

- Load aircraft glTF in the same frame as the sim (X forward, Y right, Z down). Using `Udaan-Product4.gltf` at repo root (copy or symlink into `public/` for the viewer runtime).
- Connect to telemetry WS (default `ws://localhost:8765`), expecting JSON:
  ```json
  {
    "position": [x, y, z],
    "quaternion": [w, x, y, z],
    "velocity": [vx, vy, vz],
    "rates": [p, q, r],
    "surfaces": [elevator, aileron, rudder],
    "throttle": 0.0,
    "soc": 1.0,
    "voltage": 12.0,
    "load_factor": 1.0
  }
  ```
- Apply pose each frame; overlay HUD for attitude, load factor, SoC, voltage.
- Keep render loop independent from sim tick; throttle updates to ~30–50 Hz.
