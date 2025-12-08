# AIDA
Autonomous Intelligent Decision Architecture — integration of LLMs and neural networks for autonomy.

## Minimal fixed-wing sim plan (scaffolding added)
- Python core: NumPy/SciPy for forces + integration; Bullet/pybullet for ground/contact.
- RL: Gymnasium API with SB3 (PPO) harness stubs in `scripts/`.
- Safety: 4 g clamp, 40 A fuse clamp, geofence (parking-lot box), boundary termination.
- Battery: simple SoC + internal-resistance sag model for thrust scaling.
- Viewer: React/Three (not yet implemented here) consuming telemetry via WebSocket; glTF render mesh.

## Layout
- `aida_sim/env/flight_env.py` — Gymnasium shell, actions/obs, hooks to physics.
- `aida_sim/dynamics/` — state, forces, integrator placeholders.
- `aida_sim/platform/` — Bullet world and asset resolution placeholders.
- `aida_sim/systems/battery.py` — SoC + Rint model.
- `aida_sim/safety/guards.py` — clamps and limit checks.
- `aida_sim/io/telemetry.py` — WebSocket telemetry stub.
- `scripts/` — training and rollout stubs.
- `viewer/` — notes for wiring glTF + HUD via WS.

## Next steps
1) Lock frames (world/body) and observation schema; finish dynamics integration.
2) Add action→surface mapping, reward, and termination logic inside `flight_env.py`.
3) Hook Bullet contact points for gear and friction; expose geofence.
4) Spin up WS telemetry bridge; point viewer to it.
5) Provide STEP file for collision/render conversion; expect collision mesh (Bullet) + glTF (viewer) in consistent frames. Assets present: `Udaan-Product4.stl` (collision), `Udaan-Product4.gltf` (render). Run `python scripts/prepare_assets.py` to copy into runtime locations.

## Local build/test workflow
- One-shot setup & sync: `./scripts/dev_local.sh`
	- Creates/uses `.venv`, installs `requirements.txt`, runs `scripts/prepare_assets.py`.
- Rollout stub: `python scripts/rollout.py`
- PPO stub: `python scripts/train_ppo.py`
- Telemetry WS (placeholder): `python -m aida_sim.io.telemetry` (wire `state_fn` later)
- Viewer loads `viewer/public/Udaan-Product4.gltf` (already copied by prep script)
