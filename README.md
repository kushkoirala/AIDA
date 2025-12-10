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

### Aerodynamic calibration

Use `scripts/calibrate_aero.py` to fit lift and pitching-moment coefficients
from the trim tables (airspeed vs. α_trim / δ_e). The script prints best-fit
values for `CL0`, `CL_alpha`, `Cm0`, `Cm_alpha`, and assumes a fixed
`Cm_de = -1.0 / rad`. Update `aida_sim/dynamics/forces.py` with the reported
numbers when new trim data is available.

### Propulsion model

`thrust_force` now models the 8.5″ prop pair with the conceptual-design
assumptions (240 W available, prop efficiency 0.4, disk actuator theory). The
resulting thrust automatically transitions from static to forward-flight
conditions and is limited by available shaft power. Battery defaults in
`aida_sim/systems/battery.py` match the 16‑cell, 2 Ah pack described in the
propulsion study.

## Next steps
1) Lock frames (world/body) and observation schema; finish dynamics integration.
2) Add action→surface mapping, reward, and termination logic inside `flight_env.py`.
3) Hook Bullet contact points for gear and friction; expose geofence.
4) Spin up WS telemetry bridge; point viewer to it.
5) Provide STEP file for collision/render conversion; expect collision mesh (Bullet) + glTF (viewer) in consistent frames. Assets present: `Udaan-Product4.stl` (collision), `Udaan-Product4.gltf` (render). Run `python scripts/prepare_assets.py` to copy into runtime locations.

## RL Flight Control & Neural Net Trainer

### New Features (Dec 2025)
- **RL Environment:** `aida_sim/env/flight_env_rl.py`
  - 16-dim normalized observation space
  - Dense reward shaping for stable PPO training
  - Randomized spawn for exploration
  - Flight envelope constraints: V_stall=11 m/s, V_max=25 m/s, throttle 25-65%
- **PPO Trainer:** `scripts/train_ppo_flight.py` (`--task cruise|takeoff`)
  - Actor-Critic neural network (137k params)
  - Monte Carlo rollouts with GAE advantage estimation
  - Apple Silicon MPS GPU acceleration
  - Checkpointing and evaluation
- **Telemetry Bridge:** `scripts/run_sim_with_telemetry.py`
  - WebSocket server for real-time state streaming

### Flight Envelope Analysis
- Stall speed: ~11 m/s
- Max safe speed: ~25 m/s (to stay under 3G)
- Safe throttle range: 25-65% for level flight
- G-limits by speed/alpha:
  - 15 m/s @ 10° alpha → 1.62G
  - 20 m/s @ 5° alpha → 0.96G
  - 25 m/s @ 10° alpha → 3.0G

### RL Training Workflow
```bash
cd /Users/kka/AIDA
export PYTHONPATH=$PYTHONPATH:$(pwd)
python scripts/train_ppo_flight.py --timesteps 500000 --device mps
# Takeoff-specific curriculum
python scripts/train_ppo_flight.py --task takeoff --timesteps 300000 --device mps
```

### Example PPO Results
- 327 FPS training on Apple Silicon GPU
- Mean reward improves with training
- Agent learns to stay airborne and approach target zone

## Local build/test workflow
- One-shot setup & sync: `./scripts/dev_local.sh`
	- Creates/uses `.venv`, installs `requirements.txt`, runs `scripts/prepare_assets.py`.
- Rollout stub: `python scripts/rollout.py`
- PPO stub: `python scripts/train_ppo.py`
- Telemetry WS (placeholder): `python -m aida_sim.io.telemetry` (wire `state_fn` later)
- Viewer loads `viewer/public/Udaan-Product4.gltf` (already copied by prep script)
