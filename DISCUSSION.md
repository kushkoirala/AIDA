# Flight Control & Viewer Notes

This document captures the major concepts we applied while wiring up the AIDA training stack so you can reproduce or extend the work on another machine.

## Coordinate Frames & Runway Alignment
- The physics core (FlightEnvRL) runs in a Z‑up frame, while the Three.js viewer is Y‑up. We rotate telemetry by −90° around **X** (see `SIM_TO_VIEWER_QUAT` in `viewer/public/index.html`) so altitude stays positive in both frames.
- The runway mesh is centered by default in Three.js. We offset it by half the runway length so `runway.start` in the sim matches the visual threshold.
- When the aircraft spawned pointing the wrong way, we realigned the quaternion by reading the runway heading from `runway_config.json`. Mission targets now refer to runway-relative vectors (`runway_dir`, `runway_right`), ensuring the policy learns forward motion down the runway axis.

## Mission Guidance & Reward Shaping
- `FlightEnvRL` now exposes a mission-phase state machine: climb to **300 ft**, then fly straight to **≈800 ft** forward. `_advance_mission_phase` rolls through `self.mission_phases`, and `_guidance_target` feeds the active waypoint into the observation vector so PPO always sees a relative “breadcrumb”.
- Rewards emphasize safe envelopes:
  - Penalties for climb ratio > 0.6 (vertical speed exceeds 60% of forward) and for exceeding the 500 ft ceiling.
  - Bonuses for forward progress along `runway_dir` and for reaching each waypoint; terminating with `mission_success` yields a large positive spike.
  - Rudder trim is now part of the reward: once forward speed builds, keeping yaw error small while using trim gets rewarded, encouraging the policy to “dial in” trim instead of fighting with the rudder stick.
- The episode terminates if we leave ±75 % of the runway width (relaxed from 40 %) or hit the ceiling, geofence, stall, etc. This ensures training stops when the aircraft drifts too far from the corridor while still letting it explore a wider box.

## Dynamics & Mass Modeling
- Aerodynamic derivatives come from the PropShox report. `AeroParams` now matches the CAD rollup (wing area, span, moments). We also add ~0.23 kg to the takeoff mass to represent the “tennis-ball” payload.
- Roll/yaw damping derivatives (`Cl_p`, `Cn_r`, etc.) were strengthened to prevent snap rolls, and we use a fixed inertia diag `[0.42, 0.55, 0.78]` kg·m².
- Ground reactions are simplified: instead of custom friction we clamp Z velocity when we hit the ground height. This avoids big impulses when the aircraft briefly dips below the runway due to numerical error.
- The battery uses a 3 Ah / 17 V model (see `self.battery_params`) so the telemetry power draw matches the PropShox spec.

## Viewer Enhancements
- The viewer adds a translucent plane at **500 ft** with a HUD readout that turns orange above **450 ft**, mirroring the ceiling logic in the env.
- A throttle slider supports manual override for quick debugging, and telemetry runs over `127.0.0.1` to simplify local testing (`run_all.sh` binds both the HTTP and WebSocket servers to localhost).

## Training Workflow
- PPO harness lives in `scripts/train_ppo_flight.py`. Always run it with `PYTHONPATH=$(pwd)` so Gym can import `aida_sim`.
- For takeoff missions, use `--task takeoff`. The script checkpoints every 25 iterations and writes `checkpoints/ppo_flight_final.pt` at the end; `run_all.sh` defaults to that checkpoint.
- When you tweak rewards or mission phases, re-run training from scratch—PPO needs to see the new shaping from the start. We typically run `600k` timesteps on CPU, but the process scales well if you move to a GPU-equipped Dell.

## Telemetry & Viewer Integration
- The telemetry server (`scripts/run_sim_with_telemetry.py`) wraps the PPO policy and pushes state at 50 Hz. It now defaults to `127.0.0.1` so multiple hosts can run without firewall prompts.
- The viewer reads telemetry in feet and degrees (conversion happens in `toImperialPayload`). Watch the `Ceiling` HUD entry plus the new phase-based heading/distance output to ensure the aircraft tracks the target vector as expected.

## Future Steps
- Re-introduce crosswind/downwind waypoints once the straight-leg mission is stable.
- Extend the reward to cover landing flare (altitude ramp + runway alignment) before relaxing the runway-deviation termination.
- Consider logging rudder trim history for each episode so you can seed future resets with a better `trim_memory`.

This file should help whoever trains on another machine (Dell, cloud, etc.) understand the key assumptions and constraints baked into the current code. Update it whenever we adjust mission phases, control limits, or viewer behavior. 
