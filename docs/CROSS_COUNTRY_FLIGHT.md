# Cross-Country Flight Controller

## Overview

The Cross-Country Flight Controller (triangle_controller.py) is a fully autonomous controller that flies complete cross-country missions from takeoff to landing. It implements a triangle intercept approach pattern for precision runway alignment.

Successfully demonstrated: SN65 (Lake Waltanna) to KHUT (Hutchinson Regional), 31 NM flight with autonomous landing on RWY 31.

## Flight Phases (11 Total)

| Phase | Trigger | Description |
|-------|---------|-------------|
| GROUND_ROLL | Start | Accelerate to rotation speed (54 kts) |
| ROTATION | V_rotate | Pitch up for liftoff |
| INITIAL_CLIMB | Airborne | Establish climb, 100 ft AGL |
| CLIMB | 100 ft AGL | Climb to cruise altitude (5500 ft) |
| CRUISE_TO_TP | Cruise altitude | Level flight toward turn point |
| TURN_TO_INTERCEPT | 10 nm from KHUT | Turn to intercept final approach course |
| INTERCEPT_LEG | On intercept heading | Descend toward runway |
| FINAL_APPROACH | 3 nm out, aligned | Glideslope descent (3 deg) |
| SHORT_FINAL | 0.4 nm out | Final stabilization |
| LANDING | 50 ft AGL | Flare and touchdown |
| LANDED | Less than 3 ft AGL, low speed | Mission complete |

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rotation Speed | 54 kts | Takeoff rotation |
| Climb Speed | 74 kts | Best rate of climb |
| Cruise Speed | 110 kts | Cruise power setting |
| Approach Speed | 65 kts | Final approach |
| Touchdown Speed | 50 kts | Target landing speed |
| Cruise Altitude | 5500 ft MSL | Cross-country cruise |
| Pattern Altitude | 1500 ft AGL | Traffic pattern entry |

## Architecture

### Unit Handling
All internal calculations use Imperial units (ft, ft/s, nm, kts):
- Metric to Imperial conversion only at flight_dynamics interface boundary
- Ensures numerical precision for aviation-standard calculations

### Control Loops
- Pitch control: Altitude/VS-based with trim
- Roll control: Heading intercept with bank limiting (25 deg max)
- Throttle control: Speed-based with phase-specific targets
- Rudder control: Coordinated turns, crosswind correction

## Usage

Start cross-country flight with telemetry viewer:

    cd /home/AIDA
    source .venv-linux/bin/activate
    python scripts/run_xc_sn65_khut.py

Viewer Access:
- Local: http://127.0.0.1:8000/?wshost=127.0.0.1
- Remote (Tailscale): http://TAILSCALE_IP:8000/?wshost=TAILSCALE_IP

## Results (Jan 8, 2025)

- Total time: 1177.5 seconds (~20 minutes)
- Distance: 31 NM
- Cruise altitude hold: 5751 ft (target 5500 ft)
- Landing: 0.3 nm past threshold
- Final speed: 10 kts (stopped on runway)

Phase Distribution:
- CRUISE_TO_TP: 42.1% (majority of flight)
- INTERCEPT_LEG: 23.0%
- CLIMB: 13.8%
- FINAL_APPROACH: 9.4%
- Other phases: 11.7%

## Files

| File | Description |
|------|-------------|
| scripts/triangle_controller.py | Main controller class |
| scripts/run_xc_sn65_khut.py | Cross-country flight runner with telemetry |
| viewer/public/engine_sound.js | Engine sound synthesis for viewer |
| aida_sim/io/telemetry.py | WebSocket telemetry server |
