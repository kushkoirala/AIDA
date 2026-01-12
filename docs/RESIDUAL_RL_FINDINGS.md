# Residual RL V2 - Training Findings & Next Steps

## Current Status (Jan 11, 2026)

Successfully trained Residual RL V2 with 7-control architecture (throttle, aileron, elevator, rudder, flaps, spoilers, brakes). The model completes full SN65->KHUT cross-country flights including takeoff, cruise, approach, and landing.

### What Works
- Full mission completion (takeoff to landing)
- Cruise phase tracking (holds altitude ~1719m, heading ~349 deg)
- Descent and approach phase transitions
- Landing on runway (stops at ~3ft altitude)

### Known Issues

#### 1. Left Drift During Ground Roll/Rotation
- Aircraft veers left during takeoff (heading 0 deg -> 339 deg, ~21 deg deviation)
- NN outputs nearly constant residuals regardless of phase
- Rudder residual is negative (-0.04 to -0.05) when it should be positive to correct left drift

**Root Cause Analysis:**
- NN learned a "mean" policy - same outputs for all phases
- Not reacting to centerline deviation (Y error) during ground ops
- Residual scale for rudder (0.10) may be too small for effective ground steering

#### 2. Constant NN Outputs
The NN outputs are nearly identical throughout the flight:

| Phase | Residuals [thr, ail, elev, rud, flap, spoil, brk] |
|-------|---------------------------------------------------|
| GROUND_ROLL | +0.01, -0.04, -0.09, +0.04, -0.03, -0.02, +0.0 |
| CRUISE | -0.01, +0.04, +0.11, -0.05, +0.04, +0.03, -0.0 |
| LANDING | -0.00, +0.04, +0.10, -0.04, +0.04, +0.03, -0.0 |

This suggests the NN hasn't learned phase-specific corrections.

## Reward Structure (Current)

### GROUND_ROLL
- Y < 0.3m: +5.0 reward
- Y >= 0.3m: -Y^2 x 1.0 penalty (quadratic)
- Roll error: -|phi| x 10.0
- Heading error: -heading^2 x 5.0

### ROTATION
- Y < 0.3m: +4.0 reward
- Y >= 0.3m: -Y^2 x 0.8 penalty (quadratic)
- Roll error: -|phi| x 8.0
- Heading error: -heading^2 x 3.0

## Residual Scales (Current)

| Control | Scale | Notes |
|---------|-------|-------|
| throttle | 0.15 | |
| aileron | 0.10 | |
| elevator | 0.15 | |
| rudder | 0.10 | Possibly too small for ground steering |
| flaps | 0.10 | |
| spoilers | 0.15 | |
| brakes | 0.10 | |

## Next Steps to Try

### Option 1: Increase Rudder Authority During Ground Ops
- Increase rudder residual scale from 0.10 to 0.25-0.30
- This gives NN more authority to correct drift

### Option 2: Phase-Specific Residual Scales
- Use larger scales during GROUND_ROLL/ROTATION
- Smaller scales during cruise (where expert is sufficient)

### Option 3: Force Zero Residuals During Ground Roll
- Disable NN during ground roll entirely
- Let expert handle takeoff, NN takes over after rotation
- Cleaner transition but loses learning opportunity

### Option 4: Add Y-Velocity to Observation
- Currently NN sees Y position but may not see rate of drift
- Adding Y-dot could help NN react faster to developing drift

### Option 5: Curriculum Learning
- Train ground roll phase separately first
- Then fine-tune on full mission

## Training Configuration

- Algorithm: PPO (Stable-Baselines3)
- Network: [512, 512, 256] for policy and value
- Observation: 25 dims [state(12) + expert(7) + target(3) + phase(3)]
- Action: 7 dims (continuous, tanh -> scaled by residual_scales)
- Timesteps: 2,000,000
- Parallel envs: 8
- Device: CUDA

## Files Modified
- scripts/residual_env_v2.py - Training environment with reward shaping
- scripts/run_residual_telemetry_v2.py - Inference/telemetry script
- scripts/train_residual_ppo_v2.py - Training script

## Model Checkpoints
- Latest: checkpoints/residual_ppo_v2/residual_ppo_v2_final.zip (PPO_14)
- Training logs: checkpoints/residual_ppo_v2/logs/PPO_14/
