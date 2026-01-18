# Full Residual RL Training Plan: All 7 Controls + All Phases

## Current Limitations (What We're Fixing)

The current `residual_env.py` only trains on **4 controls** (throttle, aileron, elevator, rudder) and only during **cruise/climb phases**:

```python
# CURRENT (Limited)
action_space = Box(shape=(4,))  # Missing: flaps, spoilers, brakes
controls = [[throttle, aileron, elevator, rudder]]  # Only 4!
```

The expert outputs **7 controls**: `[throttle, aileron, elevator, rudder, flaps, spoilers, brakes]`

Spoilers are critical for descent (expert uses spoilers=0.8 to achieve proper descent rate).

---

## Changes Required

### 1. Action Space: 4 -> 7 Dimensions

```python
# NEW
self.action_space = spaces.Box(
    low=-1.0, high=1.0, shape=(7,), dtype=np.float32
)
# [throttle_res, aileron_res, elevator_res, rudder_res, flaps_res, spoilers_res, brakes_res]
```

### 2. Observation Space: 19 -> 25 Dimensions

```python
# OLD: state(12) + expert_action(4) + target(3) = 19
# NEW: state(12) + expert_action(7) + target(3) + phase_one_hot(3) = 25

self.observation_space = spaces.Box(
    low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
)
```

**Phase encoding** (3 dimensions):
- `[1,0,0]` = Takeoff (GROUND_ROLL, ROTATION, INITIAL_CLIMB)
- `[0,1,0]` = Cruise (CLIMB, CRUISE_TO_TP)
- `[0,0,1]` = Approach (TURN_TO_INTERCEPT, INTERCEPT_LEG, FINAL_APPROACH, SHORT_FINAL, LANDING)

### 3. Fix KHUT Coordinates

```python
# OLD (wrong)
self.khut_x = 113000.0
self.khut_y = 1500.0

# NEW (correct)
self.khut_x = 52800.0
self.khut_y = -21300.0
```

### 4. Apply All 7 Controls to Simulator

```python
# OLD
controls = np.array([[throttle, aileron, elevator, rudder]], dtype=np.float32)

# NEW
controls = np.array([[throttle, aileron, elevator, rudder, flaps, spoilers, brakes]], dtype=np.float32)
```

### 5. Phase-Specific Reward Shaping

| Phase | Reward Focus |
|-------|--------------|
| **GROUND_ROLL** | Centerline tracking, acceleration |
| **ROTATION** | Smooth pitch up, no lateral drift |
| **INITIAL_CLIMB** | Positive climb rate, wings level |
| **CLIMB** | Altitude gain, heading toward TP |
| **CRUISE_TO_TP** | Altitude hold, progress to TP |
| **TURN_TO_INTERCEPT** | Capture runway heading |
| **INTERCEPT_LEG** | Glideslope tracking, speed control |
| **FINAL_APPROACH** | Glideslope + localizer, speed 65kts |
| **SHORT_FINAL** | Stable approach, flare preparation |
| **LANDING** | Smooth touchdown, centerline |

### 6. Residual Scale by Control Type

Different controls may need different residual scales:

```python
residual_scales = {
    'throttle': 0.15,    # ±15% throttle adjustment
    'aileron': 0.10,     # ±10% aileron (less authority during critical phases)
    'elevator': 0.10,    # ±10% elevator
    'rudder': 0.10,      # ±10% rudder
    'flaps': 0.05,       # ±5% flaps (mostly expert-driven)
    'spoilers': 0.10,    # ±10% spoilers (important for descent)
    'brakes': 0.05,      # ±5% brakes (mostly expert-driven)
}
```

---

## New Observation Vector (25 dims)

| Index | Name | Description |
|-------|------|-------------|
| 0-11 | State | [x, y, z, u, v, w, phi, theta, psi, p, q, r] |
| 12-18 | Expert Action | [thr, ail, ele, rud, flaps, spoil, brake] |
| 19 | Alt Error | (target_alt - current_alt) / 1000 |
| 20 | Heading Error | normalized to [-pi, pi] |
| 21 | Distance | to target / 10000 (normalized) |
| 22-24 | Phase One-Hot | [takeoff, cruise, approach] |

---

## New Reward Function

```python
def _compute_reward(self, state, action, expert_action):
    reward = 0.0

    # 1. Progress reward (all phases)
    if self.prev_dist_to_khut:
        reward += (self.prev_dist_to_khut - dist_to_khut) * 0.01

    # 2. Phase-specific rewards
    if phase in [GROUND_ROLL, ROTATION]:
        # Centerline tracking
        reward -= abs(y) * 0.01  # Penalize lateral deviation
        reward -= abs(phi) * 0.5  # Penalize roll

    elif phase in [INITIAL_CLIMB, CLIMB]:
        # Positive climb rate
        if climb_rate > 0:
            reward += 0.3
        # Wings reasonably level
        if abs(phi) < deg2rad(30):
            reward += 0.2

    elif phase == CRUISE_TO_TP:
        # Altitude hold
        alt_error = abs(alt - cruise_alt)
        if alt_error < 50:
            reward += 0.5
        elif alt_error < 200:
            reward += 0.2
        else:
            reward -= alt_error / 1000

    elif phase in [INTERCEPT_LEG, FINAL_APPROACH]:
        # Glideslope tracking
        target_alt = dist_to_threshold * tan(3.5 deg)
        gs_error = abs(alt - target_alt)
        if gs_error < 100:
            reward += 0.5
        else:
            reward -= gs_error / 500
        # Speed control (65-75 kts approach)
        if 60 < airspeed_kts < 80:
            reward += 0.3

    elif phase in [SHORT_FINAL, LANDING]:
        # Stable descent
        if -800 < climb_rate_fpm < -200:
            reward += 0.3
        # Proper flare
        if alt < 50 and theta > 0:
            reward += 0.5

    # 3. Smooth control (prefer small residuals)
    residual_magnitude = sqrt(sum(action**2))
    reward -= 0.05 * residual_magnitude

    # 4. Success bonus
    if landed_successfully:
        reward += 1000.0

    return reward
```

---

## Training Configuration

```python
# Recommended settings for full-flight training
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,         # Lower LR for stability
    n_steps=4096,               # Longer rollouts
    batch_size=128,
    n_epochs=10,
    gamma=0.995,                # Higher gamma for long episodes
    gae_lambda=0.95,
    clip_range=0.1,             # Smaller clip for stability
    ent_coef=0.005,             # Less exploration (expert is good)
    policy_kwargs={
        "net_arch": [512, 512, 256],  # Larger network for 7 controls
    }
)

# Train for longer
total_timesteps = 2_000_000     # 2M steps (was 500k)
max_episode_steps = 45000      # 15 minutes per episode
```

---

## Implementation Order

1. **Update `residual_env.py`**:
   - Fix KHUT coordinates
   - Expand action space to 7
   - Expand observation space to 25
   - Apply all 7 controls to simulator
   - Add phase-specific rewards

2. **Update `train_residual_ppo.py`**:
   - Increase timesteps to 2M
   - Adjust hyperparameters
   - Increase network size

3. **Test**:
   - Run environment test
   - Verify pure expert still lands
   - Start training

4. **Monitor**:
   - Tensorboard for reward curves
   - Check for crashes/stalls in logs
