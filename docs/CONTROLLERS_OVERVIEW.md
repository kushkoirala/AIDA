# AIDA Controllers Overview

## Controller Architecture

AIDA implements a **hybrid control architecture** that combines classical controllers with neural network policies using a **Residual RL** approach.

```
                         CONTROLLER HIERARCHY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ┌─────────────────────────────────────────────────────────────┐
    │                    HYBRID CONTROLLER                        │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   ┌─────────────────┐         ┌─────────────────────────┐  │
    │   │ Expert Controller│         │   Residual NN Policy    │  │
    │   │   (Classical)    │         │      (PPO-trained)      │  │
    │   ├─────────────────┤         ├─────────────────────────┤  │
    │   │ • FSM phases    │         │ • 25-dim observation    │  │
    │   │ • PID control   │         │ • 7-dim corrections     │  │
    │   │ • 100% reliable │         │ • ±15% adjustments      │  │
    │   └────────┬────────┘         └────────────┬────────────┘  │
    │            │                               │                │
    │            │  u_expert (7-dim)             │  δ_nn (7-dim)  │
    │            │                               │                │
    │            └───────────────┬───────────────┘                │
    │                            │                                │
    │                            ▼                                │
    │            ┌───────────────────────────────┐                │
    │            │   u_total = u_expert + λ·δ_nn │                │
    │            └───────────────────────────────┘                │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

---

## Controller Types

### 1. Classical Controllers (Expert Demonstrators)

Hand-crafted controllers for autonomous flight, used for:
- Expert demonstration data generation (imitation learning)
- Baseline policy in Residual RL architecture
- Safety fallback in hybrid architectures
- Baseline comparison for RL policies

| Controller | File | Purpose | Controls |
|------------|------|---------|----------|
| Traffic Pattern | `classical_mission_controller.py` | Full pattern (takeoff to landing) | 6 |
| Cross-Country | `triangle_controller.py` | Multi-leg XC with approach | 7 |
| Ground Roll | `classical_ground_roll_controller.py` | Runway acceleration only | 4 |

#### Triangle Intercept Controller (Cross-Country)

The primary expert controller for cross-country flights. Implements 11 flight phases:

```python
class XCPhase(Enum):
    GROUND_ROLL = 1       # Accelerate on runway
    ROTATION = 2          # Pitch up for liftoff
    INITIAL_CLIMB = 3     # Clear obstacles
    CLIMB = 4             # Climb to cruise altitude
    CRUISE_TO_TP = 5      # Level cruise to turning point
    TURN_TO_INTERCEPT = 6 # Procedure turn
    INTERCEPT_LEG = 7     # Intercept final approach
    FINAL_APPROACH = 8    # Stabilized approach
    SHORT_FINAL = 9       # Pre-landing
    LANDING = 10          # Flare and touchdown
    LANDED = 11           # Mission complete
```

**Key Parameters:**
- Rotation speed: 54 KTAS
- Climb speed: 74 KTAS
- Cruise speed: 110 KTAS
- Approach speed: 65 KTAS
- Touchdown speed: 50 KTAS
- Cruise altitude: 5,500 ft
- Glideslope: 3.5°

---

### 2. Neural Network Policies (RL-Trained)

#### Residual RL V2 (Current - 7 Controls)

The latest architecture where NN learns **corrections** to the expert controller:

| Aspect | Specification |
|--------|---------------|
| Observation | 25 dimensions |
| Action | 7 dimensions (corrections) |
| Network | 512 → 512 → 256 |
| Parameters | ~411k |
| Residual Scale | ±5% to ±15% per control |

**Training Script:**
```bash
python scripts/train_residual_ppo_v2.py \
    --timesteps 2000000 \
    --n-envs 16 \
    --lr 1e-4
```

**Checkpoint Location:** `checkpoints/residual_ppo_v2/`

#### Curriculum Learning (Phase-by-Phase)

PPO-trained policies for specific flight phases:

| Phase | Training Script | Checkpoint Location |
|-------|-----------------|---------------------|
| Ground Roll | `train_cessna172_curriculum.py --phase 1` | `checkpoints/phase1_ground_roll/` |
| Rotation | `train_cessna172_curriculum.py --phase 2` | `checkpoints/phase2_rotation/` |
| Initial Climb | `train_cessna172_curriculum.py --phase 3` | `checkpoints/phase3_initial_climb/` |
| Full Climb | `train_cessna172_curriculum.py --phase 4` | `checkpoints/phase4_climb/` |
| Cruise | `train_cessna172_curriculum.py --phase 5` | `checkpoints/phase5_cruise/` |

---

### 3. Hybrid Controllers

Combine NN policies with classical safety fallback:

```python
# Residual RL approach (current)
u_total = expert_controller(state) + lambda * nn_policy(state)

# Safety fallback approach (future)
if safety_monitor.is_safe(state):
    action = neural_net_policy(state)
else:
    action = classical_controller.compute_action(state)
```

---

## Controller Invocation

### Pure Expert (Classical Only)

```bash
# Cross-country flight with telemetry viewer
python scripts/run_residual_telemetry.py --pure-expert

# Traffic pattern
python scripts/run_traffic_pattern_with_telemetry.py
```

### Residual RL (Expert + NN Corrections)

```bash
# V2 model (7 controls)
python scripts/run_residual_telemetry_v2.py \
    --model checkpoints/residual_ppo_v2/best_model.zip

# V1 model (4 controls)
python scripts/run_residual_telemetry.py \
    --model checkpoints/residual_ppo/best_model.zip
```

### Training

```bash
# Residual RL V2 (recommended)
python scripts/train_residual_ppo_v2.py --timesteps 2000000 --n-envs 16

# Curriculum learning
python scripts/train_cessna172_curriculum.py --start-phase 1
```

---

## State and Action Vectors

### State Vector (12D)

| Index | Variable | Unit | Description |
|-------|----------|------|-------------|
| 0 | x | m | Position North |
| 1 | y | m | Position East |
| 2 | z | m | Position Down (neg = altitude) |
| 3 | u | m/s | Velocity forward |
| 4 | v | m/s | Velocity right |
| 5 | w | m/s | Velocity down |
| 6 | phi | rad | Roll angle |
| 7 | theta | rad | Pitch angle |
| 8 | psi | rad | Yaw angle (heading) |
| 9 | p | rad/s | Roll rate |
| 10 | q | rad/s | Pitch rate |
| 11 | r | rad/s | Yaw rate |

### Action Vector - V1 (4D)

| Index | Control | Range | Description |
|-------|---------|-------|-------------|
| 0 | throttle | [0, 1] | Engine power |
| 1 | aileron | [-1, 1] | Roll control |
| 2 | elevator | [-1, 1] | Pitch control |
| 3 | rudder | [-1, 1] | Yaw control |

### Action Vector - V2 (7D)

| Index | Control | Range | Residual Scale | Description |
|-------|---------|-------|----------------|-------------|
| 0 | throttle | [0, 1] | ±15% | Engine power |
| 1 | aileron | [-1, 1] | ±12% | Roll control |
| 2 | elevator | [-1, 1] | ±12% | Pitch control |
| 3 | rudder | [-1, 1] | ±10% | Yaw control |
| 4 | flaps | [0, 1] | ±8% | High-lift devices |
| 5 | spoilers | [0, 1] | ±12% | Speed brakes |
| 6 | brakes | [0, 1] | ±5% | Wheel brakes |

### Observation Vector - Residual V2 (25D)

| Indices | Component | Dimensions | Description |
|---------|-----------|------------|-------------|
| 0-11 | State | 12 | Aircraft state vector |
| 12-18 | Expert Action | 7 | Current expert controller output |
| 19-21 | Target | 3 | Heading error, distance, altitude error |
| 22-24 | Phase | 3 | One-hot: takeoff, cruise, approach |

---

## Data Storage

### Expert Demonstrations

Generated by classical controllers for imitation learning:

```
data/
├── expert_demos/           # Full flight demonstrations
├── scenario_demos/         # Specific scenario demos
└── xc_demos/               # Cross-country flight demos
```

Format: NumPy `.npz` files containing:
- `observations`: State vectors (N, 12) or (N, 25)
- `actions`: Control inputs (N, 4) or (N, 7)
- `phases`: Flight phase labels (N,)
- `metadata`: Flight info dict

### Model Checkpoints

```
checkpoints/
├── residual_ppo_v2/        # Latest 7-control model
│   ├── best_model.zip
│   ├── residual_ppo_v2_*.zip  # Periodic saves
│   └── logs/               # TensorBoard logs
├── residual_ppo/           # V1 4-control model
├── phase1_ground_roll/
├── phase2_rotation/
└── ...
```

Format: Stable Baselines3 `.zip` files

### Training Logs

```
checkpoints/residual_ppo_v2/logs/
└── PPO_*/
    └── events.out.tfevents.*
```

Access via: `tensorboard --logdir checkpoints/residual_ppo_v2/logs --port 6007`

---

## Performance Comparison

| Controller | Success Rate | Flight Time | Notes |
|------------|--------------|-------------|-------|
| Pure Expert | 100% | 18.5 min | Baseline, no learning |
| Residual RL V1 | TBD | TBD | 4 controls |
| Residual RL V2 | TBD | TBD | 7 controls, in training |

---

**Last Updated:** January 10, 2026
