# AIDA - Autonomous Intelligent Decision Architecture

**Advanced Reinforcement Learning Framework for Autonomous Fixed-Wing Aircraft Control**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.x-orange.svg)](https://stable-baselines3.readthedocs.io/)

---

## Executive Summary

AIDA is a research framework that combines classical control theory with modern deep reinforcement learning to achieve fully autonomous fixed-wing aircraft flight. The system has demonstrated **complete autonomous cross-country flights** from takeoff to landing, covering 31 nautical miles with precision runway alignment.

### Key Achievements

| Milestone | Description | Date |
|-----------|-------------|------|
| **Cross-Country Flight** | 31 NM autonomous flight SN65 → KHUT with precision landing | Jan 2026 |
| **Residual RL Training** | Neural network learns corrections to expert controller | Jan 2026 |
| **GPU-Accelerated Simulation** | 569,000 steps/sec with 10,000 parallel instances | Dec 2025 |
| **Real-Time 3D Visualization** | WebSocket telemetry with audio synthesis | Dec 2025 |

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Neural Network Architecture](#neural-network-architecture)
3. [Algorithms & Training Methods](#algorithms--training-methods)
4. [Hardware Configuration](#hardware-configuration)
5. [Flight Demonstrations](#flight-demonstrations)
6. [Quick Start](#quick-start)
7. [Project Structure](#project-structure)
8. [Development Guide](#development-guide)

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AIDA SYSTEM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐  │
│  │   TRAINING LAYER    │     │   INFERENCE LAYER   │     │  VISUALIZATION   │  │
│  ├─────────────────────┤     ├─────────────────────┤     ├──────────────────┤  │
│  │ • PPO Algorithm     │     │ • Policy Network    │     │ • 3D WebGL View  │  │
│  │ • Residual RL       │────▶│ • Expert Controller │────▶│ • TensorBoard    │  │
│  │ • Behavior Cloning  │     │ • Hybrid Blending   │     │ • Telemetry WS   │  │
│  │ • Curriculum Learn  │     │ • Safety Monitor    │     │ • Audio GPWS     │  │
│  └─────────────────────┘     └─────────────────────┘     └──────────────────┘  │
│           │                           │                           │            │
│           ▼                           ▼                           ▼            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        SIMULATION LAYER                                  │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                         │   │
│  │  ┌───────────────────┐   ┌───────────────────┐   ┌──────────────────┐  │   │
│  │  │  GPU Flight Sim   │   │  Gymnasium Env    │   │  Aircraft Models │  │   │
│  │  ├───────────────────┤   ├───────────────────┤   ├──────────────────┤  │   │
│  │  │ • CuPy/CUDA       │   │ • ResidualEnvV2   │   │ • Cessna 172     │  │   │
│  │  │ • 10k+ instances  │   │ • 25-dim obs      │   │ • Udaan UAV      │  │   │
│  │  │ • RK4 integration │   │ • 7-dim action    │   │ • Aero coeffs    │  │   │
│  │  │ • 6-DOF dynamics  │   │ • Phase rewards   │   │ • Mass/inertia   │  │   │
│  │  └───────────────────┘   └───────────────────┘   └──────────────────┘  │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Details

| Component | Technology | Purpose |
|-----------|------------|---------|
| **GPU Flight Dynamics** | CuPy/CUDA, NumPy | Parallel 6-DOF rigid body simulation |
| **RL Environment** | Gymnasium | Observation/action spaces, reward shaping |
| **Training** | Stable-Baselines3, PyTorch | PPO, policy networks, curriculum |
| **Expert Controller** | Classical PID/FSM | Baseline policy, safety fallback |
| **Visualization** | Three.js, WebSocket | Real-time 3D rendering, telemetry |
| **Audio** | Web Audio API | Engine synthesis, GPWS callouts |

### Software Package Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AIDA PACKAGES                                  │
├──────────────────┬──────────────────┬──────────────────┬────────────────┤
│    aida_sim/     │     scripts/     │     viewer/      │ gpu-flight-    │
│                  │                  │                  │ dynamics/      │
├──────────────────┼──────────────────┼──────────────────┼────────────────┤
│ • env/           │ • train_*.py     │ • public/        │ • python/      │
│   - flight_env   │ • run_*.py       │   - index.html   │   - flight_    │
│   - residual_env │ • generate_*.py  │   - viewer.js    │     dynamics.py│
│ • dynamics/      │ • test_*.py      │ • components/    │   - aircraft_  │
│ • systems/       │                  │                  │     database.py│
│ • io/telemetry   │                  │                  │                │
└──────────────────┴──────────────────┴──────────────────┴────────────────┘
```

---

## Neural Network Architecture

### Residual RL Architecture (Current - V2)

The Residual RL approach learns **corrections** to an expert controller rather than learning from scratch:

```
                    RESIDUAL RL ARCHITECTURE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Aircraft State (12-dim)
    ┌─────────────────────────┐
    │ Position:    x, y, z    │
    │ Velocity:    u, v, w    │
    │ Attitude:    φ, θ, ψ    │
    │ Rates:       p, q, r    │
    └───────────┬─────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────┐
    │              OBSERVATION BUILDER (25-dim)         │
    ├───────────────────────────────────────────────────┤
    │  State (12) + Expert Action (7) + Target (3)      │
    │  + Phase One-Hot (3)                              │
    │                                                   │
    │  [x,y,z,u,v,w,φ,θ,ψ,p,q,r,                       │
    │   thr,ail,ele,rud,flp,spl,brk,                   │
    │   Δhdg,dist,Δalt,                                │
    │   takeoff,cruise,approach]                        │
    └───────────────────┬───────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
    ┌─────────────────────┐     ┌─────────────────────┐
    │   EXPERT CONTROLLER │     │    POLICY NETWORK   │
    │   (Classical FSM)   │     │    (Neural Network) │
    ├─────────────────────┤     ├─────────────────────┤
    │ • Phase detection   │     │ Input: 25 dims      │
    │ • PID control laws  │     │ Hidden: 512→512→256 │
    │ • Trajectory gen    │     │ Output: 7 dims      │
    │                     │     │ Activation: Tanh    │
    └──────────┬──────────┘     └──────────┬──────────┘
               │                           │
               │  u_expert (7-dim)         │  δ_nn (7-dim)
               │                           │  scaled by λ
               ▼                           ▼
    ┌─────────────────────────────────────────────────┐
    │              RESIDUAL BLENDING                  │
    │                                                 │
    │   u_total = u_expert + λ × δ_nn                │
    │                                                 │
    │   Per-Control Residual Scales (λ):             │
    │   ┌─────────────────────────────────────────┐  │
    │   │ Throttle: ±15%  │  Flaps:    ±8%       │  │
    │   │ Aileron:  ±12%  │  Spoilers: ±12%      │  │
    │   │ Elevator: ±12%  │  Brakes:   ±5%       │  │
    │   │ Rudder:   ±10%  │                      │  │
    │   └─────────────────────────────────────────┘  │
    └───────────────────┬─────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────────────┐
    │              FLIGHT SIMULATOR                   │
    │         (GPU-accelerated 6-DOF)                 │
    └─────────────────────────────────────────────────┘
```

### Network Specifications

| Layer | Dimensions | Activation | Parameters |
|-------|------------|------------|------------|
| Input | 25 | - | - |
| Hidden 1 | 512 | ReLU | 13,312 |
| Hidden 2 | 512 | ReLU | 262,656 |
| Hidden 3 | 256 | ReLU | 131,328 |
| Policy Head | 7 (μ) + 7 (σ) | Tanh/Softplus | 3,598 |
| Value Head | 1 | Linear | 257 |
| **Total** | - | - | **~411k** |

### Observation Space (25 dimensions)

```python
observation = [
    # Aircraft State (12 dims)
    x, y, z,           # Position (m)
    u, v, w,           # Body velocities (m/s)
    phi, theta, psi,   # Euler angles (rad)
    p, q, r,           # Angular rates (rad/s)

    # Expert Action (7 dims)
    throttle,          # [0, 1]
    aileron,           # [-1, 1]
    elevator,          # [-1, 1]
    rudder,            # [-1, 1]
    flaps,             # [0, 1]
    spoilers,          # [0, 1]
    brakes,            # [0, 1]

    # Navigation Target (3 dims)
    heading_error,     # Normalized [-1, 1]
    distance_to_target,# Normalized
    altitude_error,    # Normalized

    # Flight Phase One-Hot (3 dims)
    is_takeoff,        # [0, 1]
    is_cruise,         # [0, 1]
    is_approach,       # [0, 1]
]
```

### Action Space (7 dimensions)

```python
action = [
    δ_throttle,   # Correction to throttle  [-1, 1] → ±15%
    δ_aileron,    # Correction to aileron   [-1, 1] → ±12%
    δ_elevator,   # Correction to elevator  [-1, 1] → ±12%
    δ_rudder,     # Correction to rudder    [-1, 1] → ±10%
    δ_flaps,      # Correction to flaps     [-1, 1] → ±8%
    δ_spoilers,   # Correction to spoilers  [-1, 1] → ±12%
    δ_brakes,     # Correction to brakes    [-1, 1] → ±5%
]
```

---

## Algorithms & Training Methods

### Algorithm Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TRAINING ALGORITHM PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE 1: Expert Demonstration                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Classical Controller (FSM + PID)                                 │ │
│  │  • 11 flight phases: Ground Roll → Landed                         │ │
│  │  • Generates optimal trajectories                                 │ │
│  │  • 100% success rate on known routes                              │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  STAGE 2: Behavior Cloning (Optional Warm-Start)                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Supervised Learning from Expert Data                             │ │
│  │  • Loss: MSE(π_θ(s), a_expert)                                    │ │
│  │  • 10-20 epochs, ~50k transitions                                 │ │
│  │  • Reduces PPO training time by 50-70%                            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  STAGE 3: Residual PPO Fine-Tuning                                      │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Proximal Policy Optimization with Residual Architecture          │ │
│  │  • NN learns corrections δ to expert: u = u_expert + λδ           │ │
│  │  • Inherits expert's stability, learns refinements                │ │
│  │  • Phase-specific reward shaping                                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### PPO Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 1e-4 | Conservative for stability |
| `n_steps` | 4096 | Long rollouts for 15-min episodes |
| `batch_size` | 256 | Large batch for variance reduction |
| `n_epochs` | 10 | Multiple passes per rollout |
| `gamma` | 0.995 | High discount for long episodes |
| `gae_lambda` | 0.95 | Advantage estimation |
| `clip_range` | 0.1 | Small clip for stable updates |
| `ent_coef` | 0.005 | Minimal exploration (expert baseline) |
| `vf_coef` | 0.5 | Value function loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |

### Reward Shaping by Phase

```python
# Takeoff Phase (Ground Roll → Initial Climb)
reward_takeoff = (
    + 1.0 * speed_progress      # Accelerate to rotation speed
    - 0.5 * centerline_error    # Stay on runway centerline
    + 2.0 * altitude_gain       # Reward positive climb
    - 1.0 * if crashed          # Termination penalty
)

# Cruise Phase (Climb → Cruise)
reward_cruise = (
    + 1.0 * altitude_tracking   # Maintain target altitude
    + 0.5 * heading_tracking    # Track to waypoint
    + 0.3 * speed_tracking      # Maintain cruise speed
    - 0.5 * control_effort      # Smooth control usage
)

# Approach Phase (Descent → Landing)
reward_approach = (
    + 2.0 * glideslope_track    # Follow 3° glideslope
    + 1.0 * localizer_track     # Runway centerline
    + 1.5 * airspeed_target     # Approach speed
    + 5.0 * successful_landing  # Terminal reward
)
```

### Curriculum Learning Phases

| Phase | Task | Training Focus | Success Criteria |
|-------|------|----------------|------------------|
| 1 | Ground Roll | Acceleration, steering | Reach V_rotate |
| 2 | Rotation | Pitch control | Positive climb rate |
| 3 | Initial Climb | Climb gradient | 100 ft AGL |
| 4 | Climb | Altitude tracking | Cruise altitude |
| 5 | Cruise | Level flight | Altitude/heading hold |
| 6 | Descent | Glideslope capture | 3° path |
| 7 | Landing | Flare, touchdown | Safe ground contact |

---

## Hardware Configuration

### Development Environment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HARDWARE CONFIGURATION                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PRIMARY WORKSTATION: Dell 7920                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  CPU:    Intel Xeon (16+ cores)                                   │ │
│  │  GPU:    NVIDIA RTX 4060 (8GB VRAM)                               │ │
│  │  RAM:    32+ GB DDR4                                              │ │
│  │  OS:     Windows 11 + WSL2 (Ubuntu 22.04)                         │ │
│  │  CUDA:   12.x                                                     │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  PERFORMANCE BENCHMARKS:                                                │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                   │ │
│  │  GPU Simulation (CuPy):                                           │ │
│  │  ├── 100 instances:    ~50,000 steps/sec   (2,500x real-time)    │ │
│  │  ├── 1,000 instances:  ~100,000 steps/sec  (5,000x real-time)    │ │
│  │  └── 10,000 instances: ~569,000 steps/sec  (28,450x real-time)   │ │
│  │                                                                   │ │
│  │  Training Throughput (SubprocVecEnv):                             │ │
│  │  ├── 16 parallel envs (CPU sim): ~1,300 steps/sec                │ │
│  │  └── Neural network updates: GPU (CUDA)                          │ │
│  │                                                                   │ │
│  │  Memory Usage:                                                    │ │
│  │  ├── Per environment: ~100 MB                                    │ │
│  │  ├── 16 envs + model: ~2 GB RAM                                  │ │
│  │  └── GPU VRAM: ~1 GB for inference                               │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Software Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.x | Neural networks |
| Stable-Baselines3 | 2.x | PPO implementation |
| CuPy | 12.x | GPU-accelerated NumPy |
| Gymnasium | 0.29+ | RL environment API |
| NumPy | 1.24+ | Numerical computing |
| TensorBoard | 2.x | Training visualization |
| Three.js | r150+ | 3D web rendering |

### Vectorized Environment Options

```python
# Option 1: SubprocVecEnv (Recommended for training)
# - True parallelism across CPU cores
# - Each subprocess runs independent simulation
# - 16 envs → ~16x speedup on multi-core systems
from stable_baselines3.common.vec_env import SubprocVecEnv
env = SubprocVecEnv([make_env(i) for i in range(16)], start_method='spawn')

# Option 2: DummyVecEnv (For GPU batched simulation)
# - Sequential stepping in single process
# - Use with GPU-batched simulator (1000+ instances)
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([make_env(i) for i in range(n_envs)])
```

---

## Flight Demonstrations

### Cross-Country Flight: SN65 → KHUT

**Route:** Lake Waltanna (SN65) to Hutchinson Regional Airport (KHUT)
**Distance:** 31 nautical miles
**Duration:** ~18 minutes (simulated)

```
                        CROSS-COUNTRY FLIGHT PROFILE

    Altitude (ft)
         ▲
    6000 │                    ┌────────────────────┐
         │                   ╱                      ╲
    5500 │──────────────────╱  CRUISE @ 110 KTAS    ╲
         │                 ╱                          ╲
    5000 │                ╱                            ╲ DESCENT
         │               ╱                              ╲
    4000 │              ╱                                ╲
         │             ╱ CLIMB                            ╲
    3000 │            ╱  500 fpm                           ╲
         │           ╱                                      ╲
    2000 │          ╱                                        ╲
         │         ╱                                          ╲
    1000 │        ╱                                            ╲ APPROACH
         │       ╱                                              ╲
       0 │──────╱────────────────────────────────────────────────╲────▶
         └──────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────
              SN65    5     10    15    20    25    30   KHUT  Distance (nm)

    Flight Phases:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    [GROUND_ROLL] → [ROTATION] → [INITIAL_CLIMB] → [CLIMB] →
    [CRUISE_TO_TP] → [TURN_TO_INTERCEPT] → [INTERCEPT_LEG] →
    [FINAL_APPROACH] → [SHORT_FINAL] → [LANDING] → [LANDED]
```

### 11 Autonomous Flight Phases

| Phase | Description | Key Parameters |
|-------|-------------|----------------|
| GROUND_ROLL | Accelerate on runway | Full throttle, V_rotate = 54 KTAS |
| ROTATION | Pitch up for liftoff | 10° pitch target |
| INITIAL_CLIMB | Clear obstacles | V_climb = 74 KTAS |
| CLIMB | Climb to cruise | 500 fpm, heading to TP |
| CRUISE_TO_TP | Level cruise | 5500 ft, 110 KTAS |
| TURN_TO_INTERCEPT | Procedure turn | Roll to runway heading |
| INTERCEPT_LEG | Intercept final | Glideslope capture |
| FINAL_APPROACH | Stabilized approach | 3° glideslope, 65 KTAS |
| SHORT_FINAL | Pre-landing | Flaps full, 60 KTAS |
| LANDING | Flare and touchdown | Idle thrust, 50 KTAS |
| LANDED | Mission complete | Brakes applied |

---

## Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.10+
- NVIDIA GPU with CUDA 12.x (recommended)
- WSL2 (Windows) or native Linux
- 8+ GB RAM
```

### Installation

```bash
# Clone and setup
cd /home/AIDA
python3 -m venv .venv-linux
source .venv-linux/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (WSL)
export MPLCONFIGDIR=/tmp/matplotlib-config
export CUPY_CACHE_DIR=/tmp/cupy-cache
```

### Run Expert Flight Demo

```bash
# Terminal 1: Start viewer server
cd /home/AIDA/viewer/public
python -m http.server 8000

# Terminal 2: Run cross-country flight
source .venv-linux/bin/activate
python scripts/run_residual_telemetry.py --pure-expert

# Open browser: http://localhost:8000
```

### Train Residual RL Policy

```bash
# Train with 16 parallel environments
python scripts/train_residual_ppo_v2.py \
    --timesteps 2000000 \
    --n-envs 16 \
    --lr 1e-4

# Monitor training
tensorboard --logdir checkpoints/residual_ppo_v2/logs --port 6007
```

### Run Trained Policy

```bash
# Run with trained model
python scripts/run_residual_telemetry_v2.py \
    --model checkpoints/residual_ppo_v2/best_model.zip

# Compare with pure expert
python scripts/run_residual_telemetry.py --pure-expert
```

---

## Project Structure

```
AIDA/
├── aida_sim/                    # Core simulation package
│   ├── env/                     # RL environments
│   │   ├── flight_env_cessna172.py
│   │   ├── residual_env_v2.py   # 7-control residual environment
│   │   └── waypoint_env.py
│   ├── dynamics/                # Flight physics
│   ├── systems/                 # Aircraft subsystems
│   └── io/                      # Telemetry I/O
│
├── gpu-flight-dynamics/         # CUDA parallel simulator
│   └── python/
│       ├── flight_dynamics.py   # GPU-accelerated 6-DOF
│       └── aircraft_database.py # Aircraft configurations
│
├── scripts/                     # Training and utility scripts
│   ├── train_residual_ppo_v2.py # Main training script
│   ├── run_residual_telemetry.py
│   ├── triangle_controller.py   # Expert FSM controller
│   └── classical_mission_controller.py
│
├── viewer/                      # 3D visualization
│   └── public/
│       └── index.html           # WebGL viewer + telemetry
│
├── checkpoints/                 # Trained models
│   └── residual_ppo_v2/
│       ├── best_model.zip
│       └── logs/                # TensorBoard logs
│
├── docs/                        # Documentation
│   ├── img/                     # Architecture diagrams
│   └── *.md                     # Technical docs
│
└── config/                      # Configuration files
```

---

## Development Guide

### Adding a New Aircraft

1. Add configuration to `gpu-flight-dynamics/python/aircraft_database.py`
2. Define aerodynamic coefficients (CL, CD, Cm, etc.)
3. Specify mass properties (mass, Ixx, Iyy, Izz)
4. Create environment wrapper if needed

### Modifying Reward Functions

Edit `scripts/residual_env_v2.py`:

```python
def _compute_reward(self, state, action, info):
    # Phase-specific rewards
    if self.phase in [XCPhase.GROUND_ROLL, XCPhase.ROTATION]:
        return self._takeoff_reward(state, action)
    elif self.phase in [XCPhase.CRUISE_TO_TP]:
        return self._cruise_reward(state, action)
    else:
        return self._approach_reward(state, action)
```

### TensorBoard Monitoring

```bash
# Start TensorBoard
tensorboard --logdir checkpoints/residual_ppo_v2/logs --port 6007

# Key metrics to watch:
# - rollout/ep_rew_mean: Average episode reward
# - train/loss: Combined loss
# - train/entropy_loss: Exploration metric
# - train/value_loss: Value function accuracy
```

---

## References

1. **Stable-Baselines3**: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)
2. **PPO Algorithm**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
3. **Residual RL**: Silver et al., "Residual Policy Learning" (2018)
4. **Flight Dynamics**: Stevens & Lewis, "Aircraft Control and Simulation" (3rd ed.)

---

## License

Internal research project - Kushal Koirala

---

**Last Updated: January 11, 2026
