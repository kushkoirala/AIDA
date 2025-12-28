# GPU-Accelerated Flight Dynamics Simulator

[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance 6-DOF flight dynamics simulator leveraging NVIDIA CUDA for parallel simulation of thousands of aircraft instances. Designed for reinforcement learning training and Monte Carlo analysis.

<p align="center">
  <img src="docs/architecture.png" alt="Architecture" width="600">
</p>

## 🎯 Key Features

- **Massive Parallelism**: Simulate 10,000+ aircraft simultaneously on GPU
- **High Fidelity**: Full 6-DOF equations with stability derivatives
- **Multiple Integrators**: Euler, RK2, RK4 methods
- **PyTorch Integration**: Direct GPU tensor interop for RL training
- **Validated Physics**: ISA atmosphere, realistic aerodynamics

## 🚀 Performance

Benchmarked on NVIDIA RTX 3080:

| Instances | CPU Time (ms) | GPU Time (ms) | Speedup |
|-----------|---------------|---------------|--------:|
| 100       | 45.2          | 1.2           | 37.7x   |
| 1,000     | 452.1         | 5.3           | 85.3x   |
| 5,000     | 2,261.4       | 18.7          | 120.9x  |
| 10,000    | 4,523.2       | 35.2          | 128.5x  |

**Peak throughput: 280+ million simulation steps per second**

## 📦 Installation

### Prerequisites

- NVIDIA GPU (Compute Capability 7.0+)
- CUDA Toolkit 12.x
- Python 3.8+ with pip

### Build CUDA Library

```bash
cd cuda
make CUDA_ARCH=sm_80  # Adjust for your GPU
```

### Install Python Package

```bash
pip install cupy-cuda12x torch numpy
cd python
pip install -e .
```

## 🔧 Quick Start

### Python API

```python
from flight_dynamics import FlightSimulator, ControlIndex
import numpy as np

# Create simulator with 1000 parallel instances
sim = FlightSimulator(n_instances=1000, use_gpu=True)

# Run simulation loop
for step in range(10000):
    # Set control inputs
    controls = np.zeros((1000, 4), dtype=np.float32)
    controls[:, ControlIndex.THROTTLE] = 0.5
    controls[:, ControlIndex.ELEVATOR] = np.random.uniform(-0.1, 0.1, 1000)
    sim.set_controls(controls)
    
    # Step simulation
    sim.step()
    
    # Get states for RL (as PyTorch tensor on GPU)
    states = sim.get_states_tensor(device='cuda')
```

### C/CUDA API

```c
#include "flight_dynamics.h"

// Initialize
AircraftParams params;
fd_default_params(&params);

SimConfig config;
fd_default_config(&config, 1000);

SimContext ctx;
fd_init(&ctx, &config, &params);

// Compute trim
float trim_state[12], trim_control[4];
fd_compute_trim(&params, 50.0f, 1000.0f, trim_state, trim_control);
fd_reset(&ctx, trim_state, trim_control);

// Simulation loop
for (int i = 0; i < 10000; i++) {
    fd_set_controls(&ctx, controls);
    fd_step(&ctx);
}

// Cleanup
fd_cleanup(&ctx);
```

## 📐 Physics Model

### State Vector (12 DOF)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0-2   | x, y, z | Position (NED) | m |
| 3-5   | u, v, w | Body velocity | m/s |
| 6-8   | φ, θ, ψ | Euler angles | rad |
| 9-11  | p, q, r | Angular rates | rad/s |

### Equations of Motion

**Translational Dynamics** (Newton's 2nd Law in rotating frame):
```
u̇ = Fx/m - qw + rv + gx
v̇ = Fy/m - ru + pw + gy  
ẇ = Fz/m - pv + qu + gz
```

**Rotational Dynamics** (Euler's Equations):
```
ṗ = (L + Ixz·ṙ + (Iyy - Izz)·qr) / Ixx
q̇ = (M + (Izz - Ixx)·pr) / Iyy
ṙ = (N + Ixz·ṗ + (Ixx - Iyy)·pq) / Izz
```

**Kinematic Equations**:
```
φ̇ = p + (q·sinφ + r·cosφ)·tanθ
θ̇ = q·cosφ - r·sinφ
ψ̇ = (q·sinφ + r·cosφ) / cosθ
```

### Aerodynamic Model

Stability derivative formulation:
- **Lift**: CL = CL0 + CLα·α + CLq·q̂ + CLδe·δe
- **Drag**: CD = CD0 + K·CL² (parabolic polar)
- **Pitch**: Cm = Cm0 + Cmα·α + Cmq·q̂ + Cmδe·δe
- **Side Force**: CY = CYβ·β + CYp·p̂ + CYr·r̂ + CYδr·δr
- **Roll**: Cl = Clβ·β + Clp·p̂ + Clr·r̂ + Clδa·δa
- **Yaw**: Cn = Cnβ·β + Cnp·p̂ + Cnr·r̂ + Cnδr·δr

### Atmosphere Model

International Standard Atmosphere (ISA):
- Troposphere (0-11km): T = 288.15 - 0.0065·h
- Stratosphere (11-20km): T = 216.65K (isothermal)

## 🏗️ Architecture

```
gpu-flight-dynamics/
├── cuda/
│   ├── flight_dynamics.h      # Public API header
│   ├── atmosphere.cu          # ISA atmosphere model
│   ├── aerodynamics.cu        # Force/moment computation
│   ├── equations_of_motion.cu # 6-DOF integration kernels
│   ├── flight_dynamics_api.cu # Host-side API
│   ├── benchmark.cu           # Performance benchmarks
│   ├── test_main.cu           # Unit tests
│   └── Makefile
├── python/
│   ├── flight_dynamics.py     # Python bindings
│   └── __init__.py
├── docs/
│   ├── PHYSICS.md             # Detailed physics documentation
│   └── architecture.png
├── tests/
│   └── test_physics.py
└── README.md
```

## 🧪 Testing

```bash
# Run CUDA tests
cd cuda
make test

# Run Python tests
cd python
pytest
```

## 📊 Benchmarking

```bash
# Run comprehensive benchmarks
cd cuda
make bench

# Python benchmarks
python -c "from flight_dynamics import benchmark; benchmark(10000, 1000)"
```

## 🔬 Reinforcement Learning Integration

Example PPO training loop with Stable-Baselines3:

```python
import gymnasium as gym
from stable_baselines3 import PPO
from flight_dynamics import FlightSimulator

class FlightEnv(gym.Env):
    def __init__(self, n_envs=1000):
        self.sim = FlightSimulator(n_instances=n_envs, use_gpu=True)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (12,))
        self.action_space = gym.spaces.Box(-1, 1, (4,))
    
    def step(self, actions):
        self.sim.set_controls(actions)
        self.sim.step()
        obs = self.sim.get_states()
        rewards = self._compute_rewards(obs)
        dones = self._check_termination(obs)
        return obs, rewards, dones, {}
    
    def reset(self):
        self.sim.reset()
        return self.sim.get_states()

# Train
env = FlightEnv(n_envs=1000)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

## 📚 References

1. Stevens, B. L., Lewis, F. L., & Johnson, E. N. (2015). *Aircraft Control and Simulation*. Wiley.
2. Etkin, B., & Reid, L. D. (1996). *Dynamics of Flight: Stability and Control*. Wiley.
3. NVIDIA CUDA Programming Guide

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**Kushal Koirala**
- GitHub: [@kushkoirala](https://github.com/kushkoirala)
- LinkedIn: [kushkoirala](https://linkedin.com/in/kushkoirala)
- Portfolio: [kushkoirala.github.io](https://kushkoirala.github.io)

---

*This project demonstrates GPU-accelerated simulation for aerospace applications, bridging domain expertise in flight dynamics with NVIDIA's compute platform.*
