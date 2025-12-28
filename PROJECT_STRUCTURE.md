# AIDA Project Structure

Updated: December 27, 2024

## Directory Organization

```
/home/AIDA/
├── aida_sim/              # Core simulation package
│   ├── dynamics/          # Flight dynamics and physics
│   ├── env/               # Gymnasium RL environments
│   ├── io/                # Telemetry and I/O
│   ├── platform/          # Bullet physics integration
│   ├── safety/            # Safety guards and limits
│   └── systems/           # Aircraft subsystems (battery, etc.)
│
├── assets/                # Project assets
│   ├── aircraft/          # 3D models (Cessna 172, Udaan)
│   └── papers/            # Research papers and references
│
├── bc_data/               # Behavior cloning datasets (NPZ)
│
├── checkpoints/           # Trained model checkpoints
│   ├── cessna172_curriculum/  # Cessna 172 curriculum learning
│   └── runs/              # Other training runs
│
├── config/                # Configuration files
│   └── runway_config.json
│
├── data/                  # Training data and outputs
│   └── visualizations/    # Plots and charts
│
├── docs/                  # Documentation
│   ├── dev/               # Development notes
│   ├── img/               # Images for documentation
│   ├── reference/         # Reference materials (PDFs, data)
│   ├── session_summaries/ # Training session summaries
│   └── PHASE1_TRAINING_SESSION_SUMMARY.md
│
├── gpu-flight-dynamics/   # GPU-accelerated CUDA simulator
│   ├── benchmarks/        # Performance benchmarks
│   └── python/            # Python bindings
│
├── logs/                  # Training logs and tensorboard data
│
├── scripts/               # Python scripts
│   ├── utils/             # Shell utility scripts
│   ├── train_*.py         # Training scripts
│   ├── test_*.py          # Test scripts
│   └── generate_*.py      # Data generation scripts
│
├── scripts_archive/       # Archived/old scripts
│
├── test_scripts/          # Development test scripts
│
├── viewer/                # 3D visualization viewer
│
├── .venv-linux/           # Python virtual environment
├── .git/                  # Git repository
├── .gitignore
├── README.md              # Main project README
├── requirements.txt       # Python dependencies
└── PROJECT_STRUCTURE.md   # This file
```

## Key Files

### Root Level
- `README.md` - Main project documentation
- `requirements.txt` - Python package dependencies

### Configuration
- `config/runway_config.json` - Runway parameters for takeoff simulations

### Core Simulation
- `aida_sim/env/flight_env_cessna172.py` - Cessna 172 RL environment with curriculum learning
- `aida_sim/env/flight_env_rl.py` - General RL flight environment
- `aida_sim/dynamics/` - Flight dynamics implementation
- `gpu-flight-dynamics/` - High-performance GPU simulator

### Training & Evaluation
- `scripts/train_cessna172_curriculum.py` - Curriculum learning training
- `scripts/test_phase1_ground_roll.py` - Phase 1 evaluation
- `checkpoints/cessna172_curriculum/` - Trained models and tensorboard logs

### Assets
- `assets/aircraft/Cessna172.gltf` - Cessna 172 3D model
- `assets/aircraft/Udaan-Product4.gltf` - Udaan aircraft 3D model
- `assets/papers/` - Research papers

### Documentation
- `docs/PHASE1_TRAINING_SESSION_SUMMARY.md` - Latest training session summary
- `docs/session_summaries/` - Historical session documentation
- `docs/reference/` - Reference materials and data

## Data Flow

```
┌─────────────────────┐
│  GPU Flight Sim     │
│  (CUDA Parallel)    │
└──────┬──────────────┘
       │ 500k steps/5s
       ▼
┌─────────────────────┐
│  BC Dataset (NPZ)   │
│  Expert Trajectories│
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐       ┌──────────────┐
│  AIDA RL Env        │◄──────┤ PPO Training │
│  (Gymnasium)        │       └──────────────┘
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Checkpoints/       │
│  Trained Models     │
└─────────────────────┘
```

## Training Checkpoints

Current best models:
- **Cessna 172 Phase 1 (Ground Roll)**:
  - `checkpoints/cessna172_curriculum/phase1_ground_roll/best_model.zip` (60k steps, -187 eval reward)
  - `checkpoints/cessna172_curriculum/phase1_ground_roll/phase1_ground_roll_ppo_final.zip` (200k steps)

## Development Workflow

1. **Setup**: Use Python venv in `.venv-linux/`
2. **Training**: Run scripts from `scripts/` directory
3. **Monitoring**: TensorBoard logs in `logs/` and `checkpoints/*/tensorboard/`
4. **Evaluation**: Test scripts in `scripts/test_*.py`
5. **Visualization**: Use `scripts/utils/run_cessna172_viewer.sh`

## Important Notes

- Scripts should be run from `/home/AIDA/` root directory
- Virtual environment: `.venv-linux/` (Linux/WSL)
- GPU acceleration: CUDA for flight sim, CUDA for PyTorch training
- TensorBoard: Accessible on port 6006
- Telemetry viewer: Accessible on port 8000
