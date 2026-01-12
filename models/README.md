# AIDA Trained Models

This folder contains trained neural network weights for the AIDA flight controller.

## Models

| File | Description | Controls | Size |
|------|-------------|----------|------|
| `residual_ppo_v2_final.zip` | Residual RL policy (AI+Expert) | 7 | ~9MB |

## Usage

```python
from stable_baselines3 import PPO

# Load the model
model = PPO.load("models/residual_ppo_v2_final.zip")

# Get action for observation
action, _ = model.predict(obs, deterministic=True)
```

## Notes

- Intermediate training checkpoints are in `checkpoints/` (not tracked)
- LLM models (*.gguf) are in `llm/models/` (not tracked - too large, download separately)
