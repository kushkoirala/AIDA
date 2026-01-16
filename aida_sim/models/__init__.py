"""
AIDA Neural Network Models

Contains transformer architectures for:
- NLP command parsing
- Trajectory prediction
- Imitation learning (behavioral cloning)
- Reinforcement learning (PPO)
"""

from .unified_transformer import (
    UnifiedTransformer,
    ModelConfig,
    TaskType,
    SimpleTokenizer,
    CommandLoss,
    TrajectoryLoss,
    ImitationLoss,
)

__all__ = [
    "UnifiedTransformer",
    "ModelConfig",
    "TaskType",
    "SimpleTokenizer",
    "CommandLoss",
    "TrajectoryLoss",
    "ImitationLoss",
]
