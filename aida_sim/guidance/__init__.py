"""
AIDA Guidance Module

Provides path following and navigation guidance:
- Vector Field guidance (Lyapunov-based)
- Pure Pursuit for ground track following
- Heading and altitude commands for autopilot
"""

from .vector_field import VectorFieldGuidance
from .pure_pursuit import PurePursuitGuidance
from .commands import GuidanceCommands

__all__ = [
    'VectorFieldGuidance',
    'PurePursuitGuidance',
    'GuidanceCommands',
]
