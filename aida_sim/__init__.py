"""
AIDA - Autonomous Intelligent Decision Architecture

A modular framework for autonomous fixed-wing aircraft control.

Packages:
- control: PID controllers and autopilot
- guidance: Vector field and pure pursuit path following
- trajectory: Dubins paths and trajectory planning
- dynamics: Flight physics
- env: Gymnasium RL environments
- io: Telemetry and data logging
- platform: Airports and world data
- planning: Mission planning
"""

__version__ = "1.1.0"

# Core modules
from . import dynamics
from . import env
from . import io
from . import platform
from . import planning

# New trajectory architecture (V1.1)
from . import control
from . import guidance
from . import trajectory
