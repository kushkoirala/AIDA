"""
AIDA Control Module

Provides autopilot and low-level control functionality:
- PID controllers for attitude and heading
- Unified autopilot that follows trajectories
- Airspeed management
"""

from .pid import PIDController, AltitudeController, HeadingController, AirspeedController
from .autopilot import TrajectoryAutopilot

__all__ = [
    'PIDController',
    'AltitudeController',
    'HeadingController',
    'AirspeedController',
    'TrajectoryAutopilot',
]
