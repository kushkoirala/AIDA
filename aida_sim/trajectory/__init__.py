"""
AIDA Trajectory Module

Provides trajectory planning and generation:
- Dubins paths for minimum-radius turns
- B-spline smoothing for continuous curvature
- Vertical profile generation
- Complete 4D trajectory planning
"""

from .dubins import DubinsPath, DubinsPathType, compute_dubins_path
from .trajectory import Trajectory, TrajectoryPoint
from .planner import TrajectoryPlanner

__all__ = [
    'DubinsPath',
    'DubinsPathType',
    'compute_dubins_path',
    'Trajectory',
    'TrajectoryPoint',
    'TrajectoryPlanner',
]
