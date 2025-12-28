"""
GPU-Accelerated Flight Dynamics Simulator

A high-performance 6-DOF flight dynamics simulator using NVIDIA CUDA.
"""

from flight_dynamics import (
    FlightSimulator,
    AircraftParams,
    MassProperties,
    Geometry,
    LongitudinalDerivatives,
    LateralDerivatives,
    PropulsionParams,
    StateIndex,
    ControlIndex,
    STATE_DIM,
    CONTROL_DIM,
    benchmark,
)

__version__ = "1.0.0"
__author__ = "Kushal Koirala"
__email__ = "kush.koirala@gmail.com"

__all__ = [
    "FlightSimulator",
    "AircraftParams",
    "MassProperties",
    "Geometry",
    "LongitudinalDerivatives",
    "LateralDerivatives",
    "PropulsionParams",
    "StateIndex",
    "ControlIndex",
    "STATE_DIM",
    "CONTROL_DIM",
    "benchmark",
]
