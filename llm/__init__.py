"""
AIDA LLM Integration Module

Provides natural language command interface for flight control.
"""

from .flight_commands import (
    FlightCommand,
    CommandType,
    FlightEnvelopeValidator,
    RuleBasedCommandParser,
    LLMCommandParser,
    LLM_SYSTEM_PROMPT,
)

from .command_executor import (
    FlightCommandExecutor,
    FlightTargets,
    ExecutorMode,
    TargetTrackingController,
)

__all__ = [
    "FlightCommand",
    "CommandType",
    "FlightEnvelopeValidator",
    "RuleBasedCommandParser",
    "LLMCommandParser",
    "LLM_SYSTEM_PROMPT",
    "FlightCommandExecutor",
    "FlightTargets",
    "ExecutorMode",
    "TargetTrackingController",
]
