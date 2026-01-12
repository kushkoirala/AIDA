#!/usr/bin/env python3
"""
AIDA Command Executor - Bridges LLM commands to flight controller.

Takes structured FlightCommand objects and translates them into
target states that the expert controller can track.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from flight_commands import FlightCommand, CommandType, FlightEnvelopeValidator


class ExecutorMode(Enum):
    """Executor operating modes."""
    AUTONOMOUS = "autonomous"  # Following pre-programmed route
    MANUAL = "manual"          # Following LLM/pilot commands
    MIXED = "mixed"            # Autonomous with manual overrides


@dataclass
class FlightTargets:
    """Target states for the flight controller to track."""
    heading_deg: Optional[float] = None      # Target heading (None = maintain current)
    altitude_ft: Optional[float] = None      # Target altitude
    airspeed_kts: Optional[float] = None     # Target airspeed
    vs_fpm: Optional[float] = None           # Target vertical speed
    flap_position: Optional[float] = None    # Flap setting (0-1)

    # Flags
    go_around: bool = False
    hold_current: bool = False

    # Metadata
    source: str = "autonomous"  # "autonomous" or "llm"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "heading_deg": self.heading_deg,
            "altitude_ft": self.altitude_ft,
            "airspeed_kts": self.airspeed_kts,
            "vs_fpm": self.vs_fpm,
            "flap_position": self.flap_position,
            "go_around": self.go_around,
            "hold_current": self.hold_current,
            "source": self.source,
        }


@dataclass
class CommandLogEntry:
    """Log entry for executed commands."""
    timestamp: float
    command: FlightCommand
    targets_before: FlightTargets
    targets_after: FlightTargets
    validated: bool
    validation_message: str


class FlightCommandExecutor:
    """
    Executes flight commands by updating target states.

    The executor maintains a set of target states that the expert
    controller should track. When LLM commands come in, it updates
    these targets accordingly.
    """

    def __init__(self):
        self.validator = FlightEnvelopeValidator()
        self.mode = ExecutorMode.AUTONOMOUS

        # Current targets (initialized to None = follow autonomous controller)
        self.targets = FlightTargets()

        # Command history for logging/debugging
        self.command_log: List[CommandLogEntry] = []
        self.max_log_entries = 100

        # Current aircraft state (updated externally)
        self.current_state: Dict[str, float] = {
            "heading_deg": 0.0,
            "altitude_ft": 0.0,
            "airspeed_kts": 0.0,
            "vs_fpm": 0.0,
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
        }

    def update_state(self, state: Dict[str, float]):
        """Update current aircraft state from simulation."""
        self.current_state.update(state)

    def execute(self, command: FlightCommand) -> tuple[bool, str, FlightTargets]:
        """
        Execute a flight command.

        Args:
            command: The FlightCommand to execute

        Returns:
            (success, message, new_targets)
        """
        # Validate command
        is_valid, validation_msg = self.validator.validate(command, self.current_state)

        if not is_valid:
            return False, validation_msg, self.targets

        # Store previous targets for logging
        targets_before = FlightTargets(**self.targets.__dict__)

        # Execute based on command type
        if command.command == CommandType.SET_HEADING:
            self.targets.heading_deg = command.heading_deg
            self.targets.source = "llm"
            self.mode = ExecutorMode.MANUAL
            msg = f"Setting heading to {command.heading_deg:.0f}°"

        elif command.command == CommandType.SET_ALTITUDE:
            self.targets.altitude_ft = command.altitude_ft
            self.targets.source = "llm"
            self.mode = ExecutorMode.MANUAL
            msg = f"Setting altitude to {command.altitude_ft:.0f} ft"

        elif command.command == CommandType.SET_AIRSPEED:
            self.targets.airspeed_kts = command.airspeed_kts
            self.targets.source = "llm"
            self.mode = ExecutorMode.MANUAL
            msg = f"Setting airspeed to {command.airspeed_kts:.0f} kts"

        elif command.command == CommandType.SET_VERTICAL_SPEED:
            self.targets.vs_fpm = command.vs_fpm
            self.targets.source = "llm"
            self.mode = ExecutorMode.MANUAL
            msg = f"Setting vertical speed to {command.vs_fpm:.0f} fpm"

        elif command.command == CommandType.TURN:
            # Calculate new heading from turn
            current_hdg = self.current_state.get("heading_deg", 0)
            delta = command.turn_degrees
            if command.turn_direction == "left":
                delta = -delta
            new_heading = (current_hdg + delta) % 360
            self.targets.heading_deg = new_heading
            self.targets.source = "llm"
            self.mode = ExecutorMode.MANUAL
            msg = f"Turning {command.turn_direction} {command.turn_degrees:.0f}° to heading {new_heading:.0f}°"

        elif command.command == CommandType.SET_FLAPS:
            self.targets.flap_position = command.flap_position
            self.targets.source = "llm"
            msg = f"Setting flaps to {command.flap_position * 100:.0f}%"

        elif command.command == CommandType.GO_AROUND:
            self.targets.go_around = True
            self.targets.source = "llm"
            self.mode = ExecutorMode.MANUAL
            msg = "Executing go-around"

        elif command.command == CommandType.HOLD_CURRENT:
            # Lock current values as targets
            self.targets.heading_deg = self.current_state.get("heading_deg")
            self.targets.altitude_ft = self.current_state.get("altitude_ft")
            self.targets.airspeed_kts = self.current_state.get("airspeed_kts")
            self.targets.hold_current = True
            self.targets.source = "llm"
            self.mode = ExecutorMode.MANUAL
            msg = "Holding current flight parameters"

        elif command.command == CommandType.UNKNOWN:
            return False, f"Unknown command: {command.raw_input}", self.targets

        else:
            return False, f"Unhandled command type: {command.command}", self.targets

        # Log the command
        self._log_command(command, targets_before, self.targets, True, msg)

        return True, msg, self.targets

    def clear_targets(self):
        """Clear all manual targets and return to autonomous mode."""
        self.targets = FlightTargets()
        self.mode = ExecutorMode.AUTONOMOUS
        return "Cleared all targets, returning to autonomous mode"

    def get_targets(self) -> FlightTargets:
        """Get current flight targets."""
        return self.targets

    def get_mode(self) -> ExecutorMode:
        """Get current executor mode."""
        return self.mode

    def _log_command(self, command: FlightCommand, before: FlightTargets,
                     after: FlightTargets, validated: bool, message: str):
        """Log a command execution."""
        entry = CommandLogEntry(
            timestamp=time.time(),
            command=command,
            targets_before=before,
            targets_after=after,
            validated=validated,
            validation_message=message
        )
        self.command_log.append(entry)

        # Trim log if too long
        if len(self.command_log) > self.max_log_entries:
            self.command_log = self.command_log[-self.max_log_entries:]

    def get_recent_commands(self, n: int = 10) -> List[CommandLogEntry]:
        """Get the n most recent commands."""
        return self.command_log[-n:]


class TargetTrackingController:
    """
    Modifies expert controller outputs to track LLM-specified targets.

    This sits between the expert controller and the final output,
    adjusting commands to achieve the targets set by the executor.
    """

    def __init__(self, executor: FlightCommandExecutor):
        self.executor = executor

        # PID-like gains for target tracking
        self.heading_kp = 0.05
        self.altitude_kp = 0.002
        self.airspeed_kp = 0.01
        self.vs_kp = 0.001

    def modify_expert_action(self, expert_action: np.ndarray,
                              current_state: Dict[str, float]) -> np.ndarray:
        """
        Modify expert action to track LLM targets if set.

        Args:
            expert_action: [throttle, aileron, elevator, rudder, flaps, spoilers, brakes]
            current_state: Current aircraft state

        Returns:
            Modified action array
        """
        targets = self.executor.get_targets()
        mode = self.executor.get_mode()

        # If autonomous mode, return expert action unchanged
        if mode == ExecutorMode.AUTONOMOUS:
            return expert_action

        action = expert_action.copy()

        # Heading tracking (affects aileron/rudder)
        if targets.heading_deg is not None:
            current_hdg = current_state.get("heading_deg", 0)
            hdg_error = self._normalize_heading_error(targets.heading_deg - current_hdg)

            # Simple proportional control for bank angle
            target_bank = np.clip(hdg_error * self.heading_kp, -0.5, 0.5)
            action[1] = np.clip(action[1] + target_bank, -1, 1)  # aileron
            action[3] = np.clip(action[3] + target_bank * 0.3, -1, 1)  # rudder (coordinated)

        # Altitude tracking (affects elevator/throttle)
        if targets.altitude_ft is not None:
            current_alt = current_state.get("altitude_ft", 0)
            alt_error = targets.altitude_ft - current_alt

            # Pitch adjustment for altitude
            pitch_adj = np.clip(alt_error * self.altitude_kp, -0.3, 0.3)
            action[2] = np.clip(action[2] - pitch_adj, -1, 1)  # elevator (negative = pitch up)

            # Throttle adjustment
            if alt_error > 100:  # Climbing
                action[0] = min(action[0] + 0.1, 1.0)
            elif alt_error < -100:  # Descending
                action[0] = max(action[0] - 0.1, 0.0)

        # Airspeed tracking (affects throttle)
        if targets.airspeed_kts is not None:
            current_spd = current_state.get("airspeed_kts", 0)
            spd_error = targets.airspeed_kts - current_spd

            throttle_adj = np.clip(spd_error * self.airspeed_kp, -0.2, 0.2)
            action[0] = np.clip(action[0] + throttle_adj, 0, 1)

        # Flap position
        if targets.flap_position is not None:
            action[4] = targets.flap_position

        # Go-around
        if targets.go_around:
            action[0] = 1.0  # Full throttle
            action[2] = -0.3  # Pitch up
            action[4] = 0.0  # Flaps up (gradually)
            action[5] = 0.0  # Spoilers retracted

        return action

    def _normalize_heading_error(self, error: float) -> float:
        """Normalize heading error to -180 to 180 range."""
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        return error


# Test function
def test_executor():
    """Test the command executor."""
    from flight_commands import RuleBasedCommandParser

    parser = RuleBasedCommandParser()
    executor = FlightCommandExecutor()

    # Simulate some state
    executor.update_state({
        "heading_deg": 90,
        "altitude_ft": 3000,
        "airspeed_kts": 100,
        "vs_fpm": 0,
    })

    test_commands = [
        "Turn heading 180",
        "Climb to 5000 feet",
        "Turn left 45 degrees",
        "Speed 80 knots",
    ]

    print("Testing Command Executor:")
    print("=" * 60)

    for cmd_text in test_commands:
        print(f"\nInput: '{cmd_text}'")

        # Parse command
        command = parser.parse(cmd_text)
        print(f"Parsed: {command.command.value}")

        # Execute command
        success, message, targets = executor.execute(command)
        print(f"Result: {message}")
        print(f"Targets: {targets.to_dict()}")


if __name__ == "__main__":
    test_executor()
