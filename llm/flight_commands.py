#!/usr/bin/env python3
"""
AIDA Flight Command System - LLM Integration

Translates natural language commands to structured flight commands
that can be executed by the expert controller.

Example:
    "Turn heading 25 degrees north" -> {"command": "SET_HEADING", "heading_deg": 25}
    "Climb to 5000 feet" -> {"command": "SET_ALTITUDE", "altitude_ft": 5000}
"""

import json
import re
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple


class CommandType(Enum):
    """Supported flight command types."""
    SET_HEADING = "SET_HEADING"
    SET_ALTITUDE = "SET_ALTITUDE"
    SET_AIRSPEED = "SET_AIRSPEED"
    SET_VERTICAL_SPEED = "SET_VS"
    TURN = "TURN"
    DIRECT_TO = "DIRECT_TO"
    SET_FLAPS = "SET_FLAPS"
    HOLD_CURRENT = "HOLD_CURRENT"
    GO_AROUND = "GO_AROUND"
    UNKNOWN = "UNKNOWN"


@dataclass
class FlightCommand:
    """Structured flight command."""
    command: CommandType
    heading_deg: Optional[float] = None
    altitude_ft: Optional[float] = None
    airspeed_kts: Optional[float] = None
    vs_fpm: Optional[float] = None
    turn_degrees: Optional[float] = None
    turn_direction: Optional[str] = None  # "left" or "right"
    waypoint: Optional[str] = None
    flap_position: Optional[float] = None
    confidence: float = 0.0
    raw_input: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {"command": self.command.value}
        for key, value in asdict(self).items():
            if value is not None and key != "command":
                if isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
        return result

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class FlightEnvelopeValidator:
    """
    Validates commands against Cessna 172 flight envelope.
    Ensures safety limits are not exceeded.
    """

    # Cessna 172 limits
    MIN_AIRSPEED_KTS = 45      # Stall speed (dirty)
    MAX_AIRSPEED_KTS = 140     # Vne
    SERVICE_CEILING_FT = 14000
    MIN_ALTITUDE_FT = 0
    MAX_CLIMB_RATE_FPM = 1000
    MAX_DESCENT_RATE_FPM = 1500
    MAX_BANK_ANGLE_DEG = 60

    def validate(self, command: FlightCommand, current_state: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate a command against flight envelope.

        Returns:
            (is_valid, message)
        """
        if command.altitude_ft is not None:
            if command.altitude_ft > self.SERVICE_CEILING_FT:
                return False, f"Altitude {command.altitude_ft}ft exceeds service ceiling ({self.SERVICE_CEILING_FT}ft)"
            if command.altitude_ft < self.MIN_ALTITUDE_FT:
                return False, f"Altitude cannot be negative"

        if command.airspeed_kts is not None:
            if command.airspeed_kts < self.MIN_AIRSPEED_KTS:
                return False, f"Airspeed {command.airspeed_kts}kts below stall speed ({self.MIN_AIRSPEED_KTS}kts)"
            if command.airspeed_kts > self.MAX_AIRSPEED_KTS:
                return False, f"Airspeed {command.airspeed_kts}kts exceeds Vne ({self.MAX_AIRSPEED_KTS}kts)"

        if command.vs_fpm is not None:
            if command.vs_fpm > self.MAX_CLIMB_RATE_FPM:
                return False, f"Climb rate {command.vs_fpm}fpm exceeds capability"
            if command.vs_fpm < -self.MAX_DESCENT_RATE_FPM:
                return False, f"Descent rate {abs(command.vs_fpm)}fpm too high"

        if command.heading_deg is not None:
            if not (0 <= command.heading_deg < 360):
                return False, f"Heading must be 0-359 degrees"

        return True, "Command validated"


class RuleBasedCommandParser:
    """
    Rule-based command parser for flight commands.
    Used as fallback when LLM is not available or for simple commands.
    """

    # Patterns for different command types
    HEADING_PATTERNS = [
        r"(?:turn|heading|hdg)\s*(?:to\s*)?(\d+)\s*(?:degrees?|deg|°)?",
        r"fly\s*heading\s*(\d+)",
        r"steer\s*(\d+)",
    ]

    ALTITUDE_PATTERNS = [
        r"(?:climb|descend|altitude|alt)\s*(?:to\s*)?(\d+)\s*(?:feet|ft)?",
        r"(?:climb|descend)\s*(?:to\s*)?(FL\d+)",  # Flight level
        r"maintain\s*(\d+)\s*(?:feet|ft)?",
    ]

    AIRSPEED_PATTERNS = [
        r"(?:speed|airspeed)\s*(?:to\s*)?(\d+)\s*(?:knots?|kts?)?",
        r"accelerate\s*(?:to\s*)?(\d+)",
        r"slow\s*(?:to\s*)?(\d+)",
    ]

    TURN_PATTERNS = [
        r"turn\s*(left|right)\s*(\d+)\s*(?:degrees?|deg|°)?",
        r"(\d+)\s*(?:degrees?|deg|°)?\s*(left|right)",
    ]

    VS_PATTERNS = [
        r"(?:vertical speed|vs|climb rate|descent rate)\s*(?:to\s*)?(-?\d+)",
        r"climb\s*at\s*(\d+)\s*(?:fpm|feet per minute)?",
        r"descend\s*at\s*(\d+)\s*(?:fpm|feet per minute)?",
    ]

    def parse(self, text: str) -> FlightCommand:
        """Parse natural language to flight command."""
        text = text.lower().strip()

        # Check for go-around
        if any(phrase in text for phrase in ["go around", "go-around", "missed approach"]):
            return FlightCommand(
                command=CommandType.GO_AROUND,
                confidence=0.95,
                raw_input=text
            )

        # Check for hold current
        if any(phrase in text for phrase in ["hold", "maintain current", "keep current"]):
            return FlightCommand(
                command=CommandType.HOLD_CURRENT,
                confidence=0.9,
                raw_input=text
            )

        # Try turn patterns first (more specific)
        for pattern in self.TURN_PATTERNS:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                # Handle different group orderings
                if groups[0] in ["left", "right"]:
                    direction, degrees = groups[0], float(groups[1])
                else:
                    degrees, direction = float(groups[0]), groups[1]
                return FlightCommand(
                    command=CommandType.TURN,
                    turn_degrees=degrees,
                    turn_direction=direction,
                    confidence=0.9,
                    raw_input=text
                )

        # Try heading patterns
        for pattern in self.HEADING_PATTERNS:
            match = re.search(pattern, text)
            if match:
                heading = float(match.group(1)) % 360
                return FlightCommand(
                    command=CommandType.SET_HEADING,
                    heading_deg=heading,
                    confidence=0.9,
                    raw_input=text
                )

        # Try altitude patterns
        for pattern in self.ALTITUDE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                alt_str = match.group(1)
                if alt_str.startswith("FL"):
                    altitude = float(alt_str[2:]) * 100
                else:
                    altitude = float(alt_str)
                return FlightCommand(
                    command=CommandType.SET_ALTITUDE,
                    altitude_ft=altitude,
                    confidence=0.85,
                    raw_input=text
                )

        # Try airspeed patterns
        for pattern in self.AIRSPEED_PATTERNS:
            match = re.search(pattern, text)
            if match:
                speed = float(match.group(1))
                return FlightCommand(
                    command=CommandType.SET_AIRSPEED,
                    airspeed_kts=speed,
                    confidence=0.85,
                    raw_input=text
                )

        # Try vertical speed patterns
        for pattern in self.VS_PATTERNS:
            match = re.search(pattern, text)
            if match:
                vs = float(match.group(1))
                # If "descend" in text, make negative
                if "descend" in text and vs > 0:
                    vs = -vs
                return FlightCommand(
                    command=CommandType.SET_VERTICAL_SPEED,
                    vs_fpm=vs,
                    confidence=0.85,
                    raw_input=text
                )

        # Check for flaps
        flap_match = re.search(r"flaps?\s*(?:to\s*)?(\d+|full|up)", text)
        if flap_match:
            flap_val = flap_match.group(1)
            if flap_val == "full":
                flap_position = 1.0
            elif flap_val == "up":
                flap_position = 0.0
            else:
                flap_position = min(float(flap_val) / 40.0, 1.0)  # C172 max flaps 40 deg
            return FlightCommand(
                command=CommandType.SET_FLAPS,
                flap_position=flap_position,
                confidence=0.85,
                raw_input=text
            )

        # Unknown command
        return FlightCommand(
            command=CommandType.UNKNOWN,
            confidence=0.0,
            raw_input=text
        )


# System prompt for LLM-based parsing
LLM_SYSTEM_PROMPT = """You are a flight command parser for a Cessna 172 aircraft.
Convert natural language pilot commands to JSON format.

IMPORTANT: Respond with ONLY valid JSON, no explanation or other text.

Valid command types and their JSON formats:

1. SET_HEADING - Set a specific heading
   {"command": "SET_HEADING", "heading_deg": <0-359>}

2. SET_ALTITUDE - Climb or descend to altitude
   {"command": "SET_ALTITUDE", "altitude_ft": <number>}

3. SET_AIRSPEED - Set target airspeed
   {"command": "SET_AIRSPEED", "airspeed_kts": <50-140>}

4. SET_VS - Set vertical speed
   {"command": "SET_VS", "vs_fpm": <-1500 to 1000>}

5. TURN - Turn left or right by degrees
   {"command": "TURN", "turn_degrees": <number>, "turn_direction": "left"|"right"}

6. SET_FLAPS - Set flap position
   {"command": "SET_FLAPS", "flap_position": <0.0-1.0>}

7. GO_AROUND - Execute go-around/missed approach
   {"command": "GO_AROUND"}

8. HOLD_CURRENT - Maintain current flight parameters
   {"command": "HOLD_CURRENT"}

Examples:
User: Turn heading 270
JSON: {"command": "SET_HEADING", "heading_deg": 270}

User: Climb to 5000 feet
JSON: {"command": "SET_ALTITUDE", "altitude_ft": 5000}

User: Turn left 30 degrees
JSON: {"command": "TURN", "turn_degrees": 30, "turn_direction": "left"}

User: Slow to 80 knots
JSON: {"command": "SET_AIRSPEED", "airspeed_kts": 80}
"""


class LLMCommandParser:
    """
    LLM-based command parser using llama-cpp-python.
    Falls back to rule-based parser if LLM fails.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.llm = None
        self.fallback_parser = RuleBasedCommandParser()
        self.validator = FlightEnvelopeValidator()

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the LLM model."""
        try:
            from llama_cpp import Llama

            print(f"Loading LLM from {model_path}...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=-1,  # Use all GPU layers
                verbose=False
            )
            print("LLM loaded successfully")
        except ImportError:
            print("llama-cpp-python not installed. Using rule-based parser only.")
        except Exception as e:
            print(f"Failed to load LLM: {e}. Using rule-based parser only.")

    def parse(self, text: str, use_llm: bool = True) -> FlightCommand:
        """
        Parse natural language to flight command.

        Args:
            text: Natural language command
            use_llm: Whether to use LLM (falls back to rules if unavailable)

        Returns:
            FlightCommand object
        """
        # Try LLM first if available
        if use_llm and self.llm is not None:
            try:
                return self._parse_with_llm(text)
            except Exception as e:
                print(f"LLM parsing failed: {e}, falling back to rules")

        # Fallback to rule-based
        return self.fallback_parser.parse(text)

    def _parse_with_llm(self, text: str) -> FlightCommand:
        """Parse using the LLM."""
        prompt = f"{LLM_SYSTEM_PROMPT}\n\nUser: {text}\nJSON:"

        response = self.llm(
            prompt,
            max_tokens=150,
            temperature=0.1,
            stop=["\n\n", "User:"]
        )

        json_str = response['choices'][0]['text'].strip()

        # Parse JSON response
        data = json.loads(json_str)

        # Convert to FlightCommand
        cmd_type = CommandType(data.get("command", "UNKNOWN"))

        return FlightCommand(
            command=cmd_type,
            heading_deg=data.get("heading_deg"),
            altitude_ft=data.get("altitude_ft"),
            airspeed_kts=data.get("airspeed_kts"),
            vs_fpm=data.get("vs_fpm"),
            turn_degrees=data.get("turn_degrees"),
            turn_direction=data.get("turn_direction"),
            waypoint=data.get("waypoint"),
            flap_position=data.get("flap_position"),
            confidence=0.95,
            raw_input=text
        )

    def validate_command(self, command: FlightCommand, current_state: Dict[str, float]) -> Tuple[bool, str]:
        """Validate a command against flight envelope."""
        return self.validator.validate(command, current_state)


# Test function
def test_parser():
    """Test the rule-based parser with sample commands."""
    parser = RuleBasedCommandParser()

    test_commands = [
        "Turn heading 270",
        "Climb to 5000 feet",
        "Turn left 30 degrees",
        "Descend to 3000",
        "Speed 90 knots",
        "Go around",
        "Heading 045",
        "Set flaps to 20",
        "Vertical speed 500 fpm",
        "Turn right 45 degrees",
    ]

    print("Testing Rule-Based Parser:")
    print("=" * 50)

    for cmd in test_commands:
        result = parser.parse(cmd)
        print(f"\nInput: '{cmd}'")
        print(f"Result: {result.to_dict()}")


if __name__ == "__main__":
    test_parser()
