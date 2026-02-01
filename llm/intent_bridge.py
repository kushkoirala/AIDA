"""
Intent Bridge - Converts LLM commands to validated FlightIntent objects

This module bridges the existing FlightCommand system with the new
FlightIntent schema, providing a migration path to full intent-based control.

Architecture:
    FlightCommand (legacy) → IntentBridge → FlightIntent (new) → Validator
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from flight_intent import (
    FlightIntent, MissionPhase, Priority, GoalType, Goal, Constraint,
    ConstraintType, Handover, FallbackMode, Waypoint,
    create_altitude_intent, create_heading_intent, create_land_intent,
    create_divert_intent, create_emergency_intent, create_return_to_cruise_intent
)
from intent_validator import (
    IntentValidator, ValidationResult, ValidationStatus,
    ConfidenceScore, is_intent_safe
)


# =============================================================================
# Flight Context for Intent Generation
# =============================================================================

@dataclass
class FlightContext:
    """Current flight state for context-aware intent generation"""
    # Position
    lat: float = 0.0
    lon: float = 0.0
    altitude_ft: float = 0.0
    heading_deg: float = 0.0
    airspeed_kts: float = 0.0

    # Flight plan
    origin: str = ""
    destination: str = ""
    cruise_altitude_ft: float = 5500.0
    distance_to_dest_nm: float = 0.0

    # Phase
    phase: str = "CRUISE"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlightContext":
        return cls(
            lat=data.get("lat", 0.0),
            lon=data.get("lon", 0.0),
            altitude_ft=data.get("altitude", data.get("altitude_ft", 0.0)),
            heading_deg=data.get("heading", data.get("heading_deg", 0.0)),
            airspeed_kts=data.get("airspeed", data.get("airspeed_kts", 0.0)),
            origin=data.get("origin", ""),
            destination=data.get("destination", ""),
            cruise_altitude_ft=data.get("cruise_altitude", data.get("cruise_altitude_ft", 5500.0)),
            distance_to_dest_nm=data.get("distance", data.get("distance_nm", 0.0)),
            phase=data.get("phase", "CRUISE")
        )


# =============================================================================
# Intent Bridge
# =============================================================================

class IntentBridge:
    """
    Bridges legacy FlightCommand to new FlightIntent system.

    This allows gradual migration from action-based commands to
    intent-based control with full validation and confidence scoring.
    """

    # Map phase strings to MissionPhase enum
    PHASE_MAP = {
        "STARTUP": MissionPhase.STARTUP,
        "TAXI": MissionPhase.TAXI,
        "TAKEOFF": MissionPhase.TAKEOFF,
        "CLIMB": MissionPhase.CLIMB,
        "CRUISE": MissionPhase.CRUISE,
        "DESCENT": MissionPhase.DESCENT,
        "APPROACH": MissionPhase.APPROACH,
        "FINAL_APPROACH": MissionPhase.APPROACH,
        "LANDING": MissionPhase.LANDING,
        "FLARE": MissionPhase.LANDING,
        "ROLLOUT": MissionPhase.LANDING,
        "GO_AROUND": MissionPhase.GO_AROUND,
        "EMERGENCY": MissionPhase.EMERGENCY,
        "LANDED": MissionPhase.LANDED,
        "UNKNOWN": MissionPhase.CRUISE,  # Default to cruise
    }

    def __init__(self):
        self.validator = IntentValidator()
        self._last_context = FlightContext()

    def update_context(self, context: Dict[str, Any]):
        """Update the flight context from telemetry"""
        self._last_context = FlightContext.from_dict(context)

    def get_mission_phase(self, phase_str: str) -> MissionPhase:
        """Convert phase string to MissionPhase enum"""
        return self.PHASE_MAP.get(phase_str.upper(), MissionPhase.CRUISE)

    def command_to_intent(
        self,
        action: str,
        value: Optional[float] = None,
        target: Optional[str] = None,
        raw_text: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[FlightIntent], ValidationResult]:
        """
        Convert a legacy FlightCommand to a validated FlightIntent.

        Args:
            action: Command action ("heading", "altitude", "land", etc.)
            value: Numeric value (heading degrees, altitude feet)
            target: Target identifier (airport code, waypoint name)
            raw_text: Original natural language input
            context: Optional flight context dict

        Returns:
            Tuple of (FlightIntent or None, ValidationResult)
        """
        # Update context if provided
        if context:
            self.update_context(context)

        # Get current mission phase
        current_phase = self.get_mission_phase(self._last_context.phase)

        # Create intent based on action type
        intent = None

        if action == "heading":
            intent = self._create_heading_intent(
                value, current_phase, raw_text
            )

        elif action == "altitude":
            intent = self._create_altitude_intent(
                value, current_phase, raw_text
            )

        elif action == "land":
            intent = self._create_land_intent(
                target, current_phase, raw_text
            )

        elif action == "return_to_cruise":
            intent = self._create_return_to_cruise_intent(raw_text)

        elif action == "divert":
            intent = self._create_divert_intent(
                target, raw_text
            )

        elif action == "emergency":
            intent = self._create_emergency_intent(
                target, raw_text
            )

        # Validate the intent
        if intent:
            intent.raw_input = raw_text
            result = self.validator.validate_intent(intent)
            return intent, result

        # No intent created - return invalid result
        result = ValidationResult(status=ValidationStatus.INVALID)
        return None, result

    def _create_heading_intent(
        self,
        heading: float,
        current_phase: MissionPhase,
        explanation: str
    ) -> FlightIntent:
        """Create a heading change intent"""
        return FlightIntent(
            mission_phase=current_phase,
            priority=Priority.NORMAL,
            goals=[Goal(
                type=GoalType.MAINTAIN_HEADING,
                target_heading_deg=heading
            )],
            constraints=[
                Constraint(type=ConstraintType.MAX_BANK_DEG, value=30),
            ],
            explanation=explanation or f"Turn to heading {heading:03.0f}",
            source="llm"
        )

    def _create_altitude_intent(
        self,
        altitude: float,
        current_phase: MissionPhase,
        explanation: str
    ) -> FlightIntent:
        """Create an altitude change intent"""
        current_alt = self._last_context.altitude_ft

        # Determine if this is a climb or descent
        if altitude > current_alt:
            # Climbing
            constraints = [
                Constraint(type=ConstraintType.MAX_CLIMB_RATE_FPM, value=700),
                Constraint(type=ConstraintType.MIN_SPEED_KT, value=60),  # Vy consideration
            ]
        else:
            # Descending
            constraints = [
                Constraint(type=ConstraintType.MAX_DESCENT_RATE_FPM, value=800),
                Constraint(type=ConstraintType.MAX_SPEED_KT, value=140),  # Avoid overspeed
            ]

        # Add standard altitude limits
        constraints.extend([
            Constraint(type=ConstraintType.MIN_ALTITUDE_FT, value=1500),
            Constraint(type=ConstraintType.MAX_ALTITUDE_FT, value=12000),
        ])

        return FlightIntent(
            mission_phase=current_phase,
            priority=Priority.NORMAL,
            goals=[Goal(
                type=GoalType.MAINTAIN_ALTITUDE,
                target_altitude_ft=altitude
            )],
            constraints=constraints,
            explanation=explanation or f"Change altitude to {altitude:,.0f} ft",
            source="llm"
        )

    def _create_land_intent(
        self,
        airport_code: str,
        current_phase: MissionPhase,
        explanation: str
    ) -> FlightIntent:
        """Create a landing intent"""
        # Determine priority based on current situation
        priority = Priority.NORMAL
        if current_phase == MissionPhase.EMERGENCY:
            priority = Priority.EMERGENCY

        return FlightIntent(
            mission_phase=current_phase,
            priority=priority,
            goals=[Goal(
                type=GoalType.LAND_AT,
                target_airport=airport_code.upper() if airport_code else "KHUT"
            )],
            constraints=[
                Constraint(type=ConstraintType.MIN_SPEED_KT, value=48),  # Vs0
                Constraint(type=ConstraintType.MAX_SPEED_KT, value=100),  # Approach speed
                Constraint(type=ConstraintType.MAX_BANK_DEG, value=25),  # Gentler in pattern
            ],
            handover=Handover(
                fallback_mode=FallbackMode.AUTOPILOT,
                reason="Landing requires precise control"
            ),
            explanation=explanation or f"Land at {airport_code}",
            source="llm"
        )

    def _create_return_to_cruise_intent(self, explanation: str) -> FlightIntent:
        """Create a return to cruise intent"""
        cruise_alt = self._last_context.cruise_altitude_ft

        return FlightIntent(
            mission_phase=MissionPhase.CRUISE,
            priority=Priority.NORMAL,
            goals=[Goal(
                type=GoalType.RETURN_TO_CRUISE,
                target_altitude_ft=cruise_alt
            )],
            constraints=[
                Constraint(type=ConstraintType.MAX_CLIMB_RATE_FPM, value=700),
                Constraint(type=ConstraintType.MAX_DESCENT_RATE_FPM, value=700),
            ],
            explanation=explanation or f"Return to cruise at {cruise_alt:,.0f} ft",
            source="llm"
        )

    def _create_divert_intent(
        self,
        airport_code: str,
        explanation: str
    ) -> FlightIntent:
        """Create a diversion intent"""
        return FlightIntent(
            mission_phase=MissionPhase.CRUISE,
            priority=Priority.HIGH,
            goals=[Goal(
                type=GoalType.DIVERT_TO,
                target_airport=airport_code.upper() if airport_code else "KHUT"
            )],
            constraints=[
                Constraint(type=ConstraintType.MIN_ALTITUDE_FT, value=1500),
            ],
            handover=Handover(
                fallback_mode=FallbackMode.HOLD_CURRENT,
                reason="Diverting to alternate airport"
            ),
            explanation=explanation or f"Divert to {airport_code}",
            source="llm"
        )

    def _create_emergency_intent(
        self,
        nearest_airport: str,
        explanation: str
    ) -> FlightIntent:
        """Create an emergency intent"""
        return FlightIntent(
            mission_phase=MissionPhase.EMERGENCY,
            priority=Priority.EMERGENCY,
            goals=[Goal(
                type=GoalType.DIVERT_TO,
                target_airport=nearest_airport.upper() if nearest_airport else "KHUT"
            )],
            constraints=[
                # Relaxed constraints for emergency
                Constraint(type=ConstraintType.MIN_ALTITUDE_FT, value=500),
            ],
            handover=Handover(
                fallback_mode=FallbackMode.HUMAN,
                reason="Emergency situation - pilot authority required"
            ),
            explanation=explanation or f"EMERGENCY: Divert to {nearest_airport}",
            source="llm"
        )


# =============================================================================
# Intent Executor (stub for future trajectory integration)
# =============================================================================

class IntentExecutor:
    """
    Executes validated FlightIntents.

    This is a stub that currently maps intents back to simple commands.
    In the future, this will interface with the trajectory planner.
    """

    def __init__(self):
        self.active_intent: Optional[FlightIntent] = None
        self.intent_history: list = []

    def execute(self, intent: FlightIntent) -> Dict[str, Any]:
        """
        Execute a validated FlightIntent.

        Returns dict with:
            - success: bool
            - action: str (legacy action type)
            - value: float (if applicable)
            - target: str (if applicable)
        """
        self.active_intent = intent
        self.intent_history.append(intent)

        result = {
            "success": False,
            "action": None,
            "value": None,
            "target": None,
            "intent_id": intent.intent_id
        }

        # Extract primary goal
        if not intent.goals:
            return result

        primary_goal = intent.goals[0]

        # Map goal types to legacy actions
        if primary_goal.type == GoalType.MAINTAIN_HEADING:
            result["success"] = True
            result["action"] = "heading"
            result["value"] = primary_goal.target_heading_deg

        elif primary_goal.type == GoalType.MAINTAIN_ALTITUDE:
            result["success"] = True
            result["action"] = "altitude"
            result["value"] = primary_goal.target_altitude_ft

        elif primary_goal.type == GoalType.LAND_AT:
            result["success"] = True
            result["action"] = "land"
            result["target"] = primary_goal.target_airport

        elif primary_goal.type == GoalType.DIVERT_TO:
            result["success"] = True
            result["action"] = "land"  # Divert uses same landing logic
            result["target"] = primary_goal.target_airport

        elif primary_goal.type == GoalType.RETURN_TO_CRUISE:
            result["success"] = True
            result["action"] = "altitude"
            result["value"] = primary_goal.target_altitude_ft

        return result


# =============================================================================
# Integration Functions
# =============================================================================

def process_command_with_intent(
    action: str,
    value: Optional[float] = None,
    target: Optional[str] = None,
    raw_text: str = "",
    context: Optional[Dict[str, Any]] = None,
    min_confidence: float = 0.5
) -> Dict[str, Any]:
    """
    Process a command through the intent validation pipeline.

    This is the main integration point for the LLM command server.

    Args:
        action: Command action
        value: Numeric value
        target: Target identifier
        raw_text: Original input text
        context: Flight context dict
        min_confidence: Minimum confidence threshold

    Returns:
        Dict with execution result and validation info
    """
    bridge = IntentBridge()
    executor = IntentExecutor()

    # Convert command to intent
    intent, validation = bridge.command_to_intent(
        action, value, target, raw_text, context
    )

    result = {
        "success": False,
        "action": action,
        "value": value,
        "target": target,
        "validation": validation.to_dict(),
        "intent": intent.to_dict() if intent else None
    }

    # Check validation
    if not validation.is_valid:
        result["error"] = "Intent validation failed"
        result["errors"] = [e.message for e in validation.errors]
        return result

    # Check confidence
    confidence = validation.confidence.overall_confidence
    if confidence < min_confidence:
        result["error"] = f"Confidence too low: {confidence:.2f} < {min_confidence}"
        return result

    # Execute the intent
    exec_result = executor.execute(intent)
    result.update(exec_result)

    return result


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Intent Bridge Test")
    print("=" * 60)

    # Simulate flight context
    context = {
        "lat": 38.0,
        "lon": -98.0,
        "altitude": 5500,
        "heading": 90,
        "airspeed": 120,
        "origin": "SN65",
        "destination": "KHUT",
        "cruise_altitude": 5500,
        "distance": 15.0,
        "phase": "CRUISE"
    }

    # Test commands
    test_cases = [
        ("heading", 270, None, "Turn to heading 270"),
        ("altitude", 7000, None, "Climb to 7000 feet"),
        ("land", None, "KHUT", "Land at Hutchinson"),
        ("altitude", 20000, None, "Climb to 20000 feet"),  # Should fail validation
    ]

    for action, value, target, text in test_cases:
        print(f"\n--- Testing: {text} ---")
        result = process_command_with_intent(
            action, value, target, text, context
        )
        print(f"Success: {result['success']}")
        if result.get('validation'):
            val = result['validation']
            print(f"Valid: {val['is_valid']}")
            print(f"Confidence: {val['confidence']['overall_confidence']:.3f}")
        if result.get('error'):
            print(f"Error: {result['error']}")
        if result.get('errors'):
            print(f"Errors: {result['errors']}")

    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)
