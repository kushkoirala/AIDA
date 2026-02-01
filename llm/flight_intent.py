"""
Flight Intent Schema - Structured intent format for LLM-guided autonomy

This module defines the data structures for high-level flight intents that
the LLM produces. These intents are validated and converted to trajectory
setpoints by the safety supervisor and trajectory planner.

Architecture:
    LLM → FlightIntent → Validator → Trajectory Planner → Controller
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import json


# =============================================================================
# Enums
# =============================================================================

class MissionPhase(str, Enum):
    """Current phase of flight mission"""
    STARTUP = "startup"
    TAXI = "taxi"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    LANDING = "landing"
    GO_AROUND = "go_around"
    EMERGENCY = "emergency"
    LANDED = "landed"


class Priority(str, Enum):
    """Intent priority level"""
    NORMAL = "normal"
    HIGH = "high"
    EMERGENCY = "emergency"


class GoalType(str, Enum):
    """Types of flight goals"""
    REACH_WAYPOINT = "reach_waypoint"
    HOLD_POSITION = "hold_position"
    MAINTAIN_ALTITUDE = "maintain_altitude"
    MAINTAIN_SPEED = "maintain_speed"
    MAINTAIN_HEADING = "maintain_heading"
    FOLLOW_ROUTE = "follow_route"
    LAND_AT = "land_at"
    DIVERT_TO = "divert_to"
    AVOID_AREA = "avoid_area"
    PROCEED_TO = "proceed_to"
    RETURN_TO_CRUISE = "return_to_cruise"
    HOLD_PATTERN = "hold_pattern"


class ConstraintType(str, Enum):
    """Types of flight constraints"""
    MAX_BANK_DEG = "max_bank_deg"
    MAX_PITCH_DEG = "max_pitch_deg"
    MAX_CLIMB_RATE_FPM = "max_climb_rate_fpm"
    MAX_DESCENT_RATE_FPM = "max_descent_rate_fpm"
    MIN_SPEED_KT = "min_speed_kt"
    MAX_SPEED_KT = "max_speed_kt"
    MIN_ALTITUDE_FT = "min_altitude_ft"
    MAX_ALTITUDE_FT = "max_altitude_ft"
    AVOID_AIRSPACE = "avoid_airspace"
    AVOID_TERRAIN = "avoid_terrain"
    MIN_SEPARATION_NM = "min_separation_nm"
    NO_FLY_ZONE = "no_fly_zone"


class FallbackMode(str, Enum):
    """Fallback modes when intent cannot be executed"""
    AUTOPILOT = "autopilot"
    HUMAN = "human"
    ABORT_MISSION = "abort_mission"
    HOLD_CURRENT = "hold_current"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Waypoint:
    """Geographic waypoint"""
    lat: float
    lon: float
    alt_ft: Optional[float] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"lat": self.lat, "lon": self.lon}
        if self.alt_ft is not None:
            d["alt_ft"] = self.alt_ft
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Waypoint":
        return cls(
            lat=data["lat"],
            lon=data["lon"],
            alt_ft=data.get("alt_ft"),
            name=data.get("name")
        )


@dataclass
class GeoPolygon:
    """Geographic polygon for airspace/zones"""
    points: List[Waypoint]

    def to_dict(self) -> Dict[str, Any]:
        return {"points": [p.to_dict() for p in self.points]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeoPolygon":
        return cls(points=[Waypoint.from_dict(p) for p in data["points"]])


@dataclass
class Goal:
    """A single flight goal within an intent"""
    type: GoalType
    target_waypoint: Optional[Waypoint] = None
    target_airport: Optional[str] = None  # ICAO code
    target_altitude_ft: Optional[float] = None
    target_speed_kt: Optional[float] = None
    target_heading_deg: Optional[float] = None
    target_time_utc: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"type": self.type.value}
        if self.target_waypoint is not None:
            d["target_waypoint"] = self.target_waypoint.to_dict()
        if self.target_airport is not None:
            d["target_airport"] = self.target_airport
        if self.target_altitude_ft is not None:
            d["target_altitude_ft"] = self.target_altitude_ft
        if self.target_speed_kt is not None:
            d["target_speed_kt"] = self.target_speed_kt
        if self.target_heading_deg is not None:
            d["target_heading_deg"] = self.target_heading_deg
        if self.target_time_utc is not None:
            d["target_time_utc"] = self.target_time_utc.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        target_time = None
        if data.get("target_time_utc"):
            target_time = datetime.fromisoformat(data["target_time_utc"])

        return cls(
            type=GoalType(data["type"]),
            target_waypoint=Waypoint.from_dict(data["target_waypoint"]) if data.get("target_waypoint") else None,
            target_airport=data.get("target_airport"),
            target_altitude_ft=data.get("target_altitude_ft"),
            target_speed_kt=data.get("target_speed_kt"),
            target_heading_deg=data.get("target_heading_deg"),
            target_time_utc=target_time
        )


@dataclass
class Constraint:
    """A constraint on flight execution"""
    type: ConstraintType
    value: Optional[float] = None
    airspace_id: Optional[str] = None
    geometry: Optional[GeoPolygon] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"type": self.type.value}
        if self.value is not None:
            d["value"] = self.value
        if self.airspace_id is not None:
            d["airspace_id"] = self.airspace_id
        if self.geometry is not None:
            d["geometry"] = self.geometry.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraint":
        return cls(
            type=ConstraintType(data["type"]),
            value=data.get("value"),
            airspace_id=data.get("airspace_id"),
            geometry=GeoPolygon.from_dict(data["geometry"]) if data.get("geometry") else None
        )


@dataclass
class Route:
    """A sequence of waypoints forming a route"""
    waypoints: List[Waypoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"waypoints": [wp.to_dict() for wp in self.waypoints]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Route":
        return cls(waypoints=[Waypoint.from_dict(wp) for wp in data.get("waypoints", [])])


@dataclass
class Handover:
    """Handover instructions if intent fails"""
    fallback_mode: FallbackMode
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"fallback_mode": self.fallback_mode.value}
        if self.reason is not None:
            d["reason"] = self.reason
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Handover":
        return cls(
            fallback_mode=FallbackMode(data["fallback_mode"]),
            reason=data.get("reason")
        )


@dataclass
class FlightIntent:
    """
    Complete flight intent from LLM.

    This is the primary data structure for LLM-to-controller communication.
    The LLM produces intents, which are validated and converted to trajectory
    setpoints by downstream components.
    """
    # Required fields
    mission_phase: MissionPhase
    priority: Priority
    goals: List[Goal]

    # Optional fields
    intent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp_utc: datetime = field(default_factory=datetime.utcnow)
    constraints: List[Constraint] = field(default_factory=list)
    route: Optional[Route] = None
    handover: Optional[Handover] = None
    explanation: Optional[str] = None

    # Metadata (not part of schema, added by system)
    source: str = "llm"  # "llm", "pilot", "autopilot"
    raw_input: Optional[str] = None  # Original natural language input

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        d = {
            "intent_id": self.intent_id,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "mission_phase": self.mission_phase.value,
            "priority": self.priority.value,
            "goals": [g.to_dict() for g in self.goals],
            "constraints": [c.to_dict() for c in self.constraints],
        }
        if self.route is not None:
            d["route"] = self.route.to_dict()
        if self.handover is not None:
            d["handover"] = self.handover.to_dict()
        if self.explanation is not None:
            d["explanation"] = self.explanation
        return d

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlightIntent":
        """Create from dictionary"""
        return cls(
            intent_id=data.get("intent_id", str(uuid.uuid4())[:8]),
            timestamp_utc=datetime.fromisoformat(data["timestamp_utc"]) if data.get("timestamp_utc") else datetime.utcnow(),
            mission_phase=MissionPhase(data["mission_phase"]),
            priority=Priority(data["priority"]),
            goals=[Goal.from_dict(g) for g in data["goals"]],
            constraints=[Constraint.from_dict(c) for c in data.get("constraints", [])],
            route=Route.from_dict(data["route"]) if data.get("route") else None,
            handover=Handover.from_dict(data["handover"]) if data.get("handover") else None,
            explanation=data.get("explanation")
        )

    @classmethod
    def from_json(cls, json_str: str) -> "FlightIntent":
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Factory Functions for Common Intents
# =============================================================================

def create_altitude_intent(
    target_altitude_ft: float,
    current_phase: MissionPhase = MissionPhase.CRUISE,
    priority: Priority = Priority.NORMAL,
    explanation: str = None
) -> FlightIntent:
    """Create an intent to change altitude"""
    return FlightIntent(
        mission_phase=current_phase,
        priority=priority,
        goals=[Goal(
            type=GoalType.MAINTAIN_ALTITUDE,
            target_altitude_ft=target_altitude_ft
        )],
        constraints=[
            Constraint(type=ConstraintType.MIN_ALTITUDE_FT, value=1500),
            Constraint(type=ConstraintType.MAX_ALTITUDE_FT, value=12000),
            Constraint(type=ConstraintType.MAX_CLIMB_RATE_FPM, value=1000),
            Constraint(type=ConstraintType.MAX_DESCENT_RATE_FPM, value=1000),
        ],
        explanation=explanation or f"Change altitude to {target_altitude_ft} ft"
    )


def create_heading_intent(
    target_heading_deg: float,
    current_phase: MissionPhase = MissionPhase.CRUISE,
    priority: Priority = Priority.NORMAL,
    explanation: str = None
) -> FlightIntent:
    """Create an intent to change heading"""
    return FlightIntent(
        mission_phase=current_phase,
        priority=priority,
        goals=[Goal(
            type=GoalType.MAINTAIN_HEADING,
            target_heading_deg=target_heading_deg
        )],
        constraints=[
            Constraint(type=ConstraintType.MAX_BANK_DEG, value=30),
        ],
        explanation=explanation or f"Turn to heading {target_heading_deg:03.0f}"
    )


def create_land_intent(
    airport_code: str,
    current_phase: MissionPhase = MissionPhase.CRUISE,
    priority: Priority = Priority.NORMAL,
    explanation: str = None
) -> FlightIntent:
    """Create an intent to land at an airport"""
    return FlightIntent(
        mission_phase=current_phase,
        priority=priority,
        goals=[Goal(
            type=GoalType.LAND_AT,
            target_airport=airport_code.upper()
        )],
        constraints=[
            Constraint(type=ConstraintType.MIN_SPEED_KT, value=48),  # Vs0
            Constraint(type=ConstraintType.MAX_SPEED_KT, value=120),
        ],
        handover=Handover(
            fallback_mode=FallbackMode.AUTOPILOT,
            reason="Landing requires precise control"
        ),
        explanation=explanation or f"Land at {airport_code.upper()}"
    )


def create_divert_intent(
    airport_code: str,
    reason: str = "pilot request",
    priority: Priority = Priority.HIGH,
    explanation: str = None
) -> FlightIntent:
    """Create an intent to divert to an airport"""
    return FlightIntent(
        mission_phase=MissionPhase.CRUISE,
        priority=priority,
        goals=[Goal(
            type=GoalType.DIVERT_TO,
            target_airport=airport_code.upper()
        )],
        constraints=[
            Constraint(type=ConstraintType.MIN_ALTITUDE_FT, value=1500),
        ],
        handover=Handover(
            fallback_mode=FallbackMode.HOLD_CURRENT,
            reason=reason
        ),
        explanation=explanation or f"Divert to {airport_code.upper()}: {reason}"
    )


def create_return_to_cruise_intent(
    cruise_altitude_ft: float,
    cruise_heading_deg: Optional[float] = None,
    explanation: str = None
) -> FlightIntent:
    """Create an intent to return to cruise parameters"""
    goals = [Goal(
        type=GoalType.RETURN_TO_CRUISE,
        target_altitude_ft=cruise_altitude_ft
    )]

    if cruise_heading_deg is not None:
        goals.append(Goal(
            type=GoalType.MAINTAIN_HEADING,
            target_heading_deg=cruise_heading_deg
        ))

    return FlightIntent(
        mission_phase=MissionPhase.CRUISE,
        priority=Priority.NORMAL,
        goals=goals,
        explanation=explanation or f"Return to cruise at {cruise_altitude_ft} ft"
    )


def create_emergency_intent(
    nearest_airport: str,
    emergency_type: str = "general",
    explanation: str = None
) -> FlightIntent:
    """Create an emergency diversion intent"""
    return FlightIntent(
        mission_phase=MissionPhase.EMERGENCY,
        priority=Priority.EMERGENCY,
        goals=[Goal(
            type=GoalType.DIVERT_TO,
            target_airport=nearest_airport.upper()
        )],
        constraints=[
            # Relax constraints for emergency
            Constraint(type=ConstraintType.MIN_ALTITUDE_FT, value=500),
        ],
        handover=Handover(
            fallback_mode=FallbackMode.HUMAN,
            reason=f"Emergency: {emergency_type}"
        ),
        explanation=explanation or f"EMERGENCY: Divert to {nearest_airport.upper()}"
    )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Create and serialize an intent
    intent = create_altitude_intent(
        target_altitude_ft=7000,
        current_phase=MissionPhase.CRUISE,
        explanation="Pilot requested climb to 7000 feet"
    )

    print("=== Flight Intent Example ===")
    print(intent.to_json())

    # Example: Compound intent (altitude + heading)
    compound_intent = FlightIntent(
        mission_phase=MissionPhase.CRUISE,
        priority=Priority.NORMAL,
        goals=[
            Goal(type=GoalType.MAINTAIN_ALTITUDE, target_altitude_ft=7000),
            Goal(type=GoalType.MAINTAIN_HEADING, target_heading_deg=270)
        ],
        constraints=[
            Constraint(type=ConstraintType.MAX_BANK_DEG, value=30),
            Constraint(type=ConstraintType.MAX_CLIMB_RATE_FPM, value=500),
        ],
        explanation="Climb to 7000 and turn to heading 270"
    )

    print("\n=== Compound Intent Example ===")
    print(compound_intent.to_json())

    # Test round-trip serialization
    json_str = compound_intent.to_json()
    restored = FlightIntent.from_json(json_str)
    print("\n=== Round-trip test ===")
    print(f"Original goals: {len(compound_intent.goals)}")
    print(f"Restored goals: {len(restored.goals)}")
    print(f"Match: {compound_intent.to_dict() == restored.to_dict()}")
