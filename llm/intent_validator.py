"""
Intent Validator - Schema validation and confidence scoring for FlightIntent

This module provides validation and confidence scoring for LLM-generated
flight intents. It acts as the first layer of the Safety Supervisor.

Architecture:
    LLM Output → IntentValidator → [Valid Intent + Confidence Score]
                                 → [Invalid: Rejection with reasons]
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

try:
    import jsonschema
    from jsonschema import Draft202012Validator, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    print("Warning: jsonschema not installed. Schema validation disabled.")

from flight_intent import (
    FlightIntent, MissionPhase, Priority, GoalType,
    ConstraintType, FallbackMode
)


# =============================================================================
# Validation Result Types
# =============================================================================

class ValidationStatus(str, Enum):
    """Overall validation status"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"  # Valid but with concerns


@dataclass
class ValidationIssue:
    """A single validation issue"""
    severity: str  # "error", "warning", "info"
    code: str      # Machine-readable code
    message: str   # Human-readable message
    path: str = "" # JSON path to the issue location

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path
        }


@dataclass
class ConfidenceScore:
    """Confidence scoring breakdown"""
    # Hard gates (must all be True for intent to be valid)
    schema_valid: bool = False
    constraints_feasible: bool = False
    goals_achievable: bool = False

    # Soft scores (0.0 to 1.0)
    semantic_coherence: float = 0.0
    safety_margin: float = 0.0
    specificity: float = 0.0

    # Overall score
    @property
    def hard_gates_passed(self) -> bool:
        return self.schema_valid and self.constraints_feasible and self.goals_achievable

    @property
    def soft_score(self) -> float:
        """Weighted average of soft scores"""
        weights = {
            "semantic_coherence": 0.4,
            "safety_margin": 0.4,
            "specificity": 0.2
        }
        return (
            self.semantic_coherence * weights["semantic_coherence"] +
            self.safety_margin * weights["safety_margin"] +
            self.specificity * weights["specificity"]
        )

    @property
    def overall_confidence(self) -> float:
        """Overall confidence (0.0 if hard gates fail)"""
        if not self.hard_gates_passed:
            return 0.0
        return self.soft_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hard_gates": {
                "schema_valid": self.schema_valid,
                "constraints_feasible": self.constraints_feasible,
                "goals_achievable": self.goals_achievable,
                "all_passed": self.hard_gates_passed
            },
            "soft_scores": {
                "semantic_coherence": round(self.semantic_coherence, 3),
                "safety_margin": round(self.safety_margin, 3),
                "specificity": round(self.specificity, 3),
                "weighted_average": round(self.soft_score, 3)
            },
            "overall_confidence": round(self.overall_confidence, 3)
        }


@dataclass
class ValidationResult:
    """Complete validation result"""
    status: ValidationStatus
    intent: Optional[FlightIntent] = None
    confidence: ConfidenceScore = field(default_factory=ConfidenceScore)
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == ValidationStatus.VALID

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "confidence": self.confidence.to_dict(),
            "issues": [i.to_dict() for i in self.issues],
            "intent": self.intent.to_dict() if self.intent else None
        }


# =============================================================================
# Cessna 172 Operating Limits (for constraint validation)
# =============================================================================

CESSNA_172_LIMITS = {
    # Speed limits (KIAS)
    "vne": 163,           # Never exceed
    "vno": 129,           # Max structural cruising
    "va": 99,             # Maneuvering speed (at max gross)
    "vfe": 85,            # Max flaps extended
    "vs0": 48,            # Stall speed (landing config)
    "vs1": 53,            # Stall speed (clean)
    "vy": 74,             # Best rate of climb
    "vx": 62,             # Best angle of climb
    "vref": 65,           # Final approach speed

    # Performance limits
    "max_altitude_ft": 14000,    # Service ceiling
    "min_altitude_ft": 0,        # Ground level
    "max_climb_fpm": 730,        # Max rate of climb
    "max_descent_fpm": 1000,     # Comfortable descent
    "max_bank_deg": 60,          # Structural limit
    "normal_bank_deg": 30,       # Normal ops
    "max_pitch_deg": 20,         # Normal ops

    # Fuel
    "fuel_capacity_gal": 56,
    "fuel_burn_gph": 8.5,
    "max_range_nm": 640,
}


# =============================================================================
# Intent Validator Class
# =============================================================================

class IntentValidator:
    """
    Validates FlightIntent objects against schema and operational constraints.

    This is the first layer of the Safety Supervisor, ensuring that LLM outputs
    are structurally valid and operationally feasible before being passed to
    the trajectory planner.
    """

    def __init__(self, schema_path: str = None):
        """
        Initialize validator with JSON schema.

        Args:
            schema_path: Path to JSON schema file. If None, uses default location.
        """
        self.schema = None
        self.validator = None

        if HAS_JSONSCHEMA:
            if schema_path is None:
                schema_path = os.path.join(os.path.dirname(__file__), "intent_schema.json")

            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    self.schema = json.load(f)
                self.validator = Draft202012Validator(self.schema)
            else:
                print(f"Warning: Schema file not found at {schema_path}")

    def validate(self, intent_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate an intent dictionary.

        Args:
            intent_data: Dictionary representation of FlightIntent

        Returns:
            ValidationResult with status, confidence, and any issues
        """
        result = ValidationResult(status=ValidationStatus.INVALID)
        confidence = ConfidenceScore()

        # Step 1: JSON Schema Validation
        schema_issues = self._validate_schema(intent_data)
        result.issues.extend(schema_issues)

        if not any(i.severity == "error" for i in schema_issues):
            confidence.schema_valid = True
        else:
            # Schema validation failed - return early
            result.confidence = confidence
            return result

        # Step 2: Parse into FlightIntent object
        try:
            intent = FlightIntent.from_dict(intent_data)
            result.intent = intent
        except Exception as e:
            result.issues.append(ValidationIssue(
                severity="error",
                code="PARSE_ERROR",
                message=f"Failed to parse intent: {str(e)}"
            ))
            result.confidence = confidence
            return result

        # Step 3: Constraint Feasibility Check
        constraint_issues = self._validate_constraints(intent)
        result.issues.extend(constraint_issues)

        if not any(i.severity == "error" for i in constraint_issues):
            confidence.constraints_feasible = True

        # Step 4: Goal Achievability Check
        goal_issues = self._validate_goals(intent)
        result.issues.extend(goal_issues)

        if not any(i.severity == "error" for i in goal_issues):
            confidence.goals_achievable = True

        # Step 5: Soft Scoring
        confidence.semantic_coherence = self._score_semantic_coherence(intent)
        confidence.safety_margin = self._score_safety_margin(intent)
        confidence.specificity = self._score_specificity(intent)

        # Determine final status
        result.confidence = confidence

        if confidence.hard_gates_passed:
            if result.warnings:
                result.status = ValidationStatus.WARNING
            else:
                result.status = ValidationStatus.VALID
        else:
            result.status = ValidationStatus.INVALID

        return result

    def validate_json(self, json_str: str) -> ValidationResult:
        """Validate a JSON string."""
        try:
            intent_data = json.loads(json_str)
            return self.validate(intent_data)
        except json.JSONDecodeError as e:
            result = ValidationResult(status=ValidationStatus.INVALID)
            result.issues.append(ValidationIssue(
                severity="error",
                code="JSON_PARSE_ERROR",
                message=f"Invalid JSON: {str(e)}"
            ))
            return result

    def validate_intent(self, intent: FlightIntent) -> ValidationResult:
        """Validate a FlightIntent object."""
        return self.validate(intent.to_dict())

    # -------------------------------------------------------------------------
    # Schema Validation
    # -------------------------------------------------------------------------

    def _validate_schema(self, intent_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate against JSON schema."""
        issues = []

        if not HAS_JSONSCHEMA or self.validator is None:
            # No schema validation available - assume valid but warn
            issues.append(ValidationIssue(
                severity="warning",
                code="SCHEMA_UNAVAILABLE",
                message="JSON schema validation not available"
            ))
            return issues

        errors = list(self.validator.iter_errors(intent_data))

        for error in errors:
            path = ".".join(str(p) for p in error.absolute_path) or "root"
            issues.append(ValidationIssue(
                severity="error",
                code="SCHEMA_VIOLATION",
                message=error.message,
                path=path
            ))

        return issues

    # -------------------------------------------------------------------------
    # Constraint Validation
    # -------------------------------------------------------------------------

    def _validate_constraints(self, intent: FlightIntent) -> List[ValidationIssue]:
        """Validate that constraints are feasible for Cessna 172."""
        issues = []
        limits = CESSNA_172_LIMITS

        for constraint in intent.constraints:
            if constraint.type == ConstraintType.MIN_SPEED_KT:
                if constraint.value and constraint.value < limits["vs0"]:
                    issues.append(ValidationIssue(
                        severity="error",
                        code="BELOW_STALL",
                        message=f"Min speed {constraint.value} kt is below stall speed ({limits['vs0']} kt)",
                        path="constraints"
                    ))

            elif constraint.type == ConstraintType.MAX_SPEED_KT:
                if constraint.value and constraint.value > limits["vne"]:
                    issues.append(ValidationIssue(
                        severity="error",
                        code="EXCEEDS_VNE",
                        message=f"Max speed {constraint.value} kt exceeds VNE ({limits['vne']} kt)",
                        path="constraints"
                    ))

            elif constraint.type == ConstraintType.MAX_ALTITUDE_FT:
                if constraint.value and constraint.value > limits["max_altitude_ft"]:
                    issues.append(ValidationIssue(
                        severity="warning",
                        code="ABOVE_CEILING",
                        message=f"Max altitude {constraint.value} ft exceeds service ceiling ({limits['max_altitude_ft']} ft)",
                        path="constraints"
                    ))

            elif constraint.type == ConstraintType.MAX_BANK_DEG:
                if constraint.value and constraint.value > limits["max_bank_deg"]:
                    issues.append(ValidationIssue(
                        severity="error",
                        code="EXCEEDS_BANK_LIMIT",
                        message=f"Max bank {constraint.value}° exceeds structural limit ({limits['max_bank_deg']}°)",
                        path="constraints"
                    ))

            elif constraint.type == ConstraintType.MAX_CLIMB_RATE_FPM:
                if constraint.value and constraint.value > limits["max_climb_fpm"]:
                    issues.append(ValidationIssue(
                        severity="warning",
                        code="EXCEEDS_CLIMB_RATE",
                        message=f"Max climb rate {constraint.value} fpm exceeds aircraft capability ({limits['max_climb_fpm']} fpm)",
                        path="constraints"
                    ))

        return issues

    # -------------------------------------------------------------------------
    # Goal Validation
    # -------------------------------------------------------------------------

    def _validate_goals(self, intent: FlightIntent) -> List[ValidationIssue]:
        """Validate that goals are achievable."""
        issues = []
        limits = CESSNA_172_LIMITS

        for i, goal in enumerate(intent.goals):
            goal_path = f"goals[{i}]"

            # Altitude goals
            if goal.target_altitude_ft is not None:
                if goal.target_altitude_ft < 0:
                    issues.append(ValidationIssue(
                        severity="error",
                        code="NEGATIVE_ALTITUDE",
                        message=f"Target altitude {goal.target_altitude_ft} ft is negative",
                        path=goal_path
                    ))
                elif goal.target_altitude_ft > limits["max_altitude_ft"]:
                    issues.append(ValidationIssue(
                        severity="warning",
                        code="ABOVE_CEILING",
                        message=f"Target altitude {goal.target_altitude_ft} ft exceeds service ceiling",
                        path=goal_path
                    ))

            # Speed goals
            if goal.target_speed_kt is not None:
                if goal.target_speed_kt < limits["vs0"]:
                    issues.append(ValidationIssue(
                        severity="error",
                        code="BELOW_STALL",
                        message=f"Target speed {goal.target_speed_kt} kt is below stall speed",
                        path=goal_path
                    ))
                elif goal.target_speed_kt > limits["vne"]:
                    issues.append(ValidationIssue(
                        severity="error",
                        code="EXCEEDS_VNE",
                        message=f"Target speed {goal.target_speed_kt} kt exceeds VNE",
                        path=goal_path
                    ))

            # Heading goals
            if goal.target_heading_deg is not None:
                if goal.target_heading_deg < 0 or goal.target_heading_deg > 360:
                    issues.append(ValidationIssue(
                        severity="error",
                        code="INVALID_HEADING",
                        message=f"Target heading {goal.target_heading_deg}° is out of range [0, 360]",
                        path=goal_path
                    ))

            # Airport goals
            if goal.type in [GoalType.LAND_AT, GoalType.DIVERT_TO]:
                if not goal.target_airport:
                    issues.append(ValidationIssue(
                        severity="error",
                        code="MISSING_AIRPORT",
                        message=f"Goal type {goal.type.value} requires target_airport",
                        path=goal_path
                    ))

            # Waypoint goals
            if goal.type in [GoalType.REACH_WAYPOINT, GoalType.PROCEED_TO]:
                if not goal.target_waypoint:
                    issues.append(ValidationIssue(
                        severity="error",
                        code="MISSING_WAYPOINT",
                        message=f"Goal type {goal.type.value} requires target_waypoint",
                        path=goal_path
                    ))

        return issues

    # -------------------------------------------------------------------------
    # Soft Scoring Functions
    # -------------------------------------------------------------------------

    def _score_semantic_coherence(self, intent: FlightIntent) -> float:
        """
        Score how semantically coherent the intent is.

        Checks:
        - Mission phase matches goal types
        - Priority matches situation
        - Goals are logically consistent
        """
        score = 1.0
        deductions = []

        # Check phase-goal coherence
        phase = intent.mission_phase
        goal_types = [g.type for g in intent.goals]

        # Landing phase should have landing goals
        if phase == MissionPhase.LANDING:
            if GoalType.LAND_AT not in goal_types and GoalType.MAINTAIN_ALTITUDE not in goal_types:
                score -= 0.2
                deductions.append("Landing phase without landing/altitude goal")

        # Emergency phase should have emergency priority
        if phase == MissionPhase.EMERGENCY:
            if intent.priority != Priority.EMERGENCY:
                score -= 0.3
                deductions.append("Emergency phase without emergency priority")

        # Conflicting goals (e.g., climb and descend)
        has_altitude_goals = [g for g in intent.goals if g.target_altitude_ft is not None]
        if len(has_altitude_goals) > 1:
            altitudes = [g.target_altitude_ft for g in has_altitude_goals]
            if max(altitudes) - min(altitudes) > 5000:
                score -= 0.2
                deductions.append("Large altitude spread in goals")

        # Divert should have high/emergency priority
        if GoalType.DIVERT_TO in goal_types:
            if intent.priority == Priority.NORMAL:
                score -= 0.1
                deductions.append("Divert with normal priority")

        return max(0.0, score)

    def _score_safety_margin(self, intent: FlightIntent) -> float:
        """
        Score safety margin based on how conservative the intent is.

        Higher score = more safety margin.
        """
        score = 1.0
        limits = CESSNA_172_LIMITS

        # Check altitude goals
        for goal in intent.goals:
            if goal.target_altitude_ft is not None:
                # Penalize very low altitudes
                if goal.target_altitude_ft < 1000:
                    score -= 0.3
                elif goal.target_altitude_ft < 1500:
                    score -= 0.1

                # Penalize near ceiling
                ceiling_margin = (limits["max_altitude_ft"] - goal.target_altitude_ft) / limits["max_altitude_ft"]
                if ceiling_margin < 0.1:
                    score -= 0.2

            # Check speed goals
            if goal.target_speed_kt is not None:
                # Calculate margin from stall
                stall_margin = (goal.target_speed_kt - limits["vs1"]) / limits["vs1"]
                if stall_margin < 0.2:  # Less than 20% above stall
                    score -= 0.3
                elif stall_margin < 0.3:
                    score -= 0.1

                # Calculate margin from VNE
                vne_margin = (limits["vne"] - goal.target_speed_kt) / limits["vne"]
                if vne_margin < 0.1:
                    score -= 0.3

        # Check constraints provide adequate margins
        for constraint in intent.constraints:
            if constraint.type == ConstraintType.MIN_ALTITUDE_FT:
                if constraint.value and constraint.value < 1000:
                    score -= 0.2

            if constraint.type == ConstraintType.MAX_BANK_DEG:
                if constraint.value and constraint.value > 45:
                    score -= 0.1

        # Bonus for having handover defined
        if intent.handover is not None:
            score = min(1.0, score + 0.1)

        return max(0.0, min(1.0, score))

    def _score_specificity(self, intent: FlightIntent) -> float:
        """
        Score how specific and actionable the intent is.

        Higher score = more specific/actionable.
        """
        score = 0.5  # Base score

        # Goals specificity
        for goal in intent.goals:
            # Numeric targets are more specific
            if goal.target_altitude_ft is not None:
                score += 0.1
            if goal.target_speed_kt is not None:
                score += 0.1
            if goal.target_heading_deg is not None:
                score += 0.1
            if goal.target_waypoint is not None:
                score += 0.15
            if goal.target_airport is not None:
                score += 0.1

        # Constraints add specificity
        score += min(0.2, len(intent.constraints) * 0.05)

        # Route adds specificity
        if intent.route and intent.route.waypoints:
            score += 0.1

        # Explanation adds clarity
        if intent.explanation:
            score += 0.05

        return min(1.0, score)


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_intent(intent: FlightIntent) -> ValidationResult:
    """Quick validation of a FlightIntent object."""
    validator = IntentValidator()
    return validator.validate_intent(intent)


def validate_intent_json(json_str: str) -> ValidationResult:
    """Quick validation of a JSON string."""
    validator = IntentValidator()
    return validator.validate_json(json_str)


def is_intent_safe(intent: FlightIntent, min_confidence: float = 0.6) -> Tuple[bool, str]:
    """
    Quick check if intent is safe to execute.

    Returns:
        Tuple of (is_safe, reason)
    """
    result = validate_intent(intent)

    if not result.is_valid:
        errors = "; ".join(e.message for e in result.errors)
        return False, f"Validation failed: {errors}"

    if result.confidence.overall_confidence < min_confidence:
        return False, f"Confidence too low: {result.confidence.overall_confidence:.2f} < {min_confidence}"

    return True, "Intent validated successfully"


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    from flight_intent import create_altitude_intent, create_land_intent, create_emergency_intent

    print("=" * 60)
    print("Intent Validator Test")
    print("=" * 60)

    # Test 1: Valid altitude intent
    print("\n--- Test 1: Valid Altitude Intent ---")
    intent1 = create_altitude_intent(7000, explanation="Climb to 7000 feet")
    result1 = validate_intent(intent1)
    print(f"Status: {result1.status.value}")
    print(f"Confidence: {result1.confidence.overall_confidence:.3f}")
    print(f"Hard gates passed: {result1.confidence.hard_gates_passed}")

    # Test 2: Invalid altitude (too high)
    print("\n--- Test 2: Invalid Altitude (Too High) ---")
    intent2 = create_altitude_intent(20000, explanation="Climb to 20000 feet")
    result2 = validate_intent(intent2)
    print(f"Status: {result2.status.value}")
    print(f"Issues: {[i.message for i in result2.issues]}")

    # Test 3: Landing intent
    print("\n--- Test 3: Landing Intent ---")
    intent3 = create_land_intent("KHUT")
    result3 = validate_intent(intent3)
    print(f"Status: {result3.status.value}")
    print(f"Confidence: {result3.confidence.overall_confidence:.3f}")

    # Test 4: Emergency intent
    print("\n--- Test 4: Emergency Intent ---")
    intent4 = create_emergency_intent("KHUT", "engine failure")
    result4 = validate_intent(intent4)
    print(f"Status: {result4.status.value}")
    print(f"Confidence breakdown:")
    print(f"  - Semantic coherence: {result4.confidence.semantic_coherence:.3f}")
    print(f"  - Safety margin: {result4.confidence.safety_margin:.3f}")
    print(f"  - Specificity: {result4.confidence.specificity:.3f}")

    # Test 5: JSON round-trip
    print("\n--- Test 5: JSON Round-trip ---")
    json_str = intent1.to_json()
    result5 = validate_intent_json(json_str)
    print(f"JSON validation: {result5.status.value}")

    # Test 6: Quick safety check
    print("\n--- Test 6: Quick Safety Check ---")
    safe, reason = is_intent_safe(intent1)
    print(f"Is safe: {safe}")
    print(f"Reason: {reason}")

    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)
