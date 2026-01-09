"""
Flight Scenarios for AIDA Autonomous Flight Training

Defines end-to-end flight scenarios for behavior cloning dataset generation.
Each scenario includes waypoints, flight phases, and success criteria.

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum, auto
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aida_sim.platform.airports import get_airport, TrafficPattern
from aida_sim.platform.world import World


class FlightPhase(Enum):
    """Flight phases for scenario execution."""
    GROUND_ROLL = auto()
    ROTATION = auto()
    INITIAL_CLIMB = auto()
    CLIMB = auto()
    CRUISE = auto()
    DESCENT = auto()
    APPROACH = auto()
    FINAL = auto()
    FLARE = auto()
    ROLLOUT = auto()
    # Traffic pattern specific
    CROSSWIND = auto()
    DOWNWIND = auto()
    BASE = auto()
    # Special
    GO_AROUND = auto()
    TOUCH_AND_GO = auto()


@dataclass
class Waypoint:
    """Navigation waypoint with target state."""
    name: str
    position: np.ndarray      # NED coordinates (m)
    altitude_msl_ft: float    # Target altitude
    airspeed_kts: float       # Target airspeed
    heading_deg: Optional[float] = None  # Target heading (None = direct to next)
    phase: FlightPhase = FlightPhase.CRUISE
    tolerance_m: float = 100.0  # Position tolerance for waypoint capture


@dataclass
class ScenarioConfig:
    """Configuration for a flight scenario."""
    name: str
    description: str
    departure_airport: str
    departure_runway: str
    arrival_airport: str
    arrival_runway: str
    waypoints: List[Waypoint] = field(default_factory=list)
    cruise_altitude_ft: float = 3500.0
    pattern_altitude_ft: float = 1000.0  # AGL
    max_duration_s: float = 600.0
    success_criteria: Dict = field(default_factory=dict)


class Scenario:
    """
    Flight scenario with waypoints and phase management.

    Provides the flight profile for classical controller or RL agent to follow.
    """

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.world = World(origin_airport=config.departure_airport)
        self._build_waypoints()

    def _build_waypoints(self):
        """Build waypoint list from scenario configuration."""
        self.waypoints = list(self.config.waypoints)

    def get_initial_state(self) -> Dict:
        """
        Get initial aircraft state for scenario.

        Returns:
            Dictionary with initial position, velocity, attitude
        """
        rwy = self.world.get_runway_geometry(
            self.config.departure_airport,
            self.config.departure_runway
        )

        # Start on runway, aligned with centerline
        threshold = rwy['threshold']
        heading = rwy['heading_rad']

        return {
            'position': np.array([threshold[0], threshold[1], 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'heading_rad': heading,
            'phase': FlightPhase.GROUND_ROLL,
        }

    def get_waypoints(self) -> List[Waypoint]:
        """Get list of waypoints for the scenario."""
        return self.waypoints

    def check_completion(self, position: np.ndarray, altitude_ft: float,
                         airspeed_kts: float, phase: FlightPhase) -> bool:
        """Check if scenario is complete."""
        if not self.waypoints:
            return True

        final_wp = self.waypoints[-1]

        # Check if at final waypoint
        dist = np.linalg.norm(position[:2] - final_wp.position[:2])
        if dist < final_wp.tolerance_m:
            if final_wp.phase in [FlightPhase.ROLLOUT, FlightPhase.TOUCH_AND_GO]:
                return airspeed_kts < 10.0  # Stopped or nearly stopped
            return True

        return False


# =============================================================================
# SCENARIO BUILDERS
# =============================================================================

def create_traffic_pattern_scenario(
    airport: str = "SN65",
    runway: str = "35",
    pattern_altitude_agl_ft: float = 1000.0,
    full_stop: bool = True
) -> Scenario:
    """
    Create a traffic pattern scenario.

    Takeoff -> Crosswind -> Downwind -> Base -> Final -> Landing

    Args:
        airport: Airport ICAO code
        runway: Runway designator
        pattern_altitude_agl_ft: Pattern altitude AGL
        full_stop: True for full stop landing, False for touch-and-go
    """
    world = World(origin_airport=airport)
    apt = get_airport(airport)
    pattern = world.get_pattern_waypoints(airport, runway, pattern_altitude_agl_ft)
    rwy = world.get_runway_geometry(airport, runway)

    pattern_alt_ft = apt.elevation_ft + pattern_altitude_agl_ft

    waypoints = [
        Waypoint(
            name="departure",
            position=pattern['waypoints']['departure'],
            altitude_msl_ft=apt.elevation_ft + 50,  # 50 ft AGL at liftoff
            airspeed_kts=65,  # Vx climb
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.ROTATION,
        ),
        Waypoint(
            name="upwind",
            position=pattern['waypoints']['upwind'],
            altitude_msl_ft=pattern_alt_ft,
            airspeed_kts=80,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.CLIMB,
        ),
        Waypoint(
            name="crosswind",
            position=pattern['waypoints']['crosswind'],
            altitude_msl_ft=pattern_alt_ft,
            airspeed_kts=90,
            heading_deg=(rwy['heading_deg'] + 90 * (-1 if pattern['is_left_pattern'] else 1)) % 360,
            phase=FlightPhase.CROSSWIND,
        ),
        Waypoint(
            name="downwind_abeam",
            position=pattern['waypoints']['downwind_abeam'],
            altitude_msl_ft=pattern_alt_ft,
            airspeed_kts=90,
            heading_deg=(rwy['heading_deg'] + 180) % 360,
            phase=FlightPhase.DOWNWIND,
        ),
        Waypoint(
            name="base",
            position=pattern['waypoints']['base'],
            altitude_msl_ft=pattern_alt_ft - 200,  # Start descent
            airspeed_kts=80,
            heading_deg=(rwy['heading_deg'] + 90 * (1 if pattern['is_left_pattern'] else -1)) % 360,
            phase=FlightPhase.BASE,
        ),
        Waypoint(
            name="final",
            position=pattern['waypoints']['final'],
            altitude_msl_ft=apt.elevation_ft + 200,  # ~200 ft AGL on short final
            airspeed_kts=65,  # Approach speed
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.FINAL,
        ),
        Waypoint(
            name="threshold",
            position=pattern['waypoints']['threshold'],
            altitude_msl_ft=apt.elevation_ft + 10,  # 10 ft AGL at threshold
            airspeed_kts=55,  # Just above stall
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.FLARE,
        ),
    ]

    # Add rollout or touch-and-go
    if full_stop:
        waypoints.append(Waypoint(
            name="rollout",
            position=pattern['waypoints']['threshold'] + np.array([300, 0, 0]),  # 300m down runway
            altitude_msl_ft=apt.elevation_ft,
            airspeed_kts=0,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.ROLLOUT,
        ))
    else:
        waypoints.append(Waypoint(
            name="touch_and_go",
            position=pattern['waypoints']['departure'],
            altitude_msl_ft=apt.elevation_ft + 50,
            airspeed_kts=65,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.TOUCH_AND_GO,
        ))

    config = ScenarioConfig(
        name=f"traffic_pattern_{airport}_{runway}",
        description=f"Traffic pattern at {airport} runway {runway}",
        departure_airport=airport,
        departure_runway=runway,
        arrival_airport=airport,
        arrival_runway=runway,
        waypoints=waypoints,
        pattern_altitude_ft=pattern_altitude_agl_ft,
        max_duration_s=300.0,
        success_criteria={
            'landed': True,
            'on_runway': True,
            'safe_touchdown_fpm': 300,  # Max sink rate
        },
    )

    return Scenario(config)


def create_cross_country_scenario(
    dep_airport: str = "SN65",
    dep_runway: str = "35",
    arr_airport: str = "KHUT",
    arr_runway: str = "13",
    cruise_altitude_ft: float = 3500.0
) -> Scenario:
    """
    Create a cross-country flight scenario.

    Departure -> Climb -> Cruise -> Descent -> Approach -> Landing

    Args:
        dep_airport: Departure airport ICAO
        dep_runway: Departure runway
        arr_airport: Arrival airport ICAO
        arr_runway: Arrival runway
        cruise_altitude_ft: Cruise altitude MSL
    """
    world = World(origin_airport=dep_airport)
    dep_apt = get_airport(dep_airport)
    arr_apt = get_airport(arr_airport)

    route = world.get_cross_country_route(
        dep_airport, dep_runway,
        arr_airport, arr_runway,
        cruise_altitude_ft
    )

    dep_rwy = world.get_runway_geometry(dep_airport, dep_runway)
    arr_rwy = world.get_runway_geometry(arr_airport, arr_runway)

    waypoints = [
        Waypoint(
            name="departure",
            position=np.array(dep_rwy['threshold']),
            altitude_msl_ft=dep_apt.elevation_ft + 50,
            airspeed_kts=65,
            heading_deg=dep_rwy['heading_deg'],
            phase=FlightPhase.ROTATION,
        ),
        Waypoint(
            name="top_of_climb",
            position=route['waypoints']['top_of_climb'],
            altitude_msl_ft=cruise_altitude_ft,
            airspeed_kts=100,
            heading_deg=route['course_true'],
            phase=FlightPhase.CLIMB,
        ),
        Waypoint(
            name="top_of_descent",
            position=route['waypoints']['top_of_descent'],
            altitude_msl_ft=cruise_altitude_ft,
            airspeed_kts=110,
            heading_deg=route['course_true'],
            phase=FlightPhase.CRUISE,
        ),
        Waypoint(
            name="initial_approach",
            position=route['waypoints']['initial_approach'],
            altitude_msl_ft=arr_apt.elevation_ft + 1500,
            airspeed_kts=90,
            heading_deg=(arr_rwy['heading_deg'] + 180) % 360,  # Inbound heading
            phase=FlightPhase.APPROACH,
        ),
        Waypoint(
            name="final_approach",
            position=route['waypoints']['final_approach'],
            altitude_msl_ft=arr_apt.elevation_ft + 300,
            airspeed_kts=70,
            heading_deg=arr_rwy['heading_deg'],
            phase=FlightPhase.FINAL,
        ),
        Waypoint(
            name="threshold",
            position=route['waypoints']['arr_threshold'],
            altitude_msl_ft=arr_apt.elevation_ft + 10,
            airspeed_kts=55,
            heading_deg=arr_rwy['heading_deg'],
            phase=FlightPhase.FLARE,
        ),
        Waypoint(
            name="rollout",
            position=route['waypoints']['arr_threshold'] + np.array([500, 0, 0]),
            altitude_msl_ft=arr_apt.elevation_ft,
            airspeed_kts=0,
            heading_deg=arr_rwy['heading_deg'],
            phase=FlightPhase.ROLLOUT,
        ),
    ]

    config = ScenarioConfig(
        name=f"xc_{dep_airport}_{arr_airport}",
        description=f"Cross-country from {dep_airport} to {arr_airport}",
        departure_airport=dep_airport,
        departure_runway=dep_runway,
        arrival_airport=arr_airport,
        arrival_runway=arr_runway,
        waypoints=waypoints,
        cruise_altitude_ft=cruise_altitude_ft,
        max_duration_s=1800.0,  # 30 minutes max
        success_criteria={
            'landed': True,
            'on_runway': True,
            'at_destination': True,
        },
    )

    return Scenario(config)


def create_go_around_scenario(
    airport: str = "SN65",
    runway: str = "35",
    go_around_altitude_ft: float = 100.0
) -> Scenario:
    """
    Create a go-around scenario.

    Approach -> Go-around at specified altitude -> Climb -> Re-enter pattern

    Args:
        airport: Airport ICAO code
        runway: Runway designator
        go_around_altitude_ft: AGL altitude to initiate go-around
    """
    world = World(origin_airport=airport)
    apt = get_airport(airport)
    pattern = world.get_pattern_waypoints(airport, runway, 1000.0)
    rwy = world.get_runway_geometry(airport, runway)

    pattern_alt_ft = apt.elevation_ft + 1000.0

    # Start on final approach
    waypoints = [
        Waypoint(
            name="final_start",
            position=pattern['waypoints']['final'] - np.array([500, 0, 0]),  # Further out
            altitude_msl_ft=apt.elevation_ft + 300,
            airspeed_kts=70,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.FINAL,
        ),
        Waypoint(
            name="go_around_point",
            position=pattern['waypoints']['final'],
            altitude_msl_ft=apt.elevation_ft + go_around_altitude_ft,
            airspeed_kts=60,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.GO_AROUND,
        ),
        Waypoint(
            name="climb_out",
            position=pattern['waypoints']['upwind'],
            altitude_msl_ft=pattern_alt_ft,
            airspeed_kts=80,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.CLIMB,
        ),
        # Re-enter pattern
        Waypoint(
            name="crosswind",
            position=pattern['waypoints']['crosswind'],
            altitude_msl_ft=pattern_alt_ft,
            airspeed_kts=90,
            heading_deg=(rwy['heading_deg'] + 90 * (-1 if pattern['is_left_pattern'] else 1)) % 360,
            phase=FlightPhase.CROSSWIND,
        ),
        Waypoint(
            name="downwind",
            position=pattern['waypoints']['downwind_abeam'],
            altitude_msl_ft=pattern_alt_ft,
            airspeed_kts=90,
            heading_deg=(rwy['heading_deg'] + 180) % 360,
            phase=FlightPhase.DOWNWIND,
        ),
        Waypoint(
            name="base",
            position=pattern['waypoints']['base'],
            altitude_msl_ft=pattern_alt_ft - 200,
            airspeed_kts=80,
            heading_deg=(rwy['heading_deg'] + 90 * (1 if pattern['is_left_pattern'] else -1)) % 360,
            phase=FlightPhase.BASE,
        ),
        Waypoint(
            name="final",
            position=pattern['waypoints']['final'],
            altitude_msl_ft=apt.elevation_ft + 200,
            airspeed_kts=65,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.FINAL,
        ),
        Waypoint(
            name="landing",
            position=pattern['waypoints']['threshold'],
            altitude_msl_ft=apt.elevation_ft + 10,
            airspeed_kts=55,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.FLARE,
        ),
    ]

    config = ScenarioConfig(
        name=f"go_around_{airport}_{runway}",
        description=f"Go-around practice at {airport} runway {runway}",
        departure_airport=airport,
        departure_runway=runway,
        arrival_airport=airport,
        arrival_runway=runway,
        waypoints=waypoints,
        pattern_altitude_ft=1000.0,
        max_duration_s=400.0,
        success_criteria={
            'go_around_executed': True,
            'landed': True,
        },
    )

    return Scenario(config)


def create_straight_out_departure_scenario(
    airport: str = "SN65",
    runway: str = "35",
    cruise_altitude_ft: float = 3500.0,
    cruise_distance_nm: float = 10.0
) -> Scenario:
    """
    Create a straight-out departure scenario.

    Takeoff -> Climb straight out -> Level at cruise -> Hold heading

    Args:
        airport: Departure airport
        runway: Departure runway
        cruise_altitude_ft: Target cruise altitude
        cruise_distance_nm: Distance to fly before scenario ends
    """
    world = World(origin_airport=airport)
    apt = get_airport(airport)
    rwy = world.get_runway_geometry(airport, runway)

    heading_rad = rwy['heading_rad']
    cruise_dist_m = cruise_distance_nm * 1852

    # Calculate end point
    end_north = rwy['threshold'][0] + cruise_dist_m * np.cos(heading_rad)
    end_east = rwy['threshold'][1] + cruise_dist_m * np.sin(heading_rad)
    cruise_down = -((cruise_altitude_ft * 0.3048) - world.origin.alt_msl)

    waypoints = [
        Waypoint(
            name="liftoff",
            position=np.array(rwy['threshold']) + np.array([200, 0, 0]),
            altitude_msl_ft=apt.elevation_ft + 50,
            airspeed_kts=65,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.ROTATION,
        ),
        Waypoint(
            name="climb",
            position=np.array(rwy['threshold']) + np.array([2000, 0, cruise_down * 0.3]),
            altitude_msl_ft=apt.elevation_ft + (cruise_altitude_ft - apt.elevation_ft) * 0.3,
            airspeed_kts=80,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.CLIMB,
        ),
        Waypoint(
            name="level_off",
            position=np.array(rwy['threshold']) + np.array([5000, 0, cruise_down]),
            altitude_msl_ft=cruise_altitude_ft,
            airspeed_kts=100,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.CLIMB,
        ),
        Waypoint(
            name="cruise_end",
            position=np.array([end_north, end_east, cruise_down]),
            altitude_msl_ft=cruise_altitude_ft,
            airspeed_kts=110,
            heading_deg=rwy['heading_deg'],
            phase=FlightPhase.CRUISE,
        ),
    ]

    config = ScenarioConfig(
        name=f"straight_out_{airport}_{runway}",
        description=f"Straight-out departure from {airport} runway {runway}",
        departure_airport=airport,
        departure_runway=runway,
        arrival_airport=airport,  # Same airport (no landing)
        arrival_runway=runway,
        waypoints=waypoints,
        cruise_altitude_ft=cruise_altitude_ft,
        max_duration_s=300.0,
        success_criteria={
            'reached_cruise': True,
            'heading_maintained': True,
        },
    )

    return Scenario(config)


# =============================================================================
# SCENARIO REGISTRY
# =============================================================================

SCENARIO_BUILDERS = {
    'traffic_pattern': create_traffic_pattern_scenario,
    'cross_country': create_cross_country_scenario,
    'go_around': create_go_around_scenario,
    'straight_out': create_straight_out_departure_scenario,
}


def list_scenarios() -> List[str]:
    """List available scenario types."""
    return list(SCENARIO_BUILDERS.keys())


def create_scenario(scenario_type: str, **kwargs) -> Scenario:
    """
    Create a scenario by type.

    Args:
        scenario_type: Type of scenario ('traffic_pattern', 'cross_country', etc.)
        **kwargs: Arguments for the scenario builder

    Returns:
        Configured Scenario instance
    """
    if scenario_type not in SCENARIO_BUILDERS:
        raise ValueError(f"Unknown scenario type: {scenario_type}. "
                        f"Available: {list_scenarios()}")

    return SCENARIO_BUILDERS[scenario_type](**kwargs)


if __name__ == "__main__":
    print("AIDA Flight Scenarios")
    print("=" * 60)

    # Test traffic pattern scenario
    print("\n1. Traffic Pattern at SN65 RWY 35")
    tp = create_traffic_pattern_scenario("SN65", "35")
    print(f"   Name: {tp.config.name}")
    print(f"   Waypoints: {len(tp.waypoints)}")
    for wp in tp.waypoints:
        print(f"     - {wp.name}: {wp.phase.name}, {wp.airspeed_kts} kts, {wp.altitude_msl_ft:.0f} ft")

    # Test cross-country scenario
    print("\n2. Cross-Country SN65 -> KHUT")
    xc = create_cross_country_scenario("SN65", "35", "KHUT", "13")
    print(f"   Name: {xc.config.name}")
    print(f"   Waypoints: {len(xc.waypoints)}")
    for wp in xc.waypoints:
        print(f"     - {wp.name}: {wp.phase.name}, {wp.airspeed_kts} kts")

    # Test go-around scenario
    print("\n3. Go-Around at SN65")
    ga = create_go_around_scenario("SN65", "35")
    print(f"   Name: {ga.config.name}")
    print(f"   Waypoints: {len(ga.waypoints)}")

    # Test straight-out departure
    print("\n4. Straight-Out Departure from SN65")
    so = create_straight_out_departure_scenario("SN65", "35")
    print(f"   Name: {so.config.name}")
    print(f"   Cruise altitude: {so.config.cruise_altitude_ft} ft")

    print("\n" + "=" * 60)
    print("Available scenario types:", list_scenarios())
