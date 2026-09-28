#!/usr/bin/env python3
"""
Generalized Cross-Country (XC) Controller with Triangle Intercept Pattern

This controller implements a triangle intercept approach pattern that can be
configured for any origin/destination airport pair. It uses a state machine
for flight phases and provides smooth transitions between phases.

Configuration is done via an AirportConfig dataclass that specifies:
- Airport position (in local XY coordinates from origin)
- Runway heading
- Field elevation

Usage:
    from generalized_xc_controller import GeneralizedXCController, AirportConfig

    # Define airports
    origin = AirportConfig(
        icao="SN65",
        name="Lake Waltanna",
        x_ft=0.0, y_ft=0.0,  # Origin
        elevation_ft=1448.0,
        runway_heading_deg=4.0
    )
    destination = AirportConfig(
        icao="KHUT",
        name="Hutchinson Regional",
        x_ft=173228.0, y_ft=-69882.0,  # ~35nm NW
        elevation_ft=1542.0,
        runway_heading_deg=314.0
    )

    # Create controller
    controller = GeneralizedXCController(origin, destination, cruise_altitude_ft=5500.0)

    # In flight loop
    action = controller.compute_action(state, sim_time)

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpu-flight-dynamics' / 'python'))
from flight_dynamics import StateIndex

# Unit conversion constants
FT_TO_M = 0.3048
M_TO_FT = 3.28084
NM_TO_FT = 6076.12
FT_TO_NM = 1 / 6076.12
KTS_TO_FPS = 1.68781
FPS_TO_KTS = 1 / 1.68781

# Laminar Research Cessna 172 SP Skyhawk (180 HP), X-Plane 12
# Aircraft/Laminar Research/Cessna 172 SP/Cessna_172SP.acf
# Rotate stays after the clean stall and below best-angle speed.
# Climb targets best rate. Do not substitute speeds that are not in this list.
C172_VSO_KTS = 40.0          # stall, landing configuration
C172_VS_KTS = 48.0           # stall, clean
C172_VX_KTS = 62.0           # best angle
C172_VY_KTS = 74.0           # best rate
C172_VBG_KTS = 68.0          # best glide
C172_VFE1_KTS = 110.0        # first flap limit
C172_VFEM_KTS = 85.0         # full-flap limit
C172_VNO_KTS = 120.0         # max structural cruise
C172_VNE_KTS = 163.0         # never exceed
C172_STALL_WARN_AOA_DEG = 12.0
C172_V_ROTATE_KTS = 54.0     # Vs < rotate < Vx

# Turning point sits on the runway backcourse, behind the landing threshold.
# A fixed 10 nm offset on a short leg (SN65→KAAO is ~21 nm) places that point
# past the field, so distance-to-destination grows until TURN_TO_INTERCEPT.
# Scale with the direct route: clip(0.4 × direct_nm, 3 nm, 10 nm).
TP_DISTANCE_MAX_NM = 10.0
TP_DISTANCE_MIN_NM = 3.0
TP_DISTANCE_FRACTION = 0.4


def adaptive_turn_point_distance_nm(direct_distance_nm: float,
                                    requested_nm: Optional[float] = None) -> float:
    """Route-scaled distance from the threshold back to the turning point.

    ``adaptive = clip(0.4 * direct_nm, 3 nm, 10 nm)``.

    When ``requested_nm`` is omitted, that adaptive distance is used. An
    explicit request is kept when it is shorter, and is reduced to the cap
    when it would fly past a short leg (the old default of 10 nm on
    SN65→KAAO).
    """
    direct = max(float(direct_distance_nm), 0.0)
    adaptive = TP_DISTANCE_FRACTION * direct
    adaptive = min(TP_DISTANCE_MAX_NM, max(TP_DISTANCE_MIN_NM, adaptive))
    if requested_nm is None:
        return float(adaptive)
    requested = float(requested_nm)
    if not np.isfinite(requested) or requested < 0.0:
        return float(adaptive)
    return float(min(requested, adaptive))


class XCPhase(Enum):
    """Flight phases for cross-country flight."""
    GROUND_ROLL = 1
    ROTATION = 2
    INITIAL_CLIMB = 3
    CLIMB = 4
    CRUISE_TO_TP = 5
    TURN_TO_INTERCEPT = 6
    INTERCEPT_LEG = 7
    FINAL_APPROACH = 8
    SHORT_FINAL = 9
    FLARE = 10
    ROLLOUT = 11
    LANDING = 12
    LANDED = 13


@dataclass
class AirportConfig:
    """
    Configuration for an airport.

    Positions can be specified in either:
    - Local XY coordinates (x_ft, y_ft) - used directly
    - Lat/Lon coordinates (lat, lon) - converted using reference point

    The viewer_runway_heading_deg is the heading of the runway as rendered
    in the 3D viewer. This may differ from the real runway_heading_deg due to
    simplifications in the viewer (e.g., SN65's runway is rendered at 0°
    even though the real runway is at 4°). Aircraft positioning uses this
    field to ensure the aircraft starts on the visual centerline.
    """
    icao: str                        # ICAO code (e.g., "KHUT")
    name: str                        # Full name
    x_ft: float = 0.0               # X position in feet (north from origin)
    y_ft: float = 0.0               # Y position in feet (east from origin)
    elevation_ft: float = 0.0       # Field elevation MSL in feet
    runway_heading_deg: float = 0.0  # Primary runway heading (for display)
    viewer_runway_heading_deg: Optional[float] = None  # Heading in viewer (for positioning)

    # Optional lat/lon (converted to x_ft/y_ft if provided)
    lat: Optional[float] = None
    lon: Optional[float] = None

    def get_viewer_heading_deg(self) -> float:
        """Get the runway heading as rendered in the viewer."""
        if self.viewer_runway_heading_deg is not None:
            return self.viewer_runway_heading_deg
        return self.runway_heading_deg

    def get_threshold_position(self, threshold_offset_ft: float = 2000.0):
        """
        Get runway threshold position.

        Args:
            threshold_offset_ft: Distance from airport center to threshold

        Returns:
            (threshold_x_ft, threshold_y_ft)
        """
        # Threshold is behind the runway heading (backcourse direction)
        backcourse = np.deg2rad(self.runway_heading_deg + 180)
        threshold_x = self.x_ft + threshold_offset_ft * np.cos(backcourse)
        threshold_y = self.y_ft + threshold_offset_ft * np.sin(backcourse)
        return threshold_x, threshold_y


# Pre-defined airport configurations for Kansas region
# NOTE: viewer_runway_heading_deg is the heading of the runway as rendered in the
# 3D viewer. This must match the visual runway or aircraft will start off-centerline.
KANSAS_AIRPORTS = {
    "SN65": AirportConfig(
        icao="SN65",
        name="Lake Waltanna Airport",
        x_ft=0.0,
        y_ft=0.0,
        elevation_ft=1448.0,
        runway_heading_deg=4.0,
        viewer_runway_heading_deg=0.0,  # Viewer runway is North-South (no rotation)
        lat=37.594167,
        lon=-97.615833
    ),
    "KHUT": AirportConfig(
        icao="KHUT",
        name="Hutchinson Regional Airport",
        x_ft=52800.0 * M_TO_FT,   # ~173,228 ft
        y_ft=-21300.0 * M_TO_FT,  # ~-69,882 ft
        elevation_ft=1542.0,
        runway_heading_deg=314.0,
        viewer_runway_heading_deg=314.0,  # Viewer runway matches real heading
        lat=38.066167,
        lon=-97.860500
    ),
    "KICT": AirportConfig(
        icao="KICT",
        name="Wichita Eisenhower National",
        x_ft=6156.0 * M_TO_FT,    # ~6km north
        y_ft=15000.0 * M_TO_FT,   # ~15km east
        elevation_ft=1333.0,
        runway_heading_deg=14.0,
        viewer_runway_heading_deg=14.0,  # Viewer runway matches real heading
        lat=37.649944,
        lon=-97.433056
    ),
    "KAAO": AirportConfig(
        icao="KAAO",
        name="Colonel James Jabara Airport",
        x_ft=55880.0,   # ~9.2nm north of SN65
        y_ft=115140.0,  # ~19nm east of SN65
        elevation_ft=1421.0,
        runway_heading_deg=180.0,
        viewer_runway_heading_deg=180.0,  # Viewer runway matches real heading
        lat=37.747500,
        lon=-97.221389
    ),
    "K50K": AirportConfig(
        icao="K50K",
        name="Pawnee Municipal Airport",
        x_ft=215034.0,   # ~65.5km north of SN65
        y_ft=-436524.0,  # ~133km west of SN65 (negative = west)
        elevation_ft=2200.0,
        runway_heading_deg=170.0,
        viewer_runway_heading_deg=170.0,  # Viewer runway matches real heading
        lat=38.184,
        lon=-99.127
    ),
}


def latlon_to_xy(lat: float, lon: float,
                ref_lat: float = 37.594167, ref_lon: float = -97.615833) -> tuple:
    """
    Convert lat/lon to local XY coordinates in feet.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        ref_lat: Reference latitude (origin)
        ref_lon: Reference longitude (origin)

    Returns:
        (x_ft, y_ft) - North, East from reference point
    """
    lat_to_ft = 364567.0  # ~60nm per degree * 6076 ft/nm
    lon_to_ft = 364567.0 * np.cos(np.deg2rad(ref_lat))
    x = (lat - ref_lat) * lat_to_ft
    y = (lon - ref_lon) * lon_to_ft
    return x, y


class GeneralizedXCController:
    """
    Generalized Cross-Country Controller with Triangle Intercept Pattern.

    This controller implements a complete flight from takeoff to landing
    using a triangle intercept approach pattern:

    1. Takeoff from origin airport on runway heading
    2. Turn to cruise heading toward destination
    3. Climb to cruise altitude
    4. Cruise toward turning point (TP) behind destination runway
    5. Turn to intercept final approach course
    6. Descend on glideslope
    7. Land on destination runway

    All internal calculations are in IMPERIAL units (ft, ft/s, nm).
    The controller accepts metric state from flight_dynamics and converts internally.
    """

    def __init__(self,
                 origin: AirportConfig,
                 destination: AirportConfig,
                 cruise_altitude_ft: float = 5500.0,
                 pattern_altitude_ft: float = 1500.0,
                 tp_distance_nm: Optional[float] = None):
        """
        Initialize the controller.

        Args:
            origin: Origin airport configuration
            destination: Destination airport configuration
            cruise_altitude_ft: Cruise altitude in feet MSL
            pattern_altitude_ft: Traffic pattern altitude AGL
            tp_distance_nm: Optional turning-point distance from the
                destination threshold (nm). Omit it to scale with the
                direct route (see ``adaptive_turn_point_distance_nm``).
                A request longer than that scaled distance is capped so a
                stale 10 nm value cannot fly past a short leg.
        """
        self.origin = origin
        self.destination = destination

        # Flight parameters - altitudes are AGL (simulator ground is Z=0)
        self.cruise_altitude_ft = cruise_altitude_ft
        self.pattern_altitude_ft = pattern_altitude_ft  # AGL, no elevation offset

        # Speeds in ft/s (internal unit). Rotate and climb come from the
        # X-Plane 172 SP table above: liftoff after Vs, climb at Vy.
        self.v_s = C172_VS_KTS * KTS_TO_FPS
        self.v_x = C172_VX_KTS * KTS_TO_FPS
        self.v_rotate = C172_V_ROTATE_KTS * KTS_TO_FPS
        self.v_climb = C172_VY_KTS * KTS_TO_FPS
        self.v_cruise = 110.0 * KTS_TO_FPS     # below Vno (120) and Vne (163)
        self.v_approach = 65.0 * KTS_TO_FPS
        self.v_touchdown = 50.0 * KTS_TO_FPS
        self.stall_warn_aoa_deg = C172_STALL_WARN_AOA_DEG

        # Altitude parameters - all altitudes are AGL (simulator ground is Z=0)
        # No elevation offsets since simulator has no terrain model
        self.initial_climb_alt_ft = 100.0        # 100ft AGL before turn
        self.turn_to_cruise_altitude_ft = 1000.0 # 1000ft AGL before cruise heading
        self.flare_altitude_ft = 50.0            # 50ft AGL to start flare
        self.touchdown_altitude_ft = 3.0         # 3ft AGL touchdown
        self.short_final_distance_ft = 2500.0

        # Runway parameters
        self.departure_heading = np.deg2rad(origin.runway_heading_deg)
        self.runway_heading = np.deg2rad(destination.runway_heading_deg)
        self.backcourse = np.deg2rad((destination.runway_heading_deg + 180) % 360)

        # Destination positions
        self.dest_x_ft = destination.x_ft
        self.dest_y_ft = destination.y_ft

        # Threshold position (slightly behind airport center along backcourse)
        threshold_offset_ft = 610.0 * M_TO_FT  # ~2001 ft
        self.threshold_x_ft = self.dest_x_ft + threshold_offset_ft * np.cos(self.backcourse)
        self.threshold_y_ft = self.dest_y_ft + threshold_offset_ft * np.sin(self.backcourse)

        # Aimpoint (touchdown target) - at threshold
        self.aimpoint_x_ft = self.threshold_x_ft
        self.aimpoint_y_ft = self.threshold_y_ft

        # Direct route length, then a turning point behind the threshold on
        # the backcourse. Short legs must not inherit a fixed 10 nm offset.
        direct_ft = float(np.hypot(self.dest_x_ft - origin.x_ft,
                                   self.dest_y_ft - origin.y_ft))
        self.direct_distance_nm = direct_ft * FT_TO_NM
        self.tp_distance_nm = adaptive_turn_point_distance_nm(
            self.direct_distance_nm, tp_distance_nm)
        tp_distance_ft = self.tp_distance_nm * NM_TO_FT
        self.tp_x_ft = self.threshold_x_ft + tp_distance_ft * np.cos(self.backcourse)
        self.tp_y_ft = self.threshold_y_ft + tp_distance_ft * np.sin(self.backcourse)

        # Cruise heading: direct to turning point from origin
        self.cruise_heading = np.arctan2(self.tp_y_ft - origin.y_ft,
                                          self.tp_x_ft - origin.x_ft)

        # Trigger distances
        self.tp_trigger_distance_ft = 3000.0  # ~0.5nm

        # Phase state
        self.phase = XCPhase.GROUND_ROLL
        self.phase_start_time = 0.0
        self.has_turned_to_cruise = False

        # LLM Override mode
        self.override_active = False
        self.override_heading = None
        self.override_altitude = None
        self.override_land_target = None
        self._needs_recalculate = False

        # Control gains
        self.kp_pitch = 1.0
        self.kd_pitch = 0.4
        self.kp_roll = 1.5
        self.kd_roll = 0.5
        self.turn_bank_angle = np.deg2rad(30.0)

        # Altitude hold integral state
        self.alt_integral_ft = 0.0
        self.alt_integral_max = 500.0  # Anti-windup clamp (ft·s)

        # Climb-speed integral (kt·s) so a fixed deck angle cannot accelerate
        # through Vy the way the old 8° pitch did (~100 kt).
        self.vy_integral_kts = 0.0

        # Target pitch angles
        self.pitch_rotate = np.deg2rad(10.0)
        self.pitch_climb = np.deg2rad(8.0)
        self.pitch_cruise = np.deg2rad(0.0)
        self.pitch_descent = np.deg2rad(-3.0)
        self.pitch_approach = np.deg2rad(-2.0)
        self.pitch_flare = np.deg2rad(5.0)

        # Glideslope angle
        self.glideslope_deg = 3.5

        # Print configuration
        self._print_config()

    def _print_config(self):
        """Print controller configuration summary."""
        print(f"\n{'='*60}")
        print(f"Generalized XC Controller: {self.origin.icao} -> {self.destination.icao}")
        print(f"{'='*60}")
        print(f"Origin:      {self.origin.name} ({self.origin.icao})")
        print(f"  Position:  ({self.origin.x_ft/NM_TO_FT:.1f}nm, {self.origin.y_ft/NM_TO_FT:.1f}nm)")
        print(f"  Elevation: {self.origin.elevation_ft:.0f} ft")
        print(f"  Runway:    {self.origin.runway_heading_deg:.0f}°")
        print(f"\nDestination: {self.destination.name} ({self.destination.icao})")
        print(f"  Position:  ({self.dest_x_ft/NM_TO_FT:.1f}nm, {self.dest_y_ft/NM_TO_FT:.1f}nm)")
        print(f"  Elevation: {self.destination.elevation_ft:.0f} ft")
        print(f"  Runway:    {self.destination.runway_heading_deg:.0f}°")
        print(f"\nFlight Plan:")
        print(f"  Cruise Alt:    {self.cruise_altitude_ft:.0f} ft")
        print(f"  Rotate:        {C172_V_ROTATE_KTS:.0f} kt (after Vs {C172_VS_KTS:.0f}, before Vx {C172_VX_KTS:.0f})")
        print(f"  Climb:         Vy {C172_VY_KTS:.0f} kt")
        print(f"  Cruise Hdg:    {np.rad2deg(self.cruise_heading):.0f}°")
        print(f"  Turning Point: ({self.tp_x_ft/NM_TO_FT:.1f}nm, {self.tp_y_ft/NM_TO_FT:.1f}nm)")
        print(f"  TP Distance:   {self.tp_distance_nm:.1f} nm behind threshold")
        print(f"  Direct Dist:   {self.direct_distance_nm:.1f} nm")
        print(f"{'='*60}\n")

    def reset(self):
        """Reset controller to initial state."""
        self.phase = XCPhase.GROUND_ROLL
        self.phase_start_time = 0.0
        self.has_turned_to_cruise = False
        self.alt_integral_ft = 0.0
        self.vy_integral_kts = 0.0
        self.clear_overrides()

    def clear_overrides(self):
        """Clear all LLM override commands."""
        self.override_active = False
        self.override_heading = None
        self.override_altitude = None
        self.override_land_target = None
        self._needs_recalculate = False

    def set_heading_override(self, heading_deg: float):
        """Set heading override from LLM command."""
        self.override_active = True
        self.override_heading = heading_deg
        print(f"[Controller] Heading override set: {heading_deg}°")

    def set_altitude_override(self, altitude_ft: float):
        """Set altitude override from LLM command."""
        self.override_active = True
        self.override_altitude = altitude_ft
        print(f"[Controller] Altitude override set: {altitude_ft} ft")

    def set_land_override(self, target: str = None):
        """Set landing override - resume approach to target.

        Clears heading override and flags for route recalculation on the
        next compute_action() call, which has access to current position.
        """
        self.override_active = True
        self.override_land_target = target or self.destination.icao
        self.override_heading = None  # Clear heading diversion
        self._needs_recalculate = True  # Will be consumed in compute_action
        print(f"[Controller] Landing override set: {self.override_land_target}")

    def recalculate_approach(self, x_ft: float, y_ft: float):
        """Recalculate approach geometry from current position.

        Determines the correct flight phase based on the aircraft's
        position relative to the turning point and threshold, so the
        controller can properly re-acquire the approach after a diversion.
        """
        dist_to_tp = np.sqrt((x_ft - self.tp_x_ft)**2 +
                             (y_ft - self.tp_y_ft)**2)
        dist_to_threshold = np.sqrt((x_ft - self.threshold_x_ft)**2 +
                                    (y_ft - self.threshold_y_ft)**2)

        # Check if aircraft is roughly on the inbound side of the TP
        # (i.e., between TP and threshold along the runway axis)
        tp_to_threshold_x = self.threshold_x_ft - self.tp_x_ft
        tp_to_threshold_y = self.threshold_y_ft - self.tp_y_ft
        tp_to_aircraft_x = x_ft - self.tp_x_ft
        tp_to_aircraft_y = y_ft - self.tp_y_ft
        dot = (tp_to_aircraft_x * tp_to_threshold_x +
               tp_to_aircraft_y * tp_to_threshold_y)
        past_tp_toward_threshold = dot > 0

        if dist_to_tp > self.tp_trigger_distance_ft and not past_tp_toward_threshold:
            # Far from TP and not yet past it — cruise to TP first
            self.phase = XCPhase.CRUISE_TO_TP
            print(f"[Controller] Recalculated: CRUISE_TO_TP "
                  f"({dist_to_tp / NM_TO_FT:.1f} nm to TP)")
        elif dist_to_threshold > 3 * NM_TO_FT:
            # Past TP or close to it, but far from threshold — turn to intercept
            self.phase = XCPhase.TURN_TO_INTERCEPT
            print(f"[Controller] Recalculated: TURN_TO_INTERCEPT "
                  f"({dist_to_threshold / NM_TO_FT:.1f} nm to threshold)")
        else:
            # Close to threshold — go straight to final approach
            self.phase = XCPhase.FINAL_APPROACH
            print(f"[Controller] Recalculated: FINAL_APPROACH "
                  f"({dist_to_threshold / NM_TO_FT:.1f} nm to threshold)")

    def get_phase_name(self) -> str:
        """Get current phase name."""
        return self.phase.name

    def get_status(self) -> dict:
        """Get current controller status."""
        return {
            "phase": self.phase.name,
            "origin": self.origin.icao,
            "destination": self.destination.icao,
            "cruise_altitude_ft": self.cruise_altitude_ft,
            "direct_distance_nm": self.direct_distance_nm,
            "tp_distance_nm": self.tp_distance_nm,
            "override_active": self.override_active,
            "override_heading": self.override_heading,
            "override_altitude": self.override_altitude,
        }

    def get_target_altitude_ft(self) -> float:
        """Get the current target altitude based on phase and overrides.

        Returns the altitude the controller is trying to achieve:
        - During climb/cruise: cruise_altitude_ft (or override if active)
        - During approach: calculated glideslope altitude
        - During ground operations: 0
        """
        if self.override_active and self.override_altitude is not None:
            return self.override_altitude

        if self.phase in (XCPhase.CLIMB, XCPhase.CRUISE_TO_TP,
                          XCPhase.TURN_TO_INTERCEPT):
            return self.cruise_altitude_ft
        elif self.phase in (XCPhase.INTERCEPT_LEG, XCPhase.FINAL_APPROACH):
            # Glideslope - would need current position to calculate
            # Return pattern altitude as approximation
            return self.pattern_altitude_ft
        else:
            return self.cruise_altitude_ft  # Default to cruise

    def get_target_heading_deg(self) -> float:
        """Get the current target heading based on phase and overrides.

        Returns the heading the controller is trying to achieve.
        """
        if self.override_active and self.override_heading is not None:
            return self.override_heading

        if self.phase in (XCPhase.GROUND_ROLL, XCPhase.ROTATION, XCPhase.INITIAL_CLIMB):
            return self.origin.runway_heading_deg
        elif self.phase in (XCPhase.TURN_TO_INTERCEPT, XCPhase.INTERCEPT_LEG,
                            XCPhase.FINAL_APPROACH, XCPhase.SHORT_FINAL,
                            XCPhase.FLARE, XCPhase.ROLLOUT, XCPhase.LANDING):
            return np.rad2deg(self.runway_heading) % 360
        else:
            # Cruise - heading to destination
            return np.rad2deg(np.arctan2(self.tp_y_ft, self.tp_x_ft)) % 360

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _heading_control(self, target: float, current: float,
                         roll: float, roll_rate: float) -> float:
        """Compute aileron input for heading control."""
        error = self._normalize_angle(target - current)
        target_roll = np.clip(error * 0.8, -self.turn_bank_angle, self.turn_bank_angle)
        roll_error = target_roll - roll
        return np.clip(self.kp_roll * roll_error - self.kd_roll * roll_rate, -1.0, 1.0)

    def _vy_climb_command(self, airspeed_fps: float, base_pitch: float):
        """Pitch and throttle that seek Vy (X-Plane 172 SP, 74 kt).

        Below Vx the nose stays shallow (``base_pitch``, capped at 6°) and
        power stays full so the airplane accelerates off the stall. Above
        Vx the deck angle is the 10° rotation attitude, under the 12°
        stall-warn AoA. Full throttle at that attitude trims near 88 kt
        in this thrust model, so throttle — not a steeper pitch — holds
        Vy. Climb rate is whatever excess thrust remains at that power
        (about 400 fpm here), not the book 172 rate.
        """
        err_kts = (airspeed_fps - self.v_climb) * FPS_TO_KTS
        self.vy_integral_kts += err_kts * 0.02
        self.vy_integral_kts = float(np.clip(self.vy_integral_kts, -80.0, 80.0))

        if airspeed_fps < self.v_x:
            return min(base_pitch, np.deg2rad(6.0)), 1.0

        pitch = float(np.deg2rad(min(10.0, self.stall_warn_aoa_deg)))
        # Positive speed error (fast) reduces throttle. Integral holds the
        # power setting that trims at Vy.
        thr_corr = float(np.clip(
            0.02 * err_kts + 0.008 * self.vy_integral_kts, -0.2, 0.65))
        throttle = float(np.clip(1.0 - thr_corr, 0.40, 1.0))
        return pitch, throttle

    def _pitch_control(self, target: float, current: float, pitch_rate: float) -> float:
        """Compute elevator input for pitch control."""
        error = target - current
        return -np.clip(self.kp_pitch * error - self.kd_pitch * pitch_rate, -1.0, 1.0)

    def _cruise_energy_hold(self, altitude_ft: float, climb_rate_fps: float,
                            target_altitude_ft: float,
                            airspeed_fps: Optional[float] = None,
                            trim_airspeed: bool = False):
        """PI altitude hold: vertical-speed pitch plus throttle trim.

        This is the cruise law. ``TURN_TO_INTERCEPT`` uses it too, with
        ``trim_airspeed`` set, so a banked procedure turn does not lock
        pitch at the cruise attitude and a fixed 0.7 throttle (that
        combination ballooned ~370 ft and ran 120 kt up to ~135 kt).

        Returns ``(pitch_target_rad, throttle)``.
        """
        alt_error_ft = altitude_ft - target_altitude_ft

        # Accumulate integral (anti-windup). dt matches the XC server step.
        self.alt_integral_ft += alt_error_ft * 0.02
        self.alt_integral_ft = np.clip(self.alt_integral_ft,
                                       -self.alt_integral_max, self.alt_integral_max)

        target_vs_fps = -0.08 * alt_error_ft - 0.005 * self.alt_integral_ft
        target_vs_fps = np.clip(target_vs_fps, -800 / 60, 800 / 60)
        vs_error_fps = climb_rate_fps - target_vs_fps

        pitch_correction = np.deg2rad(-0.6 * vs_error_fps)
        pitch_correction = np.clip(pitch_correction, np.deg2rad(-10.0), np.deg2rad(10.0))
        pitch_target = self.pitch_cruise + pitch_correction

        base_throttle = 0.50
        throttle = base_throttle - 0.002 * alt_error_ft
        if trim_airspeed and airspeed_fps is not None:
            # Bleed excess kinetic energy. Cruise leaves this off so an
            # already-trimmed cruise leg is unchanged.
            throttle += -0.004 * (airspeed_fps - self.v_cruise)
        throttle = float(np.clip(throttle, 0.3, 0.7))
        return pitch_target, throttle

    def _resolve_cruise_altitude_ft(self) -> float:
        """Cruise target, or an active altitude override when one is set."""
        if self.override_active and self.override_altitude:
            return float(self.override_altitude)
        return float(self.cruise_altitude_ft)

    def compute_action(self, state: np.ndarray, sim_time: float) -> np.ndarray:
        """
        Compute control action from current state.

        Args:
            state: Aircraft state in METRIC units (from flight_dynamics)
            sim_time: Simulation time in seconds

        Returns:
            Action array: [throttle, aileron, elevator, rudder, flaps, spoiler, brake]
        """
        # Extract state and convert to imperial (ft, ft/s)
        x_ft = state[StateIndex.X] * M_TO_FT
        y_ft = state[StateIndex.Y] * M_TO_FT
        z_ft = state[StateIndex.Z] * M_TO_FT
        altitude_ft = -z_ft  # Z is down, so altitude = -Z

        u_fps = state[StateIndex.U] * M_TO_FT
        v_fps = state[StateIndex.V] * M_TO_FT
        w_fps = state[StateIndex.W] * M_TO_FT
        airspeed_fps = np.sqrt(u_fps**2 + v_fps**2 + w_fps**2)

        phi = state[StateIndex.PHI]
        theta = state[StateIndex.THETA]
        psi = state[StateIndex.PSI]
        p = state[StateIndex.P]
        q = state[StateIndex.Q]

        # Body-axis climb proxy used by the approach/flare laws (unchanged).
        climb_rate_body_fps = -w_fps
        # Inertial climb rate, positive up, for cruise and the intercept turn.
        # Body-axis -w matches this only wings-level near zero pitch. In a
        # bank, -w looks like a descent (the extra alpha that holds altitude)
        # and the energy hold then pumps the elevator.
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        z_dot_fps = (-sin_theta * u_fps
                     + np.sin(phi) * cos_theta * v_fps
                     + np.cos(phi) * cos_theta * w_fps)
        climb_rate_fps = -z_dot_fps

        # Recalculate approach if flagged (after diversion + land command)
        # Never recalculate once in terminal phases (on the ground)
        if self._needs_recalculate:
            self._needs_recalculate = False
            terminal_phases = {XCPhase.SHORT_FINAL, XCPhase.FLARE,
                               XCPhase.ROLLOUT, XCPhase.LANDING, XCPhase.LANDED}
            if self.phase not in terminal_phases:
                self.recalculate_approach(x_ft, y_ft)

        # Distances in ft
        dist_to_tp_ft = np.sqrt((x_ft - self.tp_x_ft)**2 + (y_ft - self.tp_y_ft)**2)
        dist_to_threshold_ft = np.sqrt((x_ft - self.threshold_x_ft)**2 +
                                        (y_ft - self.threshold_y_ft)**2)
        dist_to_aimpoint_ft = np.sqrt((x_ft - self.aimpoint_x_ft)**2 +
                                       (y_ft - self.aimpoint_y_ft)**2)

        # Phase transitions
        if self.phase == XCPhase.GROUND_ROLL and airspeed_fps >= self.v_rotate:
            self.phase = XCPhase.ROTATION
            self.phase_start_time = sim_time
        elif self.phase == XCPhase.ROTATION and altitude_ft > self.initial_climb_alt_ft:
            self.phase = XCPhase.INITIAL_CLIMB
            self.vy_integral_kts = 0.0
        elif self.phase == XCPhase.INITIAL_CLIMB and altitude_ft > self.turn_to_cruise_altitude_ft:
            self.phase = XCPhase.CLIMB
        elif self.phase == XCPhase.CLIMB and altitude_ft >= self.cruise_altitude_ft - 50:
            self.phase = XCPhase.CRUISE_TO_TP
            self.alt_integral_ft = 0.0  # Reset integral for clean altitude hold start
        elif self.phase == XCPhase.CRUISE_TO_TP and dist_to_tp_ft < self.tp_trigger_distance_ft:
            # Don't transition while a heading override is active — pilot is
            # deliberately diverting, so stay in cruise mode.
            if not (self.override_active and self.override_heading is not None):
                self.phase = XCPhase.TURN_TO_INTERCEPT
        elif self.phase == XCPhase.TURN_TO_INTERCEPT:
            heading_error = abs(self._normalize_angle(psi - self.runway_heading))
            if heading_error < np.deg2rad(10):
                self.phase = XCPhase.INTERCEPT_LEG
        elif self.phase == XCPhase.INTERCEPT_LEG and dist_to_threshold_ft <= 3 * NM_TO_FT:
            self.phase = XCPhase.FINAL_APPROACH
        elif self.phase == XCPhase.FINAL_APPROACH and dist_to_threshold_ft <= self.short_final_distance_ft:
            self.phase = XCPhase.SHORT_FINAL
        elif self.phase == XCPhase.SHORT_FINAL and altitude_ft <= self.flare_altitude_ft:
            self.phase = XCPhase.FLARE
        elif self.phase == XCPhase.FLARE and altitude_ft <= self.touchdown_altitude_ft:
            self.phase = XCPhase.ROLLOUT
        elif self.phase == XCPhase.ROLLOUT and airspeed_fps < 30.0 * KTS_TO_FPS:
            self.phase = XCPhase.LANDING
        elif self.phase == XCPhase.LANDING and airspeed_fps < 5.0 * KTS_TO_FPS:
            self.phase = XCPhase.LANDED
            self.clear_overrides()  # Prevent re-triggering after landing

        # Control logic by phase
        if self.phase == XCPhase.GROUND_ROLL:
            # Slight aft stick at higher speeds to unload nosewheel and
            # pre-position for rotation, preventing ground bouncing.
            pre_rotate_elevator = 0.0
            if airspeed_fps > 0.8 * self.v_rotate:
                pre_rotate_elevator = self._pitch_control(np.deg2rad(2.0), theta, q)
            return np.array([1.0, 0.0, pre_rotate_elevator, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.ROTATION:
            # Smooth pitch ramp over 2 seconds instead of step input
            rotation_elapsed = sim_time - self.phase_start_time
            ramp_time = 2.0  # seconds to reach full rotation pitch
            ramp_fraction = min(rotation_elapsed / ramp_time, 1.0)
            pitch_target = ramp_fraction * self.pitch_rotate
            return np.array([1.0, 0.0,
                           self._pitch_control(pitch_target, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.INITIAL_CLIMB:
            # Accelerate through Vx, then hold Vy at the rotation attitude
            # with a throttle trim. Pitch stays under the stall-warn AoA.
            alt_blend = min((altitude_ft - self.initial_climb_alt_ft) / 200.0, 1.0)
            alt_blend = max(alt_blend, 0.0)
            base_pitch = self.pitch_rotate * (1 - alt_blend) + self.pitch_climb * alt_blend
            pitch_target, throttle = self._vy_climb_command(airspeed_fps, base_pitch)
            return np.array([throttle,
                           self._heading_control(self.departure_heading, psi, phi, p),
                           self._pitch_control(pitch_target, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.CLIMB:
            if altitude_ft > self.turn_to_cruise_altitude_ft:
                self.has_turned_to_cruise = True
            # Respect heading override during climb
            if self.override_active and self.override_heading is not None:
                hdg = np.deg2rad(self.override_heading)
            else:
                hdg = self.cruise_heading if self.has_turned_to_cruise else self.departure_heading
            pitch_target, throttle = self._vy_climb_command(airspeed_fps, self.pitch_climb)
            return np.array([throttle,
                           self._heading_control(hdg, psi, phi, p),
                           self._pitch_control(pitch_target, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.CRUISE_TO_TP:
            # Check for LLM override commands
            if self.override_active and self.override_heading is not None:
                bearing = np.deg2rad(self.override_heading)
            else:
                bearing = np.arctan2(self.tp_y_ft - y_ft, self.tp_x_ft - x_ft)

            pitch_target, throttle = self._cruise_energy_hold(
                altitude_ft, climb_rate_fps, self._resolve_cruise_altitude_ft())

            return np.array([throttle,
                           self._heading_control(bearing, psi, phi, p),
                           self._pitch_control(pitch_target, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.TURN_TO_INTERCEPT:
            # Respect heading override even during intercept turn.
            # Vertical channel is the cruise energy hold (not a frozen
            # cruise pitch / 0.7 throttle) so altitude and airspeed stay
            # near the cruise targets through the bank.
            if self.override_active and self.override_heading is not None:
                hdg = np.deg2rad(self.override_heading)
            else:
                hdg = self.runway_heading
            pitch_target, throttle = self._cruise_energy_hold(
                altitude_ft, climb_rate_fps, self._resolve_cruise_altitude_ft(),
                airspeed_fps=airspeed_fps, trim_airspeed=True)
            # Replace the vertical component of lift lost to bank. The VS
            # loop trims the residual. ~2° at 30° of bank, 0° wings-level.
            cos_phi = float(np.cos(np.clip(phi, -np.deg2rad(50.0), np.deg2rad(50.0))))
            cos_phi = max(abs(cos_phi), 0.55)
            pitch_target = pitch_target + np.deg2rad(12.0 * (1.0 / cos_phi - 1.0))
            pitch_target = float(np.clip(pitch_target, np.deg2rad(-12.0), np.deg2rad(12.0)))
            return np.array([throttle,
                           self._heading_control(hdg, psi, phi, p),
                           self._pitch_control(pitch_target, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.INTERCEPT_LEG:
            # Cross-track correction to centerline
            dx_ft = x_ft - self.aimpoint_x_ft
            dy_ft = y_ft - self.aimpoint_y_ft
            rwy_dx = np.cos(self.runway_heading)
            rwy_dy = np.sin(self.runway_heading)
            cross_track_ft = dx_ft * (-rwy_dy) + dy_ft * rwy_dx

            intercept_angle = np.clip(-cross_track_ft * 0.0005, -np.deg2rad(20), np.deg2rad(20))
            target_heading = self.runway_heading + intercept_angle

            # Glideslope target altitude (AGL, ground is Z=0)
            target_alt_ft = dist_to_aimpoint_ft * np.tan(np.deg2rad(self.glideslope_deg))
            alt_error_ft = altitude_ft - target_alt_ft

            # Speed protection
            min_speed_fps = 50.0 * KTS_TO_FPS
            speed_low = airspeed_fps < min_speed_fps

            if speed_low:
                target_descent_fps = -300 / 60
                throttle = 0.9
                spoiler = 0.0
                pitch_base = np.deg2rad(0.0)
            elif alt_error_ft > 2000:
                target_descent_fps = -3000 / 60
                throttle = 0.3
                spoiler = 0.8
                pitch_base = np.deg2rad(-8.0)
            elif alt_error_ft > 1000:
                target_descent_fps = -2500 / 60
                throttle = 0.35
                spoiler = 0.6
                pitch_base = np.deg2rad(-6.0)
            elif alt_error_ft > 500:
                target_descent_fps = -1500 / 60
                throttle = 0.4
                spoiler = 0.4
                pitch_base = np.deg2rad(-4.0)
            elif alt_error_ft > 200:
                target_descent_fps = -1000 / 60
                throttle = 0.45
                spoiler = 0.2
                pitch_base = np.deg2rad(-3.0)
            else:
                gs_descent_fps = -airspeed_fps * np.tan(np.deg2rad(self.glideslope_deg))
                target_descent_fps = gs_descent_fps - 0.05 * alt_error_ft
                throttle = 0.5
                spoiler = 0.0
                pitch_base = self.pitch_descent

            target_descent_fps = np.clip(target_descent_fps, -3000/60, 0)
            pitch_adjust = (climb_rate_body_fps - target_descent_fps) * 0.003
            pitch = pitch_base + pitch_adjust

            return np.array([throttle,
                           self._heading_control(target_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q),
                           0.0, 0.3, spoiler, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.FINAL_APPROACH:
            # Centerline tracking
            dx_ft = x_ft - self.aimpoint_x_ft
            dy_ft = y_ft - self.aimpoint_y_ft
            rwy_dx = np.cos(self.runway_heading)
            rwy_dy = np.sin(self.runway_heading)
            cross_track_ft = dx_ft * (-rwy_dy) + dy_ft * rwy_dx

            intercept_angle = np.clip(-cross_track_ft * 0.001, -np.deg2rad(15), np.deg2rad(15))
            target_heading = self.runway_heading + intercept_angle

            # Glideslope target altitude (AGL, ground is Z=0)
            target_alt_ft = dist_to_aimpoint_ft * np.tan(np.deg2rad(self.glideslope_deg))
            alt_error_ft = altitude_ft - target_alt_ft

            if alt_error_ft > 500:
                target_descent_fps = -2000 / 60
                throttle = 0.3
                spoiler = 0.5
                pitch_base = np.deg2rad(-5.0)
            elif alt_error_ft > 200:
                target_descent_fps = -1000 / 60
                throttle = 0.4
                spoiler = 0.3
                pitch_base = np.deg2rad(-3.0)
            else:
                gs_descent_fps = -airspeed_fps * np.tan(np.deg2rad(self.glideslope_deg))
                target_descent_fps = gs_descent_fps - 0.1 * alt_error_ft
                throttle = 0.45
                spoiler = 0.0
                pitch_base = self.pitch_approach

            target_descent_fps = np.clip(target_descent_fps, -2000/60, 0)
            pitch_adjust = (climb_rate_body_fps - target_descent_fps) * 0.003
            pitch = pitch_base + pitch_adjust

            return np.array([throttle,
                           self._heading_control(target_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q),
                           0.0, 0.5, spoiler, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.SHORT_FINAL:
            # Glideslope target altitude (AGL, ground is Z=0)
            target_alt_ft = dist_to_aimpoint_ft * np.tan(np.deg2rad(self.glideslope_deg))
            alt_error_ft = altitude_ft - target_alt_ft

            if alt_error_ft > 300:
                target_descent_fps = -1500 / 60
                throttle = 0.2
                spoiler = 0.4
                pitch_base = np.deg2rad(-4.0)
            elif alt_error_ft > 100:
                target_descent_fps = -800 / 60
                throttle = 0.3
                spoiler = 0.2
                pitch_base = np.deg2rad(-2.0)
            else:
                gs_descent_fps = -airspeed_fps * np.tan(np.deg2rad(self.glideslope_deg))
                target_descent_fps = gs_descent_fps - 0.1 * alt_error_ft
                throttle = 0.35
                spoiler = 0.0
                pitch_base = self.pitch_approach

            target_descent_fps = np.clip(target_descent_fps, -1500/60, 0)
            pitch_adjust = (climb_rate_body_fps - target_descent_fps) * 0.003
            pitch = pitch_base + pitch_adjust

            return np.array([throttle,
                           self._heading_control(self.runway_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q),
                           0.0, 0.7, spoiler, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.FLARE:
            # Flare: pitch up, idle throttle, arrest descent rate
            pitch_adjust = -climb_rate_body_fps * 0.005
            pitch = self.pitch_flare + pitch_adjust
            return np.array([0.0,
                           self._heading_control(self.runway_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q),
                           0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.ROLLOUT:
            # Rollout: on the ground, braking, maintain runway heading
            return np.array([0.0,
                           self._heading_control(self.runway_heading, psi, phi, p),
                           0.0,
                           0.0, 0.0, 0.0, 0.8], dtype=np.float32)

        elif self.phase == XCPhase.LANDING:
            # Landing: slow taxi/stop
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        elif self.phase == XCPhase.LANDED:
            # Fully stopped
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        return np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def create_controller(origin_icao: str, destination_icao: str,
                      cruise_altitude_ft: float = 5500.0,
                      custom_airports: dict = None,
                      tp_distance_nm: Optional[float] = None) -> GeneralizedXCController:
    """
    Factory function to create a controller for a given airport pair.

    Args:
        origin_icao: Origin airport ICAO code
        destination_icao: Destination airport ICAO code
        cruise_altitude_ft: Cruise altitude in feet
        custom_airports: Optional dict of additional AirportConfig objects
        tp_distance_nm: Optional turn-point distance (nm). Omit to scale
            with the direct route.

    Returns:
        Configured GeneralizedXCController
    """
    # Merge airport databases
    airports = dict(KANSAS_AIRPORTS)
    if custom_airports:
        airports.update(custom_airports)

    if origin_icao not in airports:
        raise ValueError(f"Unknown origin airport: {origin_icao}")
    if destination_icao not in airports:
        raise ValueError(f"Unknown destination airport: {destination_icao}")

    return GeneralizedXCController(
        origin=airports[origin_icao],
        destination=airports[destination_icao],
        cruise_altitude_ft=cruise_altitude_ft,
        tp_distance_nm=tp_distance_nm,
    )


if __name__ == "__main__":
    # Demo: Create controller for SN65 -> KHUT
    print("Testing Generalized XC Controller")
    print("="*60)

    # Method 1: Using factory function
    controller = create_controller("SN65", "KHUT", cruise_altitude_ft=5500.0)

    print(f"\nController created successfully!")
    print(f"Status: {controller.get_status()}")

    # Method 2: Using custom airport configs
    print("\n" + "="*60)
    print("Creating controller with custom airports...")

    custom_origin = AirportConfig(
        icao="XXXX",
        name="Custom Airport",
        x_ft=0.0,
        y_ft=0.0,
        elevation_ft=1500.0,
        runway_heading_deg=90.0
    )

    custom_dest = AirportConfig(
        icao="YYYY",
        name="Another Airport",
        x_ft=100000.0,  # ~16nm north
        y_ft=50000.0,   # ~8nm east
        elevation_ft=2000.0,
        runway_heading_deg=270.0
    )

    custom_controller = GeneralizedXCController(
        origin=custom_origin,
        destination=custom_dest,
        cruise_altitude_ft=4500.0
    )

    print(f"\nCustom controller created!")
    print(f"Status: {custom_controller.get_status()}")
