"""
Airport and Runway Database for AIDA Flight Simulation

Real-world airport data for the Wichita, Kansas region.
Supports multi-airport scenarios with accurate runway geometry.

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np


class TrafficPattern(Enum):
    """Traffic pattern direction."""
    LEFT = "left"
    RIGHT = "right"


class SurfaceType(Enum):
    """Runway surface types."""
    ASPHALT = "asphalt"
    CONCRETE = "concrete"
    TURF = "turf"
    GRAVEL = "gravel"


@dataclass
class RunwayEnd:
    """Single runway end (threshold) data."""
    designator: str          # e.g., "17", "35", "13L"
    lat: float               # Latitude (degrees)
    lon: float               # Longitude (degrees)
    elevation_ft: float      # Threshold elevation MSL
    heading_true: float      # True heading (degrees)
    heading_mag: float       # Magnetic heading (degrees)
    traffic_pattern: TrafficPattern = TrafficPattern.LEFT

    # Optional approach aids
    has_vasi: bool = False
    has_papi: bool = False
    glide_slope_deg: float = 3.0
    has_ils: bool = False


@dataclass
class Runway:
    """Complete runway with both ends."""
    id: str                  # e.g., "17/35", "13/31"
    length_ft: float
    width_ft: float
    surface: SurfaceType
    end_1: RunwayEnd
    end_2: RunwayEnd
    condition: str = "good"  # good, fair, poor

    @property
    def length_m(self) -> float:
        return self.length_ft * 0.3048

    @property
    def width_m(self) -> float:
        return self.width_ft * 0.3048

    def get_end(self, designator: str) -> Optional[RunwayEnd]:
        """Get runway end by designator."""
        if self.end_1.designator == designator:
            return self.end_1
        elif self.end_2.designator == designator:
            return self.end_2
        return None

    def get_opposite_end(self, designator: str) -> Optional[RunwayEnd]:
        """Get the opposite runway end."""
        if self.end_1.designator == designator:
            return self.end_2
        elif self.end_2.designator == designator:
            return self.end_1
        return None


@dataclass
class Airport:
    """Complete airport definition."""
    icao: str                # ICAO code (e.g., "KHUT") or FAA LID (e.g., "SN65")
    name: str
    city: str
    state: str
    lat: float               # Airport reference point latitude
    lon: float               # Airport reference point longitude
    elevation_ft: float      # Field elevation MSL
    runways: List[Runway] = field(default_factory=list)
    is_towered: bool = False

    # Communication frequencies
    atis_freq: Optional[float] = None
    tower_freq: Optional[float] = None
    ground_freq: Optional[float] = None
    ctaf_freq: Optional[float] = None

    @property
    def elevation_m(self) -> float:
        return self.elevation_ft * 0.3048

    def get_runway(self, runway_id: str) -> Optional[Runway]:
        """Get runway by ID (e.g., '17/35')."""
        for rwy in self.runways:
            if rwy.id == runway_id:
                return rwy
        return None

    def get_runway_by_designator(self, designator: str) -> Optional[Tuple[Runway, RunwayEnd]]:
        """Get runway and end by designator (e.g., '35')."""
        for rwy in self.runways:
            end = rwy.get_end(designator)
            if end:
                return rwy, end
        return None

    @property
    def primary_runway(self) -> Optional[Runway]:
        """Get the longest runway (typically primary)."""
        if not self.runways:
            return None
        return max(self.runways, key=lambda r: r.length_ft)


# =============================================================================
# AIRPORT DATABASE - Wichita, Kansas Region
# =============================================================================

# Lake Waltanna (SN65) - Private Grass Strip
SN65 = Airport(
    icao="SN65",
    name="Lake Waltanna Airport",
    city="Goddard",
    state="KS",
    lat=37.594167,           # N37 35.65'
    lon=-97.615833,          # W97 36.93'
    elevation_ft=1448.0,
    is_towered=False,
    ctaf_freq=122.9,
    runways=[
        Runway(
            id="17/35",
            length_ft=2100.0,
            width_ft=100.0,
            surface=SurfaceType.TURF,
            condition="good",
            end_1=RunwayEnd(
                designator="17",
                lat=37.597025,       # N37 35.818500'
                lon=-97.615547,      # W97 36.932833'
                elevation_ft=1391.0,
                heading_true=179.0,
                heading_mag=184.0,   # ~5 deg east variation
                traffic_pattern=TrafficPattern.RIGHT,
            ),
            end_2=RunwayEnd(
                designator="35",
                lat=37.591206,       # N37 35.472333'
                lon=-97.615444,      # W97 36.926667'
                elevation_ft=1401.0,
                heading_true=359.0,
                heading_mag=4.0,
                traffic_pattern=TrafficPattern.LEFT,
            ),
        ),
    ],
)


# Hutchinson Regional Airport (KHUT) - Towered
KHUT = Airport(
    icao="KHUT",
    name="Hutchinson Regional Airport",
    city="Hutchinson",
    state="KS",
    lat=38.066167,           # N38 3.97'
    lon=-97.860500,          # W97 51.63'
    elevation_ft=1542.0,
    is_towered=True,
    atis_freq=124.25,
    tower_freq=118.5,
    ground_freq=121.9,
    runways=[
        # Primary Runway 13/31 (longest, ILS equipped)
        Runway(
            id="13/31",
            length_ft=7003.0,
            width_ft=100.0,
            surface=SurfaceType.ASPHALT,
            condition="excellent",
            end_1=RunwayEnd(
                designator="13",
                lat=38.074207,       # N38 04.452432'
                lon=-97.871081,      # W97 52.264852'
                elevation_ft=1524.9,
                heading_true=138.0,
                heading_mag=134.0,
                traffic_pattern=TrafficPattern.LEFT,
                has_vasi=True,
                glide_slope_deg=3.0,
                has_ils=True,
            ),
            end_2=RunwayEnd(
                designator="31",
                lat=38.059855,       # N38 03.591295'
                lon=-97.854893,      # W97 51.293583'
                elevation_ft=1514.7,
                heading_true=318.0,
                heading_mag=314.0,
                traffic_pattern=TrafficPattern.LEFT,
            ),
        ),
        # Runway 4/22
        Runway(
            id="4/22",
            length_ft=4405.0,
            width_ft=100.0,
            surface=SurfaceType.ASPHALT,
            condition="fair",
            end_1=RunwayEnd(
                designator="4",
                lat=38.061733,       # N38 03.703997'
                lon=-97.859919,      # W97 51.595125'
                elevation_ft=1516.5,
                heading_true=42.0,
                heading_mag=38.0,
                traffic_pattern=TrafficPattern.LEFT,
                has_papi=True,
                glide_slope_deg=3.0,
            ),
            end_2=RunwayEnd(
                designator="22",
                lat=38.070662,       # N38 04.239712'
                lon=-97.849597,      # W97 50.975807'
                elevation_ft=1542.5,
                heading_true=222.0,
                heading_mag=218.0,
                traffic_pattern=TrafficPattern.LEFT,
                has_vasi=True,
                glide_slope_deg=3.4,
            ),
        ),
        # Runway 17/35
        Runway(
            id="17/35",
            length_ft=4012.0,
            width_ft=75.0,
            surface=SurfaceType.CONCRETE,
            condition="excellent",
            end_1=RunwayEnd(
                designator="17",
                lat=38.070079,       # N38 04.204763'
                lon=-97.862759,      # W97 51.765548'
                elevation_ft=1523.3,
                heading_true=177.0,
                heading_mag=173.0,
                traffic_pattern=TrafficPattern.LEFT,
                has_papi=True,
                glide_slope_deg=3.0,
            ),
            end_2=RunwayEnd(
                designator="35",
                lat=38.059075,       # N38 03.544508'
                lon=-97.862091,      # W97 51.725472'
                elevation_ft=1517.8,
                heading_true=357.0,
                heading_mag=353.0,
                traffic_pattern=TrafficPattern.LEFT,
                has_papi=True,
                glide_slope_deg=3.0,
            ),
        ),
    ],
)


# =============================================================================
# Airport Registry
# =============================================================================

AIRPORTS: Dict[str, Airport] = {
    "SN65": SN65,
    "KHUT": KHUT,
}


def get_airport(icao: str) -> Optional[Airport]:
    """Get airport by ICAO code or FAA LID."""
    return AIRPORTS.get(icao.upper())


def list_airports() -> List[str]:
    """List all available airport codes."""
    return list(AIRPORTS.keys())


def distance_between_airports(apt1: str, apt2: str) -> float:
    """
    Calculate great-circle distance between two airports in nautical miles.

    Uses Haversine formula.
    """
    a1 = get_airport(apt1)
    a2 = get_airport(apt2)
    if not a1 or not a2:
        raise ValueError(f"Airport not found: {apt1 if not a1 else apt2}")

    lat1, lon1 = np.radians(a1.lat), np.radians(a1.lon)
    lat2, lon2 = np.radians(a2.lat), np.radians(a2.lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # Earth radius in nautical miles
    R_nm = 3440.065
    return R_nm * c


def bearing_between_airports(apt1: str, apt2: str) -> float:
    """
    Calculate initial bearing from apt1 to apt2 in degrees true.
    """
    a1 = get_airport(apt1)
    a2 = get_airport(apt2)
    if not a1 or not a2:
        raise ValueError(f"Airport not found: {apt1 if not a1 else apt2}")

    lat1, lon1 = np.radians(a1.lat), np.radians(a1.lon)
    lat2, lon2 = np.radians(a2.lat), np.radians(a2.lon)

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


if __name__ == "__main__":
    # Test the airport database
    print("AIDA Airport Database")
    print("=" * 50)

    for code in list_airports():
        apt = get_airport(code)
        print(f"\n{apt.icao} - {apt.name}")
        print(f"  Location: {apt.city}, {apt.state}")
        print(f"  Coordinates: {apt.lat:.4f} N, {abs(apt.lon):.4f} W")
        print(f"  Elevation: {apt.elevation_ft:.0f} ft MSL")
        print(f"  Towered: {apt.is_towered}")
        print(f"  Runways:")
        for rwy in apt.runways:
            print(f"    {rwy.id}: {rwy.length_ft:.0f}' x {rwy.width_ft:.0f}' {rwy.surface.value}")

    print("\n" + "=" * 50)
    dist = distance_between_airports("SN65", "KHUT")
    bearing = bearing_between_airports("SN65", "KHUT")
    print(f"Distance SN65 -> KHUT: {dist:.1f} NM")
    print(f"Bearing SN65 -> KHUT: {bearing:.0f} deg true")
