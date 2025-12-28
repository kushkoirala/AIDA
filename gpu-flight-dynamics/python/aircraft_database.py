"""
Aircraft Parameter Database

Contains aerodynamic and mass properties for different aircraft models.
Data sources cited for each aircraft.

Author: Kushal Koirala
Date: 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class MassProperties:
    """Aircraft mass and inertia properties"""
    mass: float         # kg
    Ixx: float          # kg·m² (roll)
    Iyy: float          # kg·m² (pitch)
    Izz: float          # kg·m² (yaw)
    Ixz: float = 0.0    # kg·m² (product of inertia)


@dataclass
class Geometry:
    """Wing and reference geometry"""
    S: float            # Wing reference area [m²]
    b: float            # Wingspan [m]
    c: float            # Mean aerodynamic chord [m]
    e: float = 0.8      # Oswald efficiency factor
    
    @property
    def AR(self) -> float:
        """Aspect ratio"""
        return self.b ** 2 / self.S


@dataclass
class LongitudinalDerivatives:
    """Longitudinal aerodynamic stability derivatives"""
    # Lift
    CL0: float          # Zero-alpha lift
    CLa: float          # Lift curve slope [1/rad]
    CLq: float          # Pitch rate derivative
    CLde: float         # Elevator effectiveness
    
    # Drag
    CD0: float          # Parasitic drag
    K: float            # Induced drag factor
    
    # Pitching moment
    Cm0: float          # Zero-alpha moment
    Cma: float          # Static stability [1/rad] (negative = stable)
    Cmq: float          # Pitch damping
    Cmde: float         # Elevator effectiveness
    
    # Optional with defaults (must come last)
    CLmax: float = 1.5  # Max lift (stall)
    CLmin: float = -1.0 # Min lift
    CDa: float = 0.0    # Alpha drag


@dataclass
class LateralDerivatives:
    """Lateral-directional stability derivatives"""
    # Side force
    CYb: float          # Sideslip derivative [1/rad]
    
    # Rolling moment
    Clb: float          # Dihedral effect [1/rad]
    Clp: float          # Roll damping
    Clr: float          # Roll due to yaw rate
    Clda: float         # Aileron effectiveness
    
    # Yawing moment
    Cnb: float          # Directional stability [1/rad]
    Cnp: float          # Yaw due to roll rate
    Cnr: float          # Yaw damping
    
    # Optional with defaults (must come last)
    CYp: float = 0.0    # Roll rate derivative
    CYr: float = 0.0    # Yaw rate derivative
    CYda: float = 0.0   # Aileron derivative
    CYdr: float = 0.0   # Rudder derivative
    Cldr: float = 0.0   # Rudder-roll coupling
    Cnda: float = 0.0   # Adverse yaw
    Cndr: float = 0.0   # Rudder effectiveness


@dataclass
class PropulsionParams:
    """Propulsion system parameters"""
    thrust_max: float       # Maximum thrust [N]
    thrust_min: float = 0.0 # Minimum/idle thrust [N]
    tau: float = 0.5        # Engine time constant [s]


@dataclass 
class AircraftConfig:
    """Complete aircraft configuration"""
    name: str
    description: str
    mass: MassProperties
    geom: Geometry
    longi: LongitudinalDerivatives
    latdi: LateralDerivatives
    prop: PropulsionParams
    
    # Optional metadata
    source: str = ""
    notes: str = ""


# =============================================================================
# AIRCRAFT DATABASE
# =============================================================================

def get_cessna_172() -> AircraftConfig:
    """
    Cessna 172 Skyhawk - General Aviation Trainer
    
    A stable, forgiving aircraft used for flight training.
    Data from various public sources and typical values.
    """
    return AircraftConfig(
        name="Cessna 172",
        description="Single-engine general aviation trainer",
        
        mass=MassProperties(
            mass=1043.0,        # kg (2300 lbs gross)
            Ixx=1285.0,         # kg·m²
            Iyy=1825.0,         # kg·m²
            Izz=2667.0,         # kg·m²
            Ixz=0.0,            # Simplified
        ),
        
        geom=Geometry(
            S=16.17,            # m² (174 ft²)
            b=10.92,            # m (35.8 ft)
            c=1.49,             # m (4.9 ft)
            e=0.8,
        ),
        
        longi=LongitudinalDerivatives(
            CL0=0.307,
            CLa=4.41,           # 1/rad
            CLq=3.9,
            CLde=0.43,
            CD0=0.027,
            K=0.054,            # 1/(pi*AR*e)
            Cm0=0.04,
            Cma=-0.613,         # Stable (negative)
            Cmq=-12.4,
            Cmde=-1.122,
            CLmax=1.4,
            CLmin=-0.5,
            CDa=0.0,
        ),
        
        latdi=LateralDerivatives(
            # Side force derivatives (from Dynamics PDF Table 2)
            CYb=-0.42,                      # Sideslip derivative (Table 2: Cy,beta)

            # Rolling moment derivatives (Table 2)
            Clb=-0.17,                      # Dihedral effect (Table 2: Cl,beta)
            Clp=-0.81,                      # Roll damping (Table 2: Cl,p)
            Clr=-0.35,                      # Roll due to yaw rate (Table 2: Cl,r)
            Clda=0.65,                      # Aileron roll effectiveness (Table 2: Cl,delta_a)

            # Yawing moment derivatives (Table 2)
            Cnb=0.14,                       # Directional stability (Table 2: Cn,beta)
            Cnp=-0.59,                      # Adverse yaw from roll (Table 2: Cn,p)
            Cnr=-0.09,                      # Yaw damping (Table 2: Cn,r)

            # Optional derivatives (Table 2)
            CYp=0.0,                        # Not in table, assumed 0
            CYr=0.97,                       # Table 2: Cy,r
            CYda=0.0,                       # Table 2: Cy,delta_a = 0
            CYdr=0.19,                      # Rudder side force (Table 2: Cy,delta_r)
            Cldr=-0.03,                     # Rudder-roll coupling (Table 2: Cl,delta_r)
            Cnda=-0.06,                     # Aileron adverse yaw (Table 2: Cn,delta_a)
            Cndr=-0.092,                    # Rudder yaw effectiveness (Table 2: Cn,delta_r)
        ),
        
        prop=PropulsionParams(
            thrust_max=2500.0,  # N (~560 lbf)
            thrust_min=50.0,
            tau=0.5,
        ),
        
        source="Various GA references, Stevens & Lewis",
        notes="Stable trainer aircraft. Good for learning flight dynamics.",
    )


def get_f16() -> AircraftConfig:
    """
    General Dynamics F-16 Fighting Falcon
    
    High-performance fighter with relaxed static stability.
    The F-16 is statically unstable in pitch (positive Cma) and requires
    a flight control computer for stable flight.
    
    Data Source:
        NASA Technical Paper 1538 (1979)
        "Simulator Study of Stall/Post-Stall Characteristics of a 
         Fighter Airplane with Relaxed Longitudinal Static Stability"
        
        Additional data from:
        - Stevens & Lewis, "Aircraft Control and Simulation"
        - Nguyen et al., NASA TM-86309 (1979)
    
    Flight Condition: 
        Altitude: 15,000 ft (4,572 m)
        Mach: 0.6 (subsonic)
        Weight: ~20,500 lbs combat weight
    """
    
    # Unit conversions
    LBS_TO_KG = 0.453592
    FT_TO_M = 0.3048
    SLUG_FT2_TO_KG_M2 = 1.35582
    LBF_TO_N = 4.44822
    
    return AircraftConfig(
        name="F-16 Fighting Falcon",
        description="Single-engine supersonic multirole fighter (relaxed static stability)",
        
        mass=MassProperties(
            mass=9295.0,                    # kg (20,500 lbs combat)
            Ixx=12875.0 * SLUG_FT2_TO_KG_M2,   # 12,875 slug·ft² → kg·m²
            Iyy=75674.0 * SLUG_FT2_TO_KG_M2,   # 75,674 slug·ft² → kg·m²
            Izz=85552.0 * SLUG_FT2_TO_KG_M2,   # 85,552 slug·ft² → kg·m²
            Ixz=1331.0 * SLUG_FT2_TO_KG_M2,    # 1,331 slug·ft² → kg·m²
        ),
        
        geom=Geometry(
            S=27.87,                        # m² (300 ft²)
            b=9.14,                         # m (30 ft) 
            c=3.45,                         # m (11.32 ft) MAC
            e=0.87,                         # Higher for delta-ish wing
        ),
        
        longi=LongitudinalDerivatives(
            # Lift derivatives (per radian)
            CL0=0.0,                        # Symmetric airfoil, zero at alpha=0
            CLa=3.44,                       # Lift curve slope [1/rad]
            CLq=0.0,                        # Pitch rate effect (simplified)
            CLde=0.355,                     # Elevon effectiveness
            
            # Drag derivatives
            CD0=0.0175,                     # Minimum drag (clean config)
            K=0.13,                         # Induced drag factor
            
            # Pitching moment derivatives
            Cm0=0.0,                        # Zero at alpha=0 (trimmed)
            Cma=0.036,                      # UNSTABLE! (positive = unstable)
            Cmq=-1.1,                       # Pitch damping (less than stable aircraft)
            Cmde=-0.38,                     # Elevon pitch effectiveness
            
            # Optional with defaults
            CLmax=1.6,                      # Max lift (with LEF)
            CLmin=-0.8,
            CDa=0.0,
        ),
        
        latdi=LateralDerivatives(
            # Side force derivatives (from Dynamics PDF Table 2)
            CYb=-0.42,                      # Sideslip derivative (Table 2: Cy,beta)

            # Rolling moment derivatives (Table 2)
            Clb=-0.17,                      # Dihedral effect (Table 2: Cl,beta)
            Clp=-0.81,                      # Roll damping (Table 2: Cl,p)
            Clr=-0.35,                      # Roll due to yaw rate (Table 2: Cl,r)
            Clda=0.65,                      # Aileron roll effectiveness (Table 2: Cl,delta_a)

            # Yawing moment derivatives (Table 2)
            Cnb=0.14,                       # Directional stability (Table 2: Cn,beta)
            Cnp=-0.59,                      # Adverse yaw from roll (Table 2: Cn,p)
            Cnr=-0.09,                      # Yaw damping (Table 2: Cn,r)

            # Optional derivatives (Table 2)
            CYp=0.0,                        # Not in table, assumed 0
            CYr=0.97,                       # Table 2: Cy,r
            CYda=0.0,                       # Table 2: Cy,delta_a = 0
            CYdr=0.19,                      # Rudder side force (Table 2: Cy,delta_r)
            Cldr=-0.03,                     # Rudder-roll coupling (Table 2: Cl,delta_r)
            Cnda=-0.06,                     # Aileron adverse yaw (Table 2: Cn,delta_a)
            Cndr=-0.092,                    # Rudder yaw effectiveness (Table 2: Cn,delta_r)
        ),
        
        prop=PropulsionParams(
            # F110-GE-100 engine (approximate)
            thrust_max=76300.0,             # N (17,155 lbf dry) - using dry thrust for sim
            thrust_min=1000.0,              # Idle thrust
            tau=1.0,                        # Jet engine spool time
        ),
        
        source="NASA TP-1538, NASA TM-86309, Stevens & Lewis",
        notes="""
        CAUTION: This aircraft is UNSTABLE in pitch (Cma > 0).
        Real F-16 requires fly-by-wire flight control system.
        
        For simulation:
        - Use a stability augmentation system (SAS) or
        - Accept that unaugmented flight will diverge
        
        The positive Cma means:
        - Nose up perturbation → nose pitches further up
        - Requires constant pilot/computer correction
        """,
    )


def get_f16_stable() -> AircraftConfig:
    """
    F-16 with artificial stability augmentation.
    
    This version has modified Cma to simulate the effect of the 
    flight control computer, making it flyable without a control law.
    Use this for basic simulations where you don't have a flight computer.
    """
    config = get_f16()
    config.name = "F-16 (Stability Augmented)"
    config.description = "F-16 with simulated stability augmentation"
    
    # Make it artificially stable
    config.longi.Cma = -0.1  # Negative = stable
    config.longi.Cmq = -4.0  # Increased pitch damping
    
    config.notes = """
    Modified F-16 with artificial stability for simulation without FCS.
    Real F-16 Cma = +0.036 (unstable), this version Cma = -0.1 (stable).
    Use this for basic simulations; use real F-16 for control system design.
    """
    
    return config


def get_boeing_747() -> AircraftConfig:
    """
    Boeing 747-100 - Large Transport Aircraft

    Heavy, stable transport with slow dynamics.
    Data from NASA CR-2144 and public sources.

    Flight Condition:
        Altitude: 20,000 ft
        Mach: 0.5
        Weight: 636,600 lbs (heavy)
    """

    LBS_TO_KG = 0.453592
    SLUG_FT2_TO_KG_M2 = 1.35582

    return AircraftConfig(
        name="Boeing 747-100",
        description="Wide-body long-range transport",

        mass=MassProperties(
            mass=288756.0,                      # kg (636,600 lbs)
            Ixx=18.2e6 * SLUG_FT2_TO_KG_M2,     # slug·ft² → kg·m²
            Iyy=33.1e6 * SLUG_FT2_TO_KG_M2,
            Izz=49.7e6 * SLUG_FT2_TO_KG_M2,
            Ixz=0.97e6 * SLUG_FT2_TO_KG_M2,
        ),

        geom=Geometry(
            S=510.97,                           # m² (5,500 ft²)
            b=59.64,                            # m (195.7 ft)
            c=8.324,                            # m (27.3 ft) MAC
            e=0.85,
        ),

        longi=LongitudinalDerivatives(
            CL0=0.21,
            CLa=4.4,
            CLq=6.6,
            CLde=0.32,
            CD0=0.0164,
            K=0.042,
            Cm0=0.0,
            Cma=-0.683,                         # Very stable
            Cmq=-20.8,                          # High pitch damping
            Cmde=-1.05,
            CLmax=1.8,
            CLmin=-0.4,
            CDa=0.0,
        ),

        latdi=LateralDerivatives(
            # Side force derivatives (from Dynamics PDF Table 2)
            CYb=-0.42,                      # Sideslip derivative (Table 2: Cy,beta)

            # Rolling moment derivatives (Table 2)
            Clb=-0.17,                      # Dihedral effect (Table 2: Cl,beta)
            Clp=-0.81,                      # Roll damping (Table 2: Cl,p)
            Clr=-0.35,                      # Roll due to yaw rate (Table 2: Cl,r)
            Clda=0.65,                      # Aileron roll effectiveness (Table 2: Cl,delta_a)

            # Yawing moment derivatives (Table 2)
            Cnb=0.14,                       # Directional stability (Table 2: Cn,beta)
            Cnp=-0.59,                      # Adverse yaw from roll (Table 2: Cn,p)
            Cnr=-0.09,                      # Yaw damping (Table 2: Cn,r)

            # Optional derivatives (Table 2)
            CYp=0.0,                        # Not in table, assumed 0
            CYr=0.97,                       # Table 2: Cy,r
            CYda=0.0,                       # Table 2: Cy,delta_a = 0
            CYdr=0.19,                      # Rudder side force (Table 2: Cy,delta_r)
            Cldr=-0.03,                     # Rudder-roll coupling (Table 2: Cl,delta_r)
            Cnda=-0.06,                     # Aileron adverse yaw (Table 2: Cn,delta_a)
            Cndr=-0.092,                    # Rudder yaw effectiveness (Table 2: Cn,delta_r)
        ),

        prop=PropulsionParams(
            thrust_max=4 * 222400.0,            # 4x JT9D-7 engines, N
            thrust_min=4 * 10000.0,
            tau=3.0,                            # Slow spool-up
        ),

        source="NASA CR-2144, Heffley & Jewell",
        notes="Large transport, very stable, slow response. Good for autopilot testing.",
    )


def get_udaan() -> AircraftConfig:
    """
    Udaan - AIDA Research Platform

    Custom-designed small electric UAV for autonomous flight research.
    Twin-propeller electric aircraft with tennis ball payload capability.

    Design Parameters:
        - Wing Span: 4.5 ft (1.37 m)
        - Wing Area: 5.17 ft² (0.48 m²)
        - Takeoff Weight: 7.5 lbs (3.40 kg) + 0.23 kg payload
        - Power: 2x 315W electric motors with 9" propellers
        - Flight Envelope: 36-87 ft/s (11-26.5 m/s)
        - Operating Altitude: 1,300 ft MSL

    Data Source:
        PropShox Final Design Report (Dec 2024)
        - Table 1: Aircraft specifications
        - Section 3.2: Flight test data & aerodynamic coefficients
        - AIDA simulation validation data

    Flight Condition:
        Altitude: 1,300 ft (396 m) MSL
        Cruise Speed: 73.5 ft/s (22.4 m/s)
        Density: 2.288e-3 slug/ft³ (1.187 kg/m³)
    """

    # Unit conversions from PropShox report
    FT_TO_M = 0.3048
    FT2_TO_M2 = 0.09290304
    LB_TO_KG = 0.45359237

    # Mass with payload (4 tennis balls ≈ 0.23 kg)
    mass_takeoff = 7.5 * LB_TO_KG + 0.23  # kg

    # Inertias from CATIA CAD analysis (Dec 27, 2024)
    # Principal moments about CG from "Measure Inertia" tool
    Ixx = 1.147  # kg·m² (roll - M1 principal moment)
    Iyy = 1.614  # kg·m² (pitch - M2 principal moment)
    Izz = 2.709  # kg·m² (yaw - M3 principal moment)

    return AircraftConfig(
        name="Udaan",
        description="Small twin-prop electric UAV for autonomous flight research",

        mass=MassProperties(
            mass=mass_takeoff,              # 3.63 kg with payload
            Ixx=Ixx,
            Iyy=Iyy,
            Izz=Izz,
            Ixz=0.0,                        # Symmetric design
        ),

        geom=Geometry(
            S=5.17 * FT2_TO_M2,             # 0.4803 m²
            b=4.5 * FT_TO_M,                # 1.3716 m
            c=0.35018,                      # m (wing_area/span)
            e=0.75,                         # Typical for small UAV
        ),

        longi=LongitudinalDerivatives(
            # Lift derivatives (from Dynamics PDF Table 2)
            CL0=0.28,                          # Zero-alpha lift coefficient (Table 2)
            CLa=3.82,                          # 3.82/rad (Table 2: CL,alpha)
            CLq=7.6,                           # Pitch rate effect (≈2*CLα for conventional tail)
            CLde=0.73,                         # Elevator effectiveness (Table 2: CL,delta_e)

            # Drag derivatives (from Wind Tunnel Excel)
            CD0=0.018952,                      # Parasitic drag (Wind Tunnel)
            K=0.083236,                        # Induced drag factor from e=0.978 (Wind Tunnel)

            # Pitching moment derivatives (CRITICAL: Using 3D wind tunnel result!)
            Cm0=-0.0962,                       # Zero-alpha moment (Table 2)
            Cma=-0.38,                         # -0.38/rad (Table 2 - matches 7.5% static margin design)
            Cmq=-12.0,                         # Pitch damping (typical for conventional tail)
            Cmde=-1.13,                        # Elevator pitch effectiveness (Table 2: CM,delta_e)

            # Limits
            CLmax=1.005585,                    # Stall limit (Wind Tunnel)
            CLmin=-0.5,                        # Estimated negative stall
            CDa=0.30,                          # Alpha-squared drag term
        ),

        latdi=LateralDerivatives(
            # Side force derivatives (from Dynamics PDF Table 2)
            CYb=-0.42,                      # Sideslip derivative (Table 2: Cy,beta)

            # Rolling moment derivatives (Table 2)
            Clb=-0.17,                      # Dihedral effect (Table 2: Cl,beta)
            Clp=-0.81,                      # Roll damping (Table 2: Cl,p)
            Clr=-0.35,                      # Roll due to yaw rate (Table 2: Cl,r)
            Clda=0.65,                      # Aileron roll effectiveness (Table 2: Cl,delta_a)

            # Yawing moment derivatives (Table 2)
            Cnb=0.14,                       # Directional stability (Table 2: Cn,beta)
            Cnp=-0.59,                      # Adverse yaw from roll (Table 2: Cn,p)
            Cnr=-0.09,                      # Yaw damping (Table 2: Cn,r)

            # Optional derivatives (Table 2)
            CYp=0.0,                        # Not in table, assumed 0
            CYr=0.97,                       # Table 2: Cy,r
            CYda=0.0,                       # Table 2: Cy,delta_a = 0
            CYdr=0.19,                      # Rudder side force (Table 2: Cy,delta_r)
            Cldr=-0.03,                     # Rudder-roll coupling (Table 2: Cl,delta_r)
            Cnda=-0.06,                     # Aileron adverse yaw (Table 2: Cn,delta_a)
            Cndr=-0.092,                    # Rudder yaw effectiveness (Table 2: Cn,delta_r)
        ),

        prop=PropulsionParams(
            # Twin 315W electric motors (PropShox Sec. 3.2)
            # Using actuator disk model: T = 2*rho*A*vi*(V+vi)
            # Peak thrust ~28N at static, ~15N at cruise
            thrust_max=28.0,                # N (estimated from power & prop efficiency)
            thrust_min=0.0,                 # Electric motor (instant cutoff)
            tau=0.1,                        # Fast electric motor response
        ),

        source="PropShox Final Design Report (2024), AIDA Flight Test Data",
        notes="""
        Udaan is a research platform for testing autonomous flight algorithms.

        Design Features:
        - Electrically powered (2x 315W motors, 9" props, 50% efficiency)
        - Twin-prop configuration for redundancy
        - Carries 4 tennis balls (~0.23 kg) as payload
        - Designed for 1,300 ft MSL operation (parking lot runway)

        Flight Characteristics:
        - V_stall: 36.6 ft/s (11.2 m/s)
        - V_cruise: 73.5 ft/s (22.4 m/s)
        - V_max: 87.3 ft/s (26.6 m/s)
        - Stable design (negative Cma) suitable for autonomous operation
        - Strong damping (Cmq, Clp, Cnr) for gust rejection

        Current Status:
        - PPO RL training in AIDA simulation environment
        - GPU-accelerated physics for behavior cloning warmup
        - 3D visualization with Three.js viewer
        """,
    )


# =============================================================================
# AIRCRAFT REGISTRY
# =============================================================================

AIRCRAFT_DATABASE: Dict[str, callable] = {
    "cessna172": get_cessna_172,
    "f16": get_f16,
    "f16_stable": get_f16_stable,
    "747": get_boeing_747,
    "udaan": get_udaan,
}


def list_aircraft() -> list:
    """Return list of available aircraft"""
    return list(AIRCRAFT_DATABASE.keys())


def get_aircraft(name: str) -> AircraftConfig:
    """
    Get aircraft configuration by name.
    
    Args:
        name: Aircraft identifier (e.g., "f16", "cessna172", "747")
        
    Returns:
        AircraftConfig object
        
    Raises:
        KeyError: If aircraft not found
    """
    if name.lower() not in AIRCRAFT_DATABASE:
        available = ", ".join(list_aircraft())
        raise KeyError(f"Aircraft '{name}' not found. Available: {available}")
    
    return AIRCRAFT_DATABASE[name.lower()]()


def print_aircraft_info(config: AircraftConfig):
    """Print detailed aircraft information"""
    print(f"\n{'='*60}")
    print(f"  {config.name}")
    print(f"{'='*60}")
    print(f"  {config.description}")
    print(f"\n  Source: {config.source}")
    
    print(f"\n  Mass Properties:")
    print(f"    Mass:  {config.mass.mass:,.0f} kg ({config.mass.mass * 2.205:,.0f} lbs)")
    print(f"    Ixx:   {config.mass.Ixx:,.0f} kg·m²")
    print(f"    Iyy:   {config.mass.Iyy:,.0f} kg·m²")
    print(f"    Izz:   {config.mass.Izz:,.0f} kg·m²")
    
    print(f"\n  Geometry:")
    print(f"    Wing Area:  {config.geom.S:.2f} m² ({config.geom.S * 10.764:.0f} ft²)")
    print(f"    Wingspan:   {config.geom.b:.2f} m ({config.geom.b * 3.281:.1f} ft)")
    print(f"    MAC:        {config.geom.c:.2f} m ({config.geom.c * 3.281:.1f} ft)")
    print(f"    Aspect Ratio: {config.geom.AR:.2f}")
    
    print(f"\n  Key Stability Derivatives:")
    print(f"    CLa (lift slope):     {config.longi.CLa:.3f} /rad")
    print(f"    Cma (pitch stiffness): {config.longi.Cma:.3f} /rad", end="")
    if config.longi.Cma > 0:
        print("  ⚠️  UNSTABLE")
    else:
        print("  ✓ stable")
    print(f"    Cnb (yaw stiffness):   {config.latdi.Cnb:.3f} /rad")
    print(f"    Clb (dihedral):       {config.latdi.Clb:.3f} /rad")
    
    print(f"\n  Propulsion:")
    print(f"    Max Thrust: {config.prop.thrust_max:,.0f} N ({config.prop.thrust_max * 0.2248:,.0f} lbf)")
    
    if config.notes:
        print(f"\n  Notes:")
        for line in config.notes.strip().split('\n'):
            print(f"    {line.strip()}")
    
    print(f"{'='*60}\n")


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AIRCRAFT PARAMETER DATABASE")
    print("="*60)
    
    print(f"\nAvailable aircraft: {', '.join(list_aircraft())}")
    
    # Show info for each
    for name in list_aircraft():
        config = get_aircraft(name)
        print_aircraft_info(config)
