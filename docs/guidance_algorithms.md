# Fixed-Wing Aircraft Path Following Guidance Algorithms

## Overview

This document summarizes the research on path-following guidance algorithms for fixed-wing aircraft, specifically in the context of the AIDA (Autonomous Intelligent Decision Architecture) project. The goal is to enable a Cessna 172 aircraft to autonomously follow planned trajectories from departure to destination.

## Key Differences from Multirotor UAVs

Fixed-wing aircraft have fundamentally different dynamics than multirotor drones:

1. **Minimum Airspeed Requirement**: Must maintain forward velocity to generate lift (stall speed ~50 kts for C172)
2. **Turn Radius Constraints**: Cannot execute instantaneous heading changes; turns require bank angle and result in load factor
3. **Coupled Dynamics**: Altitude, airspeed, and heading are coupled through energy management
4. **Wind Effects**: Ground track differs from heading when wind is present

These constraints make path-following more challenging and require specialized guidance algorithms.

## Guidance Algorithm Categories

### 1. Geometric Methods

#### Pure Pursuit
- **Concept**: Steer toward a "lookahead point" on the path at distance L ahead
- **Formula**: Curvature κ = 2·sin(α) / L, where α is the angle to the lookahead point
- **Pros**: Simple, intuitive, works well for smooth paths
- **Cons**: Can cut corners on tight turns, oscillates if L is too short
- **Implementation**: `aida_sim/guidance/pure_pursuit.py`

#### Carrot-Chasing
- **Concept**: Similar to pure pursuit but the "carrot" moves along the path
- **Pros**: Simple implementation
- **Cons**: Performance degrades in high curvature sections

#### Line-of-Sight (LOS)
- **Concept**: Compute desired heading based on line-of-sight angle to a reference point
- **Pros**: Well-understood from marine navigation
- **Cons**: Requires careful tuning of lookahead distance

### 2. Vector Field Methods

#### Lyapunov Vector Field Guidance (LVFG)
- **Concept**: Define a heading vector field that converges to the desired path
- **Formula**: ψ_desired = ψ_path - arctan(k · e_crosstrack)
- **Pros**:
  - Guaranteed Lyapunov stability and convergence
  - Smooth transitions between path segments
  - Naturally handles wind disturbances
- **Cons**: Requires continuous path representation
- **Implementation**: `aida_sim/guidance/vector_field.py`
- **Reference**: Nelson, D.R. et al. (2007). "Vector Field Path Following for Miniature Air Vehicles"

### 3. Nonlinear Guidance Laws

#### L1 Guidance / NLGL
- **Concept**: Compute lateral acceleration to reach a virtual target point (VTP) at distance L1 ahead
- **Formula**: a_lateral = 2·V²·sin(η) / L1, where η is the angle to VTP
- **Pros**:
  - Well-proven in practice (ArduPilot, PX4)
  - Handles both straight and curved paths
  - Tunable via single parameter (L1 distance/period)
- **Cons**: Can be aggressive near waypoints
- **Used by**: ArduPilot, early PX4 versions

#### NPFG (Nonlinear Path Following Guidance)
- **Concept**: Evolution of L1 with improved handling of wind and path curvature
- **Pros**: Better wind rejection, smoother tracking
- **Used by**: PX4 (current versions)
- **Reference**: PX4 Documentation

### 4. Control-Theoretic Methods

#### Model Predictive Control (MPC)
- **Concept**: Optimize control inputs over a prediction horizon
- **Pros**: Can handle constraints explicitly, optimal performance
- **Cons**: Computationally expensive, requires accurate model

#### Sliding Mode Control
- **Concept**: Drive system to a sliding surface and maintain it there
- **Pros**: Robust to disturbances
- **Cons**: Chattering issues

## Algorithm Comparison

| Algorithm | Complexity | Tuning | Wind Handling | Turn Performance | Stability Proof |
|-----------|------------|--------|---------------|------------------|-----------------|
| Pure Pursuit | Low | 1-2 params | Moderate | Cuts corners | No |
| Vector Field | Medium | 2-3 params | Good | Smooth | Yes (Lyapunov) |
| L1/NLGL | Medium | 1-2 params | Good | Good | Yes |
| NPFG | Medium | 3-4 params | Excellent | Excellent | Yes |
| MPC | High | Many | Excellent | Optimal | Depends |

## Selected Approach for AIDA

Based on the research, we selected **Lyapunov Vector Field Guidance (LVFG)** for the following reasons:

1. **Stability Guarantees**: Lyapunov analysis proves global asymptotic convergence to the path
2. **Smooth Behavior**: No oscillations or corner-cutting
3. **Wind Robustness**: Naturally handles wind by correcting cross-track error
4. **Already Implemented**: Available in `aida_sim/guidance/vector_field.py`
5. **Tunable**: Two main parameters (k_path, max_course_change) are intuitive

### Integration Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   Trajectory    │────▶│  VectorFieldGuidance │────▶│    Autopilot    │
│    Planner      │     │                      │     │                 │
└─────────────────┘     │  - Cross-track error │     │  - Heading hold │
                        │  - Desired heading   │     │  - Altitude hold│
                        │  - Path progress     │     │  - Speed control│
                        └──────────────────────┘     └─────────────────┘
```

### Key Equations

**Cross-Track Error Correction:**
```
ψ_correction = arctan(k_path · e_crosstrack)
ψ_desired = ψ_path - ψ_correction
```

Where:
- k_path: Path following gain (typical: 0.005-0.02 per foot)
- e_crosstrack: Signed cross-track error (positive = right of path)
- ψ_path: Path heading at closest point
- ψ_desired: Commanded heading

**Maximum Course Change:**
The correction is limited to prevent excessive bank angles:
```
|ψ_correction| ≤ ψ_max (typically 30-45°)
```

## Flight Phase Considerations

### Takeoff/Climb
- Use constant pitch attitude for climb performance
- Lateral guidance tracks runway centerline initially
- Transition to enroute guidance after reaching safe altitude

### Cruise/Enroute
- Full vector field guidance active
- Cross-track error drives heading corrections
- Lookahead for altitude changes

### Descent/Approach
- Tighter cross-track tolerance (precision approach)
- Glideslope tracking takes priority
- Smaller max_course_change to prevent aggressive maneuvers

### Landing
- Switch to runway centerline tracking
- Flare and touchdown logic separate from guidance

## References

1. Nelson, D.R., Barber, D.B., McLain, T.W., Beard, R.W. (2007). "Vector Field Path Following for Miniature Air Vehicles". IEEE Transactions on Robotics.

2. Park, S., Deyst, J., How, J. (2004). "A New Nonlinear Guidance Logic for Trajectory Tracking". AIAA Guidance, Navigation, and Control Conference.

3. Kai, J.-M., Hamel, T., Samson, C. (2019). "A unified approach to fixed-wing aircraft path following guidance and control". Automatica.

4. Coulter, R.C. (1992). "Implementation of the Pure Pursuit Path Tracking Algorithm". CMU Robotics Institute.

5. PX4 Autopilot Documentation. "Fixed-wing Position Controller Tuning Guide". https://docs.px4.io/main/en/config_fw/position_tuning_guide_fixedwing.html

## Implementation Notes

### Tuning Guidelines

1. **k_path (0.005 - 0.02)**: Higher values = faster convergence but more aggressive
   - Start with 0.01 (converges in ~100 ft)
   - Increase if tracking is sluggish
   - Decrease if oscillating

2. **max_course_change (30° - 45°)**: Limits maximum correction angle
   - 30° for smooth cruise flight
   - 45° for more aggressive maneuvering
   - Match to aircraft bank angle limits

3. **Segment-specific tuning**:
   - Approach: k_path = 0.02, max_change = 20°
   - Cruise: k_path = 0.01, max_change = 45°
   - Climb: k_path = 0.01, max_change = 30°

---

*Document created: January 2026*
*Author: AIDA Development Team (with Claude Code)*
