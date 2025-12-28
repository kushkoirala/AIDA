# Phase-Based Reward Functions in AIDA

## You're Right!

The reward function **IS different** for each flight phase. The environment already has:
1. **Takeoff-specific rewards** (`_reward_takeoff`)
2. **Cruise-specific rewards** (`_reward_cruise`)
3. **Mission phase system** (climb, waypoints, landing)

Located in: [flight_env_rl.py](\\wsl.localhost\Ubuntu-22.04\home\AIDA\aida_sim\env\flight_env_rl.py)

---

## How It Works

### Task Selection
```python
task = "takeoff"  # or "takeoff_and_fly" for full mission
```

The environment automatically calls the appropriate reward function:
```python
def _reward(self, vs, action, terminated, info):
    if self.task == "takeoff":
        return self._reward_takeoff(...)  # Takeoff rewards
    return self._reward_cruise(...)       # Cruise/mission rewards
```

---

## PHASE 1: TAKEOFF REWARDS

**Goal:** Accelerate down runway, rotate, climb to 100ft

### Ground Roll Phase
```python
# Stay on centerline
reward += 0.35 * exp(-(lateral_offset / runway_width)²)
reward -= 0.6 * (lateral_offset / runway_width)
reward -= 0.08 * lateral_speed

# Stay aligned with runway
reward -= 0.06 * abs(yaw_error)
if centered AND aligned:
    reward += 0.5  # Bonus!

# Accelerate to takeoff speed
speed_ratio = forward_speed / takeoff_speed  # Target: 13.4 m/s
reward += 0.6 * speed_ratio

# Use full throttle
reward += 0.2 * (throttle - 0.4)  # Encourage high throttle
```

**Teaches:** Stay straight, accelerate, full power

### Rotation Phase
```python
# Pitch up at right speed (40% of V_takeoff)
if forward_speed > 0.4 * takeoff_speed:
    pitch_error = pitch - rotation_pitch  # Target: ~10°
    reward += 0.25 * exp(-(pitch_error / 5°)²)
```

**Teaches:** Rotate at correct speed and pitch angle

### Climb Phase
```python
# Reward altitude gain
if altitude > 0.5m:
    reward += 0.1 * altitude

# Encourage meeting 100ft target
progress_ratio = distance / takeoff_distance
climb_ratio = altitude / 100ft
reward += 0.4 * min(progress_ratio, climb_ratio)

# BIG penalty for being low near runway end
if distance > 90% AND altitude < 100ft:
    reward -= 4.0  # Don't hit the fence!

# Bonus for clean liftoff
if altitude > 1m AND distance > 5m:
    reward += 0.5
```

**Teaches:** Climb smoothly, reach 100ft before end of runway

### Takeoff Termination Rewards
```python
if terminated:
    if reason == "takeoff_success":
        reward += 30.0  # SUCCESS!
    elif reason == "runway_deviation":
        reward -= 8.0   # Ran off runway
    elif reason == "crash":
        reward -= 12.0  # Hit ground
    elif reason == "stall":
        reward -= 6.0   # Stalled
```

---

## PHASE 2: CLIMB REWARDS

**Goal:** Climb to mission altitude (91m / 300ft)

### Altitude Tracking
```python
phase = {"type": "altitude", "target": 91.4m, "tolerance": 3.0m}

# Penalize altitude error
target_altitude = phase["target"]
alt_error = altitude - target_altitude
reward -= 0.02 * abs(alt_error)

# Penalize excessive climb/descent rates
climb_rate = velocity_z
if climb_rate > 6.0 m/s:
    reward -= 0.02 * (climb_rate - 6.0)
if climb_rate < -4.0 m/s:
    reward -= 0.02 * (-4.0 - climb_rate)

# Big reward for reaching target
if within_tolerance(altitude, target):
    reward += 50.0  # Phase complete!
```

**Teaches:** Climb steadily, don't exceed limits, reach target altitude

### Airspeed During Climb
```python
# Maintain safe climb speed
target_speed = cruise_speed  # 22.4 m/s
speed_error = abs(airspeed - target_speed)
reward -= 0.05 * speed_error

# Avoid stall during climb
if airspeed < V_stall:
    reward -= 0.5  # Stall penalty
```

**Teaches:** Maintain speed while climbing

### Attitude Limits
```python
# Don't over-pitch
reward -= 0.1 * abs(pitch)
if abs(pitch) > 20°:
    reward -= 0.2 * (abs(pitch) - 20°)

# Keep wings level during climb
reward -= 0.1 * abs(roll)
```

**Teaches:** Smooth, coordinated climb

---

## PHASE 3: CRUISE REWARDS

**Goal:** Fly level, maintain speed/altitude, follow waypoints

### Speed Maintenance
```python
target_speed = 22.4 m/s  # Cruise speed
speed_error = abs(airspeed - target_speed)
reward -= 0.05 * speed_error

# Envelope protection
if airspeed < V_stall (11.2 m/s):
    reward -= 0.5
if airspeed > V_max (30 m/s):
    reward -= 0.3
```

### Altitude Hold
```python
target_altitude = 91.4m  # Or current phase target
alt_error = abs(altitude - target_altitude)
reward -= 0.02 * alt_error

# Ground proximity warning
if altitude < 2m:
    reward -= 0.5  # Don't fly low!
```

### Waypoint Navigation
```python
# Reward forward progress
forward_progress = dot(position, runway_dir)
reward += 0.2 * (forward_progress / leg_distance)

# Penalize lateral deviation
lateral_error = abs(dot(position, runway_right))
reward -= 0.05 * lateral_error

# Reward heading to target
heading_error = angle_to(target) - yaw
reward -= 0.05 * abs(heading_error)

# Big reward for reaching waypoint
if distance_to_target < tolerance:
    reward += 3.0
```

### Smooth Flight
```python
# Penalize aggressive maneuvers
reward -= 0.1 * abs(roll)
reward -= 0.1 * abs(pitch)

# Penalize hard control inputs
control_magnitude = norm([elevator, aileron, rudder])
reward -= 0.05 * control_magnitude

# Penalize high G-loads
if load_factor > 3.0:
    reward -= 0.2 * (load_factor - 3.0)
```

**Teaches:** Smooth, efficient cruise flight

---

## PHASE 4: APPROACH & LANDING (Future)

*Not yet implemented, but would include:*

```python
# Descend to pattern altitude
reward -= altitude_error_from_pattern

# Line up with runway
reward -= lateral_offset_from_centerline
reward -= heading_error_from_runway

# Descend on glideslope
glideslope_error = altitude - ideal_altitude_for_distance
reward -= abs(glideslope_error)

# Reduce speed for landing
target_speed = V_landing  # 14.5 m/s
reward -= abs(airspeed - target_speed)

# Touchdown reward
if landed_safely:
    reward += 50.0
```

---

## Mission Phase System

The environment tracks progress through mission phases:

```python
mission_phases = [
    {"name": "climb", "type": "altitude", "target": 91.4, "tolerance": 3.0},
    # Future phases:
    # {"name": "cruise", "type": "waypoint", "target": [x, y, z]},
    # {"name": "turn", "type": "heading", "target": heading},
    # {"name": "approach", "type": "landing"},
]

# Advance when phase satisfied
if within_tolerance(current_state, phase_target):
    phase_index += 1
    reward += 50.0  # Phase completion bonus!
```

---

## Why Phase-Based Rewards Work

### 1. **Clear Objectives**
Each phase has specific, measurable goals:
- Takeoff: Accelerate → Rotate → Climb to 100ft
- Climb: Reach 300ft while maintaining speed
- Cruise: Hold altitude, navigate waypoints
- Landing: Descend, align, touch down safely

### 2. **Curriculum Learning**
PPO learns in stages:
- **First:** Master takeoff (simpler, shorter task)
- **Then:** Add climb phase
- **Then:** Add cruise and navigation
- **Finally:** Complete full mission

### 3. **Reward Shaping**
Dense rewards guide learning:
- Small penalties for small errors (altitude ±1m → -0.02)
- Big penalties for big errors (crash → -12.0)
- Huge rewards for success (takeoff → +30, mission → +25)

### 4. **Termination Conditions**
Each phase can end early with specific feedback:
- `takeoff_success` → learned good takeoff
- `runway_deviation` → learned runway bounds
- `crash` → learned altitude limits
- `stall` → learned speed limits

---

## Example Learning Progression

### Episode 1-100: Takeoff Focus
```
Reward breakdown:
  Ground roll: +2.0 (staying on centerline)
  Acceleration: +3.5 (reaching 13 m/s)
  Rotation: +1.2 (pitching up)
  Climb: +5.0 (gaining altitude)
  Crash: -12.0 (hit fence)
Total: -0.3 (learning...)
```

### Episode 1000: Mastered Takeoff
```
Reward breakdown:
  Ground roll: +5.0 (perfect centerline)
  Acceleration: +8.0 (smooth acceleration)
  Rotation: +4.0 (perfect pitch)
  Climb: +10.0 (reached 100ft)
  Success: +30.0 (takeoff complete!)
Total: +57.0 (excellent!)
```

### Episode 2000: Learning Climb
```
Reward breakdown:
  Takeoff: +57.0 (mastered)
  Climb to 300ft: +20.0 (reaching target)
  Altitude hold: -5.0 (oscillating)
  Speed control: -3.0 (varying)
  Phase complete: +50.0
Total: +119.0 (progressing!)
```

### Episode 5000: Full Mission
```
Reward breakdown:
  Takeoff: +57.0
  Climb: +70.0
  Cruise: +40.0
  Navigation: +25.0
  Mission success: +25.0
Total: +217.0 (mastery!)
```

---

## How to Train Different Phases

### Train Takeoff Only
```python
env = FlightEnvRL(task="takeoff")
# PPO learns: ground roll → rotation → climb to 100ft
# Episodes end at takeoff_success or crash
```

### Train Full Mission
```python
env = FlightEnvRL(task="takeoff_and_fly")
# PPO learns: takeoff → climb → cruise → waypoints
# Episodes end at mission_complete or termination
```

### Start Airborne (Skip Takeoff)
```python
env = FlightEnvRL(task="cruise")
# Initializes already flying at 15m altitude
# PPO learns: altitude hold → navigation → landing
```

---

## Current Configuration

When you run:
```bash
python scripts/train_ppo_flight.py --device cuda
```

The default task is likely **"takeoff_and_fly"**, which means PPO will learn:
1. ✓ Takeoff from runway
2. ✓ Climb to mission altitude
3. ✓ Cruise and navigate
4. ✓ Complete all mission phases

Each with **phase-appropriate rewards**!

---

## Summary

**Q: Don't we need different rewards per phase?**

**A: We already have them!**

- ✅ `_reward_takeoff()` - Runway alignment, rotation, climb
- ✅ `_reward_cruise()` - Altitude/speed hold, navigation, smooth flight
- ✅ `mission_phases[]` - Structured phase progression
- ✅ Phase completion bonuses - +50 per phase, +25 mission success
- ✅ Termination penalties - Phase-specific crash/stall/deviation

The reward function **automatically adapts** to what the aircraft is trying to do at that moment, teaching the correct behavior for each flight phase.

---

**Prepared by:** Claude Code
**For:** Kushal Koirala - AIDA Project
**Date:** December 27, 2024
