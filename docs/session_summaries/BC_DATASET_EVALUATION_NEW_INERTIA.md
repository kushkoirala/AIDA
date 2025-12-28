# BC Dataset Evaluation - With Updated CATIA Inertias

**Generated:** December 27, 2024
**Dataset:** checkpoints/bc_dataset_udaan.npz
**Inertia Values:** Ixx=1.147, Iyy=1.614, Izz=2.709 kg·m² (from CATIA)

---

## Dataset Generation Summary

✅ **Generation successful:**
- **Episodes:** 1,000 trajectories
- **Total timesteps:** 500,000
- **Episode length:** 500 steps each
- **Generation time:** 165 seconds (2.75 minutes)
- **Performance:** 3,027 steps/second (GPU-accelerated)
- **File size:** 29.41 MB

---

## Dataset Statistics

### Altitude:
- **Mean:** 5.59 m
- **Std:** 24.96 m
- **Range:** -227.83 to 80.39 m

⚠️ **Issue:** Aircraft going underground (-227m) indicates instability

### Airspeed:
- **Mean:** 16.09 m/s
- **Std:** 9.70 m/s
- **Range:** -3.56 to 47.90 m/s

⚠️ **Issue:** Negative airspeed and high variance indicate flying backwards/tumbling

### Pitch:
- **Mean:** 6.63°
- **Std:** 48.89°
- **Range:** -179.99° to 179.98°

❌ **Critical Issue:** Aircraft flipping over (±180° pitch)

### Elevator:
- **Mean:** -4.29°
- **Std:** 8.83°
- **Range:** -15.00° to 15.00°

✓ **Good:** Elevator within limits, sign correction working

---

## Problem Analysis

### Issue 1: No Ground Collision Physics
The GPU simulator doesn't enforce ground contact. Aircraft can:
- Fly underground (negative altitude)
- Pass through the ground without stopping
- Tumble without crash detection

**Evidence:**
- Minimum altitude: -227.83 m (747 feet underground!)
- This explains the instability

### Issue 2: Initial Conditions Too Extreme
Looking at the controller in [generate_bc_dataset_gpu.py](\\wsl.localhost\Ubuntu-22.04\home\AIDA\scripts\generate_bc_dataset_gpu.py):

```python
# Random initial conditions (lines 251-262)
altitude = np.random.uniform(5.0, 60.0, n)  # 5-60m AGL
airspeed = np.random.uniform(11.2, 30.0, n)  # Above stall
pitch = np.random.uniform(-15, 15, n)        # ±15 degrees
```

**Problem:** Some initial conditions may be unrecoverable:
- Low altitude (5m) + high pitch (15°) + slow speed (11.2 m/s) = stall and crash
- Aircraft doesn't have time to recover before hitting ground

### Issue 3: Controller Not Tuned for New Inertias
The PID gains in the classical controller were tuned for the old inertias:
- **Old:** Ixx=0.42, Iyy=0.55, Izz=0.78
- **New:** Ixx=1.147, Iyy=1.614, Izz=2.709 (2.7-3.5x larger)

**Effect:**
- Same control inputs produce **smaller angular accelerations**
- Controller may be too aggressive or too sluggish
- Needs re-tuning for new mass properties

### Issue 4: Elevator Sign Convention
**Correlation(altitude_error, elevator):** -0.131 (weak)

The weak correlation suggests either:
1. Controller is fighting instability (not tracking altitude smoothly)
2. Many episodes diverging/crashing before settling
3. Initial conditions dominating behavior

---

## Comparison: Old vs New Inertias

### Previous Dataset (Ixx=0.42, Iyy=0.55, Izz=0.78):
- Mean altitude: ~30m (but still had issues)
- Pitch range: ±180° (flipping)
- Mean reward: -1.460

### Current Dataset (Ixx=1.147, Iyy=1.614, Izz=2.709):
- Mean altitude: 5.59m (worse!)
- Pitch range: ±180° (still flipping)
- Mean reward: -2.843 (significantly worse!)

**Conclusion:** The larger inertias made the problem **worse** because:
1. Slower angular response → harder to recover from disturbances
2. Same aggressive initial conditions → more crashes
3. Controller gains not adjusted → poor tracking

---

## Visualization Analysis

See: [bc_dataset_visualization_NEW_INERTIA.png](C:\Users\Administrator\Desktop\bc_dataset_visualization_NEW_INERTIA.png)

Expected to see:
- **Altitude oscillations** around 30m target (going negative = underground)
- **Pitch flipping** (±180° wraparound)
- **Elevator saturation** (±15° limits)
- **Divergent trajectories** (some episodes unstable)

---

## Root Cause Summary

The BC dataset quality issues are **NOT caused by inertia values** but by:

1. **Missing ground physics** in GPU simulator
   - Aircraft can fly underground
   - No crash detection
   - No ground contact forces

2. **Too aggressive initial conditions**
   - Low altitude + extreme states = unrecoverable
   - Need gentler initialization

3. **Controller not tuned for new dynamics**
   - PID gains designed for old (smaller) inertias
   - Need to reduce gains or increase damping

4. **No episode termination on failure**
   - Crashed episodes continue running
   - Pollutes dataset with garbage data

---

## Recommendations

### Option 1: Fix Ground Physics (Ideal)
Add ground collision to GPU simulator:
```python
# In flight_dynamics.py step() function
altitude = -states[:, StateIndex.Z]
crashed = altitude < 0
states[crashed, :] = 0  # Freeze crashed instances
```

### Option 2: Better Initial Conditions (Quick Fix)
Modify [generate_bc_dataset_gpu.py](\\wsl.localhost\Ubuntu-22.04\home\AIDA\scripts\generate_bc_dataset_gpu.py):
```python
# Start higher and more stable
altitude = np.random.uniform(20.0, 60.0, n)  # Higher minimum
airspeed = np.random.uniform(18.0, 30.0, n)  # Faster (safer)
pitch = np.random.uniform(-5, 5, n)          # Smaller range
```

### Option 3: Retune Controller Gains (Medium Effort)
Adjust PID gains for larger inertias in [generate_bc_dataset_gpu.py](\\wsl.localhost\Ubuntu-22.04\home\AIDA\scripts\generate_bc_dataset_gpu.py):
```python
# Pitch controller (lines 163-173)
self.altitude_Kp = 0.5  # Reduce from current value
self.altitude_Kd = 0.3  # Reduce damping gain
```

### Option 4: Use PPO Directly (Pragmatic)
Skip BC pre-training and train PPO from scratch:
- PPO is robust and can learn from failures
- Will explore and find stable policies
- May take longer but more reliable

---

## Decision Matrix

| Approach | Effort | Quality | Time |
|----------|--------|---------|------|
| Fix ground physics | High | Best | 2-3 hours |
| Better init conditions | Low | Good | 15 min |
| Retune controller | Medium | Good | 1 hour |
| Skip BC, use PPO only | Low | Good | 30 min setup |

---

## Immediate Next Steps

### Quick Test: Better Initial Conditions
1. Modify [generate_bc_dataset_gpu.py](\\wsl.localhost\Ubuntu-22.04\home\AIDA\scripts\generate_bc_dataset_gpu.py) lines 251-262
2. Increase minimum altitude: 5m → 30m
3. Reduce pitch variance: ±15° → ±5°
4. Regenerate dataset
5. Check if altitude stays positive

### If that works:
- Train PPO with cleaner BC dataset
- Evaluate trained policy

### If still unstable:
- Skip BC pre-training entirely
- Train PPO from random policy
- Let RL learn stable flight from scratch

---

## Status

⚠️ **BC Dataset Generated but Low Quality**

**Issues:**
- ❌ Aircraft flying underground
- ❌ Pitch flipping (±180°)
- ❌ Mean reward: -2.843 (poor)

**Cause:**
- Missing ground physics in simulator
- Initial conditions too aggressive for new (larger) inertias
- Controller gains not adjusted

**Next Action:**
- **Recommend:** Adjust initial conditions and regenerate
- **OR:** Skip BC and train PPO directly

---

**Prepared by:** Claude Code
**For:** Kushal Koirala - AIDA Project
