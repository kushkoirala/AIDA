# Phase 1 Ground Roll Training - Session Summary
**Date**: December 27, 2024
**Status**: Training halted - results not meeting expectations

---

## Session Overview

This session focused on improving Phase 1 (ground roll) training for the Cessna 172 autonomous takeoff using PPO reinforcement learning. Multiple approaches were tested to address pitch-up issues and short episode durations.

---

## Problem Statement

**Initial Issues (from previous session)**:
- Training crashed at 168k steps with CUDA error
- 150k checkpoint evaluation showed:
  - ✅ 100% success reaching rotation speed (43.3 m/s)
  - ❌ Severe pitch-up issues (attitude_limit termination)
  - ❌ Extreme lateral drift (200m off centerline)
  - ❌ Large heading errors (60°)
  - ❌ Very short episodes: 8.4 seconds vs target 10-15 seconds

---

## Approaches Tried

### Approach 1: Forced Elevator Trim (Previous Session)
**Implementation**: Added -3° nose-down elevator trim throughout ground roll phase

**Result**: ❌ FAILED
- Agent learned to add positive elevator on top of trim
- Still experienced pitch-up and attitude_limit termination
- Episode duration: 8.4 seconds

**User Feedback**: "I think instead of forcing a certain elevator angle we should let it figure out the elevator angle that can keep the aircraft nose on the ground?"

---

### Approach 2: Pure Learning (No Trim, Strong Penalties)
**Implementation**:
1. Completely removed forced elevator trim
2. Massively increased pitch-up penalties:
   - -50 × theta (was -20)
   - -200 for pitch >5° (was -50)
   - +5 bonus for keeping nose down
3. Strengthened lateral tracking rewards:
   - -100 penalty for going off runway (was -20)
   - +10 bonus for staying within 1m centerline (NEW)
   - +5 bonus for staying within 5m (NEW)
   - Quadratic penalty: -2.0 × (lateral_dev²)
4. Strengthened heading control:
   - +5 bonus for heading error <2°
   - -10 × heading_error penalty (was -3.0)

**Result**: ❌ FAILED
- Training completed Phase 1 (200k steps) with 100% success
- Episode duration: **110 steps (2.2 seconds)** - even shorter!
- Agent hitting early termination too frequently

**User Feedback**: "i have a feeling we are wasting time"

---

### Approach 3: Initial Elevator Position + Learning (Current)
**Implementation**:
1. Set initial elevator to -5° nose-down at episode start in `reset()` function
2. Let agent learn to adjust from this good starting point
3. Keep all strong reward shaping from Approach 2

**Code Changes**:
- File: `aida_sim/env/flight_env_cessna172.py`
- Lines 314-320: Added initial control setup
```python
# Set initial elevator position for ground operations
# This gives the agent a good starting point (nose-down to keep nose wheel on ground)
# but lets it learn to adjust from there
if self.task in ['ground_roll', 'rotation']:
    initial_controls = np.zeros((1, CONTROL_DIM), dtype=np.float32)
    initial_controls[0, ControlIndex.ELEVATOR] = np.deg2rad(-5.0)  # -5° nose-down initial position
    self.sim.controls = initial_controls.copy()
```

**Training Configuration**:
- 4 parallel environments (SubprocVecEnv)
- Device: CUDA (GPU acceleration)
- Total timesteps: 200,000
- Success threshold: 90%
- Training speed: ~52 FPS

**Results**: ⚠️ IMPROVED BUT INSUFFICIENT
- Phase 1 completed at 200,000 steps
- Episode duration: **124-130 steps (6.2-6.5 seconds)** vs target 10-15 seconds
- Reward progression (eval checkpoints):
  - Start: -5,260
  - 20k: -3,622
  - 30k: -1,083 (70% improvement!)
  - 40k: -1,499
  - 50k: -690
  - 60k: **-187** (96% improvement from start!)

**Analysis**:
- ✅ Reward improved dramatically (96% improvement)
- ✅ Episode duration improved from 2.2s to 6.5s (3x longer)
- ✅ Agent learning pitch and lateral control
- ❌ Still far short of 10-15 second target
- ❌ Episodes terminating early (likely attitude_limit or out_of_bounds)

**User Decision**: Halted training - results not good enough

---

## Current Reward Function Details

### Ground Roll Reward Components (flight_env_cessna172.py:730-824)

1. **Airspeed Progression** (0 → 28 m/s):
   - +50 × (speed/V_rotate) proportional reward
   - +50 bonus for reaching rotation speed
   - +0.5 × throttle during acceleration

2. **Lateral Tracking**:
   - +10 bonus: within 1m of centerline
   - +5 bonus: within 5m of centerline
   - -2.0 × (lateral_dev²): moderate deviation
   - -100: off runway (severe)

3. **Heading Control**:
   - +5 bonus: heading error <2°
   - -10 × heading_error otherwise

4. **Pitch Control**:
   - +5 bonus: nose down (theta ≤ 0°)
   - -50 × theta: any pitch-up
   - -200: pitch >5° (excessive)

5. **Roll Control**:
   - -10 × |phi|: wings not level

6. **Altitude Control**:
   - -10: altitude >2m (premature liftoff)

7. **Control Usage**:
   - -0.5 × |rudder|: excessive rudder use

8. **Survival & Termination**:
   - +0.1: per timestep
   - -50: crash
   - -30: out_of_bounds

---

## Technical Details

### Environment Configuration
- Simulation timestep (dt): 0.05s (20Hz)
- Max episode steps: 500 (25 seconds)
- Target episode duration: 200-300 steps (10-15 seconds)
- Rotation speed (V_rotate): 28 m/s (55 KIAS)
- Runway width: ~30m (standard)

### Aircraft Physics (Cessna 172)
- CL0 = 0.307 (natural pitch-up tendency)
- Cm0 = 0.04 (pitching moment bias nose-up)
- CLde = 0.43 (elevator lift effectiveness)
- Cmde = -1.122 (elevator pitch effectiveness)

### Training Infrastructure
- Framework: Stable Baselines3 PPO
- Parallel environments: 4 (SubprocVecEnv)
- GPU acceleration: CuPy for flight dynamics
- Checkpoints: Every 10k steps
- Evaluation: Every 10k steps
- TensorBoard: http://10.0.1.208:6006
- Telemetry viewer: http://10.0.1.208:8000

---

## Key Files Modified

1. **aida_sim/env/flight_env_cessna172.py**
   - Lines 70-75: Removed forced elevator trim
   - Lines 314-320: Added initial elevator position in reset()
   - Lines 401-409: Simplified control mapping (removed trim application)
   - Lines 751-824: Strengthened reward function

2. **Training Script**: `scripts/train_cessna172_curriculum.py`
   - Used with: `--start-phase 1`
   - 4 parallel environments
   - CUDA device

---

## Checkpoint Locations

All checkpoints saved in: `/home/AIDA/checkpoints/cessna172_curriculum/`

**Phase 1 Training Runs**:
- `phase1_ground_roll/tensorboard/phase1_ground_roll_1` - Early attempt with trim
- `phase1_ground_roll/tensorboard/phase1_ground_roll_2` - Intermediate
- `phase1_ground_roll/tensorboard/phase1_ground_roll_3` - Pure learning (failed)
- `phase1_ground_roll/tensorboard/phase1_ground_roll_4` - **Current run** (initial elevator position)

**Key Checkpoints from Run 4**:
- `best_model.zip` - Best eval performance (-187 reward at 60k steps)
- `rl_model_60000_steps.zip` - 60k checkpoint
- Episode length: 124 steps (6.2 seconds)

---

## Root Cause Analysis

### Why Episodes Are Still Too Short

**Hypothesis**: Agent is hitting termination conditions too early, likely:

1. **Attitude Limit (45° pitch)**: Despite penalties, agent may still pitch up enough to hit limit
2. **Out of Bounds**: Lateral deviation may still exceed runway boundaries
3. **Max Steps Not Reached**: Agent not learning to maintain control for full duration

**Evidence**:
- Episodes consistently around 124-130 steps
- Not hitting the 500 step maximum
- Rapid reward improvement suggests agent IS learning
- But learning might be focused on "survive as long as possible" rather than "maintain control for full takeoff"

### Why Pure Penalties Didn't Work

The pure penalty approach (Approach 2) likely created a reward structure where:
- Agent learned to minimize penalties by terminating quickly
- Short episodes avoid accumulating penalties
- No incentive to extend episode duration

### Why Initial Position Helped But Not Enough

Initial -5° elevator:
- ✅ Gives agent better starting point
- ✅ Episodes 3x longer than pure learning
- ❌ But agent still doesn't learn to maintain this throughout episode
- ❌ May need stronger constraints or different reward structure

---

## Potential Next Steps (Not Attempted)

### Option 1: Adjust Success Criteria
- Current: Agent must reach V_rotate (28 m/s)
- Consider: Add explicit time-based requirement (must maintain control for 10+ seconds)

### Option 2: Curriculum Within Phase 1
- Start: Just maintain nose-down for 5 seconds (no speed requirement)
- Then: Maintain nose-down for 10 seconds
- Then: Maintain nose-down while accelerating
- Finally: Full ground roll task

### Option 3: Modify Termination Conditions
- Relax attitude_limit from 45° to 60° during ground roll
- Add "soft" penalties instead of hard termination
- Let agent learn from longer episodes

### Option 4: Different Initial Conditions
- Try different elevator angles (-3°, -7°, -10°)
- Add initial throttle setting
- Add initial rudder bias for directional control

### Option 5: Behavior Cloning + RL
- Generate expert demonstrations of good ground rolls
- Pre-train policy with BC
- Fine-tune with PPO

### Option 6: Different Reward Structure
- Reward based on time maintaining good state (nose down, centerline, heading)
- De-emphasize speed progression early in training
- Progressive reward shaping

---

## User's Perspective Evolution

1. **Initial**: Wanted to visualize training, see results
2. **After pitch-up issues**: "let it figure out the elevator angle"
3. **After pure learning failed**: "we are wasting time"
4. **Final approach**: "add the elevator at the beginning but then let it adjust, perhaps 5 degrees"
5. **After completion**: Results "not good" - halted all work

User showed iterative problem-solving approach, testing in telemetry viewer, and pragmatic decision-making when approaches weren't working.

---

## Lessons Learned

1. **Forced constraints prevent learning**: Hard-coded trim didn't allow agent to learn proper control
2. **Pure penalties can backfire**: Too-strong penalties may incentivize early termination
3. **Initial conditions matter**: Starting from good state (initial elevator) significantly improved results
4. **Episode duration ≠ success**: Agent can improve rewards without achieving task goals
5. **6 seconds is not 10-15 seconds**: Even 3x improvement may not be enough for real-world requirements

---

## Session End State

- Training: **STOPPED**
- Best checkpoint: 60k steps, -187 eval reward, 124 step episodes
- Network access: TensorBoard on http://10.0.1.208:6006
- All code changes: Committed to `/home/AIDA/aida_sim/env/flight_env_cessna172.py`
- Summary saved: This document

**Next session should**: Consider fundamentally different approach or adjust task requirements before continuing RL training.
