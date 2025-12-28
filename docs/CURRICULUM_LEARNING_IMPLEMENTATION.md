# Curriculum Learning Implementation for Cessna 172

**Date:** December 27, 2024
**Status:** ✅ Implemented, Training in Progress (Phase 1)
**Based on:** Moving Window DAgger (MwDAgger) curriculum approach

---

## Executive Summary

After discovering that pure PPO training failed (flatlined at 102 steps, +70.8 reward), we implemented **curriculum learning** based on insights from the academic paper on imitation learning for fixed-wing UAVs.

**Key Innovation:** Break the complex full-mission task into 5 progressive phases, mastering each before advancing.

---

## Why Curriculum Learning?

### The Problem We Had

**Pure PPO Training (Failed):**
- Episodes stuck at 102 steps
- Reward flatlined at +70.8
- 99.4% clip fraction (policy frozen)
- Agent learned "do nothing" policy
- **Root cause:** Task too complex to learn from scratch

**Evidence from Academic Paper:**
- Baseline supervised learning: 16.375s flights (failed)
- Sequential DAgger: Decreasing flight times (failed)
- **Solution:** Moving Window DAgger (MwDAgger) curriculum learning → 180+ second flights ✅

### The Solution

**Curriculum Learning Philosophy:**
> "You can't learn to land before you learn to take off"

Just like student pilot training:
1. Ground school
2. Taxi practice
3. Takeoff training
4. Pattern work
5. Landing practice

We train the RL agent the same way!

---

## Curriculum Design

### 5-Phase Progressive Training

| Phase | Task | Duration | Goal | Success Threshold | Timesteps |
|-------|------|----------|------|-------------------|-----------|
| **1** | Ground Roll | 10s | Accelerate to 55 KIAS | 90% | 200k |
| **2** | Rotation | 15s | Liftoff to 10 ft AGL | 90% | 300k |
| **3** | Initial Climb | 30s | Climb to 500 ft | 85% | 400k |
| **4** | Full Climb | 60s | Reach 3000 ft cruise | 80% | 500k |
| **5** | Cruise | 30s | Maintain altitude/speed | 80% | 600k |

**Total training time:** ~5 hours (vs infinite with non-curriculum PPO)

---

## Phase Definitions

### Phase 1: Ground Roll

**Task:** `ground_roll`
**Duration:** 500 steps (10 seconds)

**Goals:**
1. Accelerate from 0 to rotation speed (28 m/s = 55 KIAS)
2. Stay on runway centerline (±15m lateral deviation)
3. Keep wings level (minimal roll)
4. Maintain runway heading

**Success Criteria:**
```python
{
    'airspeed_min': 28.0,       # Must reach rotation speed
    'altitude_max': 2.0,        # Stay on ground
    'heading_max_dev': 15.0,    # Stay aligned with runway
}
```

**Reward Function:**
- +50 points for speed progress (0 → 28 m/s)
- +50 points bonus for reaching rotation speed
- -0.5 per meter lateral deviation
- -20 points for going off runway
- -5 points per radian of roll
- -3 points per radian of heading error

**Max Reward:** +100 for perfect ground roll

---

### Phase 2: Rotation

**Task:** `rotation`
**Duration:** 750 steps (15 seconds)

**Goals:**
1. Complete ground roll (reach rotation speed)
2. Gentle pitch-up (~10°)
3. Liftoff to 10 ft AGL (3 m)
4. Maintain positive climb rate

**Success Criteria:**
```python
{
    'altitude_min': 3.0,        # Reach 10 ft AGL
    'airspeed_min': 30.0,       # 58 KIAS
    'pitch_max': 0.26 rad,      # 15° max pitch
}
```

**Reward Function:**
- **On ground:** +30 for speed progress, prepare for rotation
- **Airborne:**
  - +40 for altitude progress (0 → 3m)
  - +30 bonus for reaching target altitude
  - +0.3 per m/s climb rate
  - -0.5 per m/s airspeed error
  - -1.0 per radian pitch error

**Max Reward:** +100 for successful liftoff

---

### Phase 3: Initial Climb

**Task:** `initial_climb`
**Duration:** 1500 steps (30 seconds)

**Goals:**
1. Climb from liftoff to 500 ft AGL (152.4 m)
2. Maintain climb speed (38 m/s = 75 KIAS)
3. Target climb rate (3.0 m/s = ~600 fpm)
4. Wings level, coordinated flight

**Success Criteria:**
```python
{
    'altitude_min': 152.4,      # 500 ft AGL
    'airspeed_min': 35.0,       # 68 KIAS
    'climb_rate_min': 2.0,      # ~400 fpm minimum
}
```

**Reward Function:**
- +50 for altitude progress (0 → 152.4m)
- +30 bonus for reaching 500 ft
- -0.3 per m/s airspeed error
- +2 bonus if within ±3 m/s of target speed
- -0.5 per m/s climb rate error
- +3 bonus if climb rate 2.5-3.5 m/s
- -10 if descending
- -2 per radian of bank angle
- -1 per radian of pitch error (target: 12°)

**Max Reward:** +100 for reaching 500 ft with good technique

---

### Phase 4: Full Climb

**Task:** `full_climb`
**Duration:** 3000 steps (60 seconds)

**Goals:**
1. Climb to cruise altitude (3000 ft = 914.4 m)
2. Maintain climb speed (35-40 m/s)
3. Sustained climb rate (>1.5 m/s = ~300 fpm)

**Success Criteria:**
```python
{
    'altitude_min': 914.4,      # 3000 ft cruise altitude
    'airspeed_min': 35.0,       # 68 KIAS
    'climb_rate_min': 1.5,      # ~300 fpm
}
```

**Reward Function:**
- Same as existing `_reward_climb()` function
- +75 for reaching cruise altitude
- Speed and climb rate tracking
- Altitude progress rewards

**Max Reward:** +100+ for successful climb

---

### Phase 5: Cruise

**Task:** `cruise`
**Duration:** 1500 steps (30 seconds)

**Goals:**
1. Hold cruise altitude (3000 ft ± 50 ft)
2. Maintain cruise speed (56 m/s = 110 KIAS ± 6 knots)
3. Level flight (minimal pitch/roll)
4. Smooth controls

**Success Criteria:**
```python
{
    'altitude_tolerance': 15.0,     # ±50 ft
    'airspeed_tolerance': 5.0,      # ±10 knots
    'bank_max': 0.17 rad,           # 10° max bank
    'duration': 600,                # 12 seconds stable
}
```

**Reward Function:**
- Same as existing `_reward_cruise()` function
- -0.05 per meter altitude error
- +0.5 bonus if within ±15m (50 ft)
- -0.05 per m/s speed error
- +0.3 bonus if within ±3 m/s
- -0.1 per radian pitch/roll
- -0.03 for control magnitude

**Max Reward:** +100 for perfect cruise hold

---

## Transfer Learning Between Phases

Each phase initializes from the previous phase's final model:

```python
Phase 1: Train from scratch → phase1_ppo_final.zip
Phase 2: Load phase1_ppo_final.zip → train → phase2_ppo_final.zip
Phase 3: Load phase2_ppo_final.zip → train → phase3_ppo_final.zip
Phase 4: Load phase3_ppo_final.zip → train → phase4_ppo_final.zip
Phase 5: Load phase4_ppo_final.zip → train → phase5_ppo_final.zip
```

**Benefits:**
- Faster learning (warm start)
- Preserved knowledge from earlier phases
- Continuous skill progression

---

## Implementation Details

### Environment Modifications

**File:** [aida_sim/env/flight_env_cessna172.py](../aida_sim/env/flight_env_cessna172.py)

**Changes:**
1. Added `phase_configs` dict with task definitions
2. Created 3 new reward functions:
   - `_reward_ground_roll()` - Lines 718-790
   - `_reward_rotation()` - Lines 792-878
   - `_reward_initial_climb()` - Lines 880-960
3. Updated `_compute_reward()` to dispatch to phase-specific functions
4. Added phase-specific success criteria validation

### Training Script

**File:** [scripts/train_cessna172_curriculum.py](../scripts/train_cessna172_curriculum.py)

**Features:**
- Auto-start TensorBoard (http://localhost:6006)
- Sequential phase training with transfer learning
- Automatic evaluation after each phase (100 episodes)
- Success rate checking (must reach threshold to proceed)
- Checkpoint saving every 50k steps
- Best model tracking

**Usage:**
```bash
# Train full curriculum (all 5 phases)
python scripts/train_cessna172_curriculum.py --n-envs 4 --device cuda

# Start from specific phase (e.g., Phase 3)
python scripts/train_cessna172_curriculum.py --start-phase 3 --n-envs 4

# Skip evaluation (faster, no safety checks)
python scripts/train_cessna172_curriculum.py --skip-evaluation --n-envs 4
```

### Testing Script

**File:** [scripts/test_curriculum_phases.py](../scripts/test_curriculum_phases.py)

**Purpose:** Sanity check all phases before training

**Usage:**
```bash
python scripts/test_curriculum_phases.py
```

**Tests:**
1. Environment creation
2. Reset functionality
3. Episode execution
4. Reward computation
5. Termination conditions

---

## Training Progress

### Current Status

**Date:** December 27, 2024, 2:15 PM
**Phase:** Phase 1 - Ground Roll
**Status:** Training started
**Task ID:** b7173c2

**Configuration:**
- Parallel environments: 4 (SubprocVecEnv)
- Device: CUDA (RTX 4060)
- Timesteps target: 200,000
- Success threshold: 90%

**TensorBoard:** http://localhost:6006

### Expected Training Timeline

| Phase | Timesteps | Estimated Time | Completion |
|-------|-----------|----------------|------------|
| 1 | 200k | ~30 min | ⏳ In Progress |
| 2 | 300k | ~45 min | ⏸️ Pending |
| 3 | 400k | ~1 hour | ⏸️ Pending |
| 4 | 500k | ~1.25 hours | ⏸️ Pending |
| 5 | 600k | ~1.5 hours | ⏸️ Pending |

**Total:** ~5 hours for complete curriculum

---

## Success Criteria

### Phase Evaluation

After each phase training completes:

1. **Automatic Evaluation:** 100 test episodes
2. **Success Metrics:**
   - Success rate (% of episodes with reward > 50)
   - Mean episode reward
   - Mean episode length
3. **Threshold Check:**
   - Phase 1-2: Must reach 90% success rate
   - Phase 3: Must reach 85% success rate
   - Phase 4-5: Must reach 80% success rate
4. **User Decision:** If threshold not met, option to retry or continue

### Final Model Validation

After Phase 5:
1. Test on `full_mission` task (all phases combined)
2. Compare to Cessna 172 POH specifications:
   - Rotation speed: 55 KIAS
   - Climb rate: 730 fpm
   - Cruise speed: 110 KIAS
3. Visual inspection of flight trajectory (3D visualization)

---

## Comparison: Old vs New Approach

| Aspect | Pure PPO (Failed) | Curriculum PPO (New) |
|--------|------------------|---------------------|
| **Task** | Full mission from scratch | 5 progressive phases |
| **Training time** | Infinite (never learned) | ~5 hours |
| **Episode length** | Stuck at 102 steps | Progresses through phases |
| **Reward** | Flatlined at +70.8 | Progressive improvement |
| **Success rate** | 0% | Target 80-90% per phase |
| **Agent behavior** | "Do nothing" policy | Learns flight skills |
| **Inspiration** | Standard RL | Moving Window DAgger (academic paper) |

---

## Theoretical Foundation

### Moving Window DAgger (MwDAgger)

From the academic paper:

**Problem with Sequential Learning:**
- Data concentrates on early flight regions
- Agent never sees later phases
- Distribution mismatch between training and deployment

**MwDAgger Solution:**
- Time-based curriculum windows: [0, T1] → [0, T2] → [0, T3]
- Agent masters early phases before seeing later phases
- Progressive expansion of training window

**Our Adaptation:**
- Instead of time windows, we use **task phases**
- Each phase is a complete sub-task
- Transfer learning connects phases
- PPO optimization instead of imitation learning

**Key Insight:**
> "Curriculum learning is essential for complex sequential tasks"

---

## Advantages of Our Approach

1. **No Expert Demonstrations Required**
   - Pure RL (PPO) learns from scratch
   - No need for GNC controller
   - Agent can discover optimal policies

2. **GPU-Accelerated**
   - 77% GPU utilization
   - Parallel environments (4 workers)
   - Fast training (~5 hours vs days)

3. **Proven Concept**
   - Based on academic research (MwDAgger)
   - Similar problem domain (fixed-wing UAV)
   - Curriculum learning is well-established

4. **Modular Design**
   - Each phase is independent
   - Easy to retrain single phase
   - Can tune hyperparameters per phase

5. **Interpretable Progress**
   - Clear success/failure per phase
   - Observable skill progression
   - Easy to debug which phase is failing

---

## Potential Issues & Solutions

### Issue 1: Phase Not Reaching Success Threshold

**Symptom:** Phase 1 trains for 200k steps but only reaches 70% success rate

**Solutions:**
1. Increase timesteps (200k → 300k)
2. Tune reward function (adjust weight balances)
3. Adjust hyperparameters (learning rate, entropy coefficient)
4. Check for bugs in phase-specific reward function

### Issue 2: Transfer Learning Not Helping

**Symptom:** Phase 2 performs worse than random policy despite loading Phase 1 model

**Solutions:**
1. Train Phase 2 from scratch (don't load Phase 1)
2. Reduce learning rate for fine-tuning
3. Check if Phase 1 and Phase 2 tasks are compatible
4. Use progressive unfreezing (freeze early layers initially)

### Issue 3: Later Phases Too Hard

**Symptom:** Phase 4-5 never reach success threshold

**Solutions:**
1. Add intermediate phases (e.g., "climb to 1500 ft" between Phase 3 and 4)
2. Relax success thresholds (80% → 70%)
3. Increase training timesteps
4. Simplify reward function (fewer competing objectives)

### Issue 4: Ground Physics Still Problematic

**Symptom:** Aircraft slides sideways, unrealistic ground behavior

**Solutions:**
1. Add friction forces to ground contact model
2. Implement proper landing gear physics
3. Use higher ground contact damping
4. Restrict lateral controls when on ground

---

## Next Steps

### During Training (Monitoring)

1. **Watch TensorBoard:** http://localhost:6006
   - `ep_rew_mean` should increase
   - `ep_len_mean` should increase (more steps before termination)
   - `clip_fraction` should be 0.1-0.3 (healthy PPO updates)

2. **Check Phase Success:**
   - After Phase 1 completes, should see 90%+ success rate
   - Episode rewards should be mostly > 50

3. **GPU Utilization:**
   - Should be 70-80%
   - If lower, can increase `n_envs`

### After Phase 1 Completes (~30 min)

1. Review evaluation results
2. If successful (>90%), proceed to Phase 2
3. If not, retrain Phase 1 with adjustments

### After All Phases Complete (~5 hours)

1. **Test on Full Mission:**
   ```bash
   python scripts/test_cessna172_env.py \
       --model checkpoints/cessna172_curriculum/phase5_cruise/phase5_cruise_ppo_final.zip \
       --task full_mission
   ```

2. **Create Visualization:**
   ```bash
   python scripts/visualize_cessna172.py \
       --model checkpoints/cessna172_curriculum/phase5_cruise/phase5_cruise_ppo_final.zip
   ```

3. **Compare to POH:**
   - Extract rotation speed, climb rate, cruise speed
   - Compare to Cessna 172 POH specifications
   - Validate flight path makes sense

4. **Git Commit:**
   - Add curriculum learning files to staging
   - Update COMMIT_MESSAGE.md
   - Create commit with all changes

---

## Files Created/Modified

### New Files

1. **[scripts/train_cessna172_curriculum.py](../scripts/train_cessna172_curriculum.py)** (401 lines)
   - Main curriculum training script
   - Auto-start TensorBoard
   - Sequential phase training
   - Evaluation and success checking

2. **[scripts/test_curriculum_phases.py](../scripts/test_curriculum_phases.py)** (116 lines)
   - Sanity check for all phases
   - Tests environment creation and execution
   - Validates reward computation

3. **[docs/IMITATION_LEARNING_ANALYSIS.md](IMITATION_LEARNING_ANALYSIS.md)** (644 lines)
   - Analysis of why PPO training failed
   - Comparison to academic paper findings
   - Three solution options (A, B, C)
   - Detailed implementation plan

4. **[docs/CURRICULUM_LEARNING_IMPLEMENTATION.md](CURRICULUM_LEARNING_IMPLEMENTATION.md)** (This file)
   - Complete curriculum learning documentation
   - Phase definitions and reward functions
   - Training progress tracking
   - Troubleshooting guide

### Modified Files

1. **[aida_sim/env/flight_env_cessna172.py](../aida_sim/env/flight_env_cessna172.py)**
   - Added `phase_configs` dict (Lines 73-148)
   - Updated docstring with new tasks (Lines 37-48)
   - Created `_reward_ground_roll()` (Lines 718-790)
   - Created `_reward_rotation()` (Lines 792-878)
   - Created `_reward_initial_climb()` (Lines 880-960)
   - Updated `_compute_reward()` to dispatch to phase functions (Lines 492-523)

---

## References

1. **Academic Paper:** "Imitation Learning for Neural Network Autopilot in Fixed-Wing Unmanned Aerial Systems"
   - Key contribution: Moving Window DAgger (MwDAgger)
   - Proves curriculum learning essential for flight tasks
   - 180+ second flights vs 16s baseline

2. **Cessna 172 POH** - Performance specifications

3. **Schulman et al. (2017)** - "Proximal Policy Optimization Algorithms"

4. **Stable-Baselines3 Documentation** - PPO implementation

5. **[docs/IMITATION_LEARNING_ANALYSIS.md](IMITATION_LEARNING_ANALYSIS.md)** - Problem analysis and solution design

---

## Summary

We've successfully pivoted from failed pure PPO training to a **curriculum learning approach** inspired by academic research. This implementation:

✅ Breaks complex task into 5 manageable phases
✅ Uses transfer learning between phases
✅ Has clear success criteria for each phase
✅ Automates evaluation and progression
✅ Provides detailed monitoring via TensorBoard
✅ Estimated to complete in ~5 hours

**Current Status:** Phase 1 (Ground Roll) training in progress!

**Next Milestone:** Phase 1 evaluation in ~30 minutes

---

**Author:** Kushal Koirala & Claude Code
**Date:** December 27, 2024
**Training Started:** 2:15 PM
**TensorBoard:** http://localhost:6006
