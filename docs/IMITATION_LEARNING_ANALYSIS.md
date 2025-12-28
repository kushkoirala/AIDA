# Imitation Learning Analysis: Why PPO Training Failed

**Date:** December 27, 2024
**Author:** Kushal Koirala & Claude Code
**Context:** Analysis of training failure after reading academic paper on imitation learning for fixed-wing UAV autopilot

---

## Executive Summary

Our Cessna 172 PPO training exhibited **identical failure patterns** to those described in the academic paper "Imitation Learning for Neural Network Autopilot in Fixed-Wing Unmanned Aerial Systems." The paper provides critical insights into why our approach failed and offers proven solutions.

**Key Finding:** Complex flight missions cannot be learned from scratch using pure reinforcement learning OR simple imitation learning. **Curriculum learning** is essential.

---

## Problem Comparison: PDF vs Our Training

### Their Problems (From Paper)

| Method | Flight Duration | Issue |
|--------|----------------|-------|
| Baseline Supervised Learning | 16.375s | Error accumulation, cannot fly stably |
| Sequential DAgger | Decreasing over time (-0.54132 slope) | Data distribution mismatch, concentrates on early flight |
| Monte Carlo DAgger | ~60-70s (partial success) | Random sampling helps but incomplete |
| **MwDAgger (Solution)** | **180s+ (full mission)** | **Curriculum learning works** |

### Our Problems (Cessna 172 PPO)

| Phase | Episode Length | Reward | Issue |
|-------|---------------|---------|-------|
| Initial (before fixes) | 17 steps (0.34s) | -5.51 | Ground physics bug - aircraft fell through ground |
| After bug fixes | 146 steps (2.9s) with random actions | Variable | Sanity checks passed |
| **Training at 111K steps** | **102 steps (2.04s)** | **+70.8 (flatlined)** | **Agent learned "do nothing" policy** |

### Critical Similarity

**PDF Quote:** "The baseline model [...] only flies for an average of 16.375 seconds before crashing."

**Our Result:** Episodes stuck at **102 steps = 2.04 seconds** - almost same order of magnitude!

Both exhibit:
- ❌ Episodes terminating at fixed point
- ❌ No learning progress (flatlined metrics)
- ❌ Agent stuck in failure mode
- ❌ Cannot progress beyond early flight phase

---

## Why Our PPO Training Failed

### 1. **Task Too Complex for Scratch Learning**

**Our Task:**
```
Ground Roll (0-1s) → Rotation (1-2s) → Liftoff (2-3s)
→ Climb (3-20s) → Level Off (20-25s) → Cruise (25-60s)
```

**Problem:** Agent must discover ALL phases sequentially. If it fails at step 1 (ground roll), it NEVER sees step 2 (rotation).

**PDF Insight:** "Sequential data aggregation results in the data being focused on the beginning of the flight, resulting in a data distribution mismatch."

### 2. **Reinforcement Learning Exploration Issue**

**What Happened:**
1. Random policy → crashes immediately
2. Agent learns "sit still" gives +70.8 reward (better than crashing)
3. Policy converges to "do nothing"
4. **99.4% clip fraction** = policy frozen, no updates
5. **Episode length stuck at 102 steps** = hitting attitude limit from sitting still

**Why This Happens:**
- Sparse reward signal (only get +50 for takeoff IF you take off)
- Agent never discovers takeoff by random exploration
- Local optimum: "don't crash" (reward +70.8) vs "try to fly" (reward -5.0 crash)

### 3. **Missing Curriculum Learning**

**PDF Solution:** Moving Window DAgger (MwDAgger)
- **Phase 1 (0-60s):** Learn early flight (ground roll, rotation)
- **Phase 2 (0-120s):** Add climb phase
- **Phase 3 (0-180s+):** Full mission

**Our Approach:** Tried to learn entire mission at once
- Equivalent to asking a student pilot to do a solo cross-country flight on day 1
- Never works in aviation training
- Shouldn't work in RL training either

---

## Technical Evidence

### PPO Metrics at Failure (111K steps)

```
ep_len_mean: 102 steps (stuck, not increasing)
ep_rew_mean: +70.8 (flatlined)
clip_fraction: 0.994 (99.4% - policy frozen!)
approx_kl: 70,678 (exploding, was 411,996)
learning_rate: 0.0003
value_loss: 0.087
```

**Diagnosis:**
- `clip_fraction = 0.994` means **99.4% of policy updates are being clipped**
  - Healthy range: 0.1-0.3 (10-30%)
  - This means: new policy ≈ old policy (no learning!)
- `approx_kl` exploding → policy trying to make large updates but being clipped
- Episode length stuck → agent in deterministic failure mode

### Reward Breakdown

Our phase-based rewards:
```python
Takeoff:  +50 max  (acceleration, rotation, liftoff)
Climb:    +75 max  (altitude gain, climb rate)
Cruise:  +100 max  (altitude/speed hold, wings level)
Total:   +225 max
```

**Agent getting +70.8 suggests:**
- Partial takeoff reward (~40 points) - accelerating down runway
- Hitting attitude limit at 102 steps (pitch/roll too extreme)
- Never reaches climb or cruise

---

## What the PDF Teaches Us

### Key Insights from Paper

1. **Supervised Learning Alone Fails**
   - Quote: "The baseline model [...] only flies for an average of 16.375 seconds"
   - Error accumulation from compounding mistakes
   - Off-policy data doesn't match deployment distribution

2. **Sequential DAgger Also Fails**
   - Quote: "Sequential data aggregation results in [...] data distribution mismatch"
   - Negative slope (-0.54132) in flight duration
   - Network overfits to early flight regions

3. **Random Sampling Helps (Monte Carlo DAgger)**
   - Randomly sample trajectory points for training
   - Breaks concentration on early flight
   - Still incomplete - doesn't enforce progression

4. **Curriculum Learning is THE Solution (MwDAgger)**
   - Quote: "MwDAgger [...] achieves the longest flight times"
   - Time-based windows: [0, 60s] → [0, 120s] → [0, 180s+]
   - Agent masters early flight BEFORE adding later phases
   - **Result: 180+ second flights (full mission success)**

### Their Architecture

```
GNC Controller (Expert) → Neural Network (Student)
          ↓
    State: [x, y, z, u, v, w, φ, θ, ψ, p, q, r]  (12D - same as ours!)
          ↓
    Action: [throttle, elevator, aileron, rudder]  (4D - same as ours!)
```

**Exactly the same state/action space we're using!**

---

## Solutions: Three Paths Forward

### Option A: Pure Imitation Learning (MwDAgger)

**What:** Implement Moving Window DAgger with GNC expert demonstrations

**How:**
1. Create GNC controller for Cessna 172 (PID autopilot)
2. Collect expert trajectories for takeoff → climb → cruise
3. Train neural network using MwDAgger curriculum:
   - **Phase 1 (0-20s):** Ground roll + rotation + initial climb
   - **Phase 2 (0-40s):** Add climb to 1500 ft
   - **Phase 3 (0-60s+):** Full mission to 3000 ft cruise

**Pros:**
- ✅ Proven to work (paper results)
- ✅ Sample efficient (uses expert data)
- ✅ Stable learning (supervised learning baseline)

**Cons:**
- ⚠️ Requires GNC controller implementation
- ⚠️ Limited to expert's capabilities
- ⚠️ No reward optimization (just imitates)

**Implementation:**
```python
# Phase 1: Train on [0, 20s] window
for episode in range(n_episodes_phase1):
    expert_trajectory = gnc.fly_takeoff(duration=20)
    student_trajectory = model.fly(duration=20)
    dataset.add(expert_trajectory + student_trajectory)
    model.train(dataset.sample())

# Phase 2: Train on [0, 40s] window
for episode in range(n_episodes_phase2):
    expert_trajectory = gnc.fly_takeoff_and_climb(duration=40)
    student_trajectory = model.fly(duration=40)
    dataset.add(expert_trajectory + student_trajectory)
    model.train(dataset.sample())

# Phase 3: Full mission
...
```

---

### Option B: Curriculum Learning for PPO (Recommended)

**What:** Break PPO training into progressive phases, mastering each before advancing

**How:**
1. **Phase 1 (Ground Roll Only):**
   - Task: Accelerate from 0 to 55 KIAS on runway
   - Termination: Reward if reaches 55 KIAS, penalty if crashes
   - Duration: ~10 seconds max
   - **Train until 90% success rate**

2. **Phase 2 (Rotation):**
   - Task: Ground roll + gentle pitch-up to liftoff
   - Termination: Reward if reaches 10 ft altitude
   - Duration: ~15 seconds max
   - **Train until 90% success rate**

3. **Phase 3 (Initial Climb):**
   - Task: Liftoff + climb to 500 ft
   - Termination: Reward if reaches 500 ft at 70 KIAS
   - Duration: ~30 seconds max
   - **Train until 90% success rate**

4. **Phase 4 (Full Climb):**
   - Task: Climb to 3000 ft cruise altitude
   - Duration: ~60 seconds max

5. **Phase 5 (Cruise):**
   - Full mission as currently designed

**Pros:**
- ✅ No expert demonstrations needed
- ✅ PPO can optimize beyond imitation
- ✅ Proven to work (MwDAgger shows curriculum works)
- ✅ Can reuse existing environment

**Cons:**
- ⚠️ Requires multiple training runs
- ⚠️ Need to manually define phase transitions
- ⚠️ More compute time than imitation

**Implementation:**
```python
# scripts/train_cessna172_curriculum.py

# Phase 1: Ground Roll
env_phase1 = Cessna172Env(
    task='ground_roll',  # New task type
    max_episode_steps=500,  # 10 seconds
    success_criteria={'airspeed_min': 55 * 0.514}  # 55 KIAS
)
model_phase1 = PPO('MlpPolicy', env_phase1, ...)
model_phase1.learn(total_timesteps=200_000)

# Evaluate: must reach 90% success rate
success_rate = evaluate(model_phase1, env_phase1, n_episodes=100)
assert success_rate > 0.9, "Phase 1 not mastered"

# Phase 2: Rotation (initialize from Phase 1)
env_phase2 = Cessna172Env(
    task='rotation',
    max_episode_steps=750,  # 15 seconds
    success_criteria={'altitude_min': 10.0}
)
model_phase2 = PPO.load('phase1_model.zip', env=env_phase2)
model_phase2.learn(total_timesteps=300_000)

# ... continue for all phases
```

---

### Option C: Hybrid Approach (BC Warmstart + PPO + Curriculum)

**What:** Combine behavioral cloning with PPO fine-tuning, using curriculum

**How:**
1. **Step 1:** Generate expert demonstrations (GNC controller)
2. **Step 2:** Behavioral cloning (supervised learning) on expert data
3. **Step 3:** PPO fine-tuning with curriculum (as in Option B)

**Pros:**
- ✅ Best of both worlds
- ✅ BC provides strong initialization
- ✅ PPO optimizes beyond imitation
- ✅ Curriculum ensures stable learning

**Cons:**
- ⚠️ Most complex to implement
- ⚠️ Requires both GNC controller AND PPO setup

**Implementation:**
```python
# Step 1: BC Warmstart
bc_dataset = load_expert_demos('gnc_trajectories.pkl')
bc_model = train_behavioral_cloning(bc_dataset, epochs=50)

# Step 2: Convert to PPO policy
ppo_model = PPO('MlpPolicy', env, ...)
ppo_model.policy.load_state_dict(bc_model.state_dict())

# Step 3: PPO fine-tuning with curriculum
for phase in ['ground_roll', 'rotation', 'climb', 'cruise']:
    env_phase = Cessna172Env(task=phase, ...)
    ppo_model.set_env(env_phase)
    ppo_model.learn(total_timesteps=phase_timesteps[phase])
```

---

## Immediate Recommendations

### 1. **Stop Current Training** ✅ (Already Done)
User correctly identified flatlined metrics as a fundamental problem.

### 2. **Choose Path Forward**

**My Recommendation: Option B (Curriculum PPO)**

**Reasoning:**
- No need to implement GNC controller (saves time)
- Proven concept from PDF (MwDAgger curriculum works)
- Reuses existing environment/infrastructure
- PPO can discover better policies than imitation

**Estimated Timeline:**
- Phase 1 (ground roll): 200k steps (~30 min)
- Phase 2 (rotation): 300k steps (~45 min)
- Phase 3 (initial climb): 400k steps (~1 hour)
- Phase 4 (full climb): 500k steps (~1.25 hours)
- Phase 5 (cruise): 600k steps (~1.5 hours)
- **Total: ~5 hours of training**

### 3. **Implementation Plan**

**Week 1: Curriculum Framework**
- [ ] Create phase-based task types in `Cessna172Env`
- [ ] Define success criteria for each phase
- [ ] Implement phase transition logic
- [ ] Create `train_cessna172_curriculum.py`

**Week 2: Phase 1-2 Training**
- [ ] Train ground roll (target: 90% success)
- [ ] Train rotation (target: 90% success)
- [ ] Validate learned behaviors

**Week 3: Phase 3-5 Training**
- [ ] Train initial climb (target: 90% success)
- [ ] Train full climb (target: 90% success)
- [ ] Train cruise (target: 80% success)

**Week 4: Evaluation & Visualization**
- [ ] Full mission evaluation (100 episodes)
- [ ] Compare to POH specs (rotation speed, climb rate, cruise)
- [ ] 3D visualization of learned flight path
- [ ] Write paper/report

---

## Additional Insights from PDF

### Why Ground Physics Matters

**PDF mentions:** "low frequency oscillations" in aircraft states during failed learning.

**Our observation:** Agent stuck at 102 steps - likely oscillating/spinning.

**Root cause:** Our ground physics is TOO SIMPLE:
```python
if altitude < 0.0:
    self.sim.states[0, StateIndex.Z] = 0.0
    if current_w < 0:
        self.sim.states[0, StateIndex.W] = 0.0
```

**Missing:**
- Friction forces (runway-tire interaction)
- Normal forces (ground reaction)
- Rolling resistance
- Gear damping

**Impact:**
- Aircraft can spin on ground unrealistically
- No resistance to lateral motion
- Makes ground roll phase nearly impossible to learn

**Fix Needed:**
```python
def apply_ground_contact_forces(state, dt):
    if altitude < 1.0:  # On or near ground
        # Normal force (prevents sinking)
        F_normal = -F_gravity_z

        # Friction force (opposes lateral motion)
        v_lateral = sqrt(v**2 + w**2)  # Side velocity
        F_friction = -mu_friction * F_normal * (v_lateral / |v_lateral|)

        # Rolling resistance (opposes forward motion)
        F_rolling = -mu_rolling * F_normal * (u / |u|)

        # Apply forces
        state[U] += (F_friction + F_rolling) * dt / mass
```

---

## Comparison Table: All Approaches

| Approach | Sample Efficiency | Training Time | Performance Ceiling | Complexity | Expert Needed? |
|----------|------------------|---------------|-------------------|------------|----------------|
| **Pure PPO (current)** | ❌ Very Low | ❌ Infinite (doesn't learn) | N/A | ✅ Low | ❌ No |
| **Curriculum PPO (Option B)** | ⚠️ Medium | ✅ ~5 hours | 🟢 High (can optimize) | ⚠️ Medium | ❌ No |
| **Imitation (MwDAgger, Option A)** | ✅ High | ✅ ~2-3 hours | ⚠️ Medium (limited to expert) | 🔴 High | ✅ Yes |
| **Hybrid BC+PPO (Option C)** | ✅ Very High | ⚠️ ~6 hours | 🟢 Very High | 🔴 Very High | ✅ Yes |

---

## Lessons Learned

### What We Did Right
1. ✅ GPU-accelerated flight dynamics (77% GPU util)
2. ✅ Realistic Cessna 172 model from POH
3. ✅ Phase-based reward function (good concept)
4. ✅ Comprehensive sanity checks
5. ✅ Fixed critical bugs (airspeed, ground contact)

### What We Did Wrong
1. ❌ Tried to learn full mission from scratch (too complex)
2. ❌ No curriculum learning (essential for complex tasks)
3. ❌ Oversimplified ground physics (friction matters!)
4. ❌ Didn't recognize early warning signs (102 steps stuck)

### Critical Insight from PDF

**Quote:** "Moving Window DAgger (MwDAgger) [...] leverages the sequential nature of the flight task to create a curriculum for the neural network."

**Translation:** Complex sequential tasks (like flight) REQUIRE curriculum learning. You can't learn to land before learning to take off.

**Aviation Analogy:**
- Student pilot training: Ground school → Taxi → Takeoff → Maneuvers → Landing
- Our failed approach: "Here's an airplane, figure out how to do a cross-country flight"
- PDF's solution: "Let's learn ground roll first, then rotation, then climb..."

---

## Next Steps

### Immediate (Today)
1. ✅ Read and analyze PDF (DONE)
2. ✅ Create this analysis document (DONE)
3. ⏭️ **Present options to user and get decision**

### Short Term (This Week)
1. Implement curriculum learning framework
2. Define phase success criteria
3. Create phase-based environment variants
4. Write curriculum training script

### Medium Term (Next 2 Weeks)
1. Train Phase 1-2 (ground roll, rotation)
2. Train Phase 3-4 (climb)
3. Train Phase 5 (cruise)
4. Validate full mission performance

### Long Term (Next Month)
1. Add proper ground physics (friction, gear forces)
2. Implement wind/turbulence
3. Add landing phase
4. 3D visualization with GLTF model
5. Compare to POH specifications
6. Write academic paper/report

---

## References

1. **Paper:** "Imitation Learning for Neural Network Autopilot in Fixed-Wing Unmanned Aerial Systems"
   - Authors: [Check PDF for authors]
   - Key contribution: MwDAgger (Moving Window DAgger) for curriculum learning
   - Result: 180+ second autonomous flights vs 16.375s baseline

2. **Cessna 172 POH** - Performance specifications
3. **Stevens & Lewis** - Aircraft Control and Simulation (stability derivatives)
4. **Schulman et al. (2017)** - Proximal Policy Optimization
5. **Ross et al. (2011)** - DAgger: Dataset Aggregation

---

## Appendix: MwDAgger Algorithm Pseudocode

From the paper:

```python
def MwDAgger(expert, student, time_windows):
    """
    Moving Window DAgger for curriculum learning

    Args:
        expert: GNC controller (provides demonstrations)
        student: Neural network policy
        time_windows: [(0, T1), (0, T2), ..., (0, Tfinal)]
                      e.g., [(0,60), (0,120), (0,180)]
    """
    dataset = []

    for window_start, window_end in time_windows:
        print(f"Training on window [{window_start}, {window_end}]s")

        for episode in range(episodes_per_window):
            # Collect expert trajectory
            expert_traj = expert.fly(duration=window_end)
            dataset.add(expert_traj)

            # Collect student trajectory (with expert interventions)
            student_traj = []
            state = env.reset()

            for t in range(int(window_end / dt)):
                # Student chooses action
                action_student = student.predict(state)

                # Expert provides correct action
                action_expert = expert.predict(state)

                # Execute student action, log expert action
                state, _, _, _ = env.step(action_student)
                student_traj.append((state, action_expert))

                if done:
                    break

            dataset.add(student_traj)

            # Train student on aggregated dataset
            student.train(dataset.sample(batch_size))

    return student
```

**Key Differences from Sequential DAgger:**
- Sequential DAgger: Full trajectories from start
- MwDAgger: **Time-limited windows that expand**
- Result: Prevents data concentration on early flight

---

**Status:** Analysis complete, awaiting user decision on path forward.

**Recommended Action:** Implement **Option B (Curriculum PPO)** starting with Phase 1 (ground roll).

**Date:** December 27, 2024
**Next Update:** After user selects approach
