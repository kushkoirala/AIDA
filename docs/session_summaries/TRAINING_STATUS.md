# AIDA Training Status

**Date:** December 27, 2024
**Status:** 🟢 Both trainings running simultaneously

---

## Active Training Sessions

### 1. Udaan UAV Training
- **Task ID:** bb5bd0a
- **Status:** 🟢 Running
- **Configuration:**
  - Aircraft: Udaan UAV (13.5 kg)
  - Inertias: Updated CATIA values (Ixx=1.147, Iyy=1.614, Izz=2.709 kg·m²)
  - Task: takeoff_and_fly
  - Training: PPO from scratch (no BC pre-training)
  - Device: CUDA
- **Notes:**
  - Training without BC due to contaminated dataset (underground flight, flips)
  - Will learn through exploration and reward signals
  - Expected to be challenging due to smaller scale and moderate stability

### 2. Cessna 172 Training
- **Task ID:** bcc1402 (RESTARTED - now working!)
- **Status:** 🟢 Running with TRUE parallel execution
- **Configuration:**
  - Aircraft: Cessna 172 Skyhawk (1,043 kg)
  - Task: full_mission (takeoff → climb to 3000 ft → cruise)
  - Parallel environments: 4 (SubprocVecEnv - TRUE parallel)
  - Total timesteps: 1,000,000
  - Learning rate: 3e-4
  - Batch size: 128
  - Device: CUDA
  - Checkpoint dir: checkpoints/cessna172
- **Mission Phases:**
  1. Takeoff: 0 → 50 ft AGL (+50 reward)
  2. Climb: 50 ft → 3000 ft (+75 reward)
  3. Cruise: Maintain 3000 ft, 110 KIAS (+100 reward)
- **Notes:**
  - Highly stable aircraft (Cma=-0.613)
  - Well-documented flight characteristics
  - Expected to learn faster than Udaan
- **Recent Fixes:**
  - ✅ Fixed critical stall logic bug (now only checks stall when airborne)
  - ✅ Using SubprocVecEnv with 4 environments (same as successful run this morning)
  - ✅ 4 worker processes running at 65-67% CPU each
  - ✅ GPU: 77% utilization, only 1.1GB/8GB VRAM used
  - ✅ Room to scale up to 8-12 environments if needed

---

## Monitoring

### Check Training Progress

**Cessna 172:**
```bash
tensorboard --logdir=checkpoints/cessna172/tensorboard
```

**Key Metrics to Watch:**
- `rollout/ep_rew_mean` - Average episode reward (target: +50 to +100)
- `rollout/ep_len_mean` - Average episode length
- `train/entropy_loss` - Exploration vs exploitation
- `train/policy_loss` - Policy improvement
- `train/value_loss` - Value function accuracy

### View Checkpoints

**Cessna 172:**
- Auto-saved every 50k steps: `checkpoints/cessna172/cessna172_ppo_<step>.zip`
- Best model: `checkpoints/cessna172/best_model.zip`
- Final model: `checkpoints/cessna172/cessna172_ppo_final.zip`

**Udaan:**
- Check default checkpoint directory (likely `checkpoints/` or `runs/`)

---

## Expected Timeline

### Cessna 172 (1M timesteps):

| Phase | Timesteps | Expected Behavior | Mean Reward |
|-------|-----------|-------------------|-------------|
| Exploration | 0-100k | Random actions, crashes | -5 to 0 |
| Takeoff Learning | 100k-300k | Discovers acceleration, rotation | 0 to +10 |
| Climb Refinement | 300k-600k | Consistent takeoffs, learning altitude | +10 to +30 |
| Mission Mastery | 600k-1000k | Full mission completion | +30 to +100 |

**Estimated Training Time:** 2-4 hours (depends on GPU)

### Udaan:

Timeline unknown - will depend on how well it learns without BC pre-training. Expect slower progress due to:
- Smaller scale → faster dynamics
- Less stable → requires more precise control
- No curriculum from BC data

---

## GPU Usage

Both trainings are running on the same GPU (CUDA). Monitor GPU utilization:

**Windows Task Manager:**
- Performance tab → GPU
- Should see consistent 80-100% utilization

**If GPU memory issues occur:**
- Reduce parallel environments for Cessna: `--n-envs 2`
- Or stop one training temporarily

---

## Next Actions

### While Training:

1. ✅ **Monitor GPU activity** - Ensure consistent utilization
2. **Check TensorBoard** - View learning curves every 30-60 min
3. **Inspect checkpoints** - Test policies at 250k, 500k, 750k steps
4. **Compare progress** - Which aircraft learns faster?

### When Training Completes:

**Cessna 172:**
1. Visualize trained policy
2. Test in different conditions (different cruise altitudes, wind)
3. Analyze flight trajectories
4. Compare to real Cessna 172 POH performance

**Udaan:**
1. Evaluate final policy quality
2. Compare to BC-pretrained approach (if we fix ground physics later)
3. Analyze what worked / what didn't

---

## Comparison: Cessna vs Udaan

| Factor | Udaan | Cessna 172 | Prediction |
|--------|-------|------------|------------|
| **Stability** | Moderate | High | Cessna learns faster |
| **Scale** | Small (13.5 kg) | Large (1043 kg) | Cessna more realistic |
| **Documentation** | Limited | Extensive | Cessna easier to validate |
| **Complexity** | Simple UAV | Full GA aircraft | Cessna more challenging |
| **Training Data** | None (from scratch) | None (from scratch) | Equal footing |
| **Expected Success** | Moderate | High | Cessna should excel |

---

## Current Status Summary

```
🟢 Udaan Training:     RUNNING (Task bb5bd0a)
🟢 Cessna 172 Training: RUNNING (Task bc6db7e)
🟢 GPU:                 Active
⏱️ Expected completion: 2-4 hours
📊 Monitoring:          TensorBoard available
💾 Checkpoints:         Auto-saving every 50k steps
```

---

## Troubleshooting

### If training stops:

**Check task status:**
```bash
# In WSL
ps aux | grep python
```

**View output:**
- Udaan: Check task output with TaskOutput tool
- Cessna: `C:\Users\ADMINI~1\AppData\Local\Temp\claude\c--Users-Administrator-Desktop\tasks\bc6db7e.output`

### If GPU runs out of memory:

**Stop Cessna temporarily:**
- Kill task bc6db7e
- Let Udaan finish first
- Then restart Cessna

### If learning plateaus:

**For Cessna:**
- Adjust hyperparameters (increase entropy, reduce learning rate)
- Train on specific phase first (--task takeoff)
- Check reward function balance

**For Udaan:**
- May need to add ground physics to simulator
- Consider BC pre-training with fixed dataset
- Adjust reward shaping for smaller scale

---

**Last Updated:** December 27, 2024
**Both trainings initiated successfully!** 🚀

Check back in 1-2 hours to see initial progress.
