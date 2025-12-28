# Cessna 172 Visualization Plan

**Date:** December 27, 2024
**Status:** Ready to implement after training completes

---

## 3D Model Files Available

✅ **Cessna172.gltf** (134 KB) - 3D model definition
✅ **Cessna172.bin** (1.1 MB) - Binary mesh data

**Location:** `/home/AIDA/`

---

## Visualization Options

### **Option 1: Interactive 3D Flight Replay (Recommended)**

**What it shows:**
- 3D Cessna 172 model following the actual flight path
- Real-time instrument panel (airspeed, altitude, attitude)
- Control surface positions (elevator, aileron, rudder, throttle)
- Runway and environment
- Multiple camera angles (chase cam, cockpit view, ground view)

**Technology:**
- **Plotly** for interactive 3D plotting
- **GLTF loader** for 3D model
- Web-based (viewable in browser)

**Output:**
- HTML file that can be opened in any browser
- Interactive controls to pause, replay, change camera angles
- Can export as video

### **Option 2: Matplotlib 3D Animation**

**What it shows:**
- Flight trajectory path in 3D space
- Aircraft position markers
- Altitude profile
- Speed/attitude graphs

**Technology:**
- Pure Python matplotlib
- Simpler than Option 1
- Good for quick analysis

**Output:**
- Static images or animated GIF
- Publication-ready plots

### **Option 3: Full Web Viewer (Advanced)**

**What it shows:**
- Everything from Option 1
- Side-by-side comparison of multiple trained policies
- Real-time metrics dashboard
- Replay controls (speed, pause, rewind)

**Technology:**
- Three.js for 3D rendering
- React for UI
- Hosted locally or deployed

---

## What We'll Visualize

### **Trained Policy Flight:**

1. **Takeoff Phase:**
   - Start position on runway
   - Throttle application
   - Acceleration to rotation speed (55 KIAS)
   - Rotation (pitch up ~10°)
   - Liftoff and initial climb

2. **Climb Phase:**
   - Climb from 50 ft → 3000 ft
   - Maintain climb speed (75 KIAS)
   - Pitch angle (~5-7°)
   - Vertical speed (~500-800 fpm)

3. **Cruise Phase:**
   - Level flight at 3000 ft
   - Cruise speed (110 KIAS)
   - Minimal pitch/roll oscillations

### **Data to Display:**

**Flight State:**
- Position (x, y, z)
- Velocity (u, v, w)
- Attitude (roll, pitch, yaw)
- Angular rates (p, q, r)

**Controls:**
- Throttle (0-100%)
- Elevator (-20° to +20°)
- Aileron (-25° to +25°)
- Rudder (-15° to +15°)

**Performance:**
- Airspeed (KIAS)
- Altitude (ft AGL)
- Vertical speed (fpm)
- G-load
- Reward per timestep

**Instruments:**
- Artificial horizon
- Airspeed indicator
- Altimeter
- Vertical speed indicator
- Heading indicator

---

## Implementation Plan

### **Step 1: Extract Flight Data from Trained Model**

Create script: `visualize_cessna172_flight.py`

```python
# Load trained model
model = PPO.load("checkpoints/cessna172/cessna172_ppo_final.zip")

# Run evaluation episode
env = Cessna172Env()
obs, info = env.reset()

flight_data = []
for step in range(3000):  # Max 60 seconds
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Record state
    flight_data.append({
        'time': step * 0.02,
        'position': obs[:3],
        'velocity': obs[3:6],
        'attitude': obs[6:9],
        'rates': obs[9:12],
        'controls': action,
        'altitude': info['altitude'],
        'airspeed': info['airspeed'],
        'reward': reward,
    })

    if terminated or truncated:
        break

# Save for visualization
np.save('flight_trajectory.npy', flight_data)
```

### **Step 2: Load Cessna 172 GLTF Model**

```python
import json

# Load GLTF
with open('Cessna172.gltf', 'r') as f:
    gltf = json.load(f)

# Load binary data
with open('Cessna172.bin', 'rb') as f:
    bin_data = f.read()

# Extract mesh vertices, normals, textures
# (Handled by plotly's GLTF support)
```

### **Step 3: Create Interactive Visualization**

```python
import plotly.graph_objects as go

# Create 3D scene
fig = go.Figure()

# Add runway
fig.add_trace(go.Mesh3d(
    # Runway geometry
))

# Add flight path
positions = np.array([d['position'] for d in flight_data])
fig.add_trace(go.Scatter3d(
    x=positions[:, 0],
    y=positions[:, 1],
    z=-positions[:, 2],  # NED to visualization coords
    mode='lines',
    line=dict(color='blue', width=3)
))

# Add Cessna model at positions along path
# (Animation frames)
frames = []
for i, data in enumerate(flight_data):
    frame = go.Frame(
        data=[go.Mesh3d(
            # Cessna mesh transformed to position/attitude
        )],
        name=str(i)
    )
    frames.append(frame)

fig.frames = frames

# Add playback controls
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'buttons': [
            {'label': 'Play', 'method': 'animate', ...},
            {'label': 'Pause', 'method': 'animate', ...},
        ]
    }]
)

fig.write_html('cessna172_flight.html')
```

### **Step 4: Add Instrument Panel**

Create subplot with:
- Altitude vs time
- Airspeed vs time
- Attitude angles (roll, pitch, yaw)
- Control inputs (throttle, elevator, aileron, rudder)
- Reward over time

---

## Expected Results

### **Successful Training Visualization:**

**Takeoff:**
- Smooth throttle application (0 → 100% in ~5 seconds)
- Acceleration down runway centerline
- Rotation at ~55 KIAS
- Positive climb rate established

**Climb:**
- Steady climb at ~75 KIAS
- Pitch ~5-7°
- Reaching 3000 ft in ~5-7 minutes

**Cruise:**
- Level flight ±50 ft
- Speed 110 KIAS ±5 knots
- Wings level (roll < 5°)

### **Comparison Metrics:**

| Phase | Real Cessna 172 POH | Trained Model | Match? |
|-------|---------------------|---------------|--------|
| Rotation speed | 55 KIAS | ? KIAS | TBD |
| Climb rate | 730 fpm | ? fpm | TBD |
| Climb speed | 75 KIAS | ? KIAS | TBD |
| Cruise speed | 110 KIAS | ? KIAS | TBD |

---

## Timeline

1. **Wait for training to complete** (currently at 350k/1M, ~2 hours remaining)
2. **Extract flight trajectory** (~5 minutes)
3. **Create basic visualization** (~30 minutes)
4. **Add 3D model integration** (~1 hour)
5. **Polish and add instruments** (~1 hour)

**Total: 2.5-3 hours after training completes**

---

## Next Steps

Once training reaches 1,000,000 timesteps:

1. Load final model: `cessna172_ppo_final.zip`
2. Run evaluation episode (deterministic mode)
3. Record full state trajectory
4. Create visualization script
5. Generate interactive HTML viewer

---

**Files Ready:**
- ✅ Cessna172.gltf
- ✅ Cessna172.bin
- ⏳ Training in progress (350k → 1M)

**Ready to visualize once training completes!** 🚀
