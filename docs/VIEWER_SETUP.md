# AIDA 3D Telemetry Viewer

Real-time 3D visualization of autonomous flight with WebSocket telemetry streaming.

## Features

### Flight Instruments (January 2026)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VIEWER UI LAYOUT                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐                      ┌─────────────────────┐  │
│  │ CONTROLS        │                      │ PILOT INFO          │  │
│  │ • Follow        │                      │ ┌─────────────────┐ │  │
│  │ • Pause         │                      │ │ PILOT: EXPERT   │ │  │
│  │ • Reset View    │                      │ │ SIM:   10x      │ │  │
│  │ • Sound         │                      │ └─────────────────┘ │  │
│  │ • Callouts      │                      └─────────────────────┘  │
│  └─────────────────┘                                               │
│                                                                     │
│                    ┌─────────────────────────┐                      │
│                    │    COMPASS + HEADING    │                      │
│                    │       ◄── 314° ──►      │                      │
│                    └─────────────────────────┘                      │
│                                                                     │
│   ┌───────┐     ┌───────────────────────────┐     ┌───────┐        │
│   │ SPD   │     │                           │     │ ALT   │        │
│   │ 110   │     │    ATTITUDE INDICATOR     │     │ 5500  │        │
│   │ KTS   │     │    (Pitch + Roll + VS)    │     │ FT    │        │
│   │       │     │                           │     │       │        │
│   │ AOA   │     │          ═══●═══          │     │       │        │
│   │ 2.5   │     │                           │     │       │        │
│   │ DEG   │     └───────────────────────────┘     └───────┘        │
│   └───────┘                                                         │
│                                                                     │
│                    ┌─────────────────────────┐                      │
│                    │  PHASE: CRUISE_TO_TP    │                      │
│                    └─────────────────────────┘                      │
│                                                                     │
│              ┌─────────────────────────────────────┐                │
│              │          3D AIRCRAFT VIEW           │                │
│              │         (Three.js WebGL)            │                │
│              └─────────────────────────────────────┘                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### UI Elements

| Element | Location | Description |
|---------|----------|-------------|
| **Compass** | Top center | Heading with curved arc display |
| **Attitude Indicator** | Center | Pitch, roll, vertical speed |
| **Speed Tape** | Left of attitude | Airspeed (KTS) and AOA (DEG) |
| **Altitude Tape** | Right of attitude | Altitude (FT) |
| **Pilot Info** | Top right | Current controller (EXPERT/AI+EXPERT) |
| **Sim Speed** | Top right | Simulation speed multiplier |
| **Phase Display** | Below attitude | Current flight phase |
| **Controls** | Top left | Follow, pause, sound, etc. |

### Pilot Mode Indicator

Shows which controller is flying the aircraft:

| Mode | Color | Description |
|------|-------|-------------|
| **EXPERT** | Green | Pure classical controller |
| **AI+EXPERT** | Blue | Residual RL (NN + Expert) |
| **HYBRID** | Yellow | V1 model (4 controls) |

---

## Quick Start

### Local Viewing (Windows/WSL)

**Terminal 1: Start HTTP Server**
```bash
cd /home/AIDA/viewer/public
python -m http.server 8000 --bind 0.0.0.0
```

**Terminal 2: Start Flight Simulation**
```bash
cd /home/AIDA
source .venv-linux/bin/activate

# Pure expert flight
python scripts/run_residual_telemetry.py --pure-expert

# Or with trained model
python scripts/run_residual_telemetry_v2.py --model checkpoints/residual_ppo_v2/best_model.zip
```

**Browser:** http://localhost:8000

### Remote Viewing (Other LAN Device)

1. Get Windows/WSL IP:
   ```bash
   # Windows
   ipconfig | findstr "IPv4"

   # WSL
   hostname -I
   ```

2. Open browser on remote device:
   ```
   http://<IP>:8000/?wshost=<IP>
   ```
   Example: `http://10.0.1.208:8000/?wshost=10.0.1.208`

### Tailscale Remote Access

For access from anywhere via Tailscale:
```
http://100.79.9.48:8000/?wshost=100.79.9.48
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TELEMETRY ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐                ┌─────────────────────┐    │
│  │   Flight Simulator  │                │   Browser Viewer    │    │
│  │   (Python/NumPy)    │                │   (Three.js)        │    │
│  ├─────────────────────┤                ├─────────────────────┤    │
│  │ • 6-DOF dynamics    │  WebSocket    │ • 3D rendering      │    │
│  │ • Expert controller │ ───────────▶  │ • Instruments       │    │
│  │ • NN policy (opt)   │   (8765)      │ • Audio synthesis   │    │
│  │ • Telemetry server  │    20 Hz      │ • GPWS callouts     │    │
│  └─────────────────────┘                └─────────────────────┘    │
│                                                  ▲                  │
│                                    HTTP (8000)   │                  │
│                                 ──────────────────                  │
│                                 (viewer files, 3D model)            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Telemetry Data Format

The WebSocket sends JSON at 20 Hz:

```json
{
  "position": [52800, -21300, -1676],
  "quaternion": [0.99, 0.0, 0.05, 0.0],
  "velocity": [55, 0, 0],
  "rates": [0, 0, 0],
  "surfaces": [0.0, 0.0, 0.0],
  "throttle": 0.65,
  "flaps": 0.0,
  "spoilers": 0.0,
  "soc": 1.0,
  "voltage": 12.0,
  "load_factor": 1.0,
  "units": "metric",
  "model": "cessna172",
  "phase": "CRUISE_TO_TP",
  "aoa_deg": 2.5,
  "pilot_mode": "EXPERT",
  "sim_speed": 10.0
}
```

---

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| **8000** | HTTP | Serves viewer HTML/JS/3D model |
| **8765** | WebSocket | Real-time telemetry stream |
| **6006/6007** | HTTP | TensorBoard (optional) |

---

## Viewer Controls

| Control | Function |
|---------|----------|
| **Follow** | Camera follows aircraft |
| **Pause** | Pause/resume display (sim continues) |
| **Reset View** | Reset camera to default |
| **Sound** | Toggle engine audio synthesis |
| **Callouts** | Toggle GPWS-style altitude callouts |
| **Jump to Phase** | Skip to specific flight phase |

---

## Audio Features

### Engine Sound Synthesis
- Web Audio API oscillators
- Frequency varies with throttle position
- Spatial audio based on aircraft position

### Altitude Callouts (GPWS-style)
- "Five hundred" at 500 ft AGL
- "Minimums" at 200 ft AGL
- "Fifty, forty, thirty, twenty, ten" during flare

---

## URL Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wshost` | 127.0.0.1 | WebSocket server IP |
| `wsport` | 8765 | WebSocket server port |

Example: `http://localhost:8000/?wshost=10.0.1.208&wsport=8765`

---

## Troubleshooting

### "Waiting for telemetry..."

1. Check WebSocket server is running:
   ```bash
   ss -tlnp | grep 8765
   # or
   netstat -an | grep 8765
   ```

2. Verify correct `wshost` parameter in URL

3. Check Windows Firewall allows port 8765:
   ```powershell
   # Add firewall rule
   netsh advfirewall firewall add rule name="AIDA WebSocket" dir=in action=allow protocol=TCP localport=8765
   ```

### Page loads but no 3D model

1. Check HTTP server is running:
   ```bash
   ss -tlnp | grep 8000
   ```

2. Verify `Cessna172.gltf` exists:
   ```bash
   ls -la /home/AIDA/viewer/public/Cessna172.gltf
   ```

3. Check browser console (F12) for errors

### Remote device can't connect

1. Enable WSL2 mirrored networking in `%USERPROFILE%\.wslconfig`:
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

2. Restart WSL:
   ```cmd
   wsl --shutdown
   ```

3. Add Windows Firewall rules for ports 8000 and 8765

### Instruments not updating

1. Check browser console for WebSocket errors
2. Verify telemetry script is sending data:
   ```bash
   # Should see periodic output
   python scripts/run_residual_telemetry.py --pure-expert
   ```

---

## Files

| File | Purpose |
|------|---------|
| `viewer/public/index.html` | Main viewer (HTML + CSS + JS) |
| `viewer/public/Cessna172.gltf` | 3D aircraft model |
| `aida_sim/io/telemetry.py` | WebSocket telemetry server |
| `scripts/run_residual_telemetry.py` | Flight + telemetry runner |
| `scripts/run_residual_telemetry_v2.py` | V2 model runner |

---

## Customization

### Changing Simulation Speed

Edit in `run_residual_telemetry.py`:
```python
sim_speed = 10.0  # 10x faster than real-time
```

### Adding New Instruments

Edit `viewer/public/index.html`:
1. Add HTML element in the appropriate container
2. Add CSS styling
3. Update JavaScript to read from telemetry data

Example - adding a new gauge:
```html
<div id="my-gauge">
  <div class="gauge-label">FUEL</div>
  <div class="gauge-value" id="fuel-value">100</div>
  <div class="gauge-unit">%</div>
</div>
```

```javascript
// In updateInstruments()
document.getElementById('fuel-value').textContent = data.fuel_percent.toFixed(0);
```

---

**Last Updated:** January 10, 2026
