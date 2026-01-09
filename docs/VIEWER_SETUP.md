# AIDA 3D Telemetry Viewer Setup

## Quick Start

### Local Viewing (Windows/WSL)
1. Start the flight simulation:
   ```bash
   cd /home/AIDA
   source .venv-linux/bin/activate
   python scripts/run_xc_sn65_khut.py
   ```

2. Start HTTP server (if not already running):
   ```bash
   cd /home/AIDA/viewer/public
   python -m http.server 8080 --bind 0.0.0.0
   ```

3. Open browser:
   - http://127.0.0.1:8080/?wshost=127.0.0.1

### Remote Viewing (MacBook or other LAN device)
1. Get Windows LAN IP:
   ```cmd
   ipconfig | findstr "IPv4"
   ```

2. Open browser on remote device:
   - http://<LAN_IP>:8080/?wshost=<LAN_IP>
   - Example: http://10.0.1.208:8080/?wshost=10.0.1.208

## Architecture

```
┌─────────────────┐     WebSocket (8765)     ┌─────────────────┐
│ Flight Sim      │ ────────────────────────▶│ Browser Viewer  │
│ (Python/NumPy)  │                          │ (THREE.js)      │
└─────────────────┘                          └─────────────────┘
                                                    ▲
                          HTTP (8080)               │
                    ────────────────────────────────┘
                    (serves index.html, Cessna 3D model)
```

## Ports
- **8080**: HTTP server (serves viewer files)
- **8765**: WebSocket telemetry server (real-time flight data)

## Viewer Controls
- **Follow**: Toggle camera follow mode
- **Pause**: Pause/resume simulation display
- **Reset View**: Reset camera to default position
- **Sound**: Toggle engine sound
- **Callouts**: Toggle phase callouts

## URL Parameters
- `wshost`: WebSocket host IP (default: 127.0.0.1)
  - Example: `?wshost=10.0.1.208` for remote viewing

## Troubleshooting

### "Waiting for telemetry"
1. Check WebSocket server is running:
   ```bash
   ss -tlnp | grep 8765
   ```
2. Verify correct wshost parameter in URL
3. Check Windows Firewall allows port 8765

### Page loads but no 3D model
1. Check HTTP server is running:
   ```bash
   ss -tlnp | grep 8080
   ```
2. Verify Cessna172.gltf exists in viewer/public/

### Remote device can't connect
1. Ensure WSL2 mirrored networking is enabled in `%USERPROFILE%\.wslconfig`:
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```
2. Restart WSL after changing: `wsl --shutdown`
3. Check Windows Firewall allows ports 8080 and 8765

## Files
- `/home/AIDA/viewer/public/index.html` - Main viewer
- `/home/AIDA/viewer/public/Cessna172.gltf` - 3D aircraft model
- `/home/AIDA/scripts/run_xc_sn65_khut.py` - Cross-country flight script
- `/home/AIDA/scripts/cross_country_controller.py` - Flight controller
