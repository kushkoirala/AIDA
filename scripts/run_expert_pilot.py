#!/usr/bin/env python3
"""
Run the expert triangle controller (used to generate BC training data) with telemetry viewer.
This lets us verify the expert behavior that the NN is trying to imitate.
"""
import sys
import time
import asyncio
import json
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpu-flight-dynamics' / 'python'))

from flight_dynamics import FlightSimulator, StateIndex
from triangle_controller import TriangleInterceptController, XCPhase

# Try importing websockets
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print('Warning: websockets not available, telemetry server disabled')

# Constants
M_TO_FT = 3.28084
FT_TO_M = 0.3048
MPS_TO_KTS = 1.94384

# Shared state for telemetry
shared_state = {
    'position': [0, 0, 0],
    'quaternion': [1, 0, 0, 0],
    'velocity': [0, 0, 0],
    'rates': [0, 0, 0],
    'surfaces': [0, 0, 0],
    'throttle': 0.0,
    'flaps': 0.0,
    'spoilers': 0.0,
    'soc': 1.0,
    'voltage': 12.0,
    'mode': 'GROUND_ROLL',
    'waypoints': [],
    'current_waypoint': 0,
}
state_lock = threading.Lock()

async def telemetry_handler(websocket):
    """WebSocket handler for telemetry data."""
    try:
        while True:
            with state_lock:
                data = json.dumps(shared_state)
            await websocket.send(data)
            await asyncio.sleep(0.05)  # 20Hz update rate
    except websockets.exceptions.ConnectionClosed:
        pass

async def start_telemetry_server():
    """Start the WebSocket telemetry server."""
    if not WEBSOCKETS_AVAILABLE:
        return
    print('[telemetry_server] Starting on 0.0.0.0:8765')
    async with websockets.serve(telemetry_handler, '0.0.0.0', 8765):
        print('[telemetry_server] Running...')
        await asyncio.Future()  # Run forever

def run_telemetry_server():
    """Run telemetry server in a thread."""
    if not WEBSOCKETS_AVAILABLE:
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_telemetry_server())

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)
    return [
        cr*cp*cy + sr*sp*sy,  # w
        sr*cp*cy - cr*sp*sy,  # x
        cr*sp*cy + sr*cp*sy,  # y
        cr*cp*sy - sr*sp*cy   # z
    ]

def update_telemetry(sim, controller, controls):
    """Update shared telemetry state from simulator."""
    state = sim.get_states()[0]
    
    # Position (convert to viewer coordinates: x=east, y=up, z=north)
    x = state[StateIndex.X]
    y = state[StateIndex.Y]  
    z = -state[StateIndex.Z]  # Z is down in sim, up in viewer
    
    # Attitude
    roll = state[StateIndex.PHI]
    pitch = state[StateIndex.THETA]
    yaw = state[StateIndex.PSI]
    
    # Body velocities (U, V, W)
    u = state[StateIndex.U]  # forward
    v = state[StateIndex.V]  # right
    w = state[StateIndex.W]  # down
    
    # Angular rates
    p = state[StateIndex.P]
    q = state[StateIndex.Q]
    r = state[StateIndex.R]
    
    with state_lock:
        shared_state['position'] = [float(x), float(z), float(y)]
        shared_state['quaternion'] = euler_to_quaternion(roll, pitch, yaw)
        shared_state['velocity'] = [float(u), float(-w), float(v)]
        shared_state['rates'] = [float(p), float(q), float(r)]
        shared_state['surfaces'] = [float(controls[1]), float(controls[2]), float(controls[3])]  # elev, ail, rud
        shared_state['throttle'] = float(controls[0])
        shared_state['mode'] = controller.phase.name

def main():
    print('=' * 70)
    print('  EXPERT PILOT (Triangle Controller) with Telemetry')
    print('=' * 70)
    print('Departure: Lake Waltanna (SN65) RWY 35')
    print('Arrival:   Hutchinson Regional (KHUT) RWY 31')
    print('=' * 70)
    
    # Start telemetry server
    if WEBSOCKETS_AVAILABLE:
        telemetry_thread = threading.Thread(target=run_telemetry_server, daemon=True)
        telemetry_thread.start()
        print()
        print('=' * 70)
        print('  TELEMETRY SERVER RUNNING')
        print('=' * 70)
        print('Open viewer at: http://localhost:8080')
        print('Press Ctrl+C to stop')
        print('=' * 70)
        print()
    
    # Create simulator and controller
    sim = FlightSimulator(n_instances=1, dt=0.02, use_gpu=False)
    controller = TriangleInterceptController(
        cruise_altitude_ft=5500.0,
        pattern_altitude_ft=1500.0
    )
    
    # Initialize at SN65 runway
    initial_state = np.zeros((1, 12), dtype=np.float32)
    initial_state[0, 0] = -400.0  # X position (runway)
    initial_state[0, 3] = 5.0     # U (taxi speed)
    sim.reset(initial_state)
    
    dt = 0.02
    sim_time = 0.0
    last_print = -10.0
    last_phase = None
    
    print('Starting expert flight...')
    print()
    
    try:
        while sim_time < 1200.0:  # 20 minute max
            # Get current state
            state = sim.get_states()[0]
            
            # Get expert controls
            controls = controller.compute_action(state, dt)
            
            # Apply controls
            sim.set_controls(np.array([controls]))
            sim.step()
            
            # Update telemetry
            update_telemetry(sim, controller, controls)
            
            # Print status every 10 seconds or on phase change
            if sim_time - last_print >= 10.0 or controller.phase != last_phase:
                alt_m = -state[StateIndex.Z]
                alt_ft = alt_m * M_TO_FT
                speed_mps = np.sqrt(state[StateIndex.U]**2 + state[StateIndex.V]**2)
                speed_kts = speed_mps * MPS_TO_KTS
                heading = np.rad2deg(state[StateIndex.PSI]) % 360
                
                phase_name = controller.phase.name.ljust(18)
                print(f'[{sim_time:6.1f}s] {phase_name} | '
                      f'Alt: {alt_m:5.0f}m ({alt_ft:5.0f}ft) | '
                      f'Spd: {speed_mps:5.1f}m/s ({speed_kts:5.1f}kts) | '
                      f'Hdg: {heading:5.1f} | '
                      f'Thr:{controls[0]:.2f} Elv:{controls[1]:.2f}')
                
                last_print = sim_time
                last_phase = controller.phase
            
            # Check for landing
            if controller.phase == XCPhase.LANDED:
                print()
                print('=' * 70)
                print('  LANDED!')
                print('=' * 70)
                break
            
            sim_time += dt
            time.sleep(dt * 0.1)  # Run 10x faster than real-time
            
    except KeyboardInterrupt:
        print()
        print('Stopped by user')

if __name__ == '__main__':
    main()
