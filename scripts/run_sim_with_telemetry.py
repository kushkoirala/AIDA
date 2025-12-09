
import asyncio
import threading
import time
import numpy as np
import gymnasium as gym
import aida_sim.env  # Register env

from aida_sim.io.telemetry import telemetry_server

# Shared state for telemetry
shared_state = {
    "position": [0, 0, 0],
    "quaternion": [1, 0, 0, 0],
    "velocity": [0, 0, 0],
    "rates": [0, 0, 0],
    "surfaces": [0, 0, 0],
    "throttle": 0.0,
    "soc": 1.0,
    "voltage": 12.0,
    "load_factor": 1.0,
}

def physics_loop():
    """Runs the simulation physics in a separate thread."""
    env = gym.make("AIDA-Flight-v0")
    obs, info = env.reset()
    
    print("Physics loop started.")
    
    # Simple trim action (keep level-ish)
    # Action space is likely [elevator, aileron, rudder, throttle]
    # Let's check action space size or assume 4.
    # Based on flight_env.py reading earlier, it didn't explicitly show action space, 
    # but standard fixed wing is usually 4 channels.
    
    while True:
        # For now, just zero inputs to see natural dynamics
        # or random small noise to see movement
        action = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32) # 50% throttle
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Access the underlying state directly for telemetry
        # env.unwrapped gives us the FlightEnv instance
        sim_state = env.unwrapped.state
        
        # Update shared state
        # We convert numpy arrays to lists for JSON serialization
        shared_state["position"] = sim_state.position.tolist()
        shared_state["quaternion"] = sim_state.orientation.tolist()
        shared_state["velocity"] = sim_state.velocity.tolist()
        shared_state["rates"] = sim_state.body_rates.tolist()
        shared_state["surfaces"] = sim_state.surfaces.tolist()
        shared_state["throttle"] = float(action[3]) if len(action) > 3 else 0.0
        shared_state["soc"] = float(sim_state.soc)
        shared_state["voltage"] = float(sim_state.voltage)
        shared_state["load_factor"] = float(sim_state.load_factor)
        
        if terminated or truncated:
            print(f"Resetting env. Reason: {info.get('termination_reason', 'unknown')}")
            env.reset()
            
        # Real-time pacing
        time.sleep(env.unwrapped.dt)

def get_state():
    return shared_state

async def main():
    # Start physics thread
    t = threading.Thread(target=physics_loop, daemon=True)
    t.start()
    
    # Run telemetry server
    print("Starting telemetry server on ws://0.0.0.0:8765")
    await telemetry_server(get_state)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping...")
