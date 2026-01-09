#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from classical_mission_controller import MissionController, MissionPhase
from aida_sim.env.flight_env_cessna172 import Cessna172Env

def run_traffic_pattern():
    sep = "=" * 70
    print(sep)
    print("CESSNA 172 TRAFFIC PATTERN MISSION")
    print(sep)
    print("Flight Area: 10km (N-S) x 5km (E-W)")
    print("Cruise Altitude: 5000 ft (1524m)")
    print("Traffic Pattern: RIGHT-HAND")
    print(sep)

    env = Cessna172Env(
        dt=0.02,
        max_episode_steps=60000,
        task="full_mission",
        cruise_altitude_ft=5000.0
    )
    
    obs, info = env.reset()
    controller = MissionController(
        cruise_altitude_ft=5000.0, 
        pattern_altitude_ft=1000.0,
        runway_start_x=-500.0, 
        flight_area_length=10000.0, 
        flight_area_width=5000.0
    )

    max_time = 1200.0
    history = {"time": [], "x": [], "y": [], "altitude": [], "heading": [], "phase": []}
    sim_time = 0.0
    step = 0
    last_phase = None
    phase_times = {}

    print("\nStarting simulation...")
    print("-" * 70)

    while sim_time < max_time:
        action_6 = controller.compute_action(obs, sim_time)
        action = action_6
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_phase = controller.phase
        if current_phase != last_phase:
            phase_times[current_phase.name] = sim_time
            altitude = -obs[2]
            airspeed = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)
            heading_deg = np.rad2deg(obs[8]) % 360
            print("[%6.1fs] %-20s | Alt: %6.1fm (%5.0fft) | Spd: %5.1fm/s | Hdg: %5.1f | X: %7.1f Y: %7.1f" % (
                sim_time, current_phase.name, altitude, altitude*3.28084, airspeed, heading_deg, obs[0], obs[1]))
            last_phase = current_phase

        if step % 25 == 0:
            history["time"].append(sim_time)
            history["x"].append(obs[0])
            history["y"].append(obs[1])
            history["altitude"].append(-obs[2])
            history["heading"].append(np.rad2deg(obs[8]) % 360)
            history["phase"].append(current_phase.value)

        if current_phase == MissionPhase.LANDED:
            print("\n" + sep)
            print("MISSION COMPLETE! Total time: %.1fs (%.1f min)" % (sim_time, sim_time/60))
            print("Final: X=%.1fm, Y=%.1fm, Alt=%.1fm" % (obs[0], obs[1], -obs[2]))
            break

        if done:
            term_reason = info.get("termination_reason", "unknown")
            print("\nEnvironment terminated at %.1fs - Reason: %s" % (sim_time, term_reason))
            altitude = -obs[2]
            print("Position: X=%.1fm, Y=%.1fm, Alt=%.1fm" % (obs[0], obs[1], altitude))
            break

        sim_time += 0.02
        step += 1

    if sim_time >= max_time:
        print("\nTimeout after %.0fs" % max_time)

    print("\n" + sep)
    print("FLIGHT PHASE SUMMARY")
    for name, t in phase_times.items():
        print("  %-25s: %6.1fs" % (name, t))

    np.savez("/home/AIDA/scripts/traffic_pattern_data.npz", **{k: np.array(v) for k, v in history.items()})
    print("\nData saved to traffic_pattern_data.npz")
    
    env.close()
    return history

if __name__ == "__main__":
    run_traffic_pattern()
