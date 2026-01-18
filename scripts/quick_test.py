#!/usr/bin/env python3
"""Quick test to run simulation faster and verify all phases."""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpu-flight-dynamics' / 'python'))

from scripts.run_trajectory_flight import create_sn65_khut_trajectory
from flight_dynamics import FlightSimulator as FlightDynamics, StateIndex
from aida_sim.control.autopilot import TrajectoryAutopilot

M_TO_FT = 3.28084
FT_TO_M = 0.3048
NM_TO_FT = 6076.12

def main():
    # Create trajectory
    trajectory = create_sn65_khut_trajectory()

    # Initialize simulation
    sim = FlightDynamics(n_instances=1, dt=0.02, use_gpu=False)
    state = sim.get_states()[0]
    state[StateIndex.X] = 0.0
    state[StateIndex.Y] = 0.0
    state[StateIndex.Z] = -1448.0 * FT_TO_M
    state[StateIndex.U] = 50.0
    state[StateIndex.PSI] = np.deg2rad(4.0)
    sim.states[0] = state

    # Initialize autopilot
    autopilot = TrajectoryAutopilot()
    autopilot.set_trajectory(trajectory)

    # Run for 60 simulated minutes
    dt = 0.02
    sim_time = 0.0
    last_phase = None

    print("Running fast simulation for 60 simulated minutes...", flush=True)
    steps = int(60 * 60 / dt)  # 60 minutes
    for step in range(steps):
        state = sim.get_states()[0]
        action = autopilot.compute_action(state, sim_time)
        sim.set_controls(action.reshape(1, -1))
        sim.step()
        sim_time += dt

        # Print on segment change or every 5 minutes
        changed_phase = autopilot.current_segment != last_phase
        every_5min = step % int(300 / dt) == 0
        if changed_phase or every_5min:
            alt = -state[StateIndex.Z] * M_TO_FT
            hdg = np.rad2deg(state[StateIndex.PSI]) % 360
            dist = autopilot.distance_to_destination / NM_TO_FT
            print(f"T+{sim_time/60:5.1f}min | Alt: {alt:5.0f}ft | Hdg: {hdg:03.0f} | "
                  f"Dist: {dist:5.1f}nm | Seg: {autopilot.current_segment.name}", flush=True)
            last_phase = autopilot.current_segment

        if autopilot.has_landed:
            print("LANDED!", flush=True)
            break

    state = sim.get_states()[0]
    print(f"Final: T+{sim_time/60:.1f}min, Alt: {-state[StateIndex.Z]*M_TO_FT:.0f}ft, "
          f"Dist: {autopilot.distance_to_destination/NM_TO_FT:.1f}nm", flush=True)

if __name__ == "__main__":
    main()
