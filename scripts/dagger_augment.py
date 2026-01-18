#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import torch
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-flight-dynamics" / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from flight_dynamics import FlightSimulator, StateIndex
from triangle_controller import TriangleInterceptController, XCPhase
from generate_waypoint_demos import WaypointDemoGenerator
from train_waypoint_bc import WaypointPilotNN

def load_bc_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WaypointPilotNN(checkpoint["input_dim"], checkpoint["action_dim"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm_stats = {
        "obs_mean": checkpoint["obs_mean"], "obs_std": checkpoint["obs_std"],
        "wp_xy_mean": checkpoint["wp_xy_mean"], "wp_xy_std": checkpoint["wp_xy_std"],
        "wp_spd_mean": checkpoint["wp_spd_mean"], "wp_spd_std": checkpoint["wp_spd_std"],
        "wp_dist_mean": checkpoint["wp_dist_mean"], "wp_dist_std": checkpoint["wp_dist_std"],
    }
    # Add bearing normalization if present (7D waypoint format)
    if "wp_bearing_mean" in checkpoint:
        norm_stats["wp_bearing_mean"] = checkpoint["wp_bearing_mean"]
        norm_stats["wp_bearing_std"] = checkpoint["wp_bearing_std"]
        norm_stats["wp_dim"] = 7
    else:
        norm_stats["wp_dim"] = 6
    return model, norm_stats

def normalize_obs(state, waypoint, ns):
    state_norm = (state - ns["obs_mean"]) / ns["obs_std"]
    wp_dim = ns.get("wp_dim", 6)
    wp_norm = np.zeros(wp_dim, dtype=np.float32)
    wp_norm[0] = (waypoint[0] - float(ns["wp_xy_mean"][0])) / float(ns["wp_xy_std"][0])
    wp_norm[1] = (waypoint[1] - float(ns["wp_xy_mean"][1])) / float(ns["wp_xy_std"][1])
    wp_norm[2] = (waypoint[2] - float(ns["wp_xy_mean"][2])) / float(ns["wp_xy_std"][2])
    wp_norm[3] = (waypoint[3] - float(ns["wp_spd_mean"])) / float(ns["wp_spd_std"])
    wp_norm[4] = waypoint[4]
    wp_norm[5] = (waypoint[5] - float(ns["wp_dist_mean"])) / float(ns["wp_dist_std"])
    if wp_dim >= 7 and "wp_bearing_mean" in ns:
        wp_norm[6] = (waypoint[6] - float(ns["wp_bearing_mean"])) / float(ns["wp_bearing_std"])
    return np.concatenate([state_norm, wp_norm])

def collect_dagger_data(bc_model, norm_stats, num_episodes=10, mix_ratio=0.5):
    print("Collecting DAgger data: %d episodes" % num_episodes)
    sim = FlightSimulator(n_instances=1, dt=0.02, use_gpu=False)
    ctrl = TriangleInterceptController()
    wp_gen = WaypointDemoGenerator(ctrl)
    all_obs, all_act, all_wp, all_ph = [], [], [], []
    
    for ep in range(num_episodes):
        init = np.zeros((1, 12), dtype=np.float32)
        init[0, 0] = -400.0 + np.random.uniform(-50, 50)
        init[0, 3] = 5.0 + np.random.uniform(-1, 1)
        sim.reset(init)
        ctrl.reset()
        ep_obs, ep_act, ep_wp, ep_ph = [], [], [], []
        sim_time = 0.0
        
        for step in range(30000):
            state = sim.get_states()[0]
            expert_act = ctrl.compute_action(state, sim_time)
            waypoint = wp_gen.get_current_waypoint(ctrl.phase, state)
            obs_norm = normalize_obs(state, waypoint, norm_stats)
            with torch.no_grad():
                bc_act = bc_model(torch.FloatTensor(obs_norm).unsqueeze(0)).numpy()[0]
            
            ep_obs.append(state.copy())
            ep_act.append(expert_act[:4].copy())
            ep_wp.append(waypoint.copy())
            ep_ph.append(ctrl.phase.value)
            
            if np.random.random() < mix_ratio:
                action = expert_act
            else:
                action = np.zeros(7, dtype=np.float32)
                action[0] = np.clip(bc_act[0], 0.0, 1.0)
                action[1:4] = np.clip(bc_act[1:4], -1.0, 1.0)
                action[4:6] = expert_act[4:6]
            
            sim.set_controls(action.reshape(1, -1))
            sim.step()
            sim_time += 0.02
            
            if -state[StateIndex.Z] < -10 or ctrl.phase == XCPhase.LANDED:
                break
        
        all_obs.extend(ep_obs)
        all_act.extend(ep_act)
        all_wp.extend(ep_wp)
        all_ph.extend(ep_ph)
        status = "LANDED" if ctrl.phase == XCPhase.LANDED else ctrl.phase.name
        print("  Episode %d/%d: %d steps, %s" % (ep+1, num_episodes, len(ep_obs), status))
    
    return np.array(all_obs), np.array(all_act), np.array(all_wp), np.array(all_ph, dtype=np.int32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-checkpoint", default="checkpoints/waypoint_pilot/waypoint_pilot_best.pt")
    parser.add_argument("--original-data", default="bc_data/waypoint_demos_augmented.npz")
    parser.add_argument("--output", default="bc_data/waypoint_demos_dagger.npz")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--mix-ratio", type=float, default=0.3)
    args = parser.parse_args()
    
    print("=" * 70)
    print("  DAgger DATA AUGMENTATION")
    print("=" * 70)
    
    bc_model, norm_stats = load_bc_model(args.bc_checkpoint)
    dag_data = collect_dagger_data(bc_model, norm_stats, args.episodes, args.mix_ratio)
    
    print("Loading original: %s" % args.original_data)
    orig = np.load(args.original_data)
    print("Original: %d, DAgger: %d" % (len(orig["observations"]), len(dag_data[0])))
    
    combined = {
        "observations": np.concatenate([orig["observations"], dag_data[0]]),
        "actions": np.concatenate([orig["actions"], dag_data[1]]),
        "waypoints": np.concatenate([orig["waypoints"], dag_data[2]]),
        "phases": np.concatenate([orig["phases"], dag_data[3]]),
    }
    print("Combined: %d samples" % len(combined["observations"]))
    
    np.savez_compressed(args.output, **combined)
    print("Saved to: %s" % args.output)

if __name__ == "__main__":
    main()
