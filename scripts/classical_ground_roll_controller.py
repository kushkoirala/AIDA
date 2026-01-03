#!/usr/bin/env python3
"""
Classical GNC Controller for Cessna 172 Ground Roll

Uses simple control laws based on physics:
- Full throttle for acceleration
- Elevator neutral until near rotation speed, then gentle pitch-up
- Rudder for heading/centerline tracking
- Ailerons for wings level

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aida_sim.env.flight_env_cessna172 import Cessna172Env, StateIndex


class GroundRollController:
    """
    Classical controller for ground roll phase.
    
    Control strategy:
    1. Throttle: 100% (max acceleration)
    2. Elevator: Neutral until 0.9*V_rotate, then gentle pitch-up
    3. Rudder: PD control for heading/centerline
    4. Ailerons: PD control for wings level
    """
    
    def __init__(self, v_rotate=28.0, runway_heading=0.0):
        self.v_rotate = v_rotate  # Rotation speed (m/s)
        self.runway_heading = runway_heading  # Target heading (rad)
        
        # PID gains for heading control (rudder)
        self.kp_heading = 0.5
        self.kd_heading = 0.2
        
        # PID gains for roll control (ailerons)  
        self.kp_roll = 1.0
        self.kd_roll = 0.3
        
        # Elevator settings
        self.elevator_neutral = 0.0
        self.elevator_rotate = 0.3  # Gentle pitch-up command
        
    def compute_action(self, obs):
        """
        Compute control action from observation.
        
        Args:
            obs: State observation [x, y, z, u, v, w, phi, theta, psi, p, q, r]
            
        Returns:
            action: [throttle, aileron, elevator, rudder] in [-1, 1]
        """
        # Extract states
        x, y, z = obs[StateIndex.X], obs[StateIndex.Y], obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi = obs[StateIndex.PHI]      # Roll
        theta = obs[StateIndex.THETA]  # Pitch
        psi = obs[StateIndex.PSI]      # Heading
        p = obs[StateIndex.P]          # Roll rate
        q = obs[StateIndex.Q]          # Pitch rate
        r = obs[StateIndex.R]          # Yaw rate
        
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        altitude = -z
        
        # 1. THROTTLE: Always full for ground roll
        throttle = 1.0
        
        # 2. ELEVATOR: Neutral until near rotation speed
        if airspeed < 0.85 * self.v_rotate:
            # Keep nose down / neutral during acceleration
            elevator = self.elevator_neutral
        elif airspeed < self.v_rotate:
            # Gradual transition to pitch-up
            blend = (airspeed - 0.85 * self.v_rotate) / (0.15 * self.v_rotate)
            elevator = blend * self.elevator_rotate
        else:
            # At/above rotation speed - rotate!
            elevator = self.elevator_rotate
            
        # 3. RUDDER: Heading control (keep aligned with runway)
        heading_error = self._wrap_angle(self.runway_heading - psi)
        rudder = self.kp_heading * heading_error - self.kd_heading * r
        rudder = np.clip(rudder, -1.0, 1.0)
        
        # 4. AILERONS: Wings level
        roll_error = 0.0 - phi  # Target zero roll
        aileron = self.kp_roll * roll_error - self.kd_roll * p
        aileron = np.clip(aileron, -1.0, 1.0)
        
        return np.array([throttle, aileron, elevator, rudder], dtype=np.float32)
    
    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def test_controller():
    """Test the classical controller on the environment."""
    print("="*60)
    print("  CLASSICAL GROUND ROLL CONTROLLER TEST")
    print("="*60)
    
    # Create environment
    env = Cessna172Env(task="ground_roll")
    controller = GroundRollController(v_rotate=env.V_ROTATE, runway_heading=env.runway_heading)
    
    obs, info = env.reset()
    
    airspeed_init = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)
    print(f"\nInitial state:")
    print(f"  Position: X={obs[0]:.1f}, Y={obs[1]:.1f}, Alt={-obs[2]:.2f}m")
    print(f"  Airspeed: {airspeed_init:.1f} m/s")
    print(f"  V_rotate: {env.V_ROTATE:.1f} m/s")
    print(f"\nRunning ground roll...\n")
    
    total_reward = 0
    step = 0
    max_steps = 800
    
    while step < max_steps:
        # Get action from classical controller
        action = controller.compute_action(obs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Extract useful info
        airspeed = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)
        altitude = -obs[2]
        pitch = np.rad2deg(obs[7])
        heading = np.rad2deg(obs[8])
        x, y = obs[0], obs[1]
        
        # Print progress every 100 steps
        if step % 100 == 0 or airspeed >= env.V_ROTATE:
            print(f"Step {step:3d}: Speed={airspeed:5.1f} m/s, Alt={altitude:5.2f}m, "
                  f"Pitch={pitch:5.1f} deg, X={x:6.1f}, Y={y:5.1f}, "
                  f"Action=[T={action[0]:.2f}, A={action[1]:.2f}, E={action[2]:.2f}, R={action[3]:.2f}]")
        
        # Check if reached rotation speed
        if airspeed >= env.V_ROTATE:
            print(f"\n SUCCESS: ROTATION SPEED REACHED at step {step}!")
            print(f"   Airspeed: {airspeed:.1f} m/s (V_rotate = {env.V_ROTATE:.1f} m/s)")
            print(f"   Distance: {x - env.runway_start[0]:.1f}m from start")
            break
            
        if terminated or truncated:
            print(f"\n FAILED: Episode terminated: {info.get('termination_reason', 'unknown')}")
            break
    
    print(f"\n" + "="*60)
    print(f"  RESULTS")
    print(f"="*60)
    print(f"  Steps: {step}")
    print(f"  Total Reward: {total_reward:.1f}")
    print(f"  Final Airspeed: {airspeed:.1f} m/s")
    print(f"  Final Position: X={x:.1f}, Y={y:.1f}")
    success = airspeed >= env.V_ROTATE
    print(f"  Success: YES if success else NO")
    print("="*60)
    
    env.close()
    return success


if __name__ == "__main__":
    test_controller()
