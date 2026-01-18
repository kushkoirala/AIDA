#!/usr/bin/env python3
"""
Waypoint-Following Flight Environment

The NN learns to be a PILOT, not a NAVIGATOR:
- Input: aircraft state + current waypoint command
- Output: flight controls (throttle, aileron, elevator, rudder)
- Generalizes to ANY flight path (triangle, teardrop, holding, etc.)

The mission planner (separate module) decides WHERE to fly.
This NN executes the commands.

Waypoint Command Structure:
    - target_x, target_y: position (meters)
    - target_altitude: altitude (meters)
    - target_speed: airspeed (m/s)
    - waypoint_type: 'flyover', 'flyby', 'hold', 'land'

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import IntEnum

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gpu-flight-dynamics" / "python"))

from flight_dynamics import FlightSimulator, StateIndex, ControlIndex, STATE_DIM, CONTROL_DIM
from aircraft_database import get_aircraft


class WaypointType(IntEnum):
    """Types of waypoints the NN must handle."""
    FLYOVER = 0    # Fly directly over the point
    FLYBY = 1      # Fly past (turn before reaching)
    HOLD = 2       # Circle/hold at this point
    LAND = 3       # Landing waypoint (descend and touchdown)
    TAKEOFF = 4    # Takeoff waypoint (accelerate and climb)


class WaypointFollowingEnv(gym.Env):
    """
    Environment for training NN to follow waypoint commands.

    Observation Space (18D):
        Aircraft state (12D):
            [x, y, z, u, v, w, phi, theta, psi, p, q, r]

        Waypoint command (6D):
            [dist_to_wp, bearing_to_wp, altitude_error,
             speed_error, waypoint_type, turn_direction]

    Action Space (4D):
        [throttle, aileron, elevator, rudder] normalized to [-1, 1]

    The NN learns to:
        1. Fly toward waypoints
        2. Maintain commanded altitude
        3. Maintain commanded airspeed
        4. Execute different waypoint types (flyover, land, etc.)
    """

    metadata = {"render_modes": ["none"], "render_fps": 50}

    def __init__(self,
                 dt: float = 0.02,
                 max_episode_steps: int = 6000,  # 2 minutes
                 random_waypoints: bool = True):
        """
        Initialize waypoint-following environment.

        Args:
            dt: Simulation timestep
            max_episode_steps: Max steps per episode
            random_waypoints: If True, generate random waypoint sequences
        """
        super().__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.random_waypoints = random_waypoints
        self.step_count = 0

        # Aircraft performance limits
        self.V_STALL = 24.0      # m/s (~47 KIAS)
        self.V_MIN = 28.0       # m/s (~55 KIAS) safe minimum
        self.V_CRUISE = 56.0    # m/s (~110 KIAS)
        self.V_MAX = 70.0       # m/s (~135 KIAS)
        self.V_APPROACH = 33.0  # m/s (~65 KIAS)

        self.ALT_MIN = 0.0      # Ground level
        self.ALT_MAX = 3000.0   # ~10,000 ft

        self.BANK_MAX = np.deg2rad(30.0)  # Max bank angle
        self.PITCH_MAX = np.deg2rad(20.0)  # Max pitch

        # Waypoint capture radii
        self.WP_CAPTURE_RADIUS = 200.0    # meters - waypoint "reached"
        self.WP_APPROACH_RADIUS = 1000.0  # meters - start preparing for WP

        # Load aircraft
        self.aircraft = get_aircraft('cessna172')
        self.params = self._config_to_params(self.aircraft)
        self.sim = FlightSimulator(n_instances=1, params=self.params, dt=dt)

        # Observation space: state (12) + waypoint info (6) = 18
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32
        )

        # Action space: throttle, aileron, elevator, rudder
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Waypoint sequence (set by mission planner or randomly)
        self.waypoints = []
        self.current_wp_idx = 0
        self.waypoints_completed = 0

        # Episode tracking
        self.episode_reward = 0.0
        self.prev_dist_to_wp = float('inf')

    def _config_to_params(self, config):
        """Convert AircraftConfig to AircraftParams."""
        from flight_dynamics import (
            AircraftParams, MassProperties, Geometry,
            LongitudinalDerivatives, LateralDerivatives, PropulsionParams
        )

        return AircraftParams(
            mass=MassProperties(
                mass=config.mass.mass,
                Ixx=config.mass.Ixx,
                Iyy=config.mass.Iyy,
                Izz=config.mass.Izz,
                Ixz=config.mass.Ixz,
            ),
            geom=Geometry(
                S=config.geom.S,
                b=config.geom.b,
                c=config.geom.c,
                e=config.geom.e,
            ),
            longi=LongitudinalDerivatives(
                CL0=config.longi.CL0,
                CLa=config.longi.CLa,
                CLq=config.longi.CLq,
                CLde=config.longi.CLde,
                CLmax=config.longi.CLmax,
                CLmin=config.longi.CLmin,
                CD0=config.longi.CD0,
                K=config.longi.K,
                CDa=config.longi.CDa,
                Cm0=config.longi.Cm0,
                Cma=config.longi.Cma,
                Cmq=config.longi.Cmq,
                Cmde=config.longi.Cmde,
            ),
            latdi=LateralDerivatives(
                CYb=config.latdi.CYb,
                CYp=config.latdi.CYp,
                CYr=config.latdi.CYr,
                CYda=config.latdi.CYda,
                CYdr=config.latdi.CYdr,
                Clb=config.latdi.Clb,
                Clp=config.latdi.Clp,
                Clr=config.latdi.Clr,
                Clda=config.latdi.Clda,
                Cldr=config.latdi.Cldr,
                Cnb=config.latdi.Cnb,
                Cnp=config.latdi.Cnp,
                Cnr=config.latdi.Cnr,
                Cnda=config.latdi.Cnda,
                Cndr=config.latdi.Cndr,
            ),
            prop=PropulsionParams(
                thrust_max=config.prop.thrust_max,
                thrust_min=config.prop.thrust_min,
                tau=config.prop.tau,
            ),
        )

    def set_waypoints(self, waypoints):
        """
        Set waypoint sequence from mission planner.

        Args:
            waypoints: List of dicts with keys:
                - x, y: position (meters)
                - altitude: target altitude (meters)
                - speed: target airspeed (m/s)
                - type: WaypointType enum
                - heading: (optional) required heading at waypoint
        """
        self.waypoints = waypoints
        self.current_wp_idx = 0

    def _generate_random_waypoints(self):
        """Generate random waypoint sequence for training variety."""
        waypoints = []

        # Start position (will be set in reset)
        x, y = 0.0, 0.0
        alt = self.np_random.uniform(500, 1500)  # Random starting altitude

        # Generate 3-6 waypoints
        n_waypoints = self.np_random.integers(3, 7)

        for i in range(n_waypoints):
            # Random direction and distance
            heading = self.np_random.uniform(-np.pi, np.pi)
            distance = self.np_random.uniform(2000, 8000)  # 2-8 km

            x += distance * np.cos(heading)
            y += distance * np.sin(heading)

            # Random altitude change
            alt_change = self.np_random.uniform(-300, 300)
            alt = np.clip(alt + alt_change, 300, 2000)

            # Random speed
            speed = self.np_random.uniform(35, 55)  # 70-110 KIAS

            # Waypoint type (mostly flyover, occasionally others)
            wp_type_prob = self.np_random.random()
            if i == n_waypoints - 1:  # Last waypoint
                wp_type = WaypointType.FLYOVER
            elif wp_type_prob < 0.7:
                wp_type = WaypointType.FLYOVER
            elif wp_type_prob < 0.9:
                wp_type = WaypointType.FLYBY
            else:
                wp_type = WaypointType.HOLD

            waypoints.append({
                'x': x,
                'y': y,
                'altitude': alt,
                'speed': speed,
                'type': wp_type,
            })

        return waypoints

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)

        self.step_count = 0
        self.episode_reward = 0.0
        self.current_wp_idx = 0
        self.waypoints_completed = 0
        self.prev_dist_to_wp = float('inf')

        # Generate or use provided waypoints
        if self.random_waypoints or len(self.waypoints) == 0:
            self.waypoints = self._generate_random_waypoints()

        # Initialize aircraft state
        initial_state = np.zeros((1, STATE_DIM), dtype=np.float32)

        if options and 'initial_state' in options:
            # Use provided initial state
            initial_state[0] = options['initial_state']
        else:
            # Random initial state (in air, heading toward first waypoint)
            first_wp = self.waypoints[0]

            # Start 3-5km from first waypoint
            start_dist = self.np_random.uniform(3000, 5000)
            start_heading = np.arctan2(first_wp['y'], first_wp['x']) + np.pi  # Opposite

            initial_state[0, StateIndex.X] = first_wp['x'] + start_dist * np.cos(start_heading)
            initial_state[0, StateIndex.Y] = first_wp['y'] + start_dist * np.sin(start_heading)
            initial_state[0, StateIndex.Z] = -first_wp['altitude']  # NED

            # Heading toward first waypoint
            heading_to_wp = np.arctan2(
                first_wp['y'] - initial_state[0, StateIndex.Y],
                first_wp['x'] - initial_state[0, StateIndex.X]
            )
            initial_state[0, StateIndex.PSI] = heading_to_wp

            # Cruise speed
            initial_state[0, StateIndex.U] = first_wp['speed']

        self.sim.reset(initial_state=initial_state)

        # Calculate initial distance to waypoint
        state = self._get_raw_state()
        wp = self.waypoints[self.current_wp_idx]
        self.prev_dist_to_wp = np.sqrt(
            (state[StateIndex.X] - wp['x'])**2 +
            (state[StateIndex.Y] - wp['y'])**2
        )

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute one timestep."""
        self.step_count += 1

        # Apply action
        controls = self._denormalize_action(action)
        self.sim.set_controls(controls)
        self.sim.step()

        # Apply ground contact if needed
        self._apply_ground_contact()

        # Check waypoint capture
        self._check_waypoint_capture()

        # Get observation and compute reward
        obs = self._get_observation()
        terminated, term_info = self._check_termination()
        reward = self._compute_reward(obs, action, terminated, term_info)

        truncated = self.step_count >= self.max_episode_steps

        self.episode_reward += reward

        info = self._get_info()
        info.update(term_info)

        return obs, reward, terminated, truncated, info

    def _get_raw_state(self):
        """Get raw aircraft state."""
        states = self.sim.get_states()
        try:
            state = states.get()[0] if hasattr(states, 'get') else states[0]
        except:
            state = np.array(states)[0]
        return state

    def _get_observation(self):
        """Get observation: aircraft state + waypoint command."""
        state = self._get_raw_state()

        # Current waypoint
        if self.current_wp_idx < len(self.waypoints):
            wp = self.waypoints[self.current_wp_idx]
        else:
            # No more waypoints - hover at last position
            wp = self.waypoints[-1] if self.waypoints else {
                'x': 0, 'y': 0, 'altitude': 1000, 'speed': 40, 'type': WaypointType.FLYOVER
            }

        # Aircraft position and velocity
        x, y, z = state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z]
        u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]
        psi = state[StateIndex.PSI]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)

        # Distance and bearing to waypoint
        dx = wp['x'] - x
        dy = wp['y'] - y
        dist_to_wp = np.sqrt(dx**2 + dy**2)
        bearing_to_wp = np.arctan2(dy, dx)

        # Relative bearing (how much to turn)
        relative_bearing = self._normalize_angle(bearing_to_wp - psi)

        # Turn direction: -1 = left, +1 = right
        turn_direction = np.sign(relative_bearing)

        # Errors (normalized)
        altitude_error = (wp['altitude'] - altitude) / 500.0  # Normalize by 500m
        speed_error = (wp['speed'] - airspeed) / 20.0  # Normalize by 20 m/s

        # Waypoint info vector (6D)
        wp_info = np.array([
            np.clip(dist_to_wp / 5000.0, 0, 2),      # Distance (normalized by 5km)
            relative_bearing / np.pi,                 # Relative bearing [-1, 1]
            np.clip(altitude_error, -2, 2),           # Altitude error
            np.clip(speed_error, -2, 2),              # Speed error
            float(wp['type']) / 4.0,                  # Waypoint type (normalized)
            turn_direction,                           # Turn direction
        ], dtype=np.float32)

        # Combine state + waypoint info
        obs = np.concatenate([state.astype(np.float32), wp_info])

        return obs

    def _denormalize_action(self, action):
        """Convert normalized action to control inputs."""
        controls = np.zeros((1, CONTROL_DIM), dtype=np.float32)

        # Throttle: [-1, 1] → [0, 1]
        controls[0, ControlIndex.THROTTLE] = (action[0] + 1.0) / 2.0

        # Aileron: [-1, 1] → [-25°, 25°]
        controls[0, ControlIndex.AILERON] = action[1] * np.deg2rad(25.0)

        # Elevator: [-1, 1] → [-20°, 20°]
        controls[0, ControlIndex.ELEVATOR] = action[2] * np.deg2rad(20.0)

        # Rudder: [-1, 1] → [-15°, 15°]
        controls[0, ControlIndex.RUDDER] = action[3] * np.deg2rad(15.0)

        return controls

    def _apply_ground_contact(self):
        """Simple ground contact model."""
        altitude = -float(self.sim.states[0, StateIndex.Z].get() if hasattr(self.sim.states, 'get')
                         else self.sim.states[0, StateIndex.Z])

        if altitude < 0.5:
            if altitude < 0:
                self.sim.states[0, StateIndex.Z] = 0.0

            current_w = float(self.sim.states[0, StateIndex.W].get() if hasattr(self.sim.states, 'get')
                            else self.sim.states[0, StateIndex.W])
            if current_w > 0:
                self.sim.states[0, StateIndex.W] = 0.0

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _check_waypoint_capture(self):
        """Check if current waypoint has been reached."""
        if self.current_wp_idx >= len(self.waypoints):
            return

        state = self._get_raw_state()
        wp = self.waypoints[self.current_wp_idx]

        # Distance to waypoint
        dx = wp['x'] - state[StateIndex.X]
        dy = wp['y'] - state[StateIndex.Y]
        dist_to_wp = np.sqrt(dx**2 + dy**2)

        # Check capture based on waypoint type
        wp_type = wp['type']

        if wp_type == WaypointType.FLYOVER:
            # Must fly within capture radius
            if dist_to_wp < self.WP_CAPTURE_RADIUS:
                self._advance_waypoint()

        elif wp_type == WaypointType.FLYBY:
            # Can turn early - capture when approaching and past perpendicular
            if dist_to_wp < self.WP_APPROACH_RADIUS:
                # Check if we've passed the waypoint (dot product with velocity)
                u, v = state[StateIndex.U], state[StateIndex.V]
                vel_dot = dx * u + dy * v
                if vel_dot < 0:  # Moving away from waypoint
                    self._advance_waypoint()

        elif wp_type == WaypointType.LAND:
            # Must be on ground near waypoint
            altitude = -state[StateIndex.Z]
            if altitude < 3.0 and dist_to_wp < self.WP_CAPTURE_RADIUS * 2:
                self._advance_waypoint()

        # Update distance tracking
        self.prev_dist_to_wp = dist_to_wp

    def _advance_waypoint(self):
        """Move to next waypoint."""
        self.waypoints_completed += 1
        self.current_wp_idx += 1

        # Reset distance tracking for new waypoint
        if self.current_wp_idx < len(self.waypoints):
            state = self._get_raw_state()
            wp = self.waypoints[self.current_wp_idx]
            self.prev_dist_to_wp = np.sqrt(
                (state[StateIndex.X] - wp['x'])**2 +
                (state[StateIndex.Y] - wp['y'])**2
            )

    def _check_termination(self):
        """Check if episode should terminate."""
        state = self._get_raw_state()

        x, y, z = state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z]
        u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]
        phi, theta = state[StateIndex.PHI], state[StateIndex.THETA]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)

        info = {'termination_reason': None}

        # Crash
        if altitude < -5.0:
            info['termination_reason'] = 'crash'
            return True, info

        # Stall (only when airborne)
        if altitude > 50.0 and airspeed < self.V_STALL:
            info['termination_reason'] = 'stall'
            return True, info

        # Overspeed
        if airspeed > self.V_MAX * 1.2:
            info['termination_reason'] = 'overspeed'
            return True, info

        # Excessive attitude
        if abs(phi) > np.deg2rad(60) or abs(theta) > np.deg2rad(45):
            info['termination_reason'] = 'attitude_limit'
            return True, info

        # All waypoints completed
        if self.current_wp_idx >= len(self.waypoints):
            info['termination_reason'] = 'mission_complete'
            return True, info

        return False, info

    def _compute_reward(self, obs, action, terminated, term_info):
        """
        Compute reward for waypoint following.

        Rewards:
            1. Progress toward waypoint (distance reduction)
            2. Altitude tracking
            3. Speed tracking
            4. Waypoint capture bonus
            5. Smooth control bonus
        """
        reward = 0.0

        state = self._get_raw_state()
        wp = self.waypoints[min(self.current_wp_idx, len(self.waypoints)-1)]

        # Aircraft state
        x, y, z = state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z]
        u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)

        # Distance to waypoint
        dist_to_wp = np.sqrt((wp['x'] - x)**2 + (wp['y'] - y)**2)

        # 1. Progress reward (distance reduction)
        progress = self.prev_dist_to_wp - dist_to_wp
        reward += progress * 0.01  # 0.01 per meter closer

        # 2. Altitude tracking
        alt_error = abs(wp['altitude'] - altitude)
        if alt_error < 30:  # Within 100 ft
            reward += 0.2
        elif alt_error < 100:
            reward += 0.1
        else:
            reward -= 0.1 * (alt_error / 100)

        # 3. Speed tracking
        speed_error = abs(wp['speed'] - airspeed)
        if speed_error < 3:  # Within 6 knots
            reward += 0.2
        elif speed_error < 10:
            reward += 0.1
        else:
            reward -= 0.1 * (speed_error / 10)

        # 4. Waypoint capture bonus
        if self.waypoints_completed > 0 and self.step_count > 1:
            # Check if we just captured a waypoint
            if dist_to_wp > self.prev_dist_to_wp + 100:  # Jumped to new WP
                reward += 50.0  # Big bonus for waypoint capture

        # 5. Smooth control (penalize jerky inputs)
        control_penalty = 0.01 * np.sum(np.abs(action))
        reward -= control_penalty

        # 6. Survival bonus
        reward += 0.05

        # Termination rewards
        if terminated:
            reason = term_info.get('termination_reason')

            if reason == 'mission_complete':
                reward += 100.0 * self.waypoints_completed

            elif reason == 'crash':
                reward -= 100.0

            elif reason == 'stall':
                reward -= 80.0

            elif reason == 'attitude_limit':
                reward -= 50.0

        return reward

    def _get_info(self):
        """Get info dict."""
        state = self._get_raw_state()

        altitude = -state[StateIndex.Z]
        u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]
        airspeed = np.sqrt(u**2 + v**2 + w**2)

        # Current waypoint info
        if self.current_wp_idx < len(self.waypoints):
            wp = self.waypoints[self.current_wp_idx]
            dist_to_wp = np.sqrt(
                (wp['x'] - state[StateIndex.X])**2 +
                (wp['y'] - state[StateIndex.Y])**2
            )
        else:
            dist_to_wp = 0.0

        return {
            'altitude': float(altitude),
            'airspeed': float(airspeed),
            'current_waypoint': self.current_wp_idx,
            'waypoints_completed': self.waypoints_completed,
            'total_waypoints': len(self.waypoints),
            'dist_to_waypoint': float(dist_to_wp),
            'episode_reward': float(self.episode_reward),
        }

    def render(self):
        pass

    def close(self):
        pass
