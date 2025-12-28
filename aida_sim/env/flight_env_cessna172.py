"""
Cessna 172 Flight Environment for Autonomous Flight Training

Realistic general aviation environment with:
- Takeoff from 1000m runway
- Climb to cruise altitude (3000-5000 ft)
- Level cruise flight
- Larger flight volume (10km × 10km × 2000m)

Author: Kushal Koirala (with Claude Code)
Date: December 27, 2024
"""
import sys
import os
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add gpu-flight-dynamics to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gpu-flight-dynamics" / "python"))

from flight_dynamics import FlightSimulator, StateIndex, ControlIndex, STATE_DIM, CONTROL_DIM
from aircraft_database import get_aircraft


class Cessna172Env(gym.Env):
    """
    Gymnasium environment for Cessna 172 autonomous flight.

    Observation Space: 12D state vector
        [x, y, z, u, v, w, phi, theta, psi, p, q, r]

    Action Space: 4D control vector (normalized -1 to 1)
        [throttle, aileron, elevator, rudder]

    Tasks:
        - "ground_roll": Phase 1 - Accelerate from 0 to rotation speed (55 KIAS)
        - "rotation": Phase 2 - Ground roll + rotation to 10 ft AGL
        - "initial_climb": Phase 3 - Climb to 500 ft AGL
        - "full_climb": Phase 4 - Climb to cruise altitude (3000 ft)
        - "cruise": Phase 5 - Level flight at cruise altitude
        - "full_mission": Complete mission (all phases)

    Curriculum Learning:
        Train phases sequentially, mastering each before advancing:
        Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
    """

    metadata = {"render_modes": ["none"], "render_fps": 50}

    def __init__(self,
                 dt: float = 0.02,
                 max_episode_steps: int = 3000,  # 60 seconds
                 task: str = "full_mission",
                 cruise_altitude_ft: float = 3000.0):
        """
        Initialize Cessna 172 environment.

        Args:
            dt: Simulation timestep (s)
            max_episode_steps: Maximum steps per episode
            task: Mission task ("takeoff", "climb", "cruise", "full_mission")
            cruise_altitude_ft: Target cruise altitude in feet AGL
        """
        super().__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.task = task
        self.step_count = 0

        # Curriculum learning phase definitions
        self.phase_configs = {
            'ground_roll': {
                'max_steps': 500,  # 10 seconds
                'success_criteria': {
                    'airspeed_min': 28.0,  # 55 KIAS rotation speed
                    'altitude_max': 2.0,   # Must stay on ground
                    'heading_max_dev': 15.0,  # Stay on runway centerline
                    'pitch_max': np.deg2rad(5.0),  # Keep nose down during roll
                },
                'target_altitude': 0.0,
                'phase_idx': 1,
            },
            'rotation': {
                'max_steps': 750,  # 15 seconds
                'success_criteria': {
                    'altitude_min': 3.0,   # 10 ft AGL
                    'airspeed_min': 30.0,  # 58 KIAS
                    'pitch_max': np.deg2rad(15.0),  # Safe pitch angle
                },
                'target_altitude': 3.0,
                'phase_idx': 2,
            },
            'initial_climb': {
                'max_steps': 1500,  # 30 seconds
                'success_criteria': {
                    'altitude_min': 152.4,  # 500 ft AGL
                    'airspeed_min': 35.0,   # 68 KIAS
                    'climb_rate_min': 2.0,  # ~400 fpm minimum
                },
                'target_altitude': 152.4,
                'phase_idx': 3,
            },
            'full_climb': {
                'max_steps': 3000,  # 60 seconds
                'success_criteria': {
                    'altitude_min': None,  # Will be set to cruise_altitude
                    'airspeed_min': 35.0,  # 68 KIAS
                    'climb_rate_min': 1.5,  # ~300 fpm minimum
                },
                'target_altitude': None,  # Will be set to cruise_altitude
                'phase_idx': 4,
            },
            'cruise': {
                'max_steps': 1500,  # 30 seconds
                'success_criteria': {
                    'altitude_tolerance': 15.0,  # ±50 ft
                    'airspeed_tolerance': 5.0,   # ±10 knots
                    'bank_max': np.deg2rad(10.0),  # Wings level
                    'duration': 600,  # 12 seconds stable
                },
                'target_altitude': None,  # Will be set to cruise_altitude
                'phase_idx': 5,
            },
            'full_mission': {
                'max_steps': 3000,  # 60 seconds
                'success_criteria': {},
                'target_altitude': None,
                'phase_idx': 0,
            },
        }

        # Set task-specific parameters
        if task in self.phase_configs:
            phase_cfg = self.phase_configs[task]
            if max_episode_steps == 3000:  # User didn't specify
                self.max_episode_steps = phase_cfg['max_steps']

            # Update cruise altitude references
            if task in ['full_climb', 'cruise', 'full_mission']:
                phase_cfg['target_altitude'] = cruise_altitude_ft * 0.3048
                if task == 'full_climb':
                    phase_cfg['success_criteria']['altitude_min'] = cruise_altitude_ft * 0.3048
        else:
            raise ValueError(f"Unknown task: {task}. Must be one of {list(self.phase_configs.keys())}")

        self.task_config = self.phase_configs[task]

        # Load Cessna 172 configuration
        self.aircraft = get_aircraft('cessna172')
        self.params = self._config_to_params(self.aircraft)

        # Create GPU simulator (single instance for gym env)
        self.sim = FlightSimulator(n_instances=1, params=self.params, dt=dt)

        # Aircraft performance parameters (from POH - Pilot's Operating Handbook)
        self.V_STALL = 24.0        # m/s (48 KIAS stall speed, clean config)
        self.V_ROTATE = 28.0       # m/s (55 KIAS rotation speed)
        self.V_CLIMB = 38.0        # m/s (75 KIAS best rate of climb)
        self.V_CRUISE = 56.0       # m/s (110 KIAS cruise speed)
        self.V_MAX = 70.0          # m/s (135 KIAS never exceed)

        # Runway parameters (typical GA runway)
        self.runway_length = 1000.0    # m (3280 ft)
        self.runway_width = 30.0       # m (100 ft)
        self.runway_heading = 0.0      # degrees (runway 36)
        self.runway_start = np.array([0.0, -500.0, 0.0], dtype=np.float32)  # Start at south end
        self.runway_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # North direction
        self.runway_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # East direction

        # Mission altitude parameters
        self.cruise_altitude = cruise_altitude_ft * 0.3048  # ft to meters
        self.takeoff_altitude = 50.0 * 0.3048  # 50 ft AGL

        # Flight volume (10km × 10km × 2000m)
        self.flight_bounds = np.array([5000.0, 5000.0, 2000.0], dtype=np.float32)

        # Mission phase tracking
        self.mission_phases = [
            {"name": "takeoff", "type": "altitude", "target": self.takeoff_altitude, "tolerance": 3.0},
            {"name": "climb", "type": "altitude", "target": self.cruise_altitude, "tolerance": 10.0},
            {"name": "cruise", "type": "maintain", "duration": 600},  # 30 seconds cruise
        ]
        self.current_phase_idx = 0

        # Gymnasium spaces
        # Observation: [x, y, z, u, v, w, phi, theta, psi, p, q, r]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(STATE_DIM,),
            dtype=np.float32
        )

        # Action: [throttle, aileron, elevator, rudder] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(CONTROL_DIM,),
            dtype=np.float32
        )

        # Episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0

    def _config_to_params(self, config):
        """Convert AircraftConfig to AircraftParams for FlightSimulator"""
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

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial conditions.

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.current_phase_idx = 0

        # Initialize state based on task
        initial_state = np.zeros((1, STATE_DIM), dtype=np.float32)

        if self.task == "takeoff" or self.task == "full_mission":
            # Start on runway at rest
            initial_state[0, StateIndex.X] = self.runway_start[0]
            initial_state[0, StateIndex.Y] = self.runway_start[1]
            initial_state[0, StateIndex.Z] = -0.5  # On ground (slight offset)
            initial_state[0, StateIndex.U] = 0.0  # At rest
            initial_state[0, StateIndex.PSI] = self.runway_heading

        elif self.task == "climb":
            # Start airborne at 50 ft
            initial_state[0, StateIndex.X] = 0.0
            initial_state[0, StateIndex.Y] = 0.0
            initial_state[0, StateIndex.Z] = -self.takeoff_altitude
            initial_state[0, StateIndex.U] = self.V_CLIMB
            initial_state[0, StateIndex.THETA] = np.deg2rad(5.0)  # Slight climb pitch

        elif self.task == "cruise":
            # Start at cruise altitude
            initial_state[0, StateIndex.X] = 0.0
            initial_state[0, StateIndex.Y] = 0.0
            initial_state[0, StateIndex.Z] = -self.cruise_altitude
            initial_state[0, StateIndex.U] = self.V_CRUISE
            initial_state[0, StateIndex.THETA] = 0.0  # Level

        # Reset simulator
        self.sim.reset(initial_state=initial_state)

        # Set initial elevator position for ground operations
        # This gives the agent a good starting point (nose-down to keep nose wheel on ground)
        # but lets it learn to adjust from there
        if self.task in ['ground_roll', 'rotation']:
            initial_controls = np.zeros((1, CONTROL_DIM), dtype=np.float32)
            initial_controls[0, ControlIndex.ELEVATOR] = np.deg2rad(-5.0)  # -5° nose-down initial position
            self.sim.controls = initial_controls.copy()

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """
        Execute one timestep.

        Args:
            action: Control vector [throttle, aileron, elevator, rudder] in [-1, 1]

        Returns:
            observation: New state
            reward: Reward for this step
            terminated: Episode ended
            truncated: Episode truncated (max steps)
            info: Additional information
        """
        self.step_count += 1

        # Denormalize action from [-1, 1] to actual control ranges
        controls = self._denormalize_action(action)

        # Set controls and step simulator
        self.sim.set_controls(controls)
        self.sim.step()

        # Apply ground contact model (simple altitude clamping)
        # TODO: Replace with proper gear/friction model
        altitude = -float(self.sim.states[0, StateIndex.Z].get() if hasattr(self.sim.states, 'get')
                         else self.sim.states[0, StateIndex.Z])

        if altitude < 0.0:  # Underground - clamp to ground
            self.sim.states[0, StateIndex.Z] = 0.0
            # Zero out downward vertical velocity (ground contact stops descent)
            current_w = float(self.sim.states[0, StateIndex.W].get() if hasattr(self.sim.states, 'get')
                            else self.sim.states[0, StateIndex.W])
            if current_w < 0:  # Descending
                self.sim.states[0, StateIndex.W] = 0.0

        # Get new observation
        obs = self._get_observation()

        # Compute reward and check termination
        terminated, term_info = self._check_termination(obs)
        reward = self._compute_reward(obs, action, terminated, term_info)

        # Check truncation
        truncated = self.step_count >= self.max_episode_steps

        # Update tracking
        self.episode_reward += reward
        self.episode_length += 1

        # Get full info dict
        info = self._get_info()
        info.update(term_info)  # Add termination reason if present

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current state as observation"""
        states = self.sim.get_states()
        # Convert from CuPy if needed
        try:
            obs = states.get() if hasattr(states, 'get') else states
        except:
            obs = np.array(states)
        return obs[0]  # Return single instance

    def _denormalize_action(self, action):
        """
        Convert normalized action [-1, 1] to actual control ranges.

        Cessna 172 control ranges:
        - Throttle: 0 to 1
        - Aileron: -25° to +25° (±0.436 rad)
        - Elevator: -20° to +20° (±0.349 rad)
        - Rudder: -15° to +15° (±0.262 rad)
        """
        controls = np.zeros((1, CONTROL_DIM), dtype=np.float32)

        # Throttle: map [-1, 1] → [0, 1]
        controls[0, ControlIndex.THROTTLE] = (action[0] + 1.0) / 2.0

        # Aileron: map [-1, 1] → [-25°, +25°]
        controls[0, ControlIndex.AILERON] = action[1] * np.deg2rad(25.0)

        # Elevator: map [-1, 1] → [-20°, +20°]
        # Agent must learn correct elevator position to keep nose wheel on ground
        controls[0, ControlIndex.ELEVATOR] = action[2] * np.deg2rad(20.0)

        # Rudder: map [-1, 1] → [-15°, +15°]
        controls[0, ControlIndex.RUDDER] = action[3] * np.deg2rad(15.0)

        return controls

    def _get_info(self):
        """Get additional information"""
        state = self._get_observation()
        altitude = -state[StateIndex.Z]
        u = state[StateIndex.U]
        v = state[StateIndex.V]
        w = state[StateIndex.W]
        airspeed = np.sqrt(u**2 + v**2 + w**2)  # TRUE airspeed

        return {
            'altitude': float(altitude),
            'airspeed': float(airspeed),
            'mission_phase': self.mission_phases[self.current_phase_idx]['name'] if self.current_phase_idx < len(self.mission_phases) else 'complete',
            'episode_reward': float(self.episode_reward),
            'episode_length': self.episode_length,
        }

    def _check_termination(self, obs):
        """
        Check if episode should terminate.

        Termination conditions:
        - Crash (altitude < 0)
        - Stall (airspeed < V_stall WHEN AIRBORNE)
        - Out of bounds
        - Excessive attitude
        - Mission complete
        """
        altitude = -obs[StateIndex.Z]
        # Calculate TRUE airspeed (magnitude of velocity vector)
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        pitch = obs[StateIndex.THETA]
        roll = obs[StateIndex.PHI]
        x, y = obs[StateIndex.X], obs[StateIndex.Y]

        info = {'termination_reason': None}

        # Determine if aircraft is on ground (within 1m of ground level)
        on_ground = altitude < 1.0

        # Crash
        if altitude < 0.0:
            info['termination_reason'] = 'crash'
            return True, info

        # Stall - ONLY check when airborne!
        # During ground roll, low speed is expected and necessary
        if not on_ground and airspeed < self.V_STALL:
            info['termination_reason'] = 'stall'
            return True, info

        # Overspeed
        if airspeed > self.V_MAX:
            info['termination_reason'] = 'overspeed'
            return True, info

        # Out of bounds
        if abs(x) > self.flight_bounds[0] or abs(y) > self.flight_bounds[1] or altitude > self.flight_bounds[2]:
            info['termination_reason'] = 'out_of_bounds'
            return True, info

        # Excessive attitude (should never happen with Cessna 172, but safety check)
        if abs(pitch) > np.deg2rad(45) or abs(roll) > np.deg2rad(60):
            info['termination_reason'] = 'attitude_limit'
            return True, info

        # Check mission phase completion
        if self.task == "full_mission" and self.current_phase_idx < len(self.mission_phases):
            phase = self.mission_phases[self.current_phase_idx]

            if phase['type'] == 'altitude':
                # Check if altitude target reached
                if abs(altitude - phase['target']) < phase['tolerance']:
                    self.current_phase_idx += 1
                    if self.current_phase_idx >= len(self.mission_phases):
                        info['termination_reason'] = 'mission_success'
                        return True, info

        return False, info

    def _compute_reward(self, obs, action, terminated, info):
        """
        Compute phase-appropriate reward for curriculum learning.

        Curriculum phases:
        - ground_roll: Accelerate to rotation speed
        - rotation: Ground roll + liftoff
        - initial_climb: Climb to 500 ft
        - full_climb: Climb to cruise altitude
        - cruise: Hold altitude and speed
        - full_mission: All phases combined
        """
        if self.task == "ground_roll":
            return self._reward_ground_roll(obs, action, terminated, info)
        elif self.task == "rotation":
            return self._reward_rotation(obs, action, terminated, info)
        elif self.task == "initial_climb":
            return self._reward_initial_climb(obs, action, terminated, info)
        elif self.task == "full_climb":
            return self._reward_climb(obs, action, terminated, info)
        elif self.task == "cruise":
            return self._reward_cruise(obs, action, terminated, info)
        elif self.task == "full_mission":
            # Use old phase tracking for full mission
            if self.current_phase_idx == 0:
                return self._reward_takeoff(obs, action, terminated, info)
            elif self.current_phase_idx == 1:
                return self._reward_climb(obs, action, terminated, info)
            else:
                return self._reward_cruise(obs, action, terminated, info)
        else:
            return 0.0

    def _reward_takeoff(self, obs, action, terminated, info):
        """
        Reward function for takeoff phase.

        Goals:
        1. Accelerate to rotation speed (55 KIAS)
        2. Rotate at correct pitch (~10°)
        3. Climb to 50 ft AGL
        4. Stay on runway centerline
        """
        reward = 0.0

        # Extract state
        x, y, z = obs[StateIndex.X], obs[StateIndex.Y], obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi, theta, psi = obs[StateIndex.PHI], obs[StateIndex.THETA], obs[StateIndex.PSI]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)  # TRUE airspeed

        # Position along runway
        rel_pos = np.array([x, y, 0]) - self.runway_start
        distance_along = np.dot(rel_pos, self.runway_dir)
        lateral_offset = np.dot(rel_pos, self.runway_right)

        # Ground roll phase (altitude < 5m)
        if altitude < 5.0:
            # Stay on centerline
            centerline_error = abs(lateral_offset) / (self.runway_width / 2.0)
            reward += 0.4 * np.exp(-centerline_error**2)
            reward -= 0.3 * centerline_error

            # Accelerate to rotation speed
            speed_ratio = airspeed / self.V_ROTATE
            reward += 0.5 * speed_ratio

            # Use high throttle
            throttle = (action[0] + 1.0) / 2.0  # Denormalize
            reward += 0.1 * throttle

        # Rotation phase (40-60 KIAS)
        if 20.0 < airspeed < 35.0 and altitude < 10.0:
            # Pitch up toward rotation pitch
            pitch_target = np.deg2rad(10.0)
            pitch_error = abs(theta - pitch_target)
            reward += 0.3 * np.exp(-(pitch_error / np.deg2rad(5.0))**2)

        # Climb phase (altitude > 5m)
        if altitude > 5.0:
            # Reward altitude gain
            reward += 0.1 * altitude / self.takeoff_altitude

            # Maintain climb speed
            speed_error = abs(airspeed - self.V_CLIMB) / self.V_CLIMB
            reward -= 0.2 * speed_error

            # Climb angle
            if w < 0:  # Climbing (NED frame)
                reward += 0.2

        # Termination rewards
        if terminated:
            reason = info.get('termination_reason')
            if reason == 'crash':
                reward -= 15.0
            elif reason == 'stall':
                reward -= 10.0
            elif altitude >= self.takeoff_altitude:
                reward += 50.0  # Successfully reached takeoff altitude!

        # Survival bonus
        reward += 0.1

        return reward

    def _reward_climb(self, obs, action, terminated, info):
        """
        Reward function for climb phase.

        Goals:
        1. Climb to cruise altitude
        2. Maintain climb speed (75 KIAS)
        3. Smooth climb rate
        """
        reward = 0.0

        # Extract state
        z = obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        theta = obs[StateIndex.THETA]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)  # TRUE airspeed
        climb_rate = -w  # NED frame

        # Altitude tracking
        alt_error = abs(altitude - self.cruise_altitude)
        reward -= 0.02 * alt_error

        # Progress toward cruise altitude
        progress = altitude / self.cruise_altitude
        reward += 0.3 * progress

        # Maintain climb speed
        speed_error = abs(airspeed - self.V_CLIMB)
        reward -= 0.05 * speed_error

        # Positive climb rate
        if climb_rate > 0:
            reward += 0.2

        # Reasonable climb rate (not too aggressive)
        if 2.0 < climb_rate < 8.0:  # 400-1600 fpm typical for C172
            reward += 0.1

        # Pitch attitude
        pitch_error = abs(theta - np.deg2rad(5.0))  # ~5° climb pitch
        reward -= 0.05 * pitch_error

        # Termination rewards
        if terminated:
            reason = info.get('termination_reason')
            if reason == 'crash':
                reward -= 15.0
            elif reason == 'stall':
                reward -= 10.0
            elif alt_error < 10.0:  # Within 10m of cruise altitude
                reward += 75.0  # Phase complete!

        # Survival bonus
        reward += 0.1

        return reward

    def _reward_cruise(self, obs, action, terminated, info):
        """
        Reward function for cruise phase.

        Goals:
        1. Hold cruise altitude (±50 ft)
        2. Maintain cruise speed (110 KIAS)
        3. Level flight (minimal pitch/roll)
        4. Smooth controls
        """
        reward = 0.0

        # Extract state
        z = obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi, theta = obs[StateIndex.PHI], obs[StateIndex.THETA]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)  # TRUE airspeed

        # Altitude hold
        alt_error = abs(altitude - self.cruise_altitude)
        reward -= 0.05 * alt_error

        # Tight altitude tolerance bonus
        if alt_error < 15.0:  # Within 50 ft
            reward += 0.5

        # Speed hold
        speed_error = abs(airspeed - self.V_CRUISE)
        reward -= 0.05 * speed_error

        # Speed tolerance bonus
        if speed_error < 3.0:  # Within ~6 knots
            reward += 0.3

        # Level flight
        reward -= 0.1 * abs(theta)
        reward -= 0.1 * abs(phi)

        # Smooth controls (penalize large inputs)
        control_mag = np.linalg.norm(action)
        reward -= 0.03 * control_mag

        # Termination rewards
        if terminated:
            reason = info.get('termination_reason')
            if reason == 'crash':
                reward -= 15.0
            elif reason == 'stall':
                reward -= 10.0
            elif reason == 'mission_success':
                reward += 100.0  # Mission complete!

        # Survival bonus
        reward += 0.1

        return reward

    def _reward_ground_roll(self, obs, action, terminated, info):
        """
        Reward function for Phase 1: Ground Roll

        Goals:
        1. Accelerate from 0 to rotation speed (55 KIAS = 28 m/s)
        2. Stay on runway centerline
        3. Keep wings level
        4. Maintain runway heading

        Max reward: +100 for reaching rotation speed
        """
        reward = 0.0

        # Extract state
        x, y, z = obs[StateIndex.X], obs[StateIndex.Y], obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi, theta, psi = obs[StateIndex.PHI], obs[StateIndex.THETA], obs[StateIndex.PSI]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)

        # 1. Airspeed progression (0 → 28 m/s)
        # Reward proportional to speed (0-50 points)
        speed_progress = min(airspeed / self.V_ROTATE, 1.0)
        reward += 50.0 * speed_progress

        # Bonus for reaching rotation speed
        if airspeed >= self.V_ROTATE:
            reward += 50.0  # SUCCESS!

        # 2. Runway centerline tracking (CRITICAL for ground roll!)
        # Calculate lateral deviation from runway centerline
        rel_pos = np.array([x, y]) - self.runway_start[:2]
        along_runway = np.dot(rel_pos, self.runway_dir[:2])  # Distance along runway
        lateral_dev = abs(np.dot(rel_pos, self.runway_right[:2]))  # Distance from centerline

        # Strong incentive to stay on centerline
        if lateral_dev > self.runway_width / 2:
            reward -= 100.0  # SEVERE penalty for going off runway
        elif lateral_dev < 1.0:
            reward += 10.0  # Big bonus for staying within 1m of centerline
        elif lateral_dev < 5.0:
            reward += 5.0  # Bonus for staying within 5m
        else:
            # Quadratic penalty for deviation (gets worse faster)
            reward -= 2.0 * (lateral_dev ** 2)

        # 3. Maintain runway heading (aligned with centerline tracking)
        heading_error = abs(psi - self.runway_heading)
        heading_error = min(heading_error, 2 * np.pi - heading_error)  # Wrap to [-π, π]
        if heading_error < np.deg2rad(2.0):
            reward += 5.0  # Bonus for being well-aligned
        else:
            reward -= 10.0 * heading_error  # Strong penalty for heading deviation

        # 4. Rudder usage for directional control
        # Encourage using rudder to maintain heading during ground roll
        rudder = action[3]
        # Small penalty for excessive rudder to avoid oscillation
        reward -= 0.5 * abs(rudder)

        # 5. Wings level (no rolling on ground!)
        reward -= 10.0 * abs(phi)  # Stronger penalty for roll

        # 6. Pitch control (CRITICAL: keep nose down during ground roll!)
        # Agent must learn to hold elevator to keep nose wheel on ground
        if theta > 0:  # Nose up is BAD during ground roll
            reward -= 50.0 * theta  # VERY strong penalty for ANY pitch-up
        if theta > np.deg2rad(5.0):  # Excessive pitch
            reward -= 200.0  # Massive penalty
        else:
            # Reward for keeping nose down
            reward += 5.0

        # 6. Stay on ground (should not lift off in this phase)
        if altitude > 2.0:
            reward -= 10.0  # Premature liftoff

        # 6. Throttle usage (should be high for acceleration)
        throttle = (action[0] + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
        if airspeed < self.V_ROTATE:
            # Encourage high throttle during acceleration
            reward += 0.5 * throttle

        # Termination penalties
        if terminated:
            reason = info.get('termination_reason')
            if reason == 'crash':
                reward -= 50.0
            elif reason == 'out_of_bounds':
                reward -= 30.0

        # Survival bonus
        reward += 0.1

        return reward

    def _reward_rotation(self, obs, action, terminated, info):
        """
        Reward function for Phase 2: Rotation

        Goals:
        1. Complete ground roll (reach rotation speed)
        2. Gentle pitch-up (~10°)
        3. Liftoff to 10 ft AGL
        4. Maintain positive climb rate

        Max reward: +100 for successful liftoff
        """
        reward = 0.0

        # Extract state
        x, y, z = obs[StateIndex.X], obs[StateIndex.Y], obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi, theta, psi = obs[StateIndex.PHI], obs[StateIndex.THETA], obs[StateIndex.PSI]
        q = obs[StateIndex.Q]  # Pitch rate

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        climb_rate = -w  # Vertical velocity (positive up)

        on_ground = altitude < 1.0

        if on_ground:
            # Phase 2a: Still on ground - encourage acceleration
            speed_progress = min(airspeed / self.V_ROTATE, 1.0)
            reward += 30.0 * speed_progress

            # Prepare for rotation (gentle pitch-up near rotation speed)
            if airspeed > self.V_ROTATE * 0.9:
                target_pitch = np.deg2rad(5.0)  # Start gentle pitch
                pitch_error = abs(theta - target_pitch)
                reward -= 2.0 * pitch_error
                reward += 5.0  # Bonus for being ready
        else:
            # Phase 2b: Airborne - encourage climb
            # Altitude progress (0 → 10 ft = 3m)
            target_alt = 3.0
            alt_progress = min(altitude / target_alt, 1.0)
            reward += 40.0 * alt_progress

            # Bonus for reaching target altitude
            if altitude >= target_alt:
                reward += 30.0  # SUCCESS!

            # Maintain airspeed during climb
            speed_error = abs(airspeed - 30.0)  # Target ~58 KIAS
            reward -= 0.5 * speed_error

            # Positive climb rate
            if climb_rate > 0:
                reward += 0.3 * climb_rate  # Reward climb rate
            else:
                reward -= 5.0  # Descending after liftoff is bad!

            # Safe pitch angle (5-15° for initial climb)
            target_pitch = np.deg2rad(10.0)
            pitch_error = abs(theta - target_pitch)
            reward -= 1.0 * pitch_error

        # Wings level throughout
        reward -= 3.0 * abs(phi)

        # Runway tracking (while on or near ground)
        if altitude < 10.0:
            rel_pos = np.array([x, y]) - self.runway_start[:2]
            lateral_dev = abs(np.dot(rel_pos, self.runway_right[:2]))
            if lateral_dev < self.runway_width / 2:
                reward += 0.5  # Staying on runway
            else:
                reward -= 2.0 * lateral_dev  # Drifting off

        # Termination penalties
        if terminated:
            reason = info.get('termination_reason')
            if reason == 'crash':
                reward -= 50.0
            elif reason == 'stall':
                reward -= 30.0

        # Survival bonus
        reward += 0.1

        return reward

    def _reward_initial_climb(self, obs, action, terminated, info):
        """
        Reward function for Phase 3: Initial Climb

        Goals:
        1. Climb from liftoff to 500 ft AGL (152.4 m)
        2. Maintain climb speed (75 KIAS = 38 m/s)
        3. Target climb rate (500-700 fpm = 2.5-3.5 m/s)
        4. Wings level, coordinated flight

        Max reward: +100 for reaching 500 ft
        """
        reward = 0.0

        # Extract state
        z = obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi, theta = obs[StateIndex.PHI], obs[StateIndex.THETA]
        p, q, r = obs[StateIndex.P], obs[StateIndex.Q], obs[StateIndex.R]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)
        climb_rate = -w

        target_alt = 152.4  # 500 ft

        # 1. Altitude progress (0 → 500 ft)
        alt_progress = min(altitude / target_alt, 1.0)
        reward += 50.0 * alt_progress

        # Bonus for reaching target
        if altitude >= target_alt:
            reward += 30.0  # SUCCESS!

        # 2. Maintain climb speed (75 KIAS)
        speed_error = abs(airspeed - self.V_CLIMB)
        reward -= 0.3 * speed_error

        # Bonus for good speed
        if speed_error < 3.0:  # Within ~6 knots
            reward += 2.0

        # 3. Climb rate (target: 500-700 fpm = 2.5-3.5 m/s)
        target_climb_rate = 3.0  # m/s (~600 fpm)
        climb_rate_error = abs(climb_rate - target_climb_rate)
        reward -= 0.5 * climb_rate_error

        # Bonus for good climb rate
        if 2.5 <= climb_rate <= 3.5:
            reward += 3.0

        # Penalize descent
        if climb_rate < 0:
            reward -= 10.0

        # 4. Wings level (minimal bank)
        reward -= 2.0 * abs(phi)

        # 5. Safe pitch angle (10-15° for climb)
        target_pitch = np.deg2rad(12.0)
        pitch_error = abs(theta - target_pitch)
        reward -= 1.0 * pitch_error

        # 6. Coordinated flight (low sideslip)
        sideslip = v  # Lateral velocity
        reward -= 0.5 * abs(sideslip)

        # Termination penalties
        if terminated:
            reason = info.get('termination_reason')
            if reason == 'crash':
                reward -= 50.0
            elif reason == 'stall':
                reward -= 40.0
            elif reason == 'attitude_limit':
                reward -= 30.0

        # Survival bonus
        reward += 0.1

        return reward

    def render(self):
        """Render environment (not implemented)"""
        pass

    def close(self):
        """Clean up resources"""
        pass


# Convenience function for creating environment
def make_cessna172_env(task="full_mission", cruise_altitude_ft=3000.0):
    """Create Cessna 172 environment"""
    return Cessna172Env(task=task, cruise_altitude_ft=cruise_altitude_ft)
