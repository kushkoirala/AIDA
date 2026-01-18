"""
Teardrop Approach Environment for KHUT RWY 13

TRAINING STRATEGY
=================
1. BC warm-start: Train NN on triangle approach demos (RWY 31)
   - NN learns: flight dynamics, approach, glideslope, landing

2. RL fine-tune: Test on THIS environment (RWY 13 - opposite direction)
   - Aircraft starts heading toward KHUT but needs to land opposite direction
   - NN must DISCOVER teardrop/course reversal maneuver through RL
   - Only reward is: land on RWY 13

The NN is NOT told what a teardrop is - it figures it out!

Classical controller: lands RWY 31 (triangle approach)
Neural network goal:  lands RWY 13 (must discover teardrop)

Author: Kushal Koirala (with Claude Code)
Date: January 2025
"""

import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gpu-flight-dynamics" / "python"))

from flight_dynamics import FlightSimulator, StateIndex, ControlIndex, STATE_DIM, CONTROL_DIM
from aircraft_database import get_aircraft


# Unit conversions
FT_TO_M = 0.3048
M_TO_FT = 3.28084
NM_TO_M = 1852.0
M_TO_NM = 1 / 1852.0
KTS_TO_MS = 0.514444
MS_TO_KTS = 1 / 0.514444


class TeardropApproachEnv(gym.Env):
    """
    Environment for training teardrop approach to KHUT RWY 13.

    The aircraft must learn to:
    1. Navigate toward KHUT from starting position
    2. Execute teardrop entry (course reversal)
    3. Intercept final approach for RWY 13
    4. Land on RWY 13

    Observation Space: 12D state + 6D goal info = 18D
        [x, y, z, u, v, w, phi, theta, psi, p, q, r,
         dist_to_runway, bearing_to_runway, altitude_error,
         glideslope_error, localizer_error, airspeed_error]

    Action Space: 4D control vector (normalized -1 to 1)
        [throttle, aileron, elevator, rudder]
    """

    metadata = {"render_modes": ["none"], "render_fps": 50}

    def __init__(self,
                 dt: float = 0.02,
                 max_episode_steps: int = 6000,  # 2 minutes
                 start_mode: str = "cruise"):  # "cruise" or "approach"
        """
        Initialize teardrop approach environment.

        Args:
            dt: Simulation timestep
            max_episode_steps: Maximum steps per episode
            start_mode: Where to start the aircraft
                - "cruise": Start en-route, must fly full approach
                - "approach": Start closer, focus on teardrop maneuver
        """
        super().__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.start_mode = start_mode
        self.step_count = 0

        # KHUT airport parameters
        # KHUT is at approximately (52800m, -21300m) from SN65 origin
        self.khut_x = 52800.0  # meters
        self.khut_y = -21300.0  # meters
        self.field_elevation = 0.0  # meters (simplified)

        # RWY 13 parameters (opposite of RWY 31)
        # RWY 31 heading is 314°, so RWY 13 heading is 134°
        self.rwy13_heading = np.deg2rad(134.0)  # Landing direction
        self.rwy31_heading = np.deg2rad(314.0)  # Opposite

        # Runway threshold for RWY 13
        # RWY 13 threshold is at the northwest end of the runway
        rwy_length = 2134.0  # meters (~7000 ft)
        self.rwy13_threshold_x = self.khut_x - (rwy_length/2) * np.cos(self.rwy13_heading)
        self.rwy13_threshold_y = self.khut_y - (rwy_length/2) * np.sin(self.rwy13_heading)

        # Touchdown aimpoint (1000ft past threshold)
        aimpoint_dist = 300.0  # meters
        self.aimpoint_x = self.rwy13_threshold_x + aimpoint_dist * np.cos(self.rwy13_heading)
        self.aimpoint_y = self.rwy13_threshold_y + aimpoint_dist * np.sin(self.rwy13_heading)

        # Teardrop entry parameters
        # Teardrop fix: ~5nm from threshold on final approach course
        teardrop_fix_dist = 5 * NM_TO_M  # 5nm
        self.teardrop_fix_x = self.rwy13_threshold_x - teardrop_fix_dist * np.cos(self.rwy13_heading)
        self.teardrop_fix_y = self.rwy13_threshold_y - teardrop_fix_dist * np.sin(self.rwy13_heading)

        # Outbound leg: 30° offset from inbound course, for 1 minute
        self.teardrop_offset_angle = np.deg2rad(30.0)
        self.outbound_heading = self.rwy13_heading + np.pi - self.teardrop_offset_angle

        # Aircraft performance
        self.v_approach = 65 * KTS_TO_MS  # 65 knots approach speed
        self.v_touchdown = 50 * KTS_TO_MS  # 50 knots touchdown
        self.v_stall = 45 * KTS_TO_MS  # 45 knots stall
        self.v_max = 150 * KTS_TO_MS  # 150 knots max

        # Pattern altitude
        self.pattern_altitude = 457.0  # 1500 ft AGL
        self.glideslope_angle = np.deg2rad(3.0)  # 3° glideslope

        # Load aircraft
        self.aircraft = get_aircraft('cessna172')
        self.params = self._config_to_params(self.aircraft)
        self.sim = FlightSimulator(n_instances=1, params=self.params, dt=dt)

        # Observation space: base state (12) + goal info (6) = 18
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

        # Phase tracking
        self.phases = ['ENROUTE', 'TEARDROP_ENTRY', 'OUTBOUND', 'TEARDROP_TURN',
                       'INBOUND', 'FINAL', 'SHORT_FINAL', 'LANDING', 'LANDED']
        self.phase = 'ENROUTE'
        self.phase_idx = 0

        # Episode tracking
        self.episode_reward = 0.0
        self.best_distance_to_runway = float('inf')

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

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)

        self.step_count = 0
        self.episode_reward = 0.0
        self.phase = 'ENROUTE'
        self.phase_idx = 0
        self.best_distance_to_runway = float('inf')

        initial_state = np.zeros((1, STATE_DIM), dtype=np.float32)

        if self.start_mode == "cruise":
            # Start 15nm from KHUT, heading toward it at cruise altitude
            start_dist = 15 * NM_TO_M
            # Coming from southwest (similar to SN65 direction)
            start_bearing = np.deg2rad(225.0)  # Southwest of KHUT

            initial_state[0, StateIndex.X] = self.khut_x + start_dist * np.cos(start_bearing)
            initial_state[0, StateIndex.Y] = self.khut_y + start_dist * np.sin(start_bearing)
            initial_state[0, StateIndex.Z] = -1676.0  # 5500 ft

            # Heading toward KHUT
            initial_heading = start_bearing + np.pi  # Opposite of bearing = toward KHUT
            initial_state[0, StateIndex.PSI] = initial_heading

            # Cruise speed
            initial_state[0, StateIndex.U] = 56.0  # ~110 knots

        else:  # "approach" - start closer, focused on teardrop
            # Start 8nm from threshold, need to do teardrop
            start_dist = 8 * NM_TO_M

            # Position: approaching from the "wrong" side (need teardrop)
            # Coming from direction that requires course reversal
            initial_state[0, StateIndex.X] = self.teardrop_fix_x - 3 * NM_TO_M * np.cos(self.rwy13_heading)
            initial_state[0, StateIndex.Y] = self.teardrop_fix_y - 3 * NM_TO_M * np.sin(self.rwy13_heading)
            initial_state[0, StateIndex.Z] = -self.pattern_altitude

            # Heading roughly toward the fix
            initial_state[0, StateIndex.PSI] = self.rwy13_heading
            initial_state[0, StateIndex.U] = 40.0  # ~78 knots

        # Add small random perturbations for robustness
        if seed is not None:
            rng = np.random.default_rng(seed)
            initial_state[0, StateIndex.X] += rng.uniform(-500, 500)
            initial_state[0, StateIndex.Y] += rng.uniform(-500, 500)
            initial_state[0, StateIndex.Z] += rng.uniform(-50, 50)
            initial_state[0, StateIndex.PSI] += rng.uniform(-0.1, 0.1)

        self.sim.reset(initial_state=initial_state)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute one timestep."""
        self.step_count += 1

        # Denormalize and apply action
        controls = self._denormalize_action(action)
        self.sim.set_controls(controls)
        self.sim.step()

        # Get observation
        obs = self._get_observation()

        # Update phase
        self._update_phase(obs)

        # Compute reward
        terminated, term_info = self._check_termination(obs)
        reward = self._compute_reward(obs, action, terminated, term_info)

        truncated = self.step_count >= self.max_episode_steps

        self.episode_reward += reward

        info = self._get_info()
        info.update(term_info)

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Get observation: state + goal-relative info."""
        states = self.sim.get_states()
        try:
            state = states.get()[0] if hasattr(states, 'get') else states[0]
        except:
            state = np.array(states)[0]

        # Extract position and compute goal-relative info
        x, y, z = state[StateIndex.X], state[StateIndex.Y], state[StateIndex.Z]
        u, v, w = state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]
        psi = state[StateIndex.PSI]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)

        # Distance and bearing to runway threshold
        dx = self.rwy13_threshold_x - x
        dy = self.rwy13_threshold_y - y
        dist_to_runway = np.sqrt(dx**2 + dy**2)
        bearing_to_runway = np.arctan2(dy, dx)

        # Relative bearing (how much to turn)
        relative_bearing = self._normalize_angle(bearing_to_runway - psi)

        # Altitude error (from glideslope)
        glideslope_alt = dist_to_runway * np.tan(self.glideslope_angle)
        altitude_error = altitude - glideslope_alt

        # Localizer error (cross-track from final approach course)
        # Project position onto perpendicular to runway heading
        rwy_perp_x = -np.sin(self.rwy13_heading)
        rwy_perp_y = np.cos(self.rwy13_heading)
        localizer_error = (x - self.aimpoint_x) * rwy_perp_x + (y - self.aimpoint_y) * rwy_perp_y

        # Airspeed error
        target_speed = self.v_approach if altitude > 50 else self.v_touchdown
        airspeed_error = airspeed - target_speed

        # Normalize goal info
        goal_info = np.array([
            dist_to_runway / 10000.0,  # Normalize by ~10km
            relative_bearing / np.pi,   # Normalize to [-1, 1]
            altitude_error / 500.0,     # Normalize by 500m
            np.clip(altitude_error / 100.0, -1, 1),  # Glideslope normalized
            np.clip(localizer_error / 500.0, -1, 1),  # Localizer normalized
            airspeed_error / 20.0,      # Normalize by 20 m/s
        ], dtype=np.float32)

        # Combine state + goal info
        obs = np.concatenate([state, goal_info])

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

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _update_phase(self, obs):
        """Update flight phase based on current state."""
        x, y, z = obs[StateIndex.X], obs[StateIndex.Y], obs[StateIndex.Z]
        psi = obs[StateIndex.PSI]
        altitude = -z

        dist_to_threshold = np.sqrt((x - self.rwy13_threshold_x)**2 +
                                    (y - self.rwy13_threshold_y)**2)
        dist_to_fix = np.sqrt((x - self.teardrop_fix_x)**2 +
                              (y - self.teardrop_fix_y)**2)

        # Heading relative to final approach course
        heading_error = abs(self._normalize_angle(psi - self.rwy13_heading))

        if self.phase == 'ENROUTE':
            if dist_to_fix < 3 * NM_TO_M:  # Within 3nm of teardrop fix
                self.phase = 'TEARDROP_ENTRY'
                self.phase_idx = 1

        elif self.phase == 'TEARDROP_ENTRY':
            # Check if starting outbound turn
            outbound_heading_error = abs(self._normalize_angle(psi - self.outbound_heading))
            if outbound_heading_error < np.deg2rad(30):
                self.phase = 'OUTBOUND'
                self.phase_idx = 2

        elif self.phase == 'OUTBOUND':
            # After ~1 minute outbound or certain distance, start turn back
            if dist_to_fix > 4 * NM_TO_M:
                self.phase = 'TEARDROP_TURN'
                self.phase_idx = 3

        elif self.phase == 'TEARDROP_TURN':
            # Check if aligned with inbound course
            if heading_error < np.deg2rad(20):
                self.phase = 'INBOUND'
                self.phase_idx = 4

        elif self.phase == 'INBOUND':
            if dist_to_threshold < 3 * NM_TO_M:
                self.phase = 'FINAL'
                self.phase_idx = 5

        elif self.phase == 'FINAL':
            if dist_to_threshold < 1 * NM_TO_M:
                self.phase = 'SHORT_FINAL'
                self.phase_idx = 6

        elif self.phase == 'SHORT_FINAL':
            if altitude < 15:  # 50 ft
                self.phase = 'LANDING'
                self.phase_idx = 7

        elif self.phase == 'LANDING':
            if altitude < 1:
                self.phase = 'LANDED'
                self.phase_idx = 8

    def _check_termination(self, obs):
        """Check if episode should terminate."""
        x, y, z = obs[StateIndex.X], obs[StateIndex.Y], obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi, theta = obs[StateIndex.PHI], obs[StateIndex.THETA]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)

        info = {'termination_reason': None}

        # Crash
        if altitude < -1:  # Below ground
            info['termination_reason'] = 'crash'
            return True, info

        # Stall (only when airborne)
        if altitude > 10 and airspeed < self.v_stall:
            info['termination_reason'] = 'stall'
            return True, info

        # Overspeed
        if airspeed > self.v_max:
            info['termination_reason'] = 'overspeed'
            return True, info

        # Excessive attitude
        if abs(phi) > np.deg2rad(60) or abs(theta) > np.deg2rad(30):
            info['termination_reason'] = 'attitude_limit'
            return True, info

        # Out of bounds (too far from KHUT)
        dist_to_khut = np.sqrt((x - self.khut_x)**2 + (y - self.khut_y)**2)
        if dist_to_khut > 50 * NM_TO_M:  # 50nm
            info['termination_reason'] = 'out_of_bounds'
            return True, info

        # Successful landing
        if self.phase == 'LANDED':
            # Check if on runway
            dist_to_threshold = np.sqrt((x - self.rwy13_threshold_x)**2 +
                                        (y - self.rwy13_threshold_y)**2)
            if dist_to_threshold < 1000:  # Within 1km of threshold
                info['termination_reason'] = 'landed_success'
            else:
                info['termination_reason'] = 'landed_off_runway'
            return True, info

        return False, info

    def _compute_reward(self, obs, action, terminated, term_info):
        """
        Compute reward - MINIMAL SHAPING, let RL discover the maneuver.

        Key principle: Don't tell the NN HOW to do teardrop.
        Only reward:
        - Staying alive (not crashing/stalling)
        - Landing on the runway

        The NN must figure out that it needs to reverse course.
        """
        reward = 0.0

        x, y, z = obs[StateIndex.X], obs[StateIndex.Y], obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]
        phi = obs[StateIndex.PHI]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)

        # Distance to runway aimpoint
        dist_to_aimpoint = np.sqrt((x - self.aimpoint_x)**2 + (y - self.aimpoint_y)**2)

        # 1. SPARSE PROGRESS REWARD - only when getting significantly closer
        if dist_to_aimpoint < self.best_distance_to_runway - 100:  # 100m threshold
            reward += 1.0
            self.best_distance_to_runway = dist_to_aimpoint

        # 2. BASIC FLIGHT STABILITY (learned from BC, reinforce here)
        # Small reward for controlled flight
        if abs(phi) < np.deg2rad(45):  # Not in extreme bank
            reward += 0.05
        if 20 < airspeed < 80:  # Reasonable speed range
            reward += 0.05

        # 3. TERMINATION REWARDS - This is where the real learning signal is
        if terminated:
            reason = term_info.get('termination_reason')

            if reason == 'landed_success':
                # HUGE reward for landing on RWY 13!
                reward += 1000.0

                # Bonus for good touchdown
                if airspeed < 30:  # Good touchdown speed
                    reward += 200.0
                if abs(phi) < np.deg2rad(5):  # Wings level
                    reward += 100.0

            elif reason == 'landed_off_runway':
                # Partial credit - at least it landed
                reward += 100.0

            elif reason == 'crash':
                reward -= 100.0

            elif reason == 'stall':
                reward -= 100.0

            elif reason == 'attitude_limit':
                reward -= 50.0

            elif reason == 'out_of_bounds':
                reward -= 20.0

        # 4. SMALL SURVIVAL BONUS
        reward += 0.01

        return reward

    def _get_info(self):
        """Get info dict."""
        obs = self._get_observation()
        x, y, z = obs[StateIndex.X], obs[StateIndex.Y], obs[StateIndex.Z]
        u, v, w = obs[StateIndex.U], obs[StateIndex.V], obs[StateIndex.W]

        altitude = -z
        airspeed = np.sqrt(u**2 + v**2 + w**2)

        dist_to_threshold = np.sqrt((x - self.rwy13_threshold_x)**2 +
                                    (y - self.rwy13_threshold_y)**2)

        return {
            'altitude': float(altitude),
            'altitude_ft': float(altitude * M_TO_FT),
            'airspeed': float(airspeed),
            'airspeed_kts': float(airspeed * MS_TO_KTS),
            'dist_to_runway_nm': float(dist_to_threshold * M_TO_NM),
            'phase': self.phase,
            'phase_idx': self.phase_idx,
            'episode_reward': float(self.episode_reward),
        }

    def render(self):
        pass

    def close(self):
        pass
