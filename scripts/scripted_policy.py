import numpy as np

# Cascaded PID scripted controller for FlightEnvRL:
# - Inner loop: attitude/rate damping (aileron/elevator/rudder)
# - Outer loop: altitude/speed to pitch/throttle targets
# - Simple takeoff/climb logic: full throttle to Vr, rotate to climb pitch, then trim to cruise
# Outputs action in env action space: [throttle, trim_elev_cmd, trim_rudder_cmd, elevator_stick, aileron, rudder_stick]


def _wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


class ScriptedPolicy:
    def __init__(self, mission_alt=60.96, cruise_speed=22.0, runway_heading=0.0):
        self.mission_alt = mission_alt
        self.cruise_speed = cruise_speed
        self.runway_heading = runway_heading
        # Approximate env constants (from PROPSHOX)
        self.geofence_z = 150.0
        self.v_stall = 11.0
        self.v_max = 26.6
        # Gains (conservative)
        self.Kp_roll = 0.8
        self.Kd_roll = 0.25
        self.Kp_pitch = 0.8
        self.Kd_pitch = 0.35
        self.Kp_yaw = 0.4
        self.Kd_yaw = 0.1
        self.Kp_h = 0.012  # softer altitude proportional
        self.Kd_h = 0.45   # stronger damping to arrest climb
        self.Kp_v = 0.10
        self.pitch_bias = np.deg2rad(-0.5)
        self.max_cmd_deg = 4.5  # tighter authority to avoid steep climb
        # Throttle trims tuned to hold cruise with minimal excess
        self.throttle_trim = 0.20
        self.min_throttle = 0.10
        self.max_throttle = 0.40
        # Takeoff/climb targets
        self.v_rotate = 12.6  # m/s (~41.3 ft/s)
        self.climb_pitch = np.deg2rad(5.0)
        self.climb_alt = mission_alt  # stay in climb mode until near target

    def act(self, obs, info=None):
        if not isinstance(obs, np.ndarray):
            return np.zeros(6, dtype=np.float32)
        # obs layout: pos_norm(3), vel_norm(3), attitude(3), rates_norm(3),
        # airspeed_norm, altitude_norm, vz_norm, vz_acc_norm, distance_norm, heading_norm,
        # trim_norm, rudder_trim_norm, phase_norm, alt_int_norm, speed_int_norm, last_action(6)
        pos_norm = obs[0:3]
        vel_norm = obs[3:6]
        attitude = obs[6:9]
        roll, pitch, yaw = attitude
        rates_norm = obs[9:12]
        p = rates_norm[0] * np.deg2rad(60)
        q = rates_norm[1] * np.deg2rad(60)
        r = rates_norm[2] * np.deg2rad(60)
        airspeed_norm = obs[12]
        altitude_norm = obs[13]
        heading_error_norm = obs[17]

        # Estimate denormalized values
        altitude = float(altitude_norm * self.geofence_z)
        heading_error = heading_error_norm * np.pi
        airspeed = float(self.v_stall + airspeed_norm * (self.v_max - self.v_stall))

        # Cruise/hold logic only (no takeoff/climb branches for warm-start stability)
        max_cmd_deg = self.max_cmd_deg
        alt_err = self.mission_alt - altitude
        vz = obs[14] * 15.0  # denormalized vertical speed (approx)
        pitch_target = self.pitch_bias + self.Kp_h * alt_err - self.Kd_h * vz
        if vz < -2.0:
            pitch_target += np.deg2rad(2.0)  # mild dive arrest
        pitch_target = np.clip(pitch_target, np.deg2rad(-3.0), np.deg2rad(4.0))
        speed_err = self.cruise_speed - airspeed
        throttle_cmd = self.throttle_trim + self.Kp_v * speed_err
        throttle = np.clip(throttle_cmd, self.min_throttle, self.max_throttle)
        # Back off power when above target or overspeeding
        if airspeed > self.cruise_speed:
            throttle = self.min_throttle
        if altitude > self.mission_alt + 2.0:
            throttle = min(throttle, self.throttle_trim - 0.02)
        if abs(alt_err) < 3.0:
            throttle = min(throttle, self.throttle_trim - 0.01)

        manual_takeoff = False  # forced cruise mode
        # Inner loop: attitude/rate PID
        roll_cmd = -(self.Kp_roll * roll + self.Kd_roll * p)
        pitch_err = pitch_target - pitch
        pitch_cmd = -(self.Kp_pitch * pitch_err - self.Kd_pitch * q)
        yaw_cmd = self.Kp_yaw * heading_error - self.Kd_yaw * r

        # Limit to surface bounds
        max_cmd = np.deg2rad(max_cmd_deg)
        aileron = np.clip(roll_cmd, -max_cmd, max_cmd)
        elevator = np.clip(-pitch_cmd, -max_cmd, max_cmd)  # negative => trailing edge up (nose-up)
        rudder = np.clip(yaw_cmd, -max_cmd, max_cmd)

        # Normalize to [-1,1] for action space
        def norm_cmd(val):
            return float(np.clip(val / max_cmd, -1.0, 1.0))

        aileron_cmd = norm_cmd(aileron)
        elevator_cmd = norm_cmd(elevator)
        rudder_cmd = norm_cmd(rudder)

        trim_elev_cmd = 0.0
        trim_rudder_cmd = 0.0

        action = np.array([throttle, trim_elev_cmd, trim_rudder_cmd, elevator_cmd, aileron_cmd, rudder_cmd], dtype=np.float32)
        return action
