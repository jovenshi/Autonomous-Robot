"""Lane centering: steer from lateral error, heading error and yaw rate.

steer = KP_LAT * lateral + KH * heading - KD_YAW * yaw_rate, clamped to
STEER_MAX, then smoothed. Positive errors call for positive (left) steer;
a positive yaw rate means already turning left, so it damps left steer.

With LANE_GAIN_SPEED_SCALING on, the lateral and heading terms are scaled
by CRUISE_SPEED / max(speed, LANE_GAIN_MIN_SPEED): the robot's sideways
motion is speed x heading, so fixed gains are gentler at low speed. Off by
default: turn it on once the gains are tuned in tools/lane_sim.py.

With no lane measurement the last steer is held; target_speed slows the
robot (conf 0) and _check_safety stops it after LANE_LOST_TIMEOUT_S.
"""


class LaneController:
    def __init__(self, cfg):
        self.cfg = cfg
        self._steer = 0.0

    def update(self, est, yaw_rate_dps, dt, speed=None):
        # dt is accepted for the design's signature; the smoothing is per
        # loop at the fixed CONTROL_HZ, so it is not used. speed (0-1) is
        # only used with LANE_GAIN_SPEED_SCALING.
        if est is None or est.lateral_err_m is None:
            return self._steer
        cfg = self.cfg
        heading = est.heading_err_rad if est.heading_err_rad is not None else 0.0
        scale = speed_scale(speed, cfg)
        raw = scale * (cfg.KP_LAT * est.lateral_err_m + cfg.KH * heading) \
            - cfg.KD_YAW * yaw_rate_dps
        raw = max(-cfg.STEER_MAX, min(cfg.STEER_MAX, raw))
        self._steer = cfg.STEER_SMOOTH * self._steer + (1.0 - cfg.STEER_SMOOTH) * raw
        return self._steer

    def reset(self):
        # Clears the smoothing filter when a lane state starts.
        self._steer = 0.0


def speed_scale(speed, cfg):
    # Gain factor for the lateral and heading terms (1 when scaling is off).
    if not cfg.LANE_GAIN_SPEED_SCALING or speed is None:
        return 1.0
    return cfg.CRUISE_SPEED / max(speed, cfg.LANE_GAIN_MIN_SPEED)
