"""Lane centering: steer from lateral error, heading error and yaw rate.

steer = KP_LAT * lateral + KH * heading - KD_YAW * yaw_rate, clamped to
STEER_MAX, then smoothed. Positive errors call for positive (left) steer;
a positive yaw rate means already turning left, so it damps left steer.

With no lane measurement the last steer is held; target_speed slows the
robot (conf 0) and _check_safety stops it after LANE_LOST_TIMEOUT_S.
"""


class LaneController:
    def __init__(self, cfg):
        self.cfg = cfg
        self._steer = 0.0

    def update(self, est, yaw_rate_dps, dt):
        # dt is accepted for the design's signature; the smoothing is per
        # loop at the fixed CONTROL_HZ, so it is not used.
        if est is None or est.lateral_err_m is None:
            return self._steer
        cfg = self.cfg
        heading = est.heading_err_rad if est.heading_err_rad is not None else 0.0
        raw = cfg.KP_LAT * est.lateral_err_m + cfg.KH * heading - cfg.KD_YAW * yaw_rate_dps
        raw = max(-cfg.STEER_MAX, min(cfg.STEER_MAX, raw))
        self._steer = cfg.STEER_SMOOTH * self._steer + (1.0 - cfg.STEER_SMOOTH) * raw
        return self._steer

    def reset(self):
        # Clears the smoothing filter when a lane state starts.
        self._steer = 0.0
