"""Fuses the front and side lane readings into one estimate at the rear axle.

Lateral error is a confidence-weighted average of both cameras. Heading
error comes from the front camera only (the side camera sees one edge, not
a direction). Either camera may be missing or stale; a reading is used only
when it is fresh, measured something and has conf > 0.
"""
import math

from robot_types import LaneEstimate


def is_fresh(obj, now, cfg):
    # True if obj exists and its t is within MAX_OBS_AGE_S of now.
    return obj is not None and now - obj.t <= cfg.MAX_OBS_AGE_S


def side_to_axle(offset_m, heading_err_rad, cfg):
    # Moves the side camera's offset back to the rear axle:
    # offset - SIDE_CAM_X_M * sin(heading).
    if cfg.SIDE_CAM_X_M is None:
        raise ValueError("SIDE_CAM_X_M is not set in config.py: run tools/calibrate.py")
    return offset_m - cfg.SIDE_CAM_X_M * math.sin(heading_err_rad)


def fuse_lane(front, side, now, cfg):
    # front: FrontResult or None; side: SideResult or None.
    front_obs = _usable_front(front, now, cfg)
    side_obs = _usable_side(side, now, cfg)
    heading = front_obs.heading_err_rad if front_obs is not None else None

    readings = []                                   # (lateral at axle, conf, t)
    if front_obs is not None:
        readings.append((front_obs.lateral_m, front_obs.conf, front_obs.t))
    if side_obs is not None:
        side_lateral = side_to_axle(side_obs.offset_m, heading or 0.0, cfg)
        readings.append((side_lateral, side_obs.conf, side_obs.t))
    if not readings:
        return LaneEstimate(None, None, 0.0, None, now)

    total_conf = sum(conf for _, conf, _ in readings)
    lateral = sum(lat * conf for lat, conf, _ in readings) / total_conf
    conf = max(conf for _, conf, _ in readings)
    t = max(t for _, _, t in readings)
    return LaneEstimate(lateral, heading, conf, _cam_gap(readings, cfg), t)


def _usable_front(front, now, cfg):
    # The front lane observation if fresh and measured, else None.
    if not is_fresh(front, now, cfg):
        return None
    lane = front.lane
    if lane.lateral_m is None or lane.conf <= 0:
        return None
    return lane


def _usable_side(side, now, cfg):
    # The side lane observation if fresh and measured, else None.
    if not is_fresh(side, now, cfg):
        return None
    obs = side.side
    if obs.offset_m is None or obs.conf <= 0:
        return None
    return obs


def _cam_gap(readings, cfg):
    # Front minus side lateral, only when both are confident.
    if len(readings) != 2:
        return None
    (front_lat, front_conf, _), (side_lat, side_conf, _) = readings
    if min(front_conf, side_conf) < cfg.FUSE_GAP_MIN_CONF:
        return None
    return front_lat - side_lat
