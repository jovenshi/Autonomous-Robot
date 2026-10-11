"""Steering for turns and for holding a heading across a junction.

Yaw and targets are in degrees, + = left (counter-clockwise), so a positive
yaw error (target left of current yaw) gives positive (left) steer.
"""
import math


def turn_steer(yaw_deg, target_deg, cfg):
    # P on yaw error, clamped to +/-1, never below TURN_MIN_STEER so the
    # pivot doesn't stall.
    error = target_deg - yaw_deg
    if error == 0:
        return 0.0
    magnitude = min(1.0, max(cfg.TURN_MIN_STEER, abs(cfg.TURN_KP * error)))
    return math.copysign(magnitude, error)


def turn_done(yaw_deg, target_deg, front_conf, cfg):
    # Within TURN_TOL_DEG, or within TURN_HANDOFF_DEG once the front camera
    # sees the new line with conf >= TURN_HANDOFF_CONF.
    error = abs(target_deg - yaw_deg)
    if error < cfg.TURN_TOL_DEG:
        return True
    return error < cfg.TURN_HANDOFF_DEG and front_conf >= cfg.TURN_HANDOFF_CONF


def vision_turn_steer(front, direction, cfg):
    # Fallback when the IMU goes invalid mid-turn: pivot at VISION_TURN_STEER
    # (direction +1 = left, -1 = right) until the front camera sees the line
    # with conf >= TURN_HANDOFF_CONF and heading error under TURN_HANDOFF_DEG.
    steer = direction * cfg.VISION_TURN_STEER
    if front is None:
        return steer, False
    lane = front.lane
    done = (lane.conf >= cfg.TURN_HANDOFF_CONF and lane.heading_err_rad is not None
            and abs(math.degrees(lane.heading_err_rad)) < cfg.TURN_HANDOFF_DEG)
    return steer, done


def hold_steer(yaw_deg, target_deg, cfg):
    # Gentler P (HOLD_KP) to drive straight on road_yaw across a junction.
    steer = cfg.HOLD_KP * (target_deg - yaw_deg)
    return max(-cfg.STEER_MAX, min(cfg.STEER_MAX, steer))
