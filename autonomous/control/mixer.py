"""Differential-drive mixing: DriveCmd -> MotorCmd.

left = v - STEER_GAIN * steer, right = v + STEER_GAIN * steer, rescaled so
neither exceeds 1. Steer + = left, so the left wheel slows. Speed 0 with
steer pivots about the rear-axle midpoint.
"""
from robot_types import MotorCmd


def mix(cmd, cfg):
    left = cmd.speed - cfg.STEER_GAIN * cmd.steer
    right = cmd.speed + cfg.STEER_GAIN * cmd.steer
    scale = max(1.0, abs(left), abs(right))
    return MotorCmd(left=left / scale, right=right / scale)
