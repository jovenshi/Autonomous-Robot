"""Differential-drive mixing: DriveCmd -> MotorCmd.

left = v - STEER_GAIN * steer, right = v + STEER_GAIN * steer, rescaled so
neither exceeds 1. Steer + = left, so the left wheel slows. Speed 0 with
steer pivots about the rear-axle midpoint.

Before mixing, steer gets two corrections measured by tools/steer_test.py
(both do nothing until measured):
  * STEER_TRIM: the steer that makes the robot drive straight at each speed
    (motor mismatch), {speed: trim}, interpolated; 0 when stopped, so a
    robot at rest never creeps round.
  * STEER_DEADBAND: the extra steer the robot needs before it actually
    turns (tyre scrub). It is blended in over STEER_DEADBAND_RAMP, so steer
    near 0 stays continuous and noise around 0 can't make it twitch.
"""
import math

import numpy as np

from robot_types import MotorCmd


def mix(cmd, cfg):
    steer = compensate(cmd.steer, cmd.speed, cfg)
    left = cmd.speed - cfg.STEER_GAIN * steer
    right = cmd.speed + cfg.STEER_GAIN * steer
    scale = max(1.0, abs(left), abs(right))
    return MotorCmd(left=left / scale, right=right / scale)


def compensate(steer, speed, cfg):
    # steer + trim for this speed, then lifted past the steering deadband.
    steer = steer + steer_trim(speed, cfg)
    deadband = cfg.STEER_DEADBAND
    if deadband > 0 and steer != 0:
        ramp = min(1.0, abs(steer) / cfg.STEER_DEADBAND_RAMP)
        steer += math.copysign(deadband * ramp, steer)
    return steer


def steer_trim(speed, cfg):
    # STEER_TRIM interpolated for this speed; 0 when stopped or unmeasured.
    # Between 0 and the lowest measured speed it fades linearly to 0.
    if not cfg.STEER_TRIM or speed <= 0:
        return 0.0
    speeds = [0.0] + sorted(cfg.STEER_TRIM)
    trims = [0.0] + [cfg.STEER_TRIM[s] for s in speeds[1:]]
    return float(np.interp(speed, speeds, trims))
