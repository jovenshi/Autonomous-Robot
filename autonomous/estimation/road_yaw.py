"""Keeps the IMU anchored to the road.

On straight road with a confident lane lock, road_yaw = IMU yaw + camera
heading error is blended in every loop. Turn and crossing targets come from
road_yaw, so gyro drift only matters since the last anchor.

Yaw values are the IMU's unwrapped degrees (+ = left), so no wrap handling
is needed.
"""
import math

from estimation.lane_fusion import is_fresh
from robot_types import RoadYaw, State

_ANCHOR_STATES = (State.LANE_FOLLOW, State.APPROACH)


class RoadYawTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self._yaw_deg = None
        self._t_anchor = 0.0

    def update(self, imu, lane, state, now):
        # Blends yaw + heading error into road_yaw by ROAD_YAW_ALPHA when the
        # robot is on straight road with a good lane lock and turning slowly.
        if not self._can_anchor(imu, lane, state, now):
            return
        measured = imu.yaw_deg + math.degrees(lane.heading_err_rad)
        if self._yaw_deg is None:
            self._yaw_deg = measured
        else:
            self._yaw_deg += self.cfg.ROAD_YAW_ALPHA * (measured - self._yaw_deg)
        self._t_anchor = now

    def after_turn(self, direction):
        # direction +1 = left, -1 = right. New road = old +/- TURN_ANGLE_DEG,
        # used until the camera re-anchors it.
        if self._yaw_deg is not None:
            self._yaw_deg += direction * self.cfg.TURN_ANGLE_DEG

    def get(self):
        # valid is False until the first anchor.
        if self._yaw_deg is None:
            return RoadYaw(yaw_deg=0.0, t_anchor=0.0, valid=False)
        return RoadYaw(yaw_deg=self._yaw_deg, t_anchor=self._t_anchor, valid=True)

    def _can_anchor(self, imu, lane, state, now):
        cfg = self.cfg
        return (state in _ANCHOR_STATES
                and is_fresh(imu, now, cfg) and imu.valid
                and abs(imu.yaw_rate_dps) < cfg.ROAD_YAW_MAX_RATE_DPS
                and lane is not None and lane.heading_err_rad is not None
                and lane.conf >= cfg.ROAD_YAW_MIN_CONF)
