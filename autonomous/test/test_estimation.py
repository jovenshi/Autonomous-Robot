"""Tests for estimation/ (lane_fusion, road_yaw, health).

Run from the robot/ folder:  python -m pytest tests
"""
import math
import os
import sys

import numpy as np
import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from estimation.health import check_health
from estimation.lane_fusion import fuse_lane, is_fresh, side_to_axle
from estimation.road_yaw import RoadYawTracker
from robot_types import (FrontLaneObs, FrontResult, HealthWarning as W, ImuSample,
                         LaneEstimate, RoadYaw, SideLaneObs, SideResult, State)

VIEW = np.zeros((2, 2, 3), np.uint8)
NOW = 100.0


def front(lateral=0.02, heading=0.1, conf=0.9, t=NOW, mean_v=120.0):
    lane = FrontLaneObs(lateral, heading, conf, (), None, t, 1)
    return FrontResult(VIEW, lane, None, None, mean_v, t, 1)


def side(offset=0.01, conf=0.9, t=NOW, mean_v=120.0):
    obs = SideLaneObs(offset, offset is not None, False, conf, 100.0, t, 1)
    return SideResult(VIEW, obs, mean_v, t, 1)


def imu(yaw=10.0, rate=0.0, valid=True, calib=3, drift=0.0, t=NOW):
    return ImuSample(yaw, rate, valid, calib, drift, t)


@pytest.fixture(autouse=True)
def side_cam_x(monkeypatch):
    monkeypatch.setattr(config, "SIDE_CAM_X_M", 0.10)


# ---------------------------------------------------------------- lane_fusion

def test_is_fresh():
    assert is_fresh(front(t=NOW - 0.1), NOW, config)
    assert not is_fresh(front(t=NOW - 0.2), NOW, config)
    assert not is_fresh(None, NOW, config)


def test_side_to_axle_sign():
    # pointing right: the camera ahead of the axle is further right than the axle
    assert side_to_axle(0.05, 0.2, config) == pytest.approx(0.05 - 0.10 * math.sin(0.2))


def test_side_to_axle_needs_calibration(monkeypatch):
    monkeypatch.setattr(config, "SIDE_CAM_X_M", None)
    with pytest.raises(ValueError):
        side_to_axle(0.0, 0.0, config)


def test_front_only():
    est = fuse_lane(front(), None, NOW, config)
    assert (est.lateral_err_m, est.heading_err_rad, est.conf) == (0.02, 0.1, 0.9)
    assert est.cam_gap_m is None


def test_side_only_has_no_heading():
    est = fuse_lane(None, side(offset=0.03, conf=0.5), NOW, config)
    assert est.lateral_err_m == pytest.approx(0.03)
    assert est.heading_err_rad is None and est.conf == 0.5


def test_weighted_by_conf_with_gap():
    f, s = front(lateral=0.04, heading=0.0, conf=0.9), side(offset=0.01, conf=0.6)
    est = fuse_lane(f, s, NOW, config)
    assert est.lateral_err_m == pytest.approx((0.04 * 0.9 + 0.01 * 0.6) / 1.5)
    assert est.cam_gap_m == pytest.approx(0.03)
    assert est.conf == 0.9


def test_gap_needs_both_confident():
    est = fuse_lane(front(conf=0.9), side(conf=0.3), NOW, config)
    assert est.cam_gap_m is None


def test_stale_and_unmeasured_readings_are_ignored():
    est = fuse_lane(front(t=NOW - 1.0), side(offset=None), NOW, config)
    assert est.lateral_err_m is None and est.conf == 0.0 and est.t == NOW


# ---------------------------------------------------------------- road_yaw

def lane(heading=0.0, conf=0.9):
    return LaneEstimate(0.0, heading, conf, None, NOW)


def test_first_anchor_then_blend():
    tracker = RoadYawTracker(config)
    assert not tracker.get().valid
    tracker.update(imu(yaw=10.0), lane(heading=math.radians(2)), State.LANE_FOLLOW, NOW)
    assert tracker.get().yaw_deg == pytest.approx(12.0) and tracker.get().valid
    tracker.update(imu(yaw=20.0, t=NOW + 1), lane(heading=0.0), State.LANE_FOLLOW, NOW + 1)
    assert tracker.get().yaw_deg == pytest.approx(12.0 + config.ROAD_YAW_ALPHA * 8.0)
    assert tracker.get().t_anchor == NOW + 1


@pytest.mark.parametrize("kwargs", [
    dict(state=State.TURN),
    dict(imu_s=imu(rate=50.0)),
    dict(imu_s=imu(valid=False)),
    dict(lane_e=lane(conf=0.3)),
    dict(lane_e=LaneEstimate(0.0, None, 0.9, None, NOW)),
])
def test_no_anchor_when_conditions_fail(kwargs):
    tracker = RoadYawTracker(config)
    tracker.update(kwargs.get("imu_s", imu()), kwargs.get("lane_e", lane()),
                   kwargs.get("state", State.LANE_FOLLOW), NOW)
    assert not tracker.get().valid


def test_stale_imu_does_not_anchor():
    tracker = RoadYawTracker(config)
    tracker.update(imu(t=NOW - 1.0), lane(), State.LANE_FOLLOW, NOW)
    assert not tracker.get().valid


def test_after_turn_shifts_road_direction():
    tracker = RoadYawTracker(config)
    tracker.update(imu(yaw=5.0), lane(), State.APPROACH, NOW)
    tracker.after_turn(+1)
    assert tracker.get().yaw_deg == pytest.approx(95.0)
    tracker.after_turn(-1)
    assert tracker.get().yaw_deg == pytest.approx(5.0)


# ---------------------------------------------------------------- health

def health(**kw):
    args = dict(front=front(), side=None, imu=imu(), lane=lane(),
                road_yaw=RoadYaw(0.0, NOW, True), fps={"front": 30.0}, overruns=0,
                motor_error=False, now=NOW, cfg=config)
    args.update(kw)
    return check_health(**args)


def test_healthy_system_has_no_warnings():
    assert health() == []


@pytest.mark.parametrize("kw, warning", [
    (dict(imu=imu(valid=False)), W.IMU_INVALID),
    (dict(imu=imu(calib=2)), W.IMU_NOT_CALIBRATED),
    (dict(imu=imu(drift=0.5)), W.IMU_DRIFT_HIGH),
    (dict(road_yaw=RoadYaw(0.0, NOW - 10, True)), W.ROAD_YAW_STALE),
    (dict(road_yaw=RoadYaw(0.0, 0.0, False)), W.ROAD_YAW_STALE),
    (dict(fps={"front": 30.0, "side": 12.0}), W.CAM_FPS_LOW),
    (dict(front=front(mean_v=20.0)), W.CAM_TOO_DARK),
    (dict(side=side(mean_v=240.0)), W.CAM_TOO_BRIGHT),
    (dict(lane=LaneEstimate(0.0, 0.0, 0.9, 0.05, NOW)), W.CAMS_DISAGREE),
    (dict(lane=lane(conf=0.2)), W.LANE_CONF_LOW),
    (dict(overruns=2), W.LOOP_OVERRUN),
    (dict(motor_error=True), W.MOTOR_SERIAL_ERROR),
])
def test_each_warning(kw, warning):
    assert health(**kw) == [warning]


def test_missing_sensors_raise_no_warnings():
    assert health(front=None, imu=None, lane=None, road_yaw=None, fps={}) == []
