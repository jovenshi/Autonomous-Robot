"""Tests for control/ (lane_controller, turn_controller, speed_governor,
command, mixer).

Run from the robot/ folder:  python -m pytest tests
"""
import copy
import math
import os
import sys
import types

import numpy as np
import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from control.command import drive_for_state
from control.lane_controller import LaneController
from control.mixer import mix
from control.speed_governor import target_speed
from control.turn_controller import hold_steer, turn_done, turn_steer, vision_turn_steer
from robot_types import (DriveCmd, FrontLaneObs, FrontResult, ImuSample, LaneEstimate,
                         State)


def est(lateral=0.0, heading=0.0, conf=1.0):
    return LaneEstimate(lateral, heading, conf, None, 0.0)


@pytest.fixture
def no_smoothing(monkeypatch):
    monkeypatch.setattr(config, "STEER_SMOOTH", 0.0)


# ---------------------------------------------------------------- lane controller

def test_axle_right_of_line_steers_left(no_smoothing):
    steer = LaneController(config).update(est(lateral=0.05), 0.0, 1 / 30)
    assert steer == pytest.approx(config.KP_LAT * 0.05) and steer > 0


def test_pointing_left_steers_right(no_smoothing):
    assert LaneController(config).update(est(heading=-0.1), 0.0, 1 / 30) < 0


def test_yaw_rate_damps_steer(no_smoothing):
    ctrl = LaneController(config)
    calm = ctrl.update(est(lateral=0.05), 0.0, 1 / 30)
    assert ctrl.update(est(lateral=0.05), 20.0, 1 / 30) < calm


def run_lane_correction(kd, speed_mps=0.5, yaw_dps_per_steer=150.0, loops=90):
    # Closed loop on a simple kinematic robot: starts 5 cm right of the line,
    # square to the road. Steer sets the yaw rate (+ = left) seen next loop;
    # yaw rate turns the heading error, heading error moves the axle.
    # Each loop also asks an identical controller with yaw rate 0 what it
    # would have steered. Returns [(yaw_rate, lateral, steer, undamped)].
    cfg = types.SimpleNamespace(**{**vars(config), "KD_YAW": kd})
    ctrl, dt = LaneController(cfg), 1.0 / config.CONTROL_HZ
    lateral, heading, rate, log = 0.05, 0.0, 0.0, []
    for _ in range(loops):
        e = est(lateral=lateral, heading=heading)
        undamped = copy.deepcopy(ctrl).update(e, 0.0, dt)
        steer = ctrl.update(e, rate, dt)
        log.append((rate, lateral, steer, undamped))
        rate = yaw_dps_per_steer * steer
        heading -= math.radians(rate) * dt
        lateral += speed_mps * math.sin(heading) * dt
    return log


def test_kd_yaw_reduces_steer_while_turning_toward_line():
    # Right of the line (lateral +), steering left: once the robot is turning
    # left (rate +), the KD_YAW term asks for less left steer than P + H alone.
    log = run_lane_correction(config.KD_YAW)
    toward = [(steer, undamped) for rate, lateral, steer, undamped in log
              if rate > 0 and lateral > 0]
    assert len(toward) > 10
    assert all(steer < undamped for steer, undamped in toward)
    peak = max(steer for _, _, steer, _ in log)
    assert peak < max(steer for _, _, steer, _ in run_lane_correction(0.0))


def test_steer_is_clamped(no_smoothing):
    assert LaneController(config).update(est(lateral=1.0), 0.0, 1 / 30) == config.STEER_MAX


def test_smoothing_and_reset():
    ctrl = LaneController(config)
    first = ctrl.update(est(lateral=0.05), 0.0, 1 / 30)
    assert first == pytest.approx((1 - config.STEER_SMOOTH) * config.KP_LAT * 0.05)
    ctrl.reset()
    assert ctrl.update(est(lateral=None), 0.0, 1 / 30) == 0.0


def test_lost_lane_holds_last_steer(no_smoothing):
    ctrl = LaneController(config)
    last = ctrl.update(est(lateral=0.05), 0.0, 1 / 30)
    assert ctrl.update(est(lateral=None, heading=None, conf=0.0), 0.0, 1 / 30) == last
    assert ctrl.update(None, 0.0, 1 / 30) == last


# ---------------------------------------------------------------- turn controller

def test_turn_steer_direction_and_floor():
    assert turn_steer(0.0, 90.0, config) == 1.0                    # big error, clamped
    assert turn_steer(0.0, 5.0, config) == config.TURN_MIN_STEER   # small error, floored
    assert turn_steer(0.0, -5.0, config) == -config.TURN_MIN_STEER
    assert turn_steer(10.0, 10.0, config) == 0.0


def test_turn_done_tolerance_and_handoff():
    assert turn_done(88.0, 90.0, 0.0, config)          # within TURN_TOL_DEG
    assert not turn_done(80.0, 90.0, 0.0, config)      # camera hasn't found the line
    assert turn_done(80.0, 90.0, 0.8, config)          # handoff to vision
    assert not turn_done(60.0, 90.0, 0.8, config)      # too far for handoff


def front_result(conf, heading):
    lane = FrontLaneObs(0.0, heading, conf, (), None, 0.0, 1)
    return FrontResult(np.zeros((2, 2, 3), np.uint8), lane, None, None, 100.0, 0.0, 1)


def test_vision_turn():
    assert vision_turn_steer(None, -1, config) == (-config.VISION_TURN_STEER, False)
    assert vision_turn_steer(front_result(0.8, math.radians(5)), 1, config)[1]
    assert not vision_turn_steer(front_result(0.8, math.radians(40)), 1, config)[1]
    assert not vision_turn_steer(front_result(0.2, 0.0), 1, config)[1]


def test_hold_steer():
    assert hold_steer(0.0, 10.0, config) == pytest.approx(config.HOLD_KP * 10)
    assert hold_steer(0.0, -1000.0, config) == -config.STEER_MAX


# ---------------------------------------------------------------- speed governor

def test_full_speed_when_confident_and_straight():
    assert target_speed(State.LANE_FOLLOW, est(conf=1.0), config) == config.CRUISE_SPEED


def test_lost_lane_slows_to_minimum_scale():
    speed = target_speed(State.LANE_FOLLOW, est(lateral=None, heading=None, conf=0.0), config)
    assert speed == pytest.approx(config.CRUISE_SPEED * config.SPEED_MIN_SCALE)


def test_heading_error_slows():
    speed = target_speed(State.APPROACH, est(heading=config.SPEED_HEADING_SLOW_RAD * 2), config)
    assert speed == pytest.approx(config.APPROACH_SPEED * config.SPEED_MIN_SCALE)


def test_cross_and_stopped_states():
    assert target_speed(State.CROSS, est(conf=0.0), config) == config.CROSS_SPEED
    assert target_speed(State.IDLE, est(), config) == 0.0


# ---------------------------------------------------------------- command

def test_stopped_states_send_zero():
    for state in (State.IDLE, State.DECIDE, State.STOP_HOLD, State.SAFE_STOP):
        assert drive_for_state(state, est(), None, None, LaneController(config), None, config) \
            == DriveCmd(0.0, 0.0)


def test_lane_state_uses_controller_and_imu_rate(no_smoothing):
    imu = ImuSample(0.0, 20.0, True, 3, 0.0, 0.0)
    with_imu = drive_for_state(State.LANE_FOLLOW, est(lateral=0.05), imu, None,
                               LaneController(config), None, config)
    no_imu = drive_for_state(State.LANE_FOLLOW, est(lateral=0.05), None, None,
                             LaneController(config), None, config)
    assert with_imu.steer < no_imu.steer
    assert no_imu.speed == config.CRUISE_SPEED


def test_lane_steer_keeps_inner_wheel_forward(no_smoothing):
    drive = drive_for_state(State.LANE_FOLLOW, est(lateral=1.0), None, None,
                            LaneController(config), None, config)
    assert drive.steer == pytest.approx(config.LANE_STEER_SPEED_RATIO * drive.speed)
    motors = mix(drive, config)
    assert motors.left > 0 and motors.right > 0


def test_turn_and_cross_not_wired_yet():
    for state in (State.TURN, State.CROSS):
        with pytest.raises(NotImplementedError):
            drive_for_state(state, est(), None, None, LaneController(config), None, config)


# ---------------------------------------------------------------- mixer

def test_mix_straight_and_left():
    straight = mix(DriveCmd(0.5, 0.0), config)
    assert (straight.left, straight.right) == (0.5, 0.5)
    left = mix(DriveCmd(0.5, 0.2), config)
    assert left.left == pytest.approx(0.3) and left.right == pytest.approx(0.7)


def test_mix_pivot_and_rescale():
    pivot = mix(DriveCmd(0.0, 0.5), config)
    assert (pivot.left, pivot.right) == (-0.5, 0.5)
    fast = mix(DriveCmd(1.0, 0.5), config)
    assert fast.right == 1.0 and fast.left == pytest.approx(0.5 / 1.5)


@pytest.mark.parametrize("imu_s", [None, ImuSample(0.0, 30.0, False, 3, 0.0, 0.0)],
                         ids=["missing", "invalid"])
def test_missing_or_invalid_imu_steers_like_kd_yaw_zero(monkeypatch, no_smoothing, imu_s):
    # Lane following without a usable IMU = lane following with no damping.
    e = est(lateral=0.05, heading=0.05)
    without = drive_for_state(State.LANE_FOLLOW, e, imu_s, None,
                              LaneController(config), None, config).steer
    monkeypatch.setattr(config, "KD_YAW", 0.0)
    turning = ImuSample(0.0, 30.0, True, 3, 0.0, 0.0)
    kd_zero = drive_for_state(State.LANE_FOLLOW, e, turning, None,
                              LaneController(config), None, config).steer
    assert without == kd_zero
