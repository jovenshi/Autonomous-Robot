"""Tests for behavior/state_machine.py (straight-drive states).

Run from the robot/ folder:  python -m pytest tests
"""
import threading

import numpy as np
import pytest


import sys
import os
# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from behavior.state_machine import Behaviour, choose_action
from robot_types import (Action, FrontLaneObs, FrontResult, ImuSample, IntersectionObs,
                         LaneEstimate, SignType, State, UiCommand)

START, STOP = [UiCommand.START], [UiCommand.STOP]


def lane(ok=True, t=0.0):
    return LaneEstimate(0.0 if ok else None, 0.0 if ok else None, 0.9 if ok else 0.0, None, t)


def imu(calib=3, t=0.0):
    return ImuSample(0.0, 0.0, True, calib, 0.0, t)


def front_with_score(score):
    lane_obs = FrontLaneObs(0.0, 0.0, 0.9, (), None, 0.0, 1)
    inter = IntersectionObs(score, None, 0.0, 1)
    return FrontResult(np.zeros((2, 2, 3), np.uint8), lane_obs, inter, None, 100.0, 0.0, 1)


@pytest.fixture
def flags():
    return threading.Event(), threading.Event()


@pytest.fixture
def b(flags):
    return Behaviour(config, signs_enabled=flags[0], exposure_hold=flags[1])


def step(beh, now, commands=(), lane_e=None, imu_s=None, front=None, motor_error=False):
    lane_e = lane_e if lane_e is not None else lane(t=now)
    return beh.step(front, None, lane_e, None, None, imu_s, list(commands), motor_error, now)


def start(beh, now=1.0):
    step(beh, now, START)
    return step(beh, now + 0.03)


# ---------------------------------------------------------------- IDLE

def test_starts_idle_with_exposure_hold(b, flags):
    assert b.state == State.IDLE and flags[1].is_set()


def test_stays_idle_without_start(b):
    assert step(b, 1.0) == State.IDLE


def test_start_without_imu_goes_to_lane_follow(b, flags):
    assert step(b, 1.0, START) == State.LANE_FOLLOW
    assert not flags[1].is_set()               # managed exposure allowed


def test_start_waits_for_gyro_calibration(b):
    assert step(b, 1.0, START, imu_s=imu(calib=1, t=1.0)) == State.IDLE
    assert step(b, 2.0, imu_s=imu(calib=1, t=2.0)) == State.IDLE
    assert step(b, 3.0, imu_s=imu(calib=3, t=3.0)) == State.LANE_FOLLOW


def test_start_after_calibration_timeout(b, capsys):
    step(b, 1.0, START, imu_s=imu(calib=0, t=1.0))
    t = config.IMU_CALIB_WAIT_S + 0.1
    assert step(b, t, imu_s=imu(calib=0, t=t)) == State.LANE_FOLLOW
    assert "not calibrated" in capsys.readouterr().out


# ---------------------------------------------------------------- safety

def test_brief_lane_loss_is_tolerated(b):
    start(b, 1.0)
    assert step(b, 1.2, lane_e=lane(ok=False, t=1.2)) == State.LANE_FOLLOW


def test_lane_lost_too_long_stops(b):
    start(b, 1.0)
    t = 1.03 + config.LANE_LOST_TIMEOUT_S + 0.05
    assert step(b, t, lane_e=lane(ok=False, t=t)) == State.SAFE_STOP
    assert b.stop_reason == "lane lost"


def test_motor_error_stops(b):
    start(b, 1.0)
    assert step(b, 1.1, motor_error=True) == State.SAFE_STOP
    assert b.stop_reason == "motor serial error"


def test_stale_imu_does_not_stop_lane_following(b):
    # Lane following works without the IMU (no KD_YAW damping).
    start(b, 1.0)
    assert step(b, 2.0, imu_s=imu(t=1.0)) == State.LANE_FOLLOW


@pytest.mark.parametrize("state", [State.TURN, State.CROSS])
def test_stale_imu_stops_imu_steered_states(b, state):
    b._enter(state, 1.0)
    assert step(b, 2.0, imu_s=imu(t=1.0)) == State.SAFE_STOP
    assert b.stop_reason == "IMU data stale"


def test_no_safety_checks_while_idle(b):
    assert step(b, 5.0, lane_e=lane(ok=False), motor_error=True) == State.IDLE


# ---------------------------------------------------------------- commands

def test_safe_stop_needs_two_starts(b):
    start(b, 1.0)
    step(b, 1.1, motor_error=True)
    assert step(b, 2.0, START) == State.IDLE
    assert b.stop_reason is None
    assert step(b, 2.1, START) == State.LANE_FOLLOW


def test_stop_command_returns_to_idle(b):
    start(b, 1.0)
    assert step(b, 1.5, STOP) == State.IDLE
    assert step(b, 1.6) == State.IDLE           # old START not remembered


# ---------------------------------------------------------------- drift request

@pytest.fixture
def drift():
    return threading.Event()


@pytest.fixture
def bd(drift):
    return Behaviour(config, drift_request=drift)


def test_drift_requested_while_idle_and_cleared_on_start(bd, drift):
    assert drift.is_set()
    assert step(bd, 1.0, START) == State.LANE_FOLLOW
    assert not drift.is_set()
    step(bd, 2.0, STOP)
    assert bd.state == State.IDLE and drift.is_set()


def test_drift_cleared_on_safe_stop(bd, drift):
    start(bd)
    step(bd, 2.0, motor_error=True)
    assert bd.state == State.SAFE_STOP and not drift.is_set()


@pytest.mark.parametrize("state", list(State))
def test_drift_requested_only_in_idle_and_stop_hold(bd, drift, state):
    # Every exit goes through _enter, so this covers clearing on every exit.
    bd._enter(State.STOP_HOLD, 1.0)
    assert drift.is_set()
    bd._enter(state, 2.0)
    assert drift.is_set() == (state in (State.IDLE, State.STOP_HOLD))


# ---------------------------------------------------------------- junction hook

def test_approach_sets_flags_then_reports_not_built(b, flags):
    start(b, 1.0)
    assert step(b, 1.1, front=front_with_score(config.APPROACH_ON)) == State.APPROACH
    assert flags[0].is_set() and flags[1].is_set()
    with pytest.raises(NotImplementedError):
        step(b, 1.2)


def test_low_score_stays_in_lane_follow(b):
    start(b, 1.0)
    assert step(b, 1.1, front=front_with_score(0.1)) == State.LANE_FOLLOW


# ---------------------------------------------------------------- choose_action

@pytest.mark.parametrize("sign, stop_done, action", [
    (SignType.STOP, False, Action.STOP),
    (SignType.STOP, True, Action[config.DEFAULT_ACTION]),
    (SignType.LEFT, False, Action.LEFT),
    (SignType.RIGHT, False, Action.RIGHT),
    (SignType.STRAIGHT, False, Action.STRAIGHT),
    (SignType.NONE, False, Action[config.DEFAULT_ACTION]),
    (None, False, Action[config.DEFAULT_ACTION]),
])
def test_choose_action(sign, stop_done, action):
    assert choose_action(sign, stop_done, config) == action


# ---------------------------------------------------------------- recovery from a knock

def test_pivot_back_suspends_lane_lost_until_timeout(b):
    from robot_types import RoadYaw
    road = RoadYaw(yaw_deg=0.0, t_anchor=0.0, valid=True)
    start(b, 1.0)
    t = 1.1
    while t < 1.1 + config.RECOVER_TIMEOUT_S - 0.1:                  # line out of view, but pivoting
        knocked = ImuSample(45.0, 0.0, True, 3, 0.0, t)
        assert b.step(None, None, lane(ok=False, t=t), road, None, knocked, [], False, t) \
            == State.LANE_FOLLOW
        t += 0.1
    t = 1.1 + config.RECOVER_TIMEOUT_S + 0.2
    knocked = ImuSample(45.0, 0.0, True, 3, 0.0, t)
    assert b.step(None, None, lane(ok=False, t=t), road, None, knocked, [], False, t) \
        == State.SAFE_STOP
    assert b.stop_reason == "could not turn back to the road"


def test_lane_lost_still_stops_when_facing_the_road(b):
    from robot_types import RoadYaw
    road = RoadYaw(yaw_deg=0.0, t_anchor=0.0, valid=True)
    start(b, 1.0)
    t = 1.1 + config.LANE_LOST_TIMEOUT_S + 0.1
    straight = ImuSample(5.0, 0.0, True, 3, 0.0, t)
    assert b.step(None, None, lane(ok=False, t=t), road, None, straight, [], False, t) \
        == State.SAFE_STOP
