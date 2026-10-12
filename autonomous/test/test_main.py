"""Tests for main.py's control_step and shutdown, with the camera and
Sabertooth replaced by fakes (no hardware, no motors).

Run from the robot/ folder:  python -m pytest tests
"""
import os
import queue
import sys
import threading

import cv2
import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
import main
from loop_log import LoopLog
from behavior.state_machine import Behaviour
from control.lane_controller import LaneController
from estimation.road_yaw import RoadYawTracker
from perception.pipeline import process_front_frame
from robot_types import Frame, ImuSample, LatestValue, MotorCmd, RoadYaw, State, UiCommand
from test_perception import BANDS, CX, H, W, scene


class FakeSaber:
    def __init__(self):
        self.sent, self.calls = [], []
        self.serial_error = False

    def set(self, cmd):
        self.sent.append(cmd)

    def stop(self):
        self.calls.append("stop")

    def close(self):
        self.calls.append("close")


class FakeCam:
    error_count = 0

    def __init__(self):
        self.calls = []

    def fps(self):
        return 30.0

    def stop(self):
        self.calls.append("cam_stop")


@pytest.fixture(autouse=True)
def bands(monkeypatch):
    monkeypatch.setattr(config, "FRONT_BANDS", BANDS)
    monkeypatch.setattr(config, "FRONT_LANE_SIZE", (W, H))
    monkeypatch.setattr(config, "STEER_SMOOTH", 0.0)


def make_system():
    flags = threading.Event(), threading.Event()
    return main.System(
        cfg=config, front_slot=LatestValue(), signs_enabled=flags[0], exposure_hold=flags[1],
        front_cam=FakeCam(), front_exposure=None, saber=FakeSaber(),
        lane_ctrl=LaneController(config), road_yaw=RoadYawTracker(config),
        behaviour=Behaviour(config, *flags), commands=queue.Queue(),
    )


@pytest.fixture
def system():
    return make_system()


def publish(system, t, line_offset_px=0.0, road=True):
    img = scene(line_x=lambda y: CX + line_offset_px) if road else scene(half_road=0, line_w=0)
    big = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
    system.front_slot.set(process_front_frame(Frame(big, t, int(t * 30), "front"), False, config))


def test_idle_sends_zero_every_loop(system, capsys):
    for i in range(3):
        publish(system, 1.0 + i / 30)
        main.control_step(system, 1.0 + i / 30)
    assert system.saber.sent == [MotorCmd(0.0, 0.0)] * 3
    assert "IDLE" in capsys.readouterr().out


def test_start_drives_and_corrects_toward_line(system):
    publish(system, 1.0, line_offset_px=20)          # line right of robot -> steer right
    system.commands.put(UiCommand.START)
    main.control_step(system, 1.0)
    assert system.behaviour.state == State.LANE_FOLLOW
    cmd = system.saber.sent[-1]
    assert cmd.left > cmd.right > 0                  # turning right, moving forward


def test_lane_lost_stops_motors(system):
    publish(system, 1.0)
    system.commands.put(UiCommand.START)
    main.control_step(system, 1.0)
    t = 1.0
    while system.behaviour.state != State.SAFE_STOP and t < 3.0:
        t += 1 / 30
        publish(system, t, road=False)
        main.control_step(system, t)
    assert system.behaviour.state == State.SAFE_STOP
    assert system.behaviour.stop_reason == "lane lost"
    assert system.saber.sent[-1] == MotorCmd(0.0, 0.0)
    assert t - 1.0 == pytest.approx(config.LANE_LOST_TIMEOUT_S, abs=0.1)


def test_stop_key_returns_to_idle(system):
    publish(system, 1.0)
    system.commands.put(UiCommand.START)
    main.control_step(system, 1.0)
    system.commands.put(UiCommand.STOP)
    main.control_step(system, 1.03)
    assert system.behaviour.state == State.IDLE
    assert system.saber.sent[-1] == MotorCmd(0.0, 0.0)


def test_recent_overruns_window(system):
    system.overrun_times.extend([1.0, 1.5, 2.2])
    assert main._recent_overruns(system, 2.4) == 2


def test_shutdown_stops_motors_first_and_survives_failures(system):
    def broken_close():
        raise RuntimeError("port gone")
    system.saber.close = broken_close
    main.shutdown(system)
    assert system.saber.calls == ["stop"]
    assert system.front_cam.calls == ["cam_stop"]


# ---------------------------------------------------------------- IMU wiring

class FakeImu:
    def __init__(self, log):
        self.log = log

    def close(self):
        self.log.append("imu_close")


@pytest.fixture
def with_imu(system):
    system.imu = FakeImu(system.saber.calls)
    return system


def imu_sample(t, yaw=0.0, rate=0.0, valid=True, calib=3, drift=0.0):
    return ImuSample(yaw, rate, valid, calib, drift, t)


def drive_once(system, t, imu_s=None, line_offset_px=20):
    # START then one loop at t with a lane offset; returns the drive command.
    publish(system, t, line_offset_px=line_offset_px)
    if imu_s is not None:
        system.imu_slot.set(imu_s)
    system.commands.put(UiCommand.START)
    main.control_step(system, t)
    return system.last_drive


def test_fresh_imu_rate_damps_lane_steer(system):
    # Line right of the robot -> steer right (-). Already turning right
    # (rate -) -> KD_YAW makes the steer less negative.
    plain = drive_once(system, 1.0).steer
    damped = drive_once(make_system(), 1.0, imu_sample(1.0, rate=-30.0)).steer
    assert plain < 0 and plain < damped


def test_stale_imu_is_ignored_for_steering(system):
    plain = drive_once(system, 1.0).steer
    other = make_system()
    stale = 1.0 - 2 * config.MAX_OBS_AGE_S
    assert drive_once(other, 1.0, imu_sample(stale, rate=-30.0)).steer == pytest.approx(plain)
    assert other.behaviour.state == State.LANE_FOLLOW          # no SAFE_STOP


def test_road_yaw_anchors_from_imu_slot(with_imu):
    # Road yaw updates with the state from the previous loop, so the first
    # anchor is on the loop after START.
    drive_once(with_imu, 1.0, imu_sample(1.0, yaw=30.0), line_offset_px=0)
    assert not with_imu.road_yaw.get().valid
    drive_once(with_imu, 1.03, imu_sample(1.03, yaw=30.0), line_offset_px=0)
    road = with_imu.road_yaw.get()
    assert road.valid and road.yaw_deg == pytest.approx(30.0, abs=1.0)


def test_stale_imu_does_not_anchor_road_yaw(with_imu):
    drive_once(with_imu, 1.0, imu_sample(0.5, yaw=30.0), line_offset_px=0)
    drive_once(with_imu, 1.03, imu_sample(0.5, yaw=30.0), line_offset_px=0)
    assert not with_imu.road_yaw.get().valid


def test_imu_warnings_and_status_line(with_imu, capsys):
    with_imu.imu_slot.set(imu_sample(1.0, yaw=12.5, valid=False, calib=2, drift=0.05))
    publish(with_imu, 1.0)
    main.control_step(with_imu, 1.0)
    out = capsys.readouterr().out
    assert "IMU_INVALID" in out and "IMU_NOT_CALIBRATED" in out
    assert "yaw   +12.5 deg" in out and "calib 2" in out and "drift +0.050" in out
    assert "road   --" in out


def test_status_line_shows_road_yaw_age():
    road = RoadYaw(10.0, 2.0, True)
    line = main._imu_line(imu_sample(5.0, yaw=11.0), road, 5.0)
    assert "road   +10.0 deg ( 3.0 s ago)" in line


def test_shutdown_closes_imu_after_motors(with_imu):
    main.shutdown(with_imu)
    assert with_imu.saber.calls == ["stop", "close", "imu_close"]


def test_stale_imu_steers_like_kd_yaw_zero(system, monkeypatch):
    stale = drive_once(system, 1.0, imu_sample(1.0 - 2 * config.MAX_OBS_AGE_S, rate=-30.0)).steer
    monkeypatch.setattr(config, "KD_YAW", 0.0)
    assert drive_once(make_system(), 1.0, imu_sample(1.0, rate=-30.0)).steer == stale


def test_status_line_shows_yaw_rate():
    assert "rate  -12.5 deg/s" in main._imu_line(imu_sample(1.0, rate=-12.5), None, 1.0)


# ---------------------------------------------------------------- loop log

def test_control_step_logs_one_row_per_loop(system, tmp_path):
    system.loop_log = LoopLog(str(tmp_path), config)
    system.imu_slot.set(imu_sample(1.0, yaw=3.0, rate=-30.0))
    drive_once(system, 1.0)
    later = 1.0 + 2 * config.MAX_OBS_AGE_S            # IMU sample now stale
    publish(system, later, line_offset_px=20)
    main.control_step(system, later)
    system.loop_log.close()
    with open(system.loop_log.path) as f:
        lines = f.read().splitlines()
    assert lines[0].startswith("# KD_YAW=") and lines[1].startswith("t,state,")
    rows = [line.split(",") for line in lines[2:]]
    assert len(rows) == 2
    assert rows[0][1] == "LANE_FOLLOW" and rows[0][5:8] == ["3", "-30", "1"]
    assert rows[1][5:8] == ["", "", "0"]            # stale sample: blank, imu_valid 0
    assert float(rows[1][8]) == pytest.approx(system.last_drive.steer, abs=1e-6)


class FakeLog:
    def __init__(self, log):
        self.log = log

    def write(self, *args):
        pass

    def close(self):
        self.log.append("log_close")


def test_shutdown_flushes_loop_log_before_camera(with_imu):
    # The camera stop can hang on a failed camera; the log must be safe first.
    order = with_imu.saber.calls
    with_imu.loop_log = FakeLog(order)
    with_imu.front_cam.calls = order
    main.shutdown(with_imu)
    assert order == ["stop", "close", "imu_close", "log_close", "cam_stop"]


def test_build_refuses_without_bands(monkeypatch):
    monkeypatch.setattr(config, "FRONT_BANDS", None)
    with pytest.raises(SystemExit):
        main.build_system(config)
