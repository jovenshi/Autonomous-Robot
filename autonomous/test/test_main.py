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
from behavior.state_machine import Behaviour
from control.lane_controller import LaneController
from estimation.road_yaw import RoadYawTracker
from perception.pipeline import process_front_frame
from robot_types import Frame, LatestValue, MotorCmd, State, UiCommand
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


@pytest.fixture
def system():
    flags = threading.Event(), threading.Event()
    return main.System(
        cfg=config, front_slot=LatestValue(), signs_enabled=flags[0], exposure_hold=flags[1],
        front_cam=FakeCam(), front_exposure=None, saber=FakeSaber(),
        lane_ctrl=LaneController(config), road_yaw=RoadYawTracker(config),
        behaviour=Behaviour(config, *flags), commands=queue.Queue(),
    )


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


def test_build_refuses_without_bands(monkeypatch):
    monkeypatch.setattr(config, "FRONT_BANDS", None)
    with pytest.raises(SystemExit):
        main.build_system(config)
