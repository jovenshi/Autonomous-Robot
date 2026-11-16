"""Recording a run (D-pad Up, UiCommand.RECORD) and driving by hand while
stopped (PS5 triggers and left stick). No hardware: fake Sabertooth, camera,
controller and recording writer."""
import json
import os
import queue
import sys
import threading

import numpy as np
import pygame
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
import main  # noqa: E402
from behavior.state_machine import Behaviour  # noqa: E402
from control.manual import ManualControl  # noqa: E402
from robot_types import (Frame, ImuSample, ManualInput, MotorCmd, RecordStatus, RunMode,  # noqa: E402
                         State, UiCommand)
from run_recorder import RunRecorder  # noqa: E402
from test_main import make_system, publish  # noqa: E402
from test_operator import FakePS5, event, pad  # noqa: E402,F401
from test_ui import bare_packet, screen_at  # noqa: E402
from tools.recording import read_index, read_meta  # noqa: E402
from ui.course import course_module  # noqa: E402
from ui.gamepad import read_manual  # noqa: E402
from ui.link import read_commands, read_ui  # noqa: E402
from ui.screen import VIEWS, _read_input  # noqa: E402

md = course_module("manual.manual_drive")
DT = 1.0 / config.CONTROL_HZ


@pytest.fixture(autouse=True)
def bands_and_smoothing():
    # test_main's autouse fixture sets FRONT_BANDS and STEER_SMOOTH = 0 for its
    # systems; imported fixtures are not autouse here, so do the same.
    from test_main import BANDS, H, W
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "FRONT_BANDS", BANDS)
    mp.setattr(config, "FRONT_LANE_SIZE", (W, H))
    mp.setattr(config, "STEER_SMOOTH", 0.0)
    yield
    mp.undo()


def held(r2=0.0, l2=0.0, stick=0.0, t=1.0):
    return ManualInput(r2=r2, l2=l2, stick_x=stick, t=t)


# ---------------------------------------------------------------- manual/manual_drive.py

def test_manual_drive_ramps_like_robot_controller():
    drive = md.ManualDrive(config)
    speed, _ = drive.update(1.0, 0.0, 0.0, None, 0.1)
    assert speed == pytest.approx(md.ACCEL_RATE * 0.1)
    for _ in range(50):
        drive.update(1.0, 0.0, 0.0, None, 0.1)
    assert drive.speed == 1.0
    speed, _ = drive.update(0.0, 0.0, 0.0, None, 0.1)
    assert speed == pytest.approx(1.0 - md.DECEL_RATE * 0.1)


def test_manual_drive_pivots_at_standstill_and_settles_to_zero():
    drive = md.ManualDrive(config)
    for _ in range(100):
        _, turn = drive.update(0.0, 0.0, 1.0, None, DT)
    assert turn == pytest.approx(md.PIVOT_TURN)
    left, right = md.mix(0.0, turn, config)
    assert left > 0 > right                      # + turn = right: spin clockwise
    for _ in range(1000):
        drive.update(0.0, 0.0, 0.0, None, DT)
    assert not drive.moving                      # smoothing snaps to exactly 0


def test_manual_drive_heading_hold_steers_back():
    drive = md.ManualDrive(config)
    drive.speed = 0.5
    imu = ImuSample(yaw_deg=0.0, yaw_rate_dps=0.0, valid=True, calib_gyro=3, drift_dps=0.0, t=0)
    drive.update(0.5, 0.0, 0.0, imu, DT)         # takes heading 0
    left_of = ImuSample(yaw_deg=5.0, yaw_rate_dps=0.0, valid=True, calib_gyro=3, drift_dps=0.0,
                        t=0)
    _, turn = drive.update(0.5, 0.0, 0.0, left_of, DT)
    assert drive.hold_yaw == 0.0 and turn > 0    # drifted left -> turn right


# ---------------------------------------------------------------- control/manual.py

def test_manual_control_drives_only_while_stopped():
    hand = ManualControl(config)
    hand.take(held(r2=1.0, stick=0.5, t=1.0))
    assert hand.step(State.LANE_FOLLOW, None, 1.0) is None
    drive, motor = hand.step(State.IDLE, None, 1.0)
    assert drive.speed > 0 and drive.steer < 0   # stick right -> DriveCmd steer + = left, so < 0
    assert motor.left > motor.right
    assert hand.step(State.SAFE_STOP, None, 1.0 + DT) is not None


def test_manual_control_stale_reading_stops_at_once():
    hand = ManualControl(config)
    hand.take(held(r2=1.0, t=1.0))
    for i in range(10):
        assert hand.step(State.IDLE, None, 1.0 + i * DT) is not None
    late = 1.0 + config.MANUAL_INPUT_TIMEOUT_S + 0.01
    assert hand.step(State.IDLE, None, late) is None
    assert hand.drive.speed == 0.0               # no ramp down: stopped


def test_manual_control_released_controls_leave_the_robot_command():
    hand = ManualControl(config)
    hand.take(held(t=1.0))
    assert hand.step(State.IDLE, None, 1.0) is None


def test_manual_drive_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "MANUAL_DRIVE_ENABLED", False)
    hand = ManualControl(config)
    hand.take(held(r2=1.0, t=1.0))
    assert hand.step(State.IDLE, None, 1.0) is None


# ---------------------------------------------------------------- gamepad and link

def test_dpad_up_sends_record_once_until_the_hat_is_centred(pad):
    gamepad, ps5 = pad
    ps5.control_request["reqArrowUp"] = False
    ps5.held = {"reqArrowUp"}
    assert gamepad.poll([], State.IDLE) == [UiCommand.RECORD]
    assert gamepad.poll([], State.LANE_FOLLOW) == []                 # still held
    ps5.held = set()
    gamepad.poll([event(pygame.JOYHATMOTION, joy=0, instance_id=0, hat=0, value=(0, 0))],
                 State.LANE_FOLLOW)
    ps5.held = {"reqArrowUp"}
    assert gamepad.poll([], State.LANE_FOLLOW) == [UiCommand.RECORD]


def test_gamepad_reads_triggers_and_stick(pad):
    gamepad, ps5 = pad
    ps5.control_request.update(reqR2Value=0.75, reqL2Value=0.0, reqLeftJoyMade=True,
                               reqLeftJoyXValue=-127)
    gamepad.poll([], State.IDLE)
    inp = gamepad.manual_input
    assert (inp.r2, inp.l2, inp.stick_x) == (0.75, 0.0, -1.0)
    gamepad._detach("test")
    gamepad.poll([], State.IDLE)
    assert gamepad.manual_input is None          # no controller: nothing sent, main stops


def test_stick_inside_deadzone_reads_zero():
    assert read_manual({"reqLeftJoyMade": False, "reqLeftJoyXValue": 40}).stick_x == 0.0


def test_stick_readings_reach_main_separately_from_commands(pad):
    gamepad, ps5 = pad
    screen, _ = screen_at(0.0)
    cmd_q = queue.Queue()
    ps5.control_request.update(reqR2Value=0.5)
    ps5.held = {"reqCross"}
    _read_input([], screen, gamepad, cmd_q)
    commands, manual = read_ui(cmd_q)
    assert commands == [UiCommand.START] and manual.r2 == 0.5
    cmd_q.put(held(r2=0.2))
    cmd_q.put(UiCommand.RECORD)
    assert read_commands(cmd_q) == [UiCommand.RECORD]


# ---------------------------------------------------------------- screen

@pytest.mark.parametrize("view", VIEWS)
def test_views_draw_recording_and_manual(view):
    screen, _ = screen_at(5.0)
    rec = RecordStatus(name="run_x", started_t=1.0, saved=120, dropped=3)
    for packet in (bare_packet(t=5.0, recording=rec), bare_packet(t=5.0, manual=True),
                   bare_packet(t=5.0, state=State.LANE_FOLLOW, recording=rec)):
        screen.update(packet)
        assert screen.draw(screen.packet, view).shape == (480, 800, 3)


# ---------------------------------------------------------------- run recorder

class FakeWriter:
    made = []

    def __init__(self, out_dir, png_level, queue_size):
        self.out_dir, self.saved, self.dropped, self.frames = out_dir, 0, 0, []
        self.closed = False
        FakeWriter.made.append(self)

    def start(self, meta):
        self.meta = meta

    def submit(self, frame, exposure_us=None, gain=None):
        self.frames.append(frame)
        self.saved += 1

    def close(self):
        self.closed = True


@pytest.fixture
def recorder(tmp_path):
    FakeWriter.made = []
    return RunRecorder(str(tmp_path), config, writer_factory=FakeWriter)


def recording_system(recorder):
    system = make_system()
    system.recorder = recorder
    system.manual = ManualControl(config)
    return system


def step(system, t, *commands):
    for c in commands:
        system.commands.put(c)
    publish(system, t)
    main.control_step(system, t)


def test_record_starts_recording_and_the_run(recorder):
    system = recording_system(recorder)
    step(system, 1.0, UiCommand.RECORD)
    assert recorder.active and system.behaviour.state == State.LANE_FOLLOW
    assert FakeWriter.made[0].meta["mode"] == "DRIVE"
    assert os.path.basename(FakeWriter.made[0].out_dir).startswith("run_")
    step(system, 1.0 + DT, UiCommand.RECORD)     # again: STOP, ends both
    assert not recorder.active and system.behaviour.state == State.IDLE
    recorder.close()
    assert FakeWriter.made[0].closed


def test_stop_and_the_run_ending_stop_the_recording(recorder):
    system = recording_system(recorder)
    step(system, 1.0, UiCommand.RECORD)
    step(system, 1.0 + DT, UiCommand.STOP)
    assert not recorder.active
    step(system, 2.0, UiCommand.RECORD)
    assert recorder.active
    t = 2.0
    while system.behaviour.state != State.SAFE_STOP and t < 4.0:   # lane lost
        t += DT
        publish(system, t, road=False)
        main.control_step(system, t)
    assert system.behaviour.state == State.SAFE_STOP and not recorder.active


def test_record_after_safe_stop_only_clears_it(recorder):
    system = recording_system(recorder)
    system.behaviour._enter(State.SAFE_STOP, 0.5, "test")
    system.prev_state = State.SAFE_STOP
    step(system, 1.0, UiCommand.RECORD)
    assert system.behaviour.state == State.IDLE and not recorder.active


def test_record_in_sign_reading_records_without_driving(recorder):
    system = recording_system(recorder)
    system.behaviour.mode = RunMode.QUIZ_SIGNS
    step(system, 1.0, UiCommand.RECORD)
    assert recorder.active and system.behaviour.state == State.IDLE
    assert FakeWriter.made[0].meta["mode"] == "QUIZ_SIGNS"
    step(system, 1.0 + DT, UiCommand.MODE_DRIVE)     # a mode change ends it
    assert not recorder.active


def test_record_in_intersection_mode_starts_that_mode(recorder):
    system = recording_system(recorder)
    step(system, 1.0, UiCommand.MODE_INTERSECTION, UiCommand.RECORD)
    assert system.behaviour.mode == RunMode.QUIZ_INTERSECTION
    assert FakeWriter.made[0].meta["mode"] == "QUIZ_INTERSECTION"
    assert system.behaviour.state == State.LANE_FOLLOW


def test_camera_frames_go_to_the_recording_only_while_recording(recorder):
    frame = Frame(np.zeros((4, 4, 3), np.uint8), 1.0, 1, "front")
    recorder.front(frame)                        # not recording: dropped
    recorder.start("DRIVE", with_side=True)
    recorder.front(frame, 1000, 1.0)
    recorder.side(frame)
    assert recorder.status().saved == 2
    recorder.stop()
    recorder.front(frame)
    assert [len(w.frames) for w in FakeWriter.made] == [1, 1]
    assert FakeWriter.made[1].out_dir.endswith("side")


def test_real_recording_is_replayable(tmp_path):
    recorder = RunRecorder(str(tmp_path), config)
    recorder.loop_log_path = "logs/run.csv"
    path = recorder.start("DRIVE", with_side=False)
    for seq in range(3):
        recorder.front(Frame(np.full((8, 8, 3), seq, np.uint8), 1.0 + seq / 30, seq, "front"))
    recorder.close()
    assert [row["seq"] for row in read_index(path)] == ["0", "1", "2"]
    meta = read_meta(path)
    assert meta["mode"] == "DRIVE" and meta["loop_log"] == "logs/run.csv"
    assert json.load(open(os.path.join(path, "meta.json")))["camera"] == "front"


# ---------------------------------------------------------------- hand driving in main

def test_idle_is_driven_by_hand_and_drift_is_paused(recorder):
    system = recording_system(recorder)
    drift = threading.Event()
    system.behaviour = Behaviour(config, system.signs_enabled, system.exposure_hold, drift)
    assert drift.is_set()
    system.manual.take(held(r2=1.0, t=1.0))
    step(system, 1.0)
    sent = system.saber.sent[-1]
    assert sent.left > 0 and sent.right > 0 and not drift.is_set()
    system.manual.take(held(t=1.0 + DT))
    for i in range(1, 30):                       # released: ramps to a stop
        step(system, 1.0 + i * DT)
    assert system.saber.sent[-1] == MotorCmd(0.0, 0.0) and drift.is_set()


def test_hand_input_is_ignored_while_driving_itself(recorder):
    system = recording_system(recorder)
    step(system, 1.0, UiCommand.START)
    system.manual.take(held(l2=1.0, t=1.0 + DT))   # full reverse held
    step(system, 1.0 + DT)
    sent = system.saber.sent[-1]
    assert sent.left > 0 and sent.right > 0       # still lane following forwards


def test_ui_packet_carries_recording_and_manual(recorder):
    system = recording_system(recorder)
    system.ui_packet_q = queue.Queue(maxsize=1)
    system.manual.take(held(r2=1.0, t=1.0))
    step(system, 1.0, UiCommand.RECORD)
    # RECORD started the run, so the hand input is not used this loop
    packet = system.ui_packet_q.get_nowait()
    assert packet.recording is not None and packet.recording.name.startswith("run_")
    assert packet.manual is False
    step(system, 1.0 + DT, UiCommand.STOP)
    system.ui_packet_q.get_nowait()
    system.manual.take(held(r2=1.0, t=1.0 + 2 * DT))
    step(system, 1.0 + 2 * DT)
    packet = system.ui_packet_q.get_nowait()
    assert packet.recording is None and packet.manual is True


# ---------------------------------------------------------------- a new run starts fresh

def imu_at(yaw, t):
    return ImuSample(yaw_deg=yaw, yaw_rate_dps=0.0, valid=True, calib_gyro=3, drift_dps=0.0, t=t)


def test_new_run_forgets_the_last_runs_road(recorder):
    # run_20261115_223428, 1625 s: road anchored at 0 deg by an earlier run,
    # robot driven back by hand and left facing -179 deg; START pivoted it
    # 180 deg towards the old road. A run now starts with no road.
    system = recording_system(recorder)
    system.imu = object()                        # an IMU is fitted
    system.road_yaw._yaw_deg = 0.0               # the earlier run's road
    system.imu_slot.set(imu_at(-179.0, 1.0))
    step(system, 1.0, UiCommand.START)
    assert system.behaviour.state == State.LANE_FOLLOW
    assert system.saber.sent[-1].left > 0 and system.saber.sent[-1].right > 0   # no pivot
    assert system.road_yaw.get().valid is False or \
        abs(system.road_yaw.get().yaw_deg + 179.0) < 10   # re-anchored on this road
    for i in range(1, 10):
        system.imu_slot.set(imu_at(-179.0, 1.0 + i * DT))
        step(system, 1.0 + i * DT)
        sent = system.saber.sent[-1]
        assert sent.left > 0 and sent.right > 0


def test_road_is_kept_through_a_run():
    from estimation.road_yaw import RoadYawTracker
    tracker = RoadYawTracker(config)
    tracker._yaw_deg = 10.0
    tracker.after_turn(+1)                       # a left turn mid-run carries the road with it
    assert tracker.get().yaw_deg == 10.0 + config.TURN_ANGLE_DEG
    tracker.reset()
    assert tracker.get().valid is False


def test_run_start_is_not_a_zone_change_to_pause_for():
    from robot_types import LocationObs, Zone
    b = Behaviour(config)
    b.mode = RunMode.QUIZ_INTERSECTION

    def loc(zone, s):
        return LocationObs(zone=zone, sign_read_ready=False, pause_point_ready=False,
                           turn_point_ready=False, dist_to_centre_m=s, conf=1.0, source="dashes",
                           t=0.0)
    b.step(None, None, None, None, None, None, [], False, 1.0, loc(Zone.READ_SIGN, -0.8))
    b.step(None, None, None, None, None, None, [UiCommand.START], False, 1.1,
           loc(Zone.READ_SIGN, -0.8))
    assert b.state != State.IDLE
    b.step(None, None, None, None, None, None, [], False, 1.2, loc(Zone.CONTINUE, None))
    assert not b.paused                          # the location tracker's reset, not a zone change
    b.step(None, None, None, None, None, None, [], False, 1.3, loc(Zone.READ_SIGN, -0.9))
    assert b.paused                              # a real zone change still pauses


def test_zone_bounds():
    assert config.READ_SIGN_LATEST_M == -0.93
    assert config.ZONE_BOUNDS_M == (-0.93, -0.65, -0.55, -0.20, -0.05, 0.30)
    assert list(config.ZONE_BOUNDS_M) == sorted(config.ZONE_BOUNDS_M)
