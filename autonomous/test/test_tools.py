"""Tests for tools/: recording, record, tune_colors, replay, imu_drift_test,
weave_report (calibrate: test_calibrate.py).
No camera, no windows, no motors.

Run from the robot/ folder:  python -m pytest tests
"""
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from robot_types import Frame
from test_perception import BANDS, CX, H, W, scene
from tools import record, replay, tune_colors
from tools.recording import RecordingWriter, iter_frames, read_index, read_meta


@pytest.fixture(autouse=True)
def lane_config(monkeypatch):
    monkeypatch.setattr(config, "FRONT_LANE_SIZE", (W, H))


def make_recording(path, offsets_px, dt=1 / 30):
    writer = RecordingWriter(str(path), png_level=1, queue_size=100)
    writer.start({"camera": "front"})
    for i, off in enumerate(offsets_px):
        img = cv2.resize(scene(line_x=lambda y, o=off: CX + o), (640, 480),
                         interpolation=cv2.INTER_NEAREST)
        writer.submit(Frame(img, 10.0 + i * dt, i + 1, "front"), 8000, 2.0)
    writer.close()
    return writer


# ---------------------------------------------------------------- recording

def test_recording_round_trip_is_lossless(tmp_path):
    rec = tmp_path / "rec"
    writer = make_recording(rec, [0, 10, 20])
    assert writer.saved == 3 and writer.dropped == 0
    rows = read_index(str(rec))
    assert [r["seq"] for r in rows] == ["1", "2", "3"] and rows[0]["exposure_us"] == "8000"
    frames = list(iter_frames(str(rec)))
    expected = cv2.resize(scene(line_x=lambda y: CX + 10), (640, 480),
                          interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(frames[1].image, expected)
    assert frames[2].t == pytest.approx(10.0 + 2 / 30) and frames[2].seq == 3
    assert read_meta(str(rec)) == {"camera": "front"}


def test_writer_skips_when_queue_full(tmp_path):
    writer = RecordingWriter(str(tmp_path / "rec"), 1, queue_size=1)
    frame = Frame(np.zeros((4, 4, 3), np.uint8), 0.0, 1, "front")
    assert writer.submit(frame) and not writer.submit(frame)
    assert writer.dropped == 1


def test_existing_folder_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        RecordingWriter(str(tmp_path), 1, 10).start({})


# ---------------------------------------------------------------- record

class FakeCsi:
    def __init__(self, cam_num, size, cfg):
        self.n = 0

    def open(self):
        pass

    def read(self):
        time.sleep(0.005)
        self.n += 1
        return np.full((480, 640, 3), self.n % 255, np.uint8)

    def get_exposure(self):
        return 9000, 1.5

    def close(self):
        pass


def test_record_saves_only_while_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "CsiSource", FakeCsi)
    writer = RecordingWriter(str(tmp_path / "rec"), 1, 100)
    recording = threading.Event()
    stream, _ = record.build(config, writer, recording)
    writer.start(record.make_meta(config))
    stream.start()
    time.sleep(0.1)
    assert writer.saved == 0 and writer._queue.qsize() == 0
    recording.set()
    time.sleep(0.15)
    recording.clear()
    stream.stop()
    writer.close()
    rows = read_index(str(tmp_path / "rec"))
    assert len(rows) >= 5 and rows[0]["exposure_us"] == "9000"
    assert read_meta(str(tmp_path / "rec"))["frame_size"] == list(config.FRONT_SIZE)


# ---------------------------------------------------------------- tune_colors

def test_config_with_overrides_only_given_values():
    cfg = tune_colors.config_with({"YELLOW_H_MIN": 5})
    assert cfg.YELLOW_H_MIN == 5 and cfg.YELLOW_H_MAX == config.YELLOW_H_MAX
    assert config.YELLOW_H_MIN != 5                 # the config module itself is untouched
    assert cfg.CONTROL_HZ == config.CONTROL_HZ


def test_render_grid_uses_real_masks(monkeypatch):
    monkeypatch.setattr(config, "FRONT_BANDS", BANDS)
    frame = Frame(cv2.resize(scene(), (640, 480), interpolation=cv2.INTER_NEAREST), 0.0, 1, "front")
    grid, masks = tune_colors.render(frame, tune_colors.config_with({}))
    assert grid.shape == (2 * H, 2 * W, 3)
    assert masks.yellow[120, int(CX)] == 255
    narrow = tune_colors.config_with({"YELLOW_H_MIN": 100, "YELLOW_H_MAX": 120})
    _, masks = tune_colors.render(frame, narrow)
    assert not masks.yellow.any()


def test_render_without_bands(monkeypatch):
    monkeypatch.setattr(config, "FRONT_BANDS", None)
    frame = Frame(scene(), 0.0, 1, "front")
    grid, _ = tune_colors.render(frame, tune_colors.config_with({}))
    assert grid.shape == (2 * H, 2 * W, 3)


# ---------------------------------------------------------------- replay

def test_replay_prints_one_line_per_frame(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "FRONT_BANDS", BANDS)
    monkeypatch.setattr(config, "STEER_SMOOTH", 0.0)
    rec = tmp_path / "rec"
    make_recording(rec, [0, 20, -20])
    replay.run(str(rec), speed=0, window=False, cfg=config)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    steers = [float(line.split("steer ")[1].split()[0]) for line in lines]
    assert steers[0] == pytest.approx(0.0, abs=0.02)
    assert steers[1] < 0 < steers[2]                    # line right -> steer right, and back


def test_replay_needs_bands(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FRONT_BANDS", None)
    with pytest.raises(SystemExit):
        replay.run(str(tmp_path), speed=0, window=False, cfg=config)


# ---------------------------------------------------------------- imu_drift_test

from robot_types import ImuSample, MotorCmd  # noqa: E402
from tools import imu_drift_test  # noqa: E402


def imu_sample(yaw=0.0, rate=0.0, t=0.0, drift=0.0):
    return ImuSample(yaw, rate, True, 3, drift, t)


def test_yaw_change_per_second():
    assert imu_drift_test.yaw_change_dps(imu_sample(yaw=12.0, t=1.1), imu_sample(yaw=10.0, t=1.0)) \
        == pytest.approx(20.0)
    assert imu_drift_test.yaw_change_dps(imu_sample(t=1.0), None) is None
    assert imu_drift_test.yaw_change_dps(imu_sample(t=1.0), imu_sample(t=1.0)) is None


@pytest.mark.parametrize("rate,change,flag", [
    (20.0, 19.0, False),          # agree (turning left)
    (-20.0, -19.0, False),        # agree (turning right)
    (20.0, -19.0, True),          # disagree
    (-20.0, 19.0, True),
    (1.0, -19.0, False),          # gyro rate too small to judge
    (20.0, -1.0, False),          # yaw change too small to judge
    (20.0, None, False),          # no previous sample
])
def test_sign_mismatch(rate, change, flag):
    assert imu_drift_test.sign_mismatch(rate, change, config) == flag


def test_status_line_flags_mismatch():
    line = imu_drift_test.status_line(imu_sample(yaw=5.0, rate=20.0), -19.0, True)
    assert "SIGN MISMATCH" in line and "+20.00" in line and "-19.00" in line
    assert "SIGN MISMATCH" not in imu_drift_test.status_line(imu_sample(), None, False)


class FakeDriftSaber:
    def __init__(self):
        self.sent = []

    def set(self, cmd):
        self.sent.append(cmd)


def test_report_feeds_motors_and_counts_mismatches(monkeypatch, capsys):
    monkeypatch.setattr(config, "DRIFT_TEST_PRINT_HZ", 1000)
    saber, run = FakeDriftSaber(), imu_drift_test.Run()
    samples = iter([imu_sample(yaw=0.0, rate=20.0, t=0.0), imu_sample(yaw=-2.0, rate=20.0, t=0.1)])

    class Slot:
        def get(self):
            return next(samples, None) or run.last

    imu_drift_test.report(Slot(), saber, run, 0.02, config)
    assert saber.sent and saber.sent[0] == MotorCmd(config.DRIFT_TEST_SPEED, config.DRIFT_TEST_SPEED)
    assert run.first.t == 0.0 and run.mismatches == 1
    assert "SIGN MISMATCH" in capsys.readouterr().out


def test_summary():
    run = imu_drift_test.Run(first=imu_sample(yaw=0.0, t=0.0),
                             last=imu_sample(yaw=1.0, t=10.0, drift=0.05))
    text = imu_drift_test.summary(run)
    assert "+0.050" in text and "+1.00 deg" in text and "+0.100 deg/s" in text
    assert imu_drift_test.summary(imu_drift_test.Run()) == "No IMU samples received"


# ---------------------------------------------------------------- weave_report

import datetime  # noqa: E402

from loop_log import LoopLog  # noqa: E402
from robot_types import DriveCmd, LaneEstimate, State  # noqa: E402
from tools import weave_report  # noqa: E402


def write_weave_log(directory):
    # Known numbers. IDLE rows (huge error) must be ignored. Segment 1:
    # lateral +/-10 mm, heading +/-0.02 rad, steer flips 3 times in 0.3 s.
    # A SAFE_STOP row splits it from segment 2 (0.1 s, no lateral, no flips).
    rows = [(0.0, State.IDLE, 0.5, 0.5, 0.0)]
    for i, sign in enumerate((1, -1, 1, -1)):
        rows.append((1.0 + i * 0.1, State.LANE_FOLLOW, sign * 0.01, sign * 0.02, sign * 0.1))
    rows.append((1.4, State.SAFE_STOP, -0.01, 0.02, 0.0))
    rows += [(2.0, State.LANE_FOLLOW, None, 0.02, 0.1), (2.1, State.LANE_FOLLOW, None, 0.02, 0.1)]
    log = LoopLog(str(directory), config, start=datetime.datetime(2026, 10, 11, 16, 0, 0))
    for t, state, lateral, heading, steer in rows:
        log.write(t, state, LaneEstimate(lateral, heading, 0.9, None, t), None,
                  DriveCmd(speed=0.7, steer=steer))
    log.close()
    return log.path


def test_weave_report_known_numbers(tmp_path):
    gains, rows = weave_report.read_log(write_weave_log(tmp_path))
    assert gains["KD_YAW"] == str(config.KD_YAW)
    s = weave_report.summarize(rows)
    assert s["rows"] == 6
    assert s["seconds"] == pytest.approx(0.4)
    assert s["lat_rms_mm"] == pytest.approx(10.0)
    assert s["lat_peak_mm"] == pytest.approx(10.0)
    assert s["head_rms_deg"] == pytest.approx(math.degrees(0.02))
    assert s["flips_per_s"] == pytest.approx(3 / 0.4)


def test_weave_report_no_lane_follow_rows():
    s = weave_report.summarize([{"t": "0", "state": "IDLE", "lateral_err_m": "0.1",
                                 "heading_err_rad": "0", "steer": "0"}])
    assert s["rows"] == 0 and s["lat_rms_mm"] is None and s["flips_per_s"] is None
    assert "lat rms -- peak -- mm" in weave_report.format_line("x.csv", {}, s)


def test_weave_report_one_line_per_log(tmp_path, monkeypatch, capsys):
    a = write_weave_log(tmp_path / "a")
    b = write_weave_log(tmp_path / "b")
    monkeypatch.setattr(sys, "argv", ["weave_report", a, b])
    weave_report.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("run_20261011_160000.csv  KD_YAW=")
    assert "lat rms 10.0 peak 10.0 mm" in lines[0] and "steer flips 7.50/s" in lines[0]
