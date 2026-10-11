"""Tests for tools/: recording, record, tune_colors, calibrate, replay.
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
from tools import calibrate, record, replay, tune_colors
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


# ---------------------------------------------------------------- calibrate

FX, U0, V0 = 500.0, 320.0, 240.0
CAM_X, CAM_H, PITCH = 0.10, 0.25, math.radians(45)


def project(forward, lateral):
    """Virtual pinhole camera on the centreline, CAM_X ahead of the axle,
    CAM_H up, pitched down by PITCH. Ground (forward, lateral) -> image px."""
    c, s = math.cos(PITCH), math.sin(PITCH)
    dx = forward - CAM_X
    zc = dx * c + CAM_H * s
    yc = -dx * s + CAM_H * c
    return FX * lateral / zc + U0, FX * yc / zc + V0


def ground_to_image_h():
    pts = [(f, l) for f in (0.2, 0.4, 0.6) for l in (-0.2, 0.0, 0.2)]
    src = np.float64(pts)
    dst = np.float64([project(f, l) for f, l in pts])
    return cv2.findHomography(src, dst, 0)[0]


def chessboard_image(near_m, square_m=0.025):
    """The board (10 x 7 squares, 9 x 6 inner corners) on green ground as the
    virtual camera sees it."""
    px_per_m = 2000.0
    sq = int(square_m * px_per_m)
    margin = sq
    tex_w, tex_h = 10 * sq + 2 * margin, 7 * sq + 2 * margin
    tex = np.full((tex_h, tex_w, 3), 255, np.uint8)
    for r in range(7):
        for c in range(10):
            if (r + c) % 2 == 0:
                tex[margin + r * sq:margin + (r + 1) * sq, margin + c * sq:margin + (c + 1) * sq] = 0
    first_row_y = margin + 6 * sq                     # nearest inner-corner row in the texture
    centre_x = margin + 5 * sq                        # middle inner-corner column
    g_from_tex = np.array([[0, -1 / px_per_m, near_m + first_row_y / px_per_m],
                           [1 / px_per_m, 0, -centre_x / px_per_m],
                           [0, 0, 1]])
    img_from_tex = ground_to_image_h() @ g_from_tex
    return cv2.warpPerspective(tex, img_from_tex, (640, 480), flags=cv2.INTER_LINEAR,
                               borderValue=(40, 150, 40))


def true_ground(lane_x, lane_y):
    inv = np.linalg.inv(ground_to_image_h())
    p = inv @ np.array([lane_x * 2, lane_y * 2, 1.0])   # lane px -> full px (x2)
    return p[:2] / p[2]


def calib_args(**kw):
    argv = ["x.png", "--near-m", "0.30", "--road-width-m", "0.30", "--no-window"]
    for key, value in kw.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return calibrate.parse_args(argv)


def test_calibration_matches_virtual_camera():
    bands, report, preview = calibrate.calibrate(chessboard_image(0.30), calib_args(), (W, H))
    assert len(bands) == 5 and preview.shape == (H, W, 3)
    assert report["rms_m"] < 0.002
    assert abs(report["offset_m"]) < 0.003 and abs(report["yaw_deg"]) < 0.5
    for y_top, y_bottom, forward_m, half_road_px, m_per_px in bands:
        y_mid = (y_top + y_bottom) / 2
        forward_true, _ = true_ground(CX, y_mid)
        lat_l, lat_r = true_ground(CX - 40, y_mid)[1], true_ground(CX + 40, y_mid)[1]
        assert forward_m == pytest.approx(forward_true, abs=0.003)
        assert m_per_px == pytest.approx((lat_r - lat_l) / 80, rel=0.02)
        assert half_road_px == round(0.15 / m_per_px)
    assert [b[2] for b in bands] == sorted([b[2] for b in bands], reverse=True)  # far first


def test_calibration_extends_rows_on_request():
    bands, _, _ = calibrate.calibrate(chessboard_image(0.30), calib_args(top_row=60, bottom_row=200),
                                      (W, H))
    assert bands[0][0] >= 60 and bands[-1][1] <= 200
    assert bands[-1][2] == pytest.approx(true_ground(CX, (bands[-1][0] + bands[-1][1]) / 2)[0],
                                         abs=0.005)


def test_band_rows_and_widths_are_integers():
    bands, _, _ = calibrate.calibrate(chessboard_image(0.30), calib_args(), (W, H))
    assert all(isinstance(b[0], int) and isinstance(b[3], int) for b in bands)


def test_no_board_is_reported():
    with pytest.raises(SystemExit):
        calibrate.calibrate(np.full((480, 640, 3), 120, np.uint8), calib_args(), (W, H))


def test_order_grid_handles_flips_and_rotation():
    rows, cols = 6, 9
    yy, xx = np.mgrid[0:rows, 0:cols]
    want = np.dstack([100 + 10 * xx, 200 - 10 * yy]).astype(np.float32)  # row 0 lowest, col 0 left
    flipped = want[::-1, ::-1].reshape(-1, 2)
    assert np.array_equal(calibrate.order_grid(flipped, (9, 6)), want)
    # board turned 90 deg: 9 corners along forward, detector rows run up the image
    yy, xx = np.mgrid[0:9, 0:6]
    want_rot = np.dstack([100 + 10 * xx, 200 - 10 * yy]).astype(np.float32)
    detector = want_rot.transpose(1, 0, 2).reshape(-1, 2)
    assert np.array_equal(calibrate.order_grid(detector, (9, 6)), want_rot)


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
