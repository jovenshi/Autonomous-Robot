"""Tests for tools/calibrate.py and tools/config_writer.py.

Board photos come from a virtual pinhole camera on the robot's centreline,
so every true ground position is known. No camera, no windows.

Run from the autonomous/ folder:  python -m pytest test/test_calibrate.py
"""
import json
import math
import os
import sys
import types

import cv2
import numpy as np
import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from tools import calibrate, config_writer  # noqa: E402

LANE = (320, 240)
FRAME = (640, 480)
FX, U0, V0 = 500.0, 320.0, 240.0
CAM_X, CAM_H, PITCH = 0.10, 0.25, math.radians(45)
SQUARE = 0.025


def cam_point(forward, lateral):
    """Ground (forward, lateral) -> camera coordinates (x right, y down, z ahead)."""
    c, s = math.cos(PITCH), math.sin(PITCH)
    dx = forward - CAM_X
    return np.array([lateral, -dx * s + CAM_H * c, dx * c + CAM_H * s])


def project(forward, lateral):
    x, y, z = cam_point(forward, lateral)
    return FX * x / z + U0, FX * y / z + V0


def ground_to_image_h():
    pts = [(f, l) for f in (0.2, 0.4, 0.6) for l in (-0.2, 0.0, 0.2)]
    return cv2.findHomography(np.float64(pts), np.float64([project(f, l) for f, l in pts]), 0)[0]


def true_ground(lane_x, lane_y):
    p = np.linalg.inv(ground_to_image_h()) @ np.array([lane_x * 2, lane_y * 2, 1.0])
    return p[:2] / p[2]


def board_texture(square_m=SQUARE, px_per_m=2000.0):
    sq = int(square_m * px_per_m)
    tex = np.full((7 * sq + 2 * sq, 10 * sq + 2 * sq, 3), 255, np.uint8)
    for r in range(7):
        for c in range(10):
            if (r + c) % 2 == 0:
                tex[sq + r * sq:sq + (r + 1) * sq, sq + c * sq:sq + (c + 1) * sq] = 0
    return tex, sq


def chessboard_image(near_m, lateral_m=0.0, square_m=SQUARE):
    """The board (9 x 6 inner corners, long side across) on green ground."""
    px_per_m = 2000.0
    tex, sq = board_texture(square_m, px_per_m)
    first_row_y, centre_x = sq + 6 * sq, sq + 5 * sq
    g_from_tex = np.array([[0, -1 / px_per_m, near_m + first_row_y / px_per_m],
                           [1 / px_per_m, 0, lateral_m - centre_x / px_per_m],
                           [0, 0, 1]])
    return cv2.warpPerspective(tex, ground_to_image_h() @ g_from_tex, FRAME,
                               flags=cv2.INTER_LINEAR, borderValue=(40, 150, 40))


def cfg_for_test(**overrides):
    values = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    values.update(FRONT_LANE_SIZE=LANE, FRONT_SIZE=FRAME, FRONT_BANDS=None, FRONT_GROUND_H=None,
                  FRONT_CAM_MATRIX=None, FRONT_DIST_COEFFS=None, CALIB_SQUARE_M=SQUARE,
                  CALIB_PATTERN=(9, 6), ROAD_WIDTH_M=0.30, CALIB_CHECK_NEAR_M=None,
                  CALIB_CHECK_LATERAL_M=None)
    values.update(overrides)
    return types.SimpleNamespace(**values)


SPOTS = [  # (near_m, lateral_m, role): what the virtual camera can see
    (0.22, 0.0, "fit"), (0.26, -0.04, "fit"), (0.26, 0.04, "fit"), (0.30, 0.0, "fit"),
    (0.34, 0.0, "fit"), (0.38, -0.05, "fit"), (0.38, 0.05, "fit"), (0.31, 0.03, "check"),
]


def make_session(folder, cfg, spots=SPOTS, entered=None):
    """Photos at the true spots; positions.csv gets `entered` (default the truth)."""
    calibrate.write_session_meta(folder, cfg)
    rows = []
    for i, (near, lat, role) in enumerate(spots):
        name = f"board_{i + 1:02d}.png"
        cv2.imwrite(os.path.join(folder, name), chessboard_image(near, lat))
        e_near, e_lat = entered[i] if entered and i in entered else (near, lat)
        rows.append({"image": name, "near_m": e_near, "lateral_m": e_lat, "role": role})
    calibrate.write_positions(folder, rows)
    return rows


# ---------------------------------------------------------------- the fit

def test_fit_matches_virtual_camera(tmp_path):
    cfg = cfg_for_test()
    make_session(str(tmp_path), cfg)
    result = calibrate.fit_session(str(tmp_path), cfg)
    assert result["rms_m"] < 0.0015
    assert all(b["found"] for b in result["boards"])
    assert result["check"]["passed"] and result["check"]["rms_m"] < 0.002
    for y_top, y_bottom, forward_m, half_road_px, m_per_px in result["bands"]:
        y_mid = (y_top + y_bottom) / 2
        lat_l, lat_r = true_ground(159.5 - 40, y_mid)[1], true_ground(159.5 + 40, y_mid)[1]
        assert forward_m == pytest.approx(true_ground(159.5, y_mid)[0], abs=0.002)
        assert m_per_px == pytest.approx((lat_r - lat_l) / 80, rel=0.02)
        assert half_road_px == round(0.15 / m_per_px)
    assert [b[2] for b in result["bands"]] == sorted([b[2] for b in result["bands"]], reverse=True)
    assert abs(result["camera_offset_m"]) < 0.003 and abs(result["camera_yaw_deg"]) < 0.5
    assert result["extrapolated"] == []
    assert os.path.exists(tmp_path / "preview.png")
    assert json.load(open(tmp_path / "result.json"))["bands"] == [list(b) for b in result["bands"]]


def test_misplaced_board_is_flagged_with_its_direction(tmp_path):
    # Board 4 is really at 0.30 m but entered as 0.292 (8 mm short).
    cfg = cfg_for_test()
    make_session(str(tmp_path), cfg, entered={3: (0.292, 0.0)})
    result = calibrate.fit_session(str(tmp_path), cfg)
    notes = {b["name"]: calibrate.board_note(b, cfg) for b in result["boards"]}
    level, note = notes["board_04.png"]
    assert level == "warn" and "farther" in note
    assert all(level == "ok" for name, (level, _) in notes.items() if name != "board_04.png")


def test_existing_band_rows_are_kept_and_compared(tmp_path):
    old = [(60, 80, 0.5, 100, 0.001), (100, 120, 0.4, 100, 0.001)]
    cfg = cfg_for_test(FRONT_BANDS=old)
    make_session(str(tmp_path), cfg)
    result = calibrate.fit_session(str(tmp_path), cfg)
    assert [tuple(b[:2]) for b in result["bands"]] == [(60, 80), (100, 120)]
    assert "was 50.0 cm" in calibrate.band_lines(result)[0]
    new_rows = calibrate.fit_session(str(tmp_path), cfg, new_rows=True)["bands"]
    assert len(new_rows) == cfg.CALIB_BAND_COUNT


def test_rows_outside_the_boards_are_reported(tmp_path):
    cfg = cfg_for_test()
    make_session(str(tmp_path), cfg)
    result = calibrate.fit_session(str(tmp_path), cfg, top_row=5, bottom_row=235)
    assert result["extrapolated"]
    assert any("outside the rows" in text for _, text in calibrate.report_lines(result, cfg))


def test_too_few_boards_and_missing_board(tmp_path):
    cfg = cfg_for_test()
    calibrate.write_session_meta(str(tmp_path), cfg)
    cv2.imwrite(str(tmp_path / "blank.png"), np.full((480, 640, 3), 120, np.uint8))
    calibrate.write_positions(str(tmp_path), [
        {"image": "blank.png", "near_m": 0.3, "lateral_m": 0.0, "role": "fit"}])
    with pytest.raises(ValueError, match="at least"):
        calibrate.fit_session(str(tmp_path), cfg)


def test_order_grid_handles_flips_and_rotation():
    yy, xx = np.mgrid[0:6, 0:9]
    want = np.dstack([100 + 10 * xx, 200 - 10 * yy]).astype(np.float32)  # row 0 lowest, col 0 left
    assert np.array_equal(calibrate.order_grid(want[::-1, ::-1].reshape(-1, 2), (9, 6)), want)
    yy, xx = np.mgrid[0:9, 0:6]                         # board turned 90 deg
    want_rot = np.dstack([100 + 10 * xx, 200 - 10 * yy]).astype(np.float32)
    detector = want_rot.transpose(1, 0, 2).reshape(-1, 2)
    assert np.array_equal(calibrate.order_grid(detector, (9, 6)), want_rot)


# ---------------------------------------------------------------- check

def test_check_passes_at_the_spot_and_fails_when_moved(tmp_path):
    cfg = cfg_for_test()
    make_session(str(tmp_path), cfg)
    values = calibrate.fit_config_values(calibrate.fit_session(str(tmp_path), cfg))
    cfg = cfg_for_test(**values)
    assert (cfg.CALIB_CHECK_NEAR_M, cfg.CALIB_CHECK_LATERAL_M) == (0.31, 0.03)
    good = calibrate.check_photo(chessboard_image(0.31, 0.03), cfg)
    assert good["passed"] and good["rms_m"] < 0.002
    moved = calibrate.check_photo(chessboard_image(0.32, 0.03), cfg)    # board 1 cm off the marks
    assert not moved["passed"]
    lines = calibrate.check_lines(moved, cfg)
    assert lines[0][0] == "bad" and "farther" in lines[1][1]


def test_check_needs_a_calibration():
    with pytest.raises(ValueError, match="FRONT_GROUND_H"):
        calibrate.check_photo(chessboard_image(0.3), cfg_for_test())


# ---------------------------------------------------------------- lens

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
DIST = np.array([-0.30, 0.10, 0.0, 0.0, 0.0])        # strong barrel distortion


def lens_views(n=15, seed=1):
    """Corner sets of a board held at varied angles, through the distorted lens."""
    rng = np.random.default_rng(seed)
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * SQUARE
    views = []
    while len(views) < n:
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.12, 0.02), rng.uniform(0.3, 0.5)])
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K, DIST)
        pts = pts.reshape(-1, 2)
        if pts.min() > 5 and pts[:, 0].max() < 635 and pts[:, 1].max() < 475:
            views.append(pts)
    return views


def test_lens_is_recovered_and_cells_and_duplicates_counted():
    views = lens_views()
    lens, rms = calibrate.calibrate_lens(views, (9, 6), SQUARE, FRAME)
    assert rms < 0.05
    assert lens.dist_coeffs[0] == pytest.approx(-0.30, abs=0.03)
    cells = set().union(*(calibrate.lens_cells(v, FRAME) for v in views))
    assert len(cells) >= 7
    assert not calibrate.is_new_view(views[0] + 3, [views[0]], 40)
    assert calibrate.is_new_view(views[0] + 100, [views[0]], 40)


def test_fit_with_lens_corrects_edge_boards():
    # Boards seen through the distorted lens: without the lens the fit is
    # poor; with the lens values the corners fit to well under a millimetre.
    lens = calibrate.Lens(K, DIST)
    spots = [(f, l) for f in (0.22, 0.30, 0.38) for l in (-0.06, 0.0, 0.06)]

    def board(near, lat, use_lens):
        ground = calibrate.ground_points(6, 9, near, SQUARE, lat)
        cam = np.array([cam_point(f, l) for f, l in ground.reshape(-1, 2)])
        raw, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), K, DIST)
        grid = calibrate.frame_to_lane(raw.reshape(-1, 2), (480, 640), LANE,
                                       lens if use_lens else None).reshape(6, 9, 2)
        return calibrate.Board("b", near, lat, "fit", grid, ground)

    _, rms_raw = calibrate.fit_boards([board(f, l, False) for f, l in spots])
    H, rms_lens = calibrate.fit_boards([board(f, l, True) for f, l in spots])
    assert rms_lens < 0.0005 < rms_raw
    gmap = calibrate.GroundMap(H, LANE, FRAME, lens)
    raw, _ = cv2.projectPoints(np.array([cam_point(0.30, 0.05)]), np.zeros(3), np.zeros(3), K, DIST)
    raw_lane = raw.reshape(-1, 2) * 0.5
    assert gmap.to_ground(raw_lane)[0] == pytest.approx([0.30, 0.05], abs=0.001)


def test_lens_impact_reports_millimetres():
    cfg = cfg_for_test(FRONT_BANDS=[(60, 80, 0.5, 100, 0.001), (200, 220, 0.3, 100, 0.001)])
    assert calibrate.lens_impact_mm(calibrate.Lens(K, DIST), cfg) > 5
    assert calibrate.lens_impact_mm(calibrate.Lens(K, np.zeros(5)), cfg) < 0.01


# ---------------------------------------------------------------- command line

def test_cli_fit_prints_and_applies(tmp_path, monkeypatch, capsys):
    cfg_values = vars(cfg_for_test())
    for name, value in cfg_values.items():
        monkeypatch.setattr(config, name, value)
    session = tmp_path / "s"
    session.mkdir()
    make_session(str(session), cfg_for_test())
    target = tmp_path / "config.py"
    target.write_text(open(config_writer.config_path()).read())
    monkeypatch.setattr(config_writer, "config_path", lambda: str(target))
    calibrate.main(["fit", str(session), "--apply"])
    out = capsys.readouterr().out
    assert "check photo" in out and "PASS" in out and "Written to config.py" in out
    written = {}
    exec(compile(target.read_text(), "config", "exec"), written)
    assert written["CALIB_CHECK_NEAR_M"] == 0.31 and len(written["FRONT_BANDS"]) == 5
    assert np.array(written["FRONT_GROUND_H"]).shape == (3, 3)
    assert os.path.exists(str(target) + ".bak")


# ---------------------------------------------------------------- config_writer

SOURCE = '''"""doc"""
A = 1                           # first value
BANDS = [
    (1, 2),
]
B = None                        # MEASURED: b
'''


def test_writer_replaces_only_named_values_and_keeps_comments():
    out = config_writer.replace_values(SOURCE, {"B": 0.5, "BANDS": [(3, 4), (5, 6)]})
    assert "A = 1                           # first value" in out
    assert "B = 0.5                         # MEASURED: b" in out
    assert "BANDS = [\n    (3, 4),\n    (5, 6),\n]" in out
    ns = {}
    exec(out, ns)
    assert ns["BANDS"] == [(3, 4), (5, 6)] and ns["A"] == 1


def test_writer_refuses_unknown_names_and_writes_atomically(tmp_path):
    with pytest.raises(KeyError, match="NOPE"):
        config_writer.replace_values(SOURCE, {"NOPE": 1})
    path = tmp_path / "config.py"
    path.write_text(SOURCE)
    backup = config_writer.write_values(str(path), {"A": np.float64(2.5)})
    assert open(backup).read() == SOURCE
    assert "A = 2.5" in path.read_text()


def test_real_config_has_every_name_calibration_writes():
    names = set(config_writer.assignment_spans(open(config_writer.config_path()).read()))
    result = {"bands": [(1, 2, 0.5, 3, 0.001)], "H": np.eye(3).tolist(), "square_m": 0.025,
              "road_width_m": 0.3, "check": {"near_m": 0.8, "lateral_m": 0.0}}
    lens_values = calibrate.lens_config_values(calibrate.Lens(K, DIST))
    assert set(calibrate.fit_config_values(result)) | set(lens_values) <= names


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
