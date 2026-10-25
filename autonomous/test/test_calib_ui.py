"""Tests for tools/calib_ui.py and tools/guide_draw.py: the guided screens,
driven by button names and taps with a fake camera showing virtual boards.
No camera, no display (the display loop test uses SDL's dummy driver).

Run from the autonomous/ folder:  python -m pytest test/test_calib_ui.py
"""
import os
import sys
import time

import numpy as np
import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from test_calibrate import SPOTS, cfg_for_test, chessboard_image  # noqa: E402
from tools import calib_ui, calibrate, config_writer, guide_draw  # noqa: E402


class FakeCamera:
    """Shows one image at a time, with the same fast detection as LiveBoard."""

    def __init__(self, cfg):
        self.cfg, self.view = cfg, None

    def show(self, image):
        corners = calibrate.find_board_corners(image, self.cfg.CALIB_PATTERN, fast=True)
        self.view = calib_ui.BoardView(image, corners, time.monotonic())

    def latest(self):
        return self.view


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    target = tmp_path / "config.py"
    target.write_text(open(config_writer.config_path()).read())
    monkeypatch.setattr(config_writer, "config_path", lambda: str(target))
    return cfg_for_test(DISPLAY_SIZE=(800, 480), CALIB_POSITIONS=tuple(SPOTS),
                        CALIB_DIR=str(tmp_path / "calibration"), CALIB_MIN_FIT_BOARDS=4)


def drawn(app):
    img = app.draw()
    assert img.shape == (480, 800, 3)
    return img


def place_every_spot(app, cam, spots=SPOTS):
    for near, lat, role in spots:
        assert app.page == "place"
        cam.show(chessboard_image(near, lat))
        drawn(app)
        app.press("capture")
        assert app.page == "confirm" and app.entry == [near, lat]
        drawn(app)
        app.press("keep")


def finish_fit(app):
    assert app.page == "fitting"
    drawn(app)
    app.tick()
    assert app.page == "results"
    drawn(app)


def test_full_session_then_apply_then_check(cfg, capsys):
    cam = FakeCamera(cfg)
    app = calib_ui.CalibApp(cfg, cam)
    assert app.page == "welcome"
    drawn(app)
    app.press("square:0.0001")
    assert app.square_m == pytest.approx(0.0251)
    app.press("square:-0.0001")
    app.key("enter")                                  # START is the green button
    assert app.page == "place" and os.path.isdir(app.folder)
    place_every_spot(app, cam)
    finish_fit(app)
    assert app.result["check"]["passed"] and app.selected is None
    app.press("preview")
    drawn(app)
    app.key("escape")                                 # BACK
    app.press("apply")
    assert app.page == "applied" and cfg.FRONT_GROUND_H is not None
    assert cfg.CALIB_CHECK_NEAR_M == 0.31
    drawn(app)
    app.press("done")
    assert app.done
    out = capsys.readouterr().out
    assert "[calib] Spot 1 of 8" in out and "CHECK photo" in out and "Wrote FRONT_BANDS" in out

    check = calib_ui.CalibApp(cfg, cam, mode="check")
    assert check.page == "check"
    cam.show(chessboard_image(0.31, 0.03))
    drawn(check)
    check.press("check")
    assert check.page == "check_result" and check.check_result["passed"]
    drawn(check)


def test_misplaced_board_is_selected_and_retaken(cfg):
    cam = FakeCamera(cfg)
    app = calib_ui.CalibApp(cfg, cam)
    app.press("start")
    for near, lat, role in SPOTS:
        cam.show(chessboard_image(near, lat))
        app.press("capture")
        if (near, lat) == (0.30, 0.0):
            app.press("near:-0.001")                  # entered 8 mm short
            for _ in range(7):
                app.key("down")
        app.press("keep")
    finish_fit(app)
    assert app.selected == "board_04.png"             # the flagged board is preselected
    app.press("retake")
    assert app.page == "place" and app.spot == 3
    cam.show(chessboard_image(0.30, 0.0))
    app.press("capture")
    assert app.entry == [0.292, 0.0]                  # starts from what was entered
    for _ in range(8):
        app.key("up")
    app.press("keep")
    finish_fit(app)                                   # straight back to the fit
    assert all(calibrate.board_note(b, cfg)[0] == "ok" for b in app.result["boards"])


def test_tapping_a_result_row_selects_it(cfg):
    cam = FakeCamera(cfg)
    app = calib_ui.CalibApp(cfg, cam)
    app.press("start")
    place_every_spot(app, cam)
    finish_fit(app)
    row = next(z for z in app.zones if z.name == "row:board_02.png")
    app.tap((row.rect[0] + 5, row.rect[1] + 5))
    assert app.selected == "board_02.png"


def test_no_board_disables_capture_and_skip_finish(cfg):
    cam = FakeCamera(cfg)
    app = calib_ui.CalibApp(cfg, cam)
    app.press("start")
    cam.show(np.full((480, 640, 3), 120, np.uint8))
    drawn(app)
    app.press("capture")
    assert app.page == "place"                         # disabled: nothing happened
    finish = next(b for b in app.buttons() if b and b.name == "finish")
    assert not finish.enabled
    place_every_spot(app, cam, SPOTS[:4])              # four fit boards...
    assert next(b for b in app.buttons() if b and b.name == "finish").enabled
    app.press("finish")
    finish_fit(app)
    assert app.result["check"] is None


def test_resume_goes_to_the_next_spot_or_the_fit(cfg, capsys):
    cam = FakeCamera(cfg)
    app = calib_ui.CalibApp(cfg, cam)
    app.press("start")
    place_every_spot(app, cam, SPOTS[:3])
    app.press("quit")
    assert app.done and "--resume" in capsys.readouterr().out
    again = calib_ui.CalibApp(cfg, cam, folder=app.folder)
    assert again.page == "place" and again.spot == 3
    place_every_spot(again, cam, SPOTS[3:])
    finish_fit(again)
    done = calib_ui.CalibApp(cfg, cam, folder=app.folder)
    finish_fit(done)


def test_check_without_calibration_explains(cfg):
    app = calib_ui.CalibApp(cfg, FakeCamera(cfg), mode="check")
    assert app.page == "message" and "first" in app.message
    drawn(app)


def test_lens_flow_and_refit(cfg, monkeypatch):
    cfg.CALIB_LENS_MIN_PHOTOS, cfg.CALIB_LENS_MIN_CELLS = 4, 1
    cfg.FRONT_BANDS = [(60, 80, 0.5, 100, 0.001)]
    cam = FakeCamera(cfg)
    session = calib_ui.CalibApp(cfg, cam)              # a finished session to refit later
    session.press("start")
    place_every_spot(session, cam)
    monkeypatch.setattr(calibrate, "calibrate_lens", lambda sets, pattern, sq, size: (
        calibrate.Lens(np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]), np.zeros(5)), 0.2))
    app = calib_ui.CalibApp(cfg, cam, mode="lens")
    app.press("start")
    assert app.page == "lens"
    for near, lat, _ in SPOTS[:4]:
        cam.show(chessboard_image(near, lat))
        drawn(app)
        app.press("capture")
    cam.show(chessboard_image(*SPOTS[0][:2]))
    drawn(app)                                         # same view again: CAPTURE disabled
    assert not next(b for b in app.buttons() if b and b.name == "capture").enabled
    app.press("compute")
    assert app.page == "lens_result"
    drawn(app)
    app.press("apply")
    assert cfg.FRONT_CAM_MATRIX is not None and app.page == "message"
    drawn(app)
    app.press("refit")
    finish_fit(app)
    assert app.result["lens_used"]


def test_guide_helpers():
    assert guide_draw.wrap("one two three four", 18, 10_000) == ["one two three four"]
    assert len(guide_draw.wrap("one two three four", 18, 60)) > 1
    assert guide_draw.clip_text("x" * 200, 14, 100).endswith("...")
    zones = [guide_draw.Zone((0, 0, 10, 10), "a"), guide_draw.Zone((5, 5, 10, 10), "b")]
    assert guide_draw.hit(zones, (6, 6)) == "b" and guide_draw.hit(zones, (1, 1)) == "a"
    assert guide_draw.hit(zones, (50, 50)) is None


def test_display_loop_runs_and_stops(cfg, monkeypatch):
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    class TwoFrames:
        done, frames = False, 0

        def draw(self):
            self.frames += 1
            self.done = self.frames >= 2
            return np.zeros((480, 800, 3), np.uint8)

        def tick(self):
            pass

    app = TwoFrames()
    assert calib_ui.run(app, cfg) and app.frames == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
