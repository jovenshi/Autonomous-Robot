"""Stage 8 (docs/quiz3_plan.md): the side camera's surface box, its fixes in
the LocationTracker, the measuring tool and the wiring in main. Real side
frames from junction_side_1-3 and synthetic ones; no camera needed."""
import os
import sys
import threading

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
import main  # noqa: E402
from estimation.health import check_health  # noqa: E402
from estimation.location import LocationTracker  # noqa: E402
from perception.side_surface import classify_surface, process_side_frame  # noqa: E402
from robot_types import (AnnotationKind, Frame, HealthWarning, LocationEvidence,  # noqa: E402
                         SideResult, Surface)
from test_intersection_quiz import make_quiz_system  # noqa: E402
from test_location import cfg_with, junction_scene  # noqa: E402

SIDE_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "side")
W, H = config.SIDE_SIZE
GRASS_BGR, ROAD_BGR = (150, 175, 160), (110, 85, 80)   # the side webcam's washed-out colours


def side_image(road_frac):
    # A side frame of grass whose SIDE_BOX is road_frac road (left part).
    img = np.empty((H, W, 3), np.uint8)
    img[:] = GRASS_BGR
    x, y, w, h = config.SIDE_BOX
    img[y:y + h, x:x + int(round(w * road_frac))] = ROAD_BGR
    img[H - 40:] = (230, 230, 235)                        # the bright floor beyond the mat
    return img


def road_mask(road_frac):
    from perception.color import make_side_masks
    return make_side_masks(Frame(side_image(road_frac), 0.0, 1, "side"), config)[0]


# ---------------------------------------------------------------- surface box

@pytest.mark.parametrize("frac,surface", [(0.0, Surface.GRASS), (0.15, Surface.GRASS),
                                          (0.4, None), (0.7, Surface.ROAD), (1.0, Surface.ROAD)])
def test_box_is_classed_by_its_road_fraction(frac, surface):
    got, measured = classify_surface(road_mask(frac), config)
    assert got == surface and measured == pytest.approx(frac, abs=0.03)


@pytest.mark.parametrize("name,surface", [
    ("junction_side_3_000224.jpg", Surface.GRASS),   # s -0.30, approaching
    ("junction_side_1_000385.jpg", Surface.GRASS),
    ("junction_side_3_000286.jpg", Surface.ROAD),    # s 0.00, the cross road beside the robot
    ("junction_side_1_000517.jpg", Surface.ROAD),
    ("junction_side_2_000278.jpg", Surface.ROAD),
    ("junction_side_2_000206.jpg", None),            # s -0.40, a dark band crossing the box
])
def test_real_side_frames(name, surface):
    frame = Frame(cv2.imread(os.path.join(SIDE_DATA, name)), 1.0, 7, "side")
    result = process_side_frame(frame, config)
    assert result.surface == surface
    assert (result.t, result.seq, result.side) == (1.0, 7, None)
    kinds = [a.kind for a in result.annotations]
    assert kinds == [AnnotationKind.RECT, AnnotationKind.TEXT]
    assert all(a.cam_id == "side" for a in result.annotations)


# ---------------------------------------------------------------- tracker

ON = cfg_with(SIDE_FIXES_ENABLED=True)


def side(surface, seq, t=0.0):
    return SideResult(np.zeros((2, 2, 3), np.uint8), None, 100.0, t, seq, surface)


def ev(cross=None, seq=1, dashes=(), t=0.0):
    return LocationEvidence(cross, None, None, (), t, seq, tuple(dashes))


def held_tracker(cfg, s):
    # A pass begun at s on the crossing line, then frames with no dashes and
    # no speed: s holds and the estimate is suspect.
    tracker, seq = LocationTracker(cfg), 0
    for _ in range(cfg.LOCATION_START_FRAMES):
        seq += 1
        tracker.update(ev(cross=-s, seq=seq), 0.0, 0.0)
    for _ in range(3):
        seq += 1
        tracker.update(ev(seq=seq), 0.0, 0.0)
    assert tracker.suspect and tracker.get().source == "held"
    return tracker, seq


def feed_side(tracker, seq, side_seq, surfaces):
    for surface in surfaces:
        side_seq += 1
        obs = tracker.update(None, 0.0, 0.0, None, side(surface, side_seq))
    return obs, side_seq


def test_side_fix_when_held_and_near():
    tracker, seq = held_tracker(ON, ON.SIDE_FIXES_M["ROAD"] - 0.05)
    obs, n = feed_side(tracker, seq, 0, [Surface.GRASS] * 3 + [Surface.ROAD])
    assert obs.source == "held"                            # one ROAD frame: not settled
    obs, n = feed_side(tracker, seq, n, [Surface.ROAD] * (ON.SIDE_SURFACE_FRAMES - 1))
    assert obs.source == "side" and -obs.dist_to_centre_m == ON.SIDE_FIXES_M["ROAD"]
    assert not tracker.suspect


def test_side_fixes_are_off_by_default_and_need_a_held_estimate():
    for cfg in (config, ON):
        tracker, seq = LocationTracker(cfg), 0
        for _ in range(cfg.LOCATION_START_FRAMES):            # measured by the crossing line
            seq += 1
            tracker.update(ev(cross=0.2, seq=seq), 0.0, 0.0)
        obs, _ = feed_side(tracker, seq, 0, [Surface.GRASS] * 3 + [Surface.ROAD] * 3)
        assert obs.source == "crossing"                       # not suspect: no side fix
    tracker, seq = held_tracker(config, config.SIDE_FIXES_M["ROAD"])
    obs, _ = feed_side(tracker, seq, 0, [Surface.GRASS] * 3 + [Surface.ROAD] * 3)
    assert obs.source == "held"                               # SIDE_FIXES_ENABLED False


def test_side_fix_far_from_s_or_used_twice_is_ignored():
    tracker, seq = held_tracker(ON, ON.SIDE_FIXES_M["ROAD"] - ON.SIDE_FIX_GATE_M - 0.05)
    obs, n = feed_side(tracker, seq, 0, [Surface.GRASS] * 3 + [Surface.ROAD] * 3)
    assert obs.source == "held"                               # a seam well before the corner
    tracker, seq = held_tracker(ON, ON.SIDE_FIXES_M["ROAD"])
    obs, n = feed_side(tracker, seq, 0, [Surface.GRASS] * 3 + [Surface.ROAD] * 3)
    assert obs.source == "side"
    for i in range(3):                                        # suspect again
        tracker.update(ev(seq=seq + 10 + i), 0.0, 0.0)
    obs, n = feed_side(tracker, seq, n, [Surface.GRASS] * 3 + [Surface.ROAD] * 3)
    assert obs.source == "held"                               # ROAD already used this pass


def test_mixed_and_repeated_side_frames_change_nothing():
    tracker, seq = held_tracker(ON, ON.SIDE_FIXES_M["ROAD"])
    obs, n = feed_side(tracker, seq, 0, [Surface.GRASS] * 3 + [None, Surface.ROAD, None])
    assert obs.source == "held"
    tracker.update(None, 0.0, 0.0, None, side(Surface.ROAD, 50))
    obs = tracker.update(None, 0.0, 0.0, None, side(Surface.ROAD, 50))    # same frame again
    assert obs.source == "held"


# ---------------------------------------------------------------- measuring tool

def make_two_camera_recording(path):
    # Pushed from 1.6 m before the centre to 0.4 m past it at 0.1 m/s; the
    # side box is road between -0.18 and +0.12 (as on the mat), with a short
    # dark band at -0.45 (a seam).
    from tools.recording import RecordingWriter
    front, side_w = RecordingWriter(str(path), 1, 400), RecordingWriter(str(path / "side"), 1, 400)
    front.start({"camera": "front"})
    side_w.start({"camera": "side"})
    for i, s in enumerate(np.arange(-1.6, 0.4, 0.01)):
        t = 100.0 + i * 0.1
        front.submit(Frame(junction_scene(s), t, i + 1, "front"))
        road = 1.0 if -0.18 <= s < 0.12 or -0.46 <= s < -0.43 else 0.0
        side_w.submit(Frame(side_image(road), t + 0.05, i + 1, "side"))
    front.close()
    side_w.close()


def test_side_fixes_tool_measures_the_changes(tmp_path, capsys):
    from tools import side_fixes
    rec = tmp_path / "junction_side"
    make_two_camera_recording(rec)
    changes = side_fixes.side_changes(str(rec), config, str(tmp_path / "out"))
    names = [name for name, _ in changes]
    assert names == ["ROAD", "GRASS", "ROAD", "GRASS"]        # the seam, then the cross road
    found = side_fixes.near_junction(changes, config)
    assert found["ROAD"][0] == pytest.approx(-0.18, abs=0.04)
    assert found["GRASS"][0] == pytest.approx(0.12, abs=0.04)
    lines = side_fixes.summary_lines([found], config)
    assert lines[0].startswith("ROAD: median") and lines[1].startswith("GRASS: median")
    assert len(os.listdir(tmp_path / "out")) > 100


# ---------------------------------------------------------------- main

def test_side_camera_is_optional(monkeypatch, capsys):
    assert main._build_side_camera(cfg_with(SIDE_CAM_ENABLED=False), None) is None

    def fails(self):
        raise RuntimeError("Cannot open webcam")
    monkeypatch.setattr(main.CameraStream, "start", fails)
    assert main._build_side_camera(config, None) is None
    assert "side camera did not start" in capsys.readouterr().out


class FakeSideCam:
    def __init__(self, rate):
        self.rate, self.stopped = rate, False

    def fps(self):
        return self.rate

    def stop(self):
        self.stopped = True


def test_control_loop_uses_the_side_camera(tmp_path):
    from loop_log import LoopLog
    sys_ = make_quiz_system()
    sys_.side_cam = FakeSideCam(19.0)
    sys_.side_slot.set(side(Surface.ROAD, 1, t=1.0))
    sys_.loop_log = LoopLog(str(tmp_path), config)
    main.control_step(sys_, 1.0)
    assert main._fps(sys_) == {"front": 30.0, "side": 19.0}
    sys_.loop_log.close()
    with open(sys_.loop_log.path) as f:
        assert f.read().splitlines()[-1].endswith(",ROAD")
    main.shutdown(sys_)
    assert sys_.side_cam.stopped


def test_side_camera_fps_has_its_own_minimum():
    args = (None, None, None, None, None)
    ok = {"front": 30.0, "side": config.SIDE_CAM_FPS_MIN + 1}
    low = {"front": 30.0, "side": config.SIDE_CAM_FPS_MIN - 1}
    assert HealthWarning.CAM_FPS_LOW not in check_health(*args, ok, 0, False, 0.0, config)
    assert HealthWarning.CAM_FPS_LOW in check_health(*args, low, 0, False, 0.0, config)


def test_display_draws_the_side_box():
    from test_ui import bare_packet, screen_at
    screen, _ = screen_at(1.0)
    result = process_side_frame(Frame(side_image(1.0), 1.0, 3, "side"), config)
    screen.update(bare_packet(side=result))
    img = screen.draw(screen.packet, "camera")
    assert img.shape == (config.DISPLAY_SIZE[1], config.DISPLAY_SIZE[0], 3)
