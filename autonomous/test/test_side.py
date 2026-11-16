"""Stage 8 (docs/quiz3_plan.md): the side camera's band, its edges as fixes
in the LocationTracker, the measuring tool and the wiring in main. Real
side frames from truth_2 (wide view, tape positions) and synthetic ones; no
camera needed."""
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
from estimation.location import band_edge  # noqa: E402
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


@pytest.mark.parametrize("name,surface,frac", [
    ("truth_2_000423.jpg", Surface.GRASS, 0.00),     # tape -0.50: grass across the band
    ("truth_2_000523.jpg", None, 0.21),              # -0.30: the corner's grass leaving
    ("truth_2_000629.jpg", Surface.ROAD, 0.91),      # -0.10: the cross road
    ("truth_2_000670.jpg", Surface.ROAD, 0.89),      # 0.00
    ("truth_2_000808.jpg", Surface.GRASS, 0.00),     # +0.25: past the cross road
])
def test_real_side_frames(name, surface, frac):
    frame = Frame(cv2.imread(os.path.join(SIDE_DATA, name)), 1.0, 7, "side")
    result = process_side_frame(frame, config)
    assert result.surface == surface and result.road_frac == pytest.approx(frac, abs=0.03)
    assert (result.t, result.seq, result.side) == (1.0, 7, None)
    kinds = [a.kind for a in result.annotations]
    assert kinds == [AnnotationKind.RECT, AnnotationKind.TEXT]
    assert all(a.cam_id == "side" for a in result.annotations)


# ---------------------------------------------------------------- band edges

def test_band_edge_is_interpolated_between_frames():
    assert band_edge((1.0, 0.2), (1.1, 0.4), config) is None
    name, t = band_edge((1.0, 0.3), (1.1, 0.7), config)
    assert name == "ENTRY" and t == pytest.approx(1.05)
    name, t = band_edge((2.0, 0.9), (2.1, 0.4), config)
    assert name == "EXIT" and t == pytest.approx(2.08)


# ---------------------------------------------------------------- tracker

OFF = cfg_with(SIDE_FIXES_ENABLED=False)
ENTRY, EXIT = config.SIDE_EDGES_M["ENTRY"], config.SIDE_EDGES_M["EXIT"]
STEP = 0.01                                   # m per front frame, 0.1 s apart


def side(frac, seq, t):
    surface = Surface.ROAD if frac >= 0.6 else Surface.GRASS if frac <= 0.2 else None
    return SideResult(np.zeros((2, 2, 3), np.uint8), None, 100.0, t, seq, surface, (), frac)


def ev(cross=None, seq=1, dashes=(), t=0.0):
    return LocationEvidence(cross, None, None, (), t, seq, tuple(dashes))


def drive_through(cfg, true_start, est_start, edge_at, frac_after, steps):
    # A pass begun on the crossing line at est_start while the robot is
    # really at true_start; both then move STEP per front frame (dash
    # travel). A side frame halfway between front frames turns the band's
    # road fraction to frac_after once the true s passes edge_at.
    # Returns the tracker and the true s at the end.
    tracker, seq, t = LocationTracker(cfg), 0, 0.0
    dashes = [0.70 + 0.115 * i for i in range(4)]
    for _ in range(cfg.LOCATION_START_FRAMES):
        seq += 1
        tracker.update(ev(cross=-est_start, seq=seq, t=t, dashes=dashes), 0.0, t)
    true, frac = true_start, 1.0 - frac_after
    for i in range(steps):
        seq, t, true = seq + 1, t + 0.1, true + STEP
        dashes = [d - STEP for d in dashes]
        if dashes[0] < 0.66:
            dashes = dashes[1:] + [dashes[-1] + 0.115]
        tracker.update(ev(seq=seq, t=t, dashes=dashes), 0.0, t)
        if true + STEP / 2 >= edge_at:
            frac = frac_after
        tracker.update(None, 0.0, t + 0.05, None, side(frac, seq, t + 0.05))
    return tracker, true


def test_entry_edge_corrects_a_lagging_estimate():
    # The estimate 4 cm behind: at ENTRY it is set right, then follows.
    tracker, true = drive_through(config, ENTRY - 0.10, ENTRY - 0.14, ENTRY, 0.9, 15)
    obs = tracker.get()
    assert obs.source in ("side", "dashes")
    assert -obs.dist_to_centre_m == pytest.approx(true, abs=0.006)
    assert (obs.side_edge, round(obs.side_error_m, 2)) == ("ENTRY", 0.04)


def test_exit_edge_corrects_and_is_measured_once():
    tracker, true = drive_through(config, EXIT - 0.10, EXIT - 0.15, EXIT, 0.1, 15)
    obs = tracker.get()
    assert -obs.dist_to_centre_m == pytest.approx(true, abs=0.006)
    assert obs.side_edge == "EXIT" and obs.side_error_m == pytest.approx(0.05, abs=0.006)


def test_fixes_off_only_report_the_error():
    tracker, true = drive_through(OFF, ENTRY - 0.10, ENTRY - 0.14, ENTRY, 0.9, 15)
    obs = tracker.get()
    assert -obs.dist_to_centre_m == pytest.approx(true - 0.04, abs=0.006)   # still 4 cm behind
    assert obs.source == "dashes" and obs.side_error_m == pytest.approx(0.04, abs=0.006)


def test_edge_far_from_the_estimate_is_ignored():
    # A seam or shadow crossing the band 0.3 m before the corner.
    tracker, _ = drive_through(config, ENTRY - 0.40, ENTRY - 0.40, ENTRY - 0.30, 0.9, 15)
    obs = tracker.get()
    assert obs.side_edge == "" and obs.side_error_m is None and obs.source == "dashes"


def test_no_edge_without_a_pass_or_with_repeated_frames():
    tracker = LocationTracker(config)
    tracker.update(None, 0.0, 0.0, None, side(0.1, 1, 0.0))
    obs = tracker.update(None, 0.0, 0.1, None, side(0.9, 2, 0.1))
    assert obs.dist_to_centre_m is None and tracker.side_edge == ""
    tracker.update(None, 0.0, 0.2, None, side(0.9, 2, 0.1))         # same frame again
    assert tracker.side_edge == ""


# ---------------------------------------------------------------- measuring tool

def make_two_camera_recording(path):
    # Pushed from 1.6 m before the centre to 0.4 m past it at 0.1 m/s,
    # held 2 s at every 0.1 m; the band is road between ENTRY and EXIT,
    # with a short dark band at -0.45 (a seam).
    from tools.recording import RecordingWriter
    front, side_w = RecordingWriter(str(path), 1, 900), RecordingWriter(str(path / "side"), 1, 900)
    front.start({"camera": "front"})
    side_w.start({"camera": "side"})
    t, i, s = 100.0, 0, -1.6
    while s < 0.4:
        hold = abs(s * 10 - round(s * 10)) < 1e-6
        for _ in range(20 if hold else 1):
            i, t = i + 1, t + 0.1
            front.submit(Frame(junction_scene(s), t, i, "front"))
            road = 1.0 if ENTRY <= s < EXIT or -0.46 <= s < -0.43 else 0.0
            side_w.submit(Frame(side_image(road), t + 0.05, i, "side"))
        s = round(s + 0.01, 2)
    front.close()
    side_w.close()


def test_side_fixes_tool_measures_the_edges(tmp_path):
    from tools import side_fixes
    rec = tmp_path / "junction_side"
    make_two_camera_recording(rec)
    track = side_fixes.tape_track(str(rec), config, -1.6 + 0.0)
    assert track[1][0] == pytest.approx(-1.6) and track[1][-1] == pytest.approx(0.3, abs=0.11)
    edges = side_fixes.side_edges(str(rec), track, config, str(tmp_path / "out"))
    names = [name for name, _ in edges]
    assert names == ["ENTRY", "EXIT", "ENTRY", "EXIT"]          # the seam, then the cross road
    found = side_fixes.near_expected(edges, config)
    assert found["ENTRY"][0] == pytest.approx(ENTRY, abs=0.02)
    assert found["EXIT"][0] == pytest.approx(EXIT, abs=0.02)
    lines = side_fixes.summary_lines([found], config)
    assert lines[0].startswith("ENTRY: median") and lines[1].startswith("EXIT: median")
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
    sys_.side_slot.set(side(1.0, 1, 1.0))
    sys_.loop_log = LoopLog(str(tmp_path), config)
    main.control_step(sys_, 1.0)
    assert main._fps(sys_) == {"front": 30.0, "side": 19.0}
    sys_.loop_log.close()
    with open(sys_.loop_log.path) as f:
        assert f.read().splitlines()[-1].endswith(",ROAD,,")    # no edge yet
    main.shutdown(sys_)
    assert sys_.side_cam.stopped


def test_side_check_is_printed_once_per_edge(capsys):
    from robot_types import LocationObs, Zone
    sys_ = make_quiz_system()
    loc = LocationObs(Zone.TURN, True, True, True, 0.2, 1.0, "side", 0.0,
                      side_edge="ENTRY", side_error_m=0.042)
    main._report_side_check(sys_, loc)
    main._report_side_check(sys_, loc)
    out = capsys.readouterr().out.splitlines()
    assert out == ["Side camera ENTRY edge: estimate was 4.2 cm behind (corrected)"]


def test_quiz_panel_shows_the_side_check():
    from robot_types import LocationObs, Zone
    from test_ui import bare_packet
    from ui.panels import quiz_rows
    loc = LocationObs(Zone.TURN, True, True, True, 0.2, 1.0, "side", 0.0,
                      side_edge="EXIT", side_error_m=-0.051)
    rows = dict((label, value) for label, value, _ in quiz_rows(bare_packet(location=loc), config))
    assert rows["side check"] == "exit -5.1 cm"


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


def test_edge_during_a_front_camera_stall_is_ignored():
    # straight_3: the front frames stop for 3 s; an edge then can't be
    # matched to the estimate, so it neither corrects nor reports.
    tracker, seq = LocationTracker(config), 0
    dashes = [0.70 + 0.115 * i for i in range(4)]
    for i in range(config.LOCATION_START_FRAMES + 3):
        seq += 1
        tracker.update(ev(cross=-(EXIT - 0.05), seq=seq, t=i * 0.1, dashes=dashes), 0.0, i * 0.1)
    t = seq * 0.1 + 3.0                                        # long after the last front frame
    tracker.update(None, 0.0, t, None, side(0.9, 1, t))
    obs = tracker.update(None, 0.0, t + 0.05, None, side(0.1, 2, t + 0.05))
    assert obs.side_edge == "" and -obs.dist_to_centre_m == pytest.approx(EXIT - 0.05)
