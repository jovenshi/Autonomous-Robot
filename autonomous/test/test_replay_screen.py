"""tools/replay_screen.py: playing a recording on the display, without the
display (Playback and Picker draw BGR images; no pygame)."""
import csv
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from loop_log import COLUMNS  # noqa: E402
from robot_types import Frame  # noqa: E402
from test_perception import BANDS, CX, H, W, scene  # noqa: E402
from tools import replay_screen as rs  # noqa: E402
from tools.recording import RecordingWriter  # noqa: E402

N, FPS = 30, 30.0


@pytest.fixture(autouse=True)
def bands(monkeypatch):
    monkeypatch.setattr(config, "FRONT_BANDS", BANDS)
    monkeypatch.setattr(config, "FRONT_LANE_SIZE", (W, H))


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make_recording(tmp_path, with_log=True, with_side=True):
    rec = str(tmp_path / "run_20261115_120000")
    writer = RecordingWriter(rec, 1, 100)
    log_path = str(tmp_path / "run.csv") if with_log else None
    writer.start({"camera": "front", "mode": "QUIZ_INTERSECTION", "loop_log": log_path})
    img = cv2.resize(scene(line_x=lambda y: CX), (640, 480), interpolation=cv2.INTER_NEAREST)
    for i in range(N):
        writer.submit(Frame(img, 10.0 + i / FPS, i, "front"))
    writer.close()
    if with_side:
        side = RecordingWriter(os.path.join(rec, "side"), 1, 100)
        side.start({"camera": "side"})
        for i in range(0, N, 2):
            side.submit(Frame(np.full((240, 320, 3), 90, np.uint8), 10.0 + i / FPS, i, "side"))
        side.close()
    if with_log:
        with open(log_path, "w", newline="") as f:
            f.write("# gains\n")
            out = csv.writer(f)
            out.writerow(COLUMNS)
            for i in range(N):
                row = dict.fromkeys(COLUMNS, "")
                row.update(t=f"{10.0 + i / FPS:.4f}", state="LANE_FOLLOW" if i < 20 else "SAFE_STOP",
                           speed="0.2" if i < 20 else "0", steer="0.05", lane_conf="0.9",
                           lateral_err_m="0.01", yaw_deg="1.5", yaw_rate_dps="0.2",
                           mode="QUIZ_INTERSECTION", zone="READ_SIGN" if i > 10 else "",
                           s="-0.9" if i > 10 else "", loc_source="dashes")
                out.writerow([row[c] for c in COLUMNS])
    return rec


def test_plays_in_real_time_and_stops_at_the_end(tmp_path):
    clock = Clock()
    pb = rs.Playback(make_recording(tmp_path), config, clock=clock)
    assert pb.index == 0 and pb.row["state"] == "LANE_FOLLOW"
    assert pb.tick()                               # first frame due at once
    assert not pb.tick()                           # next one a frame later
    clock.now += 1 / FPS
    assert pb.tick() and pb.index == 2
    for _ in range(200):
        clock.now += 1 / FPS
        pb.tick()
    assert pb.index == N - 1 and pb.paused
    assert pb.row["state"] == "SAFE_STOP"


def test_buttons_step_seek_speed_and_back(tmp_path):
    pb = rs.Playback(make_recording(tmp_path), config, clock=Clock())
    pb.press("step")
    assert pb.paused and pb.index == 1
    pb.press("fwd5")
    assert pb.index == N - 1
    pb.press("back5")
    assert pb.index == 0
    pb.press("speed")
    assert pb.speed == 2.0
    assert pb.press("back") == "back"
    pb.paused, pb.index = True, N - 1
    pb.press("play")                               # at the end: plays again from the start
    assert pb.index == 0 and not pb.paused


def test_side_frames_follow_the_front_clock(tmp_path):
    pb = rs.Playback(make_recording(tmp_path), config, clock=Clock())
    pb.press("step")                               # front frame 1: side frame 0 is the newest
    assert pb.side_index == 0
    pb.press("step")
    assert pb.side_index == 1 and pb.side is not None


def test_draws_with_loop_log_and_the_timeline_seeks(tmp_path):
    pb = rs.Playback(make_recording(tmp_path), config, clock=Clock())
    img = pb.draw()
    assert img.shape == (480, 800, 3)
    labels = [label for label, _, _ in pb._telemetry_rows()]
    assert labels[0] == "state" and "yaw (rate)" in labels
    x, y, w, h = rs.CHART_RECT
    pb.tap((x + w * 3 // 4, y + h // 2))
    assert pb.index == pytest.approx(0.75 * (N - 1), abs=1)
    pb.draw()
    for zone in pb.zones:                          # every button answers a tap
        if zone.name != "chart":
            zx, zy, zw, zh = zone.rect
            pb.tap((zx + zw // 2, zy + zh // 2))


def test_without_loop_log_shows_the_replays_own_numbers(tmp_path):
    pb = rs.Playback(make_recording(tmp_path, with_log=False, with_side=False), config,
                     clock=Clock())
    assert pb.log_path is None and pb.row is None and pb.side_rows == []
    labels = [label for label, _, _ in pb._telemetry_rows()]
    assert labels[:3] == ["lateral", "heading", "lane conf"]
    assert pb.draw().shape == (480, 800, 3)


def test_moved_loop_log_is_found_in_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "AUTONOMOUS_DIR", str(tmp_path))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run_x.csv").write_text("t\n")
    assert rs.find_loop_log({"loop_log": "/elsewhere/run_x.csv"}) == \
        str(tmp_path / "logs" / "run_x.csv")
    assert rs.find_loop_log({}) is None


def test_picker_lists_newest_first_and_pages(tmp_path):
    base = tmp_path / "recordings"
    for i in range(rs.PICKER_ROWS + 2):
        d = base / f"rec_{i:02d}"
        d.mkdir(parents=True)
        (d / "frames.csv").write_text("seq,t,file,exposure_us,gain\n")
        (d / "meta.json").write_text(json.dumps({"camera": "front"}))
        os.utime(d, (1000 + i, 1000 + i))
    (base / "not_a_recording").mkdir()
    picker = rs.Picker(config, str(base))
    assert len(picker.recordings) == rs.PICKER_ROWS + 2
    assert os.path.basename(picker.recordings[0]) == f"rec_{rs.PICKER_ROWS + 1:02d}"
    assert picker.draw().shape == (480, 800, 3)
    first = picker.zones[0]
    assert picker.tap((first.rect[0] + 5, first.rect[1] + 5)) == picker.recordings[0]
    older = next(z for z in picker.zones if z.name == "older")
    picker.tap((older.rect[0] + 5, older.rect[1] + 5))
    assert picker.page == 1
    picker.draw()
    quit_zone = next(z for z in picker.zones if z.name == "quit")
    assert picker.tap((quit_zone.rect[0] + 5, quit_zone.rect[1] + 5)) == "quit"
