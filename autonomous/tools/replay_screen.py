"""Play a recording on the robot's 7" display: front and side camera with the
robot's own annotations, telemetry, and a timeline of the whole run.

    python -m tools.replay_screen                         # pick a recording on the screen
    python -m tools.replay_screen recordings/run_20261115_223949
    python -m tools.replay rec_dir --screen               # the same, from tools.replay
    (or the REPLAY tile in python -m tools.menu)

Over SSH it opens on the robot's screen, like main.py; nothing else is needed.

Each front frame goes through process_front_frame (lane, crossing line,
flare, dashes, signs at SIGN_HZ) and each side frame through
process_side_frame, and both are drawn with the display's overlays
(ui/overlay.py), so what you see is what the robot saw. The motors are never
touched.

Telemetry is the robot's own, from the loop log of the run (meta.json names
it; recordings made by main.py, RECORD): state, speed, steer, lane, yaw,
zone, s and sign at each frame's time, and a timeline of state, steer, speed
and lateral error with a playhead (tap it to jump there). Recordings made by
tools/record.py have no loop log: the panel then shows the replay's own lane
measurement and junction location (s only moves on camera fixes: no speed
command was recorded).

Layout (800 x 480): banner along the top; front camera (400 x 300) and side
camera (240 x 180) side by side; timeline under the front camera; telemetry
under the side camera; the 160 px button column on the right (PLAY/PAUSE,
STEP, -5 s, +5 s, SPEED, BACK).
Keys: space play/pause, n or right arrow step, left arrow back 5 s, q / Esc back.
"""
import argparse
import bisect
import csv
import os
import sys
import time

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from estimation.location import LocationTracker  # noqa: E402
from perception.front_lane import LaneTracker  # noqa: E402
from perception.pipeline import SignPacer, process_front_frame  # noqa: E402
from perception.side_surface import process_side_frame  # noqa: E402
from perception.signs import load_sign_model, sign_label  # noqa: E402
from robot_types import State  # noqa: E402
from tools.guide_draw import Button, Zone, clip_text, draw_button, hit  # noqa: E402
from tools.recording import load_frame, read_index, read_meta  # noqa: E402
from ui.overlay import (AMBER, BG, BLUE, GREEN, LINE, MUTED, PANEL, PURPLE, RED, TEXT,  # noqa: E402
                        YELLOW, draw_annotations, draw_front_overlay, draw_no_data,
                        draw_side_overlay, draw_sign_overlay)
from ui.panels import state_colour  # noqa: E402
from ui.text import draw_text, text_width  # noqa: E402

AUTONOMOUS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = "replay"
BANNER_H = 28
MAIN_W = 640
FRONT_RECT = (0, BANNER_H, 400, 300)
SIDE_RECT = (400, BANNER_H, 240, 180)
CHART_RECT = (0, BANNER_H + 300, 400, 480 - BANNER_H - 300)
TELEMETRY_RECT = (400, BANNER_H + 180, 240, 480 - BANNER_H - 180)
SPEEDS = (0.25, 0.5, 1.0, 2.0)
SEEK_S = 5.0
LOG_MATCH_S = 0.2                 # s; a loop row further than this from the frame isn't its own
STATE_TINTS = {State.APPROACH: YELLOW, State.CROSS: AMBER, State.REACQUIRE: BLUE}
PICKER_ROWS = 8


# ---------------------------------------------------------------- loop log

def read_loop_log(path, t0=-np.inf, t1=np.inf):
    # The loop log's rows from t0 to t1 as dicts (t as float), '#' lines skipped.
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    out = []
    for row in rows:
        try:
            row["t"] = float(row["t"])
        except (KeyError, ValueError):
            continue
        if t0 <= row["t"] <= t1:
            out.append(row)
    return out


def find_loop_log(meta, override=None):
    # The run's loop log path, or None: --log, meta.json's loop_log, or a
    # file of that name in logs/ (the recording was copied from elsewhere).
    for path in (override, meta.get("loop_log")):
        if not path:
            continue
        if os.path.exists(path):
            return path
        local = os.path.join(AUTONOMOUS_DIR, config.LOOP_LOG_DIR, os.path.basename(path))
        if os.path.exists(local):
            return local
    return None


def _f(row, key):
    # A float cell, or None when empty.
    value = row.get(key, "") if row is not None else ""
    try:
        return float(value) if value != "" else None
    except ValueError:
        return None


# ---------------------------------------------------------------- playback

class Playback:
    """One recording: frames, the side frames, the loop log, the play head,
    and the picture for the current frame. No pygame here."""

    def __init__(self, rec_dir, cfg, model=None, log_path=None, clock=time.monotonic):
        if cfg.FRONT_BANDS is None:
            raise SystemExit("FRONT_BANDS is not set in config.py: run python -m tools.calib_ui first")
        self.rec_dir, self.cfg, self.model, self.clock = rec_dir, cfg, model, clock
        self.name = os.path.basename(os.path.normpath(rec_dir))
        self.meta = read_meta(rec_dir)
        self.rows = _present(rec_dir, read_index(rec_dir))
        if not self.rows:
            raise SystemExit(f"{rec_dir}: no frames")
        self.times = [float(r["t"]) for r in self.rows]
        side_dir = os.path.join(rec_dir, "side")
        self.side_rows = _present(side_dir, read_index(side_dir)) if os.path.isdir(side_dir) else []
        self.side_times = [float(r["t"]) for r in self.side_rows]
        self.log_path = find_loop_log(self.meta, log_path)
        self.log = read_loop_log(self.log_path, self.times[0] - 1, self.times[-1] + 1) \
            if self.log_path else []
        self.log_times = [r["t"] for r in self.log]
        self.index, self.paused, self.speed = 0, False, 1.0
        self.zones = []
        self._due = None
        self._shown_at = []               # wall times of recent frames, for the fps shown
        self._chart = None
        self._reset_trackers()
        self._process()

    # ---------------------------------------------------------------- timing and control

    @property
    def t(self):
        return self.times[self.index]

    @property
    def duration(self):
        return self.times[-1] - self.times[0]

    def tick(self):
        # Advances to the next frame when it is due. True if the frame changed.
        now = self.clock()
        if self.paused:
            self._due = None
            return False
        if self.index >= len(self.rows) - 1:
            self.paused = True            # the end: stay on the last frame
            return False
        if self._due is None:
            self._due = now
        if now < self._due:
            return False
        gap = (self.times[self.index + 1] - self.t) / self.speed
        self.index += 1
        self._process()
        # Behind (slow Pi): the next frame is due a gap after this one is
        # shown, so playback slows rather than skipping frames.
        self._due = max(self._due + gap, self.clock() - gap)
        return True

    def wait_s(self):
        # How long until tick has work (for the display loop's sleep).
        if self.paused or self._due is None:
            return 0.03
        return max(0.0, min(0.03, self._due - self.clock()))

    def press(self, name):
        # A button or key. Returns 'back' to leave, else None.
        if name == "play":
            if self.index >= len(self.rows) - 1:
                self.seek(self.times[0])  # at the end: play again from the start
            self.paused = not self.paused
        elif name == "step":
            self.paused = True
            if self.index < len(self.rows) - 1:
                self.index += 1
                self._process()
        elif name == "back5":
            self.seek(self.t - SEEK_S)
        elif name == "fwd5":
            self.seek(self.t + SEEK_S)
        elif name == "speed":
            self.speed = SPEEDS[(SPEEDS.index(self.speed) + 1) % len(SPEEDS)]
        elif name == "back":
            return "back"
        self._due = None
        return None

    def tap(self, pos):
        # A tap: a button, or the timeline (jump to that time).
        name = hit(self.zones, pos)
        if name == "chart":
            x, _, w, _ = CHART_RECT
            self.seek(self.times[0] + (pos[0] - x) / w * self.duration)
            self._due = None
            return None
        return self.press(name) if name else None

    def seek(self, t):
        # Jumps to the frame at or before t. The trackers start again: the
        # lane search window and the junction pass build up from there.
        target = max(0, bisect.bisect_right(self.times, t) - 1)
        if target == self.index:
            return
        self.index = target
        self._reset_trackers()
        self._process()

    # ---------------------------------------------------------------- processing

    def _reset_trackers(self):
        self.lane_tracker, self.pacer = LaneTracker(), SignPacer()
        self.locator = LocationTracker(self.cfg)
        self.side_index, self.side = None, None
        self.location = None

    def _process(self):
        # The current front frame (and the side frame at its time) through
        # the robot's perception.
        cfg = self.cfg
        frame = load_frame(self.rec_dir, self.rows[self.index])
        self.front = process_front_frame(frame, True, cfg, self.lane_tracker, self.model,
                                         self.pacer)
        side_i = bisect.bisect_right(self.side_times, self.t) - 1
        if side_i >= 0 and side_i != self.side_index:
            side_frame = load_frame(os.path.join(self.rec_dir, "side"), self.side_rows[side_i],
                                    "side")
            self.side, self.side_index = process_side_frame(side_frame, cfg), side_i
        self.row = self.log_row(self.t)
        speed = _f(self.row, "speed") or 0.0
        self.location = self.locator.update(self.front.location, speed, self.t, self.front.sign,
                                            self.side)
        self._shown_at = (self._shown_at + [self.clock()])[-15:]

    def log_row(self, t):
        # The loop log row nearest t, or None (no log, or none close enough).
        if not self.log:
            return None
        i = bisect.bisect_left(self.log_times, t)
        near = [j for j in (i - 1, i) if 0 <= j < len(self.log)]
        j = min(near, key=lambda k: abs(self.log_times[k] - t))
        return self.log[j] if abs(self.log_times[j] - t) <= LOG_MATCH_S else None

    def fps(self):
        shown = self._shown_at
        if len(shown) < 2 or shown[-1] == shown[0]:
            return 0.0
        return (len(shown) - 1) / (shown[-1] - shown[0])

    # ---------------------------------------------------------------- drawing

    def draw(self):
        # The whole screen at DISPLAY_SIZE (BGR); records the tap zones.
        w, h = self.cfg.DISPLAY_SIZE
        img = np.empty((h, w, 3), np.uint8)
        img[:] = BG
        self.zones = []
        self._draw_banner(img)
        _put(img, FRONT_RECT, self._front_image(FRONT_RECT[2:]))
        _put(img, SIDE_RECT, self._side_image(SIDE_RECT[2:]))
        self._draw_chart(img)
        self._draw_telemetry(img)
        self._draw_column(img)
        return img

    def _draw_banner(self, img):
        img[:BANNER_H, :MAIN_W] = PANEL
        cv2.line(img, (0, BANNER_H - 1), (MAIN_W, BANNER_H - 1), LINE, 1)
        mode = self.meta.get("mode")
        label = self.cfg.MODE_LABELS.get(mode, mode) if mode else "recording"
        draw_text(img, clip_text(f"{self.name}  ·  {label}", 15, 330), (10, BANNER_H - 8), 15,
                  TEXT, bold=True)
        elapsed = self.t - self.times[0]
        draw_text(img, f"{elapsed:5.1f} / {self.duration:.1f} s   {self.index + 1}/{len(self.rows)}"
                       f"   {self.fps():4.1f} fps", (MAIN_W - 10, BANNER_H - 8), 15, MUTED,
                  align="right")

    def _front_image(self, size):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        result = self.front
        img[:] = cv2.resize(result.view, size, interpolation=cv2.INTER_LINEAR)
        scale = size[0] / result.view.shape[1]
        draw_front_overlay(img, result, scale)
        if result.location is not None:
            draw_annotations(img, result.location.annotations, scale)
        draw_sign_overlay(img, result.sign, None, self.cfg, scale)
        return img

    def _side_image(self, size):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        if self.side is None:
            draw_no_data(img, "side camera" if self.side_rows else "no side camera")
            return img
        img[:] = cv2.resize(self.side.view, size, interpolation=cv2.INTER_LINEAR)
        scale = size[0] / self.side.view.shape[1]
        draw_side_overlay(img, self.side, self.cfg, scale)
        draw_annotations(img, self.side.annotations, scale)
        return img

    def _draw_telemetry(self, img):
        x, y, w, h = TELEMETRY_RECT
        img[y:y + h, x:x + w] = PANEL
        rows = self._telemetry_rows()
        step = min(22, (h - 26) // max(1, len(rows)))
        draw_text(img, "ROBOT (loop log)" if self.row is not None else
                  ("REPLAY (no loop log)" if not self.log else "REPLAY (no log row here)"),
                  (x + 10, y + 18), 12, MUTED, bold=True)
        for i, (label, value, colour) in enumerate(rows):
            ry = y + 26 + (i + 1) * step - 4
            draw_text(img, label, (x + 10, ry), 14, MUTED)
            draw_text(img, clip_text(value, 14, w - 90), (x + w - 10, ry), 14, colour, bold=True,
                      align="right")

    def _telemetry_rows(self):
        cfg, row = self.cfg, self.row
        lane, loc, sign = self.front.lane, self.location, self.front.sign
        if row is None:
            rows = [("lateral", _cm(lane.lateral_m), TEXT),
                    ("heading", _deg(lane.heading_err_rad), TEXT),
                    ("lane conf", f"{lane.conf:.2f}", TEXT)]
            rows += _zone_rows(loc.zone.name if loc.dist_to_centre_m is not None else "",
                               -loc.dist_to_centre_m if loc.dist_to_centre_m is not None else None,
                               loc.source, cfg)
            read = sign_label(sign.sign, cfg) if sign is not None and sign.bbox is not None else "--"
            rows.append(("sign read", read, GREEN if read != "--" else MUTED))
            if self.side is not None and self.side.surface is not None:
                rows.append(("side surface", self.side.surface.name.lower(), TEXT))
            return rows
        state = row.get("state", "")
        colour = state_colour(State[state]) if state in State.__members__ else TEXT
        yaw, rate = _f(row, "yaw_deg"), _f(row, "yaw_rate_dps")
        rows = [("state", state.replace("_", " "), colour),
                ("speed / steer", f"{_f(row, 'speed') or 0:.2f} / {_f(row, 'steer') or 0:+.2f}", TEXT),
                ("lateral", _cm(_f(row, "lateral_err_m")), TEXT),
                ("heading", _deg(_f(row, "heading_err_rad")), TEXT),
                ("lane conf", "--" if _f(row, "lane_conf") is None else f"{_f(row, 'lane_conf'):.2f}",
                 TEXT),
                ("yaw (rate)", "--" if yaw is None else f"{yaw:+.1f}° ({rate or 0:+.0f}°/s)", TEXT)]
        rows += _zone_rows(row.get("zone", ""), _f(row, "s"), row.get("loc_source", ""), cfg)
        voted = row.get("sign", "")
        rows.append(("sign", cfg.SIGN_PHRASES.get(voted, voted) or "--",
                     GREEN if voted else MUTED))
        return rows

    def _draw_chart(self, img):
        x, y, w, h = CHART_RECT
        if self._chart is None:
            self._chart = self._chart_background((w, h))
        img[y:y + h, x:x + w] = self._chart
        frac = 0.0 if self.duration <= 0 else (self.t - self.times[0]) / self.duration
        px = x + int(round(frac * (w - 1)))
        cv2.line(img, (px, y + 2), (px, y + h - 3), TEXT, 2)
        self.zones.append(Zone((x, y, w, h), "chart"))

    def _chart_background(self, size):
        # The whole run: state stripe, then steer, speed and lateral against
        # time; drawn once.
        w, h = size
        img = np.empty((h, w, 3), np.uint8)
        img[:] = BG
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), LINE, 1)
        t0, dur = self.times[0], max(self.duration, 1e-6)
        to_x = lambda t: int(round((t - t0) / dur * (w - 1)))
        if not self.log:
            draw_text(img, "no loop log: timeline only", (10, h // 2 + 5), 14, MUTED)
            for t in self.side_times[::10]:
                cv2.line(img, (to_x(t), h - 8), (to_x(t), h - 3), LINE, 1)
            return img
        top, stripe, legend = 14, 8, 16
        for r0, r1 in zip(self.log, self.log[1:] + [None]):
            x0 = max(0, to_x(r0["t"]))
            x1 = min(w - 1, to_x(r1["t"]) if r1 is not None else w - 1)
            state = r0.get("state", "")
            colour = STATE_TINTS.get(State[state], state_colour(State[state])) \
                if state in State.__members__ else MUTED
            cv2.rectangle(img, (x0, 3), (max(x0, x1), 3 + stripe), colour, -1)
        mid, half = (top + h - legend) // 2, (h - top - legend) // 2 - 4
        cv2.line(img, (0, mid), (w, mid), LINE, 1)
        series = (("steer", self.cfg.STEER_MAX, BLUE), ("speed", 1.0, TEXT),
                  ("lateral_err_m", self.cfg.UI_CHART_LAT_M, PURPLE))
        for key, limit, colour in series:
            pts = [(to_x(r["t"]), mid - int(max(-1, min(1, v / limit)) * half))
                   for r in self.log for v in [_f(r, key)] if v is not None]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, colour, 1, cv2.LINE_AA)
        for i, (label, colour) in enumerate((("steer", BLUE), ("speed", TEXT),
                                             ("lateral", PURPLE))):
            draw_text(img, label, (8 + i * 62, h - 5), 12, colour, bold=True)
        draw_text(img, "tap to jump", (w - 8, h - 5), 12, MUTED, align="right")
        return img

    def _draw_column(self, img):
        w, h = self.cfg.DISPLAY_SIZE
        x0 = MAIN_W
        img[:, x0:] = PANEL
        at_end = self.index >= len(self.rows) - 1
        play = "PLAY" if self.paused else "PAUSE"
        buttons = [Button("play", play, "from start" if at_end and self.paused else "",
                          primary=self.paused),
                   Button("step", "STEP", "one frame"),
                   Button("back5", "-5 s"), Button("fwd5", "+5 s"),
                   Button("speed", f"x{self.speed:g}", "speed"),
                   Button("back", "BACK")]
        slot = h // len(buttons)
        for i, button in enumerate(buttons):
            rect = (x0, i * slot, w - x0, slot)
            draw_button(img, rect, button)
            self.zones.append(Zone(rect, button.name))


def _present(rec_dir, rows):
    # The rows whose frame file exists (a recording cut short lists more).
    return [r for r in rows if os.path.exists(os.path.join(rec_dir, r["file"]))]


def _put(img, rect, picture):
    x, y, w, h = rect
    img[y:y + h, x:x + w] = picture
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), LINE, 1)


def _cm(m):
    return "--" if m is None else f"{m * 100:+.1f} cm"


def _deg(rad):
    return "--" if rad is None else f"{np.degrees(rad):+.1f}°"


def _zone_rows(zone, s, source, cfg):
    if not zone:
        return [("zone", "no junction", MUTED)]
    return [("zone", cfg.ZONE_PHRASES.get(zone, zone), YELLOW),
            ("s", f"{s:+.2f} m ({source})" if s is not None else "--", TEXT)]


# ---------------------------------------------------------------- picking a recording

def list_recordings(base):
    # Recording folders under base (each with frames.csv), newest first.
    if not os.path.isdir(base):
        return []
    found = [os.path.join(base, d) for d in os.listdir(base)
             if os.path.exists(os.path.join(base, d, "frames.csv"))]
    return sorted(found, key=os.path.getmtime, reverse=True)


class Picker:
    """Tap a recording to play it; newest first, PICKER_ROWS a page."""

    def __init__(self, cfg, base):
        self.cfg, self.page, self.zones = cfg, 0, []
        self.recordings = list_recordings(base)

    def pages(self):
        return max(1, -(-len(self.recordings) // PICKER_ROWS))

    def draw(self):
        w, h = self.cfg.DISPLAY_SIZE
        img = np.empty((h, w, 3), np.uint8)
        img[:] = BG
        self.zones = []
        draw_text(img, "REPLAY: pick a recording", (16, 34), 22, TEXT, bold=True)
        draw_text(img, f"newest first   page {self.page + 1}/{self.pages()}", (16, 56), 14, MUTED)
        if not self.recordings:
            draw_text(img, f"No recordings in {self.cfg.RECORD_DIR}/", (16, 110), 18, AMBER)
        row_h = (h - 72) // PICKER_ROWS
        start = self.page * PICKER_ROWS
        for i, path in enumerate(self.recordings[start:start + PICKER_ROWS]):
            y = 70 + i * row_h
            rect = (10, y, MAIN_W - 20, row_h - 6)
            cv2.rectangle(img, (rect[0], y), (rect[0] + rect[2], y + rect[3]), PANEL, -1)
            summary = _summary(path)
            room = MAIN_W - 48 - text_width(summary, 14) - 16
            draw_text(img, clip_text(os.path.basename(path), 18, room), (24, y + row_h // 2 + 2), 18,
                      TEXT, bold=True)
            draw_text(img, summary, (MAIN_W - 24, y + row_h // 2 + 2), 14, MUTED, align="right")
            self.zones.append(Zone(rect, path))
        img[:, MAIN_W:] = PANEL
        slot = h // 4
        for i, button in enumerate([Button("newer", "NEWER", enabled=self.page > 0),
                                    Button("older", "OLDER",
                                           enabled=self.page < self.pages() - 1),
                                    None, Button("quit", "QUIT")]):
            if button is None:
                continue
            rect = (MAIN_W, i * slot, w - MAIN_W, slot)
            draw_button(img, rect, button)
            if button.enabled:
                self.zones.append(Zone(rect, button.name))
        return img

    def tap(self, pos):
        # -> a recording's path, 'quit', or None.
        name = hit(self.zones, pos)
        if name == "newer":
            self.page = max(0, self.page - 1)
        elif name == "older":
            self.page = min(self.pages() - 1, self.page + 1)
        elif name == "quit" or (name and os.path.isdir(name)):
            return name
        return None


def _summary(path):
    # "Intersection Detection · 52 s · side" from meta.json and frames.csv.
    try:
        meta, rows = read_meta(path), read_index(path)
    except (OSError, ValueError):
        return ""
    parts = []
    if meta.get("mode"):
        parts.append(config.MODE_LABELS.get(meta["mode"], meta["mode"]))
    if len(rows) > 1:
        parts.append(f"{float(rows[-1]['t']) - float(rows[0]['t']):.0f} s")
    if os.path.isdir(os.path.join(path, "side")):
        parts.append("side")
    parts.append(meta.get("created", "")[5:16].replace("T", " "))
    return "  ·  ".join(p for p in parts if p)


# ---------------------------------------------------------------- display loop

def run(rec_dir, cfg, model, log_path=None):
    # Picker (unless rec_dir) -> playback -> BACK returns to the picker (or
    # quits when started on one recording). False if no display opened.
    from ui.screen import _set_display_env, open_display, show
    _set_display_env()
    display = open_display(cfg)
    if display is None:
        return False
    pygame, surface = display
    keys = {pygame.K_SPACE: "play", pygame.K_n: "step", pygame.K_RIGHT: "step",
            pygame.K_LEFT: "back5", pygame.K_q: "back", pygame.K_ESCAPE: "back"}
    picker = Picker(cfg, os.path.join(AUTONOMOUS_DIR, cfg.RECORD_DIR)) if rec_dir is None else None
    try:
        while True:
            path = rec_dir or _pick(pygame, surface, show, picker)
            if path is None:
                return True
            show(pygame, surface, _loading(cfg, path))
            print(f"[{TAG}] {path}")
            playback = Playback(path, cfg, model, log_path)
            print(f"[{TAG}] {len(playback.rows)} front frames, {len(playback.side_rows)} side, "
                  f"loop log: {playback.log_path or 'none'}")
            if _play(pygame, surface, show, playback, keys) == "quit" or rec_dir is not None:
                return True
    finally:
        pygame.quit()


def _pick(pygame, surface, show, picker):
    while True:
        show(pygame, surface, picker.draw())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                chosen = picker.tap(event.pos)
                if chosen == "quit":
                    return None
                if chosen:
                    return chosen
        time.sleep(0.05)


def _play(pygame, surface, show, playback, keys):
    # Until BACK ('back') or the window closes ('quit').
    dirty = True
    while True:
        for event in pygame.event.get():
            action = None
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN and event.key in keys:
                action = playback.press(keys[event.key])
            elif event.type == pygame.MOUSEBUTTONDOWN:
                action = playback.tap(event.pos)
            else:
                continue
            if action == "back":
                return "back"
            dirty = True
        if playback.tick() or dirty:
            show(pygame, surface, playback.draw())
            dirty = False
        time.sleep(playback.wait_s())


def _loading(cfg, path):
    w, h = cfg.DISPLAY_SIZE
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG
    draw_text(img, f"Loading {os.path.basename(os.path.normpath(path))} ...", (w // 2, h // 2),
              22, TEXT, bold=True, align="centre")
    return img


def open_model(path, cfg):
    try:
        return load_sign_model(path, cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"[{TAG}] {e}; sign candidates are boxed but not read")
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description="Play a recording on the robot's display.")
    p.add_argument("recording", nargs="?", default=None,
                   help="folder made by RECORD or tools/record.py (default: pick on the screen)")
    p.add_argument("--model", default=config.SIGN_MODEL_PATH, help="sign model (.npz)")
    p.add_argument("--log", default=None, help="loop log CSV (default: the one meta.json names)")
    args = p.parse_args(argv)
    try:
        if not run(args.recording, config, open_model(args.model, config), args.log):
            print(f"[{TAG}] No display: run this on the Pi with its screen attached.")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
