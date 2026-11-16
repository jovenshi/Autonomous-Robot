"""Tune the colour tests on the robot's 7" touch display: the yellow line,
the road, and the PAUSE sign's red. Run it whenever the lighting changes.

Stop main.py first (it owns the camera), then from the autonomous/ folder:

    python -m tools.tune_colors                                  # live camera
    python -m tools.tune_colors recordings/front_20261011_140000 # a recording

or tap TUNE COLOURS in tools/menu.py. Put the robot on the course where it
sees the yellow line and the road, as it does when driving.

Screens, one colour at a time:
    1 YELLOW LINE  tap the yellow line in 3-5 places: near, far, in shade
    2 ROAD         tap the road; then tap NOT ROAD and tap the mat beside it
    3 PAUSE RED    hold the PAUSE sign in view and tap its red (no sign: NEXT)
    REVIEW         old -> new values and the lane found with them;
                   APPLY writes config.py (config.py.bak keeps the old one)

Each tap sets the step's values so that every tap passes (NOT ROAD taps
fail), with a margin: tap the palest and the darkest parts too. Rings on the
picture show each tap: green passes as intended, red doesn't. Fine-tune
along the bottom: tap a value, then -/+. The tint shows where the test
passes, worked out by the real perception/color.py code; MASK shows the bare
mask, where speckle is easier to see. A step without taps keeps its values.

The live camera's exposure is locked after CAM_WARMUP_S, as main.py does,
so the view looks like what the robot sees when driving.

Keys: Enter = the green button, Esc = back, Left/Right = recorded frames,
Space = freeze the live view, q = quit. Quitting without APPLY prints the
values in the terminal.

The white mask (WHITE_*) is not tuned here: nothing uses it any more.
"""
import argparse
import os
import sys
import time
import types
from dataclasses import dataclass

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from perception.color import (front_lane_roi, make_front_masks, mask_red,  # noqa: E402
                              mask_road, mask_yellow, road_v_cut)
from perception.front_lane import process_front_lane  # noqa: E402
from robot_types import LatestValue  # noqa: E402
from tools import config_writer  # noqa: E402
from tools.recording import load_frame, read_index  # noqa: E402
from tools.guide_draw import (PRIMARY, Button, Zone, draw_adjuster, draw_banner,  # noqa: E402
                              draw_column, draw_level_line, draw_paragraph, hit, say)
from ui.overlay import BG, GREEN, LINE, MUTED, PANEL, RED, TEXT, draw_front_overlay  # noqa: E402
from ui.text import draw_text, text_width  # noqa: E402

TAG = "colours"
MAIN_W = 640                             # px; content area left of the button column
PIC_BOX = (480, 360)                     # px; the picture on a step screen
REVIEW_BOX = (400, 300)                  # px; the picture on the review screen
H_MARGIN = 3                             # hue added either side of the tapped hues
SV_MARGIN = 25                           # saturation margin past the taps
YELLOW_V_FRAC = 0.5                      # yellow's brightness floor: this x the darkest tap
                                         # (kept loose, as perception/color.py says, for shade)
ROAD_V_MARGIN = 10                       # brightness margin for the road's dark cut
ADJ_STEPS = (("-10", -10), ("-1", -1), ("+1", 1), ("+10", 10))


@dataclass(frozen=True)
class Param:
    name: str                            # in config.py
    label: str                           # on its chip
    help: str                            # above the -/+ buttons
    hi: int = 255                        # range is 0..hi


@dataclass(frozen=True)
class Step:
    key: str                             # also the FrontMasks field it tunes
    title: str
    banner: str                          # one line over the picture
    tip: str                             # printed in the terminal when the step opens
    tint: tuple                          # BGR, painted where the test passes
    params: tuple


STEPS = (
    Step("yellow", "Yellow line", "Tap the yellow line: near, far and in shade",
         "Tap the yellow line in 3-5 places: near the robot, far away, and wherever it is "
         "darker or paler. The values are set so every tap passes. Pink shows what the "
         "robot will call yellow: it should cover the line and nothing else.",
         (255, 0, 255),
         (Param("YELLOW_H_MIN", "hue from", "Lowest hue that counts as yellow", 179),
          Param("YELLOW_H_MAX", "hue to", "Highest hue that counts as yellow", 179),
          Param("YELLOW_S_MIN", "colour min", "Paler than this is not yellow (saturation)"),
          Param("YELLOW_V_MIN", "bright min", "Darker than this is not yellow"))),
    Step("road", "Road", "Tap the road in several places",
         "Tap the road in 3-5 places, including any shadow or shine on it. Then tap NOT ROAD "
         "and tap the mat beside the road. Blue should cover the road up to its edges, not "
         "the mat. The dark cut is chosen each frame from the light, between the two limits.",
         (255, 120, 0),
         (Param("ROAD_S_MAX", "colour max", "More colourful than this is not road"),
          Param("ROAD_V_T_MIN", "dark cut min", "The dark cut never goes below this"),
          Param("ROAD_V_T_MAX", "dark cut max", "The dark cut never goes above this"))),
    Step("red", "PAUSE red", "Hold up the PAUSE sign and tap its red",
         "Hold the PAUSE sign where the robot would see it and tap its red in 2-3 places. "
         "Cyan should fill the red octagon. No sign to hand? NEXT keeps the red values.",
         (255, 255, 0),
         (Param("RED_H_LOW_MAX", "hue to (low)", "Red hues from 0 up to this", 179),
          Param("RED_H_HIGH_MIN", "hue from (high)", "Red hues from this up to 179", 179),
          Param("RED_S_MIN", "colour min", "Paler than this is not red (saturation)"))),
)
PARAMS = {p.name: p for step in STEPS for p in step.params}
PAIRS = (("YELLOW_H_MIN", "YELLOW_H_MAX"), ("ROAD_V_T_MIN", "ROAD_V_T_MAX"))
NOT_ROAD_BANNER = "NOT ROAD: tap the mat beside the road"


@dataclass
class Tap:
    hsv: tuple                           # (H, S, V) at the tap, median of 3 x 3
    miss: bool                           # a NOT ROAD tap: must fail the test
    frame_key: tuple                     # (seq, t) of the frame it was taken on
    xy: tuple                            # lane-size pixel


# ---------------------------------------------------------------- values

def config_with(overrides, base=None):
    # A copy of every config value, with the overrides swapped in.
    base = config if base is None else base
    values = {name: getattr(base, name) for name in dir(base) if name.isupper()}
    values.update(overrides)
    return types.SimpleNamespace(**values)


def fit_values(step_key, taps, misses):
    # The values that make every tap pass (and every NOT ROAD tap fail), with
    # margins. taps, misses: (N, 3) HSV arrays. Returns only the values the
    # taps say something about.
    taps, misses = np.asarray(taps, int).reshape(-1, 3), np.asarray(misses, int).reshape(-1, 3)
    out = {}
    if step_key == "yellow" and len(taps):
        h, s, v = taps.T
        out = {"YELLOW_H_MIN": h.min() - H_MARGIN, "YELLOW_H_MAX": h.max() + H_MARGIN,
               "YELLOW_S_MIN": s.min() - SV_MARGIN, "YELLOW_V_MIN": v.min() * YELLOW_V_FRAC}
    elif step_key == "road":
        if len(taps):
            out["ROAD_S_MAX"] = taps[:, 1].max() + SV_MARGIN
            out["ROAD_V_T_MIN"] = taps[:, 2].max() + ROAD_V_MARGIN
        if len(misses):
            out["ROAD_V_T_MAX"] = misses[:, 2].min() - ROAD_V_MARGIN
        if len(taps) and len(misses) and out["ROAD_V_T_MIN"] > out["ROAD_V_T_MAX"]:
            # Road and mat overlap in brightness: cut halfway between them.
            mid = (taps[:, 2].max() + misses[:, 2].min()) // 2
            out["ROAD_V_T_MIN"] = out["ROAD_V_T_MAX"] = mid
    elif step_key == "red" and len(taps):
        h = taps[:, 0]
        low, high = h[h < 90], h[h >= 90]          # red wraps around 0 / 179
        if len(low):
            out["RED_H_LOW_MAX"] = low.max() + H_MARGIN
        if len(high):
            out["RED_H_HIGH_MIN"] = high.min() - H_MARGIN
        out["RED_S_MIN"] = taps[:, 1].min() - SV_MARGIN
    return {name: int(value) for name, value in out.items()}


def tidy(values, moved=None):
    # Clamps every value to its range and keeps each min <= its max. moved:
    # the name just changed by hand, which wins; otherwise the max follows.
    for name, param in PARAMS.items():
        values[name] = int(min(max(values[name], 0), param.hi))
    for lo, hi in PAIRS:
        if values[lo] > values[hi]:
            if moved == hi:
                values[lo] = values[hi]
            else:
                values[hi] = values[lo]
    return values


def passes(step_key, cfg, hsv_list, v_cut=None):
    # Which (H, S, V) values pass the step's test, by the real mask code.
    if not len(hsv_list):
        return np.zeros(0, bool)
    img = np.asarray(hsv_list, np.uint8).reshape(1, -1, 3)
    if step_key == "yellow":
        mask = mask_yellow(img, cfg)
    elif step_key == "red":
        mask = mask_red(img, cfg)
    else:
        mask = mask_road(img, None, cfg, v_cut=v_cut)
    return mask[0] > 0


def sample_hsv(hsv, xy):
    # Median H, S, V of the 3 x 3 pixels around xy (one stray pixel can't
    # widen the range).
    x, y = xy
    patch = hsv[max(0, y - 1):y + 2, max(0, x - 1):x + 2].reshape(-1, 3)
    return tuple(int(c) for c in np.median(patch, axis=0))


def tint(view, mask, colour):
    # The view with mask pixels painted in colour, half transparent.
    out = view.copy()
    out[mask > 0] = (0.5 * out[mask > 0] + 0.5 * np.array(colour)).astype(np.uint8)
    return out


def print_values(values, heading):
    print(f"\n# {heading}")
    for name in PARAMS:
        print(f"{name} = {values[name]}")


# ---------------------------------------------------------------- frame sources

class RecordingFrames:
    """Frames of a recording, one at a time; PREV / NEXT move through them."""
    live = False

    def __init__(self, rec_dir):
        self.rec_dir = rec_dir
        try:
            self.rows = read_index(rec_dir)
        except FileNotFoundError:
            self.rows = []
        if not self.rows:
            raise SystemExit(f"[{TAG}] No frames in {rec_dir}")
        self.index, self._cached = 0, (None, None)

    def latest(self):
        if self._cached[0] != self.index:
            self._cached = (self.index, load_frame(self.rec_dir, self.rows[self.index]))
        return self._cached[1]

    def step(self, delta):
        self.index = min(max(self.index + delta, 0), len(self.rows) - 1)

    def label(self):
        return f"{self.index + 1} / {len(self.rows)}"

    def stop(self):
        pass


class LiveFrames:
    """The front camera in its own thread, exposure locked as main.py does.
    FREEZE holds one frame so taps land where you aimed."""
    live = True

    def __init__(self, cfg):
        from hal.camera import CameraStream, CsiSource
        from hal.exposure import ExposureControl
        source = CsiSource(cfg.FRONT_CAM_NUM, cfg.FRONT_SIZE, cfg)
        self.slot, self.frozen = LatestValue(), None
        self.stream = CameraStream(source, lambda frame: frame, self.slot, "front", cfg)
        self.exposure = ExposureControl(source, "front", cfg)

    def start(self):
        self.stream.start()
        self.exposure.lock_after_warmup()

    def latest(self):
        return self.frozen if self.frozen is not None else self.slot.get()

    def toggle_freeze(self):
        self.frozen = None if self.frozen is not None else self.slot.get()

    def label(self):
        return "FROZEN" if self.frozen is not None else "LIVE"

    def stop(self):
        self.stream.stop()


# ---------------------------------------------------------------- the screens

class TuneApp:
    """Step screens, review and saved. draw() -> image; tap() / key() /
    press() act on it; the same calls drive it in the tests."""

    def __init__(self, cfg, frames):
        self.cfg, self.frames = cfg, frames
        self.size = tuple(cfg.DISPLAY_SIZE)
        self.start_values = {name: getattr(cfg, name) for name in PARAMS}
        self.values = dict(self.start_values)
        self.taps = {step.key: [] for step in STEPS}
        self.chip = {step.key: 0 for step in STEPS}
        self.step_i, self.page = 0, "step"
        self.miss_mode, self.mask_view = False, False
        self.zones, self.shown, self.done, self.backup = [], None, False, None
        self._tips_shown = set()
        self.go("step", 0)

    @property
    def step(self):
        return STEPS[self.step_i]

    def changed(self):
        return {n: v for n, v in self.values.items() if v != self.start_values[n]}

    def cfg_now(self):
        return config_with(self.values, self.cfg)

    def go(self, page, step_i=None):
        self.page = page
        if step_i is not None:
            self.step_i, self.miss_mode = step_i, False
        name = f"{page}:{self.step_i}" if page == "step" else page
        if name not in self._tips_shown:
            self._tips_shown.add(name)
            if page == "step":
                say(TAG, f"Step {self.step_i + 1} of {len(STEPS)}, {self.step.title}. "
                         f"{self.step.tip}")
            elif page == "review":
                say(TAG, "Review: check the lane is found on a few views (move the robot or "
                         "step through frames), then APPLY to write config.py.")

    # ------------------------------------------------------------ drawing

    def draw(self):
        img = np.empty((self.size[1], self.size[0], 3), np.uint8)
        img[:] = BG
        self.zones = []
        getattr(self, f"_draw_{self.page}")(img[:, :MAIN_W])
        self.zones += draw_column(img, self.buttons(), self.size)
        return img

    def buttons(self):
        if self.page == "step":
            last = self.step_i == len(STEPS) - 1
            return [Button("next", "NEXT", "review" if last else STEPS[self.step_i + 1].title,
                           primary=True),
                    Button("undo", "UNDO", "last tap", enabled=bool(self.taps[self.step.key])),
                    Button("back", "BACK", enabled=self.step_i > 0),
                    Button("quit", "QUIT", "no save")]
        if self.page == "review":
            count = len(self.changed())
            return [Button("apply", "APPLY", f"{count} to config.py" if count else "no changes",
                           primary=True, enabled=count > 0),
                    Button("back", "BACK"), None, Button("quit", "QUIT", "no save")]
        return [Button("done", "DONE", primary=True), None, None, None]

    def _draw_step(self, img):
        step, cfg = self.step, self.cfg_now()
        frame = self.frames.latest()
        self.shown = None
        if frame is None:
            draw_text(img, "Waiting for the camera...", (20, 60), 22, TEXT)
            return
        masks = make_front_masks(frame, cfg, with_signs=step.key == "red")
        mask = getattr(masks, step.key)
        h, w = mask.shape
        v_cut = road_v_cut(masks.hsv, front_lane_roi(cfg, w, h), cfg) if step.key == "road" \
            else None
        scale = min(PIC_BOX[0] / w, PIC_BOX[1] / h)
        pw, ph = int(w * scale), int(h * scale)
        if self.mask_view:
            pic = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (pw, ph),
                             interpolation=cv2.INTER_NEAREST)
        else:
            pic = cv2.resize(tint(masks.bgr, mask, step.tint), (pw, ph))
        img[:ph, :pw] = pic
        self.shown = (frame, masks, scale, (pw, ph))
        taps = self.taps[step.key]
        right = self._tap_results(step, cfg, taps, v_cut)
        key = (frame.seq, frame.t)
        for tap, ok in zip(taps, right):
            if tap.frame_key == key:
                centre = (int((tap.xy[0] + 0.5) * scale), int((tap.xy[1] + 0.5) * scale))
                cv2.circle(img, centre, 9, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.circle(img, centre, 9, GREEN if ok else RED, 2, cv2.LINE_AA)
                if tap.miss:
                    cv2.line(img, (centre[0] - 5, centre[1]), (centre[0] + 5, centre[1]),
                             GREEN if ok else RED, 2, cv2.LINE_AA)
        banner = NOT_ROAD_BANNER if self.miss_mode else step.banner
        draw_banner(img[:ph, :pw], banner, size=16)
        self._draw_side(img, (pw, 0, MAIN_W - pw, ph), step, mask, taps, right, v_cut)
        self._draw_values(img, (0, ph, MAIN_W, img.shape[0] - ph), step)

    def _tap_results(self, step, cfg, taps, v_cut):
        # For each tap: does it come out as intended (pass, or fail for NOT ROAD)?
        if not taps:
            return []
        passed = passes(step.key, cfg, [t.hsv for t in taps], v_cut)
        return [bool(p) != t.miss for p, t in zip(passed, taps)]

    def _draw_side(self, img, rect, step, mask, taps, right, v_cut):
        x, y, w, h = rect
        tx, tw = x + 10, w - 16
        draw_text(img, f"STEP {self.step_i + 1} OF {len(STEPS)}", (tx, y + 22), 13, MUTED,
                  bold=True)
        draw_text(img, step.title, (tx, y + 46), 20, TEXT, bold=True)
        ty = y + 74
        if taps:
            good = sum(right)
            level = "ok" if good == len(taps) else "warn"
            ty = draw_level_line(img, level, f"{good} of {len(taps)} taps right", (tx, ty),
                                 tw, 14) + 4
        else:
            ty = draw_paragraph(img, "No taps yet", (tx, ty), tw, 14, MUTED) + 4
        ty = draw_paragraph(img, f"Tinted: {np.count_nonzero(mask) / mask.size:.1%} of view",
                            (tx, ty), tw, 14, TEXT) + 2
        if v_cut is not None:
            ty = draw_paragraph(img, f"Dark cut now: {v_cut}", (tx, ty), tw, 14, TEXT) + 2
        if taps:
            hue, sat, val = taps[-1].hsv
            draw_paragraph(img, f"Last tap: H{hue} S{sat} V{val}", (tx, ty), tw, 13, MUTED)
        bx, bw, half = x + 6, w - 12, (w - 18) // 2
        rows = [y + h - 44 * i - 42 for i in range(4)]       # bottom row first
        self._draw_source(img, (bx, rows[0], bw, 38))
        self.zones.append(_side_button(img, (bx, rows[1], bw, 38), "RESET", "reset"))
        self.zones.append(_side_button(img, (bx, rows[2], half, 38), "TINT", "view:tint",
                                       not self.mask_view))
        self.zones.append(_side_button(img, (bx + half + 6, rows[2], half, 38), "MASK",
                                       "view:mask", self.mask_view))
        if step.key == "road":
            self.zones.append(_side_button(img, (bx, rows[3], half, 38), "ROAD", "mode:road",
                                           not self.miss_mode))
            self.zones.append(_side_button(img, (bx + half + 6, rows[3], half, 38), "NOT ROAD",
                                           "mode:not", self.miss_mode))

    def _draw_source(self, img, rect):
        # PREV / NEXT frame for a recording, FREEZE for the live camera.
        x, y, w, h = rect
        if self.frames.live:
            frozen = self.frames.frozen is not None
            self.zones.append(_side_button(img, rect, "UNFREEZE" if frozen else "FREEZE",
                                           "freeze", frozen))
            return
        bw = 40
        self.zones.append(_side_button(img, (x, y, bw, h), "<", "frame:-1"))
        self.zones.append(_side_button(img, (x + w - bw, y, bw, h), ">", "frame:1"))
        draw_text(img, self.frames.label(), (x + w // 2, y + h // 2 + 5), 13, TEXT,
                  align="centre")

    def _draw_values(self, img, rect, step):
        # One chip per value (tap to pick), then -/+ for the picked one.
        x, y, w, h = rect
        n, gap = len(step.params), 8
        cw = (w - gap * (n + 1)) // n
        for i, param in enumerate(step.params):
            cx = x + gap + i * (cw + gap)
            chosen = i == self.chip[step.key]
            box = (cx, y + 6, cw, 42)
            cv2.rectangle(img, box[:2], (cx + cw, y + 48), PANEL, -1, cv2.LINE_AA)
            cv2.rectangle(img, box[:2], (cx + cw, y + 48), GREEN if chosen else LINE,
                          2 if chosen else 1, cv2.LINE_AA)
            value = self.values[param.name]
            moved = value != self.start_values[param.name]
            draw_text(img, param.label, (cx + 8, y + 22), 12, MUTED)
            draw_text(img, str(value), (cx + 8, y + 43), 18, GREEN if moved else TEXT, bold=True)
            if moved:
                draw_text(img, f"was {self.start_values[param.name]}", (cx + cw - 6, y + 43),
                          12, MUTED, align="right")
            self.zones.append(Zone(box, f"chip:{i}"))
        param = step.params[self.chip[step.key]]
        self.zones += draw_adjuster(img, (x + gap, y + 54, w - 2 * gap, h - 58),
                                    f"{param.help} (0-{param.hi})",
                                    str(self.values[param.name]), "adj", ADJ_STEPS)

    def _draw_review(self, img):
        cfg = self.cfg_now()
        frame = self.frames.latest()
        if frame is None:
            draw_text(img, "Waiting for the camera...", (20, 60), 22, TEXT)
            return
        masks = make_front_masks(frame, cfg, with_signs=True)
        h, w = masks.yellow.shape
        scale = min(REVIEW_BOX[0] / w, REVIEW_BOX[1] / h)
        pw, ph = int(w * scale), int(h * scale)
        view = tint(tint(masks.bgr, masks.road, STEPS[1].tint), masks.yellow, STEPS[0].tint)
        pic = cv2.resize(view, (pw, ph))
        lane = process_front_lane(masks, cfg, frame.t, frame.seq) \
            if cfg.FRONT_BANDS is not None else None
        draw_front_overlay(pic, types.SimpleNamespace(lane=lane), scale)
        img[:ph, :pw] = pic
        tx, tw = pw + 12, MAIN_W - pw - 18
        draw_text(img, "Review", (tx, 30), 22, TEXT, bold=True)
        if lane is None:
            level, text = "warn", "No FRONT_BANDS yet: calibrate the camera to see the lane."
        elif lane.line_px is None:
            level, text = "bad", "No lane found in this view."
        else:
            good = lane.conf >= cfg.LANE_SEARCH_MIN_CONF
            level, text = ("ok" if good else "warn"), f"Lane found, conf {lane.conf:.2f}"
        ty = draw_level_line(img, level, text, (tx, 62), tw, 15) + 4
        ty = draw_paragraph(img, f"PAUSE red: {np.count_nonzero(masks.red) / masks.red.size:.1%}"
                                 " of view", (tx, ty), tw, 14, TEXT) + 6
        ty = draw_paragraph(img, "Pink = yellow line, blue = road. Check a few views before "
                                 "APPLY.", (tx, ty), tw, 13, MUTED)
        self._draw_source(img, (tx - 6, ph - 42, tw + 6, 38))
        self._draw_changes(img, ph + 10)

    def _draw_changes(self, img, top):
        # Every value: old -> new, the changed ones in green.
        draw_text(img, "Values (config.py now -> new)", (12, top + 14), 14, MUTED, bold=True)
        names = list(PARAMS)
        rows = (len(names) + 1) // 2
        row_h = min(26, (img.shape[0] - top - 26) // rows)
        for i, name in enumerate(names):
            x = 12 + (i // rows) * (MAIN_W // 2)
            y = top + 40 + (i % rows) * row_h
            old, new = self.start_values[name], self.values[name]
            draw_text(img, name, (x, y), 14, TEXT if old != new else MUTED)
            vx = x + MAIN_W // 2 - 24
            if old != new:
                draw_text(img, f"{old} -> {new}", (vx, y), 15, GREEN, bold=True, align="right")
            else:
                draw_text(img, str(old), (vx, y), 14, MUTED, align="right")

    def _draw_applied(self, img):
        draw_text(img, "Saved", (20, 50), 28, GREEN, bold=True)
        y = draw_paragraph(img, f"{len(self.changed())} colour values written to config.py. "
                                f"The old file is {os.path.basename(self.backup)}.",
                           (20, 90), MAIN_W - 40, 18)
        y = draw_paragraph(img, "main.py reads them when it next starts. Tune again whenever "
                                "the lighting changes.", (20, y + 8), MAIN_W - 40, 18, MUTED)
        self._draw_changes(img, y + 10)

    # ------------------------------------------------------------ input

    def tap(self, pos):
        if self.page == "step" and self.shown is not None:
            pw, ph = self.shown[3]
            if 0 <= pos[0] < pw and 0 <= pos[1] < ph:
                self.add_tap(pos)
                return
        name = hit(self.zones, pos)
        if name is not None:
            self.press(name)

    def key(self, name, shift=False):
        if name == "q":
            self.quit()
        elif name == "enter":
            primary = [b for b in self.buttons() if b is not None and b.primary and b.enabled]
            if primary:
                self.press(primary[0].name)
        elif name == "escape":
            self.press("back")
        elif name in ("left", "right") and not self.frames.live:
            self.frames.step(-1 if name == "left" else 1)
        elif name == "space" and self.frames.live:
            self.frames.toggle_freeze()

    def press(self, name):
        # A column button (if on screen and enabled) or a zone's name.
        for button in self.buttons():
            if button is not None and button.name == name:
                if button.enabled:
                    self._on_button(name)
                return
        kind, _, arg = name.partition(":")
        if kind == "chip":
            self.chip[self.step.key] = int(arg)
        elif kind == "adj":
            self.adjust(int(arg))
        elif kind == "view":
            self.mask_view = arg == "mask"
        elif kind == "mode":
            self.miss_mode = arg == "not"
        elif kind == "reset":
            self.reset_step()
        elif kind == "frame":
            self.frames.step(int(arg))
        elif kind == "freeze":
            self.frames.toggle_freeze()

    def _on_button(self, name):
        if name == "next":
            if self.step_i < len(STEPS) - 1:
                self.go("step", self.step_i + 1)
            else:
                self.go("review")
        elif name == "back":
            if self.page == "review":
                self.go("step", len(STEPS) - 1)
            else:
                self.go("step", self.step_i - 1)
        elif name == "undo":
            self.taps[self.step.key].pop()
            self.refit()
        elif name == "apply":
            self.apply()
        elif name in ("quit", "done"):
            self.quit()

    # ------------------------------------------------------------ actions

    def add_tap(self, pos):
        frame, masks, scale, _ = self.shown
        h, w = masks.hsv.shape[:2]
        xy = (min(int(pos[0] / scale), w - 1), min(int(pos[1] / scale), h - 1))
        hsv = sample_hsv(masks.hsv, xy)
        miss = self.miss_mode and self.step.key == "road"
        self.taps[self.step.key].append(Tap(hsv, miss, (frame.seq, frame.t), xy))
        self.refit()
        say(TAG, f"{'NOT road' if miss else self.step.title} tap at {xy}: "
                 f"H {hsv[0]}  S {hsv[1]}  V {hsv[2]}")

    def refit(self):
        # The step's values from its taps; values no tap decides go back to
        # config.py's. Hand adjustments are redone after tapping.
        step = self.step
        taps = self.taps[step.key]
        for param in step.params:
            self.values[param.name] = self.start_values[param.name]
        self.values.update(fit_values(step.key, [t.hsv for t in taps if not t.miss],
                                      [t.hsv for t in taps if t.miss]))
        tidy(self.values)

    def adjust(self, delta):
        param = self.step.params[self.chip[self.step.key]]
        self.values[param.name] += delta
        tidy(self.values, moved=param.name)

    def reset_step(self):
        self.taps[self.step.key] = []
        for param in self.step.params:
            self.values[param.name] = self.start_values[param.name]

    def apply(self):
        changed = self.changed()
        self.backup = config_writer.write_values(config_writer.config_path(), changed)
        say(TAG, f"Wrote {len(changed)} values to config.py: "
                 + ", ".join(f"{n} {self.start_values[n]} -> {v}" for n, v in changed.items())
                 + f". The old file is {self.backup}.")
        self.go("applied")

    def quit(self):
        if self.changed() and self.backup is None:
            print_values(self.values, "Not saved. To keep these values, paste into config.py:")
        self.done = True

    def tick(self):
        pass


def _side_button(img, rect, label, name, selected=False):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), PRIMARY if selected else PANEL, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), GREEN if selected else LINE, 1, cv2.LINE_AA)
    size = 15 if text_width(label, 15, True) < w - 8 else 12
    draw_text(img, label, (x + w // 2, y + h // 2 + 5), size, TEXT, bold=True, align="centre")
    return Zone(rect, name)


# ---------------------------------------------------------------- display loop

KEYS = {"K_RETURN": "enter", "K_KP_ENTER": "enter", "K_ESCAPE": "escape", "K_q": "q",
        "K_LEFT": "left", "K_RIGHT": "right", "K_SPACE": "space"}


def run(app, cfg):
    # Shows the app full screen until it is done; returns False if no display.
    from ui.screen import _set_display_env, open_display, show
    _set_display_env()
    display = open_display(cfg)
    if display is None:
        return False
    pygame, surface = display
    keys = {getattr(pygame, k): v for k, v in KEYS.items()}
    period, next_t = 1.0 / cfg.CALIB_UI_HZ, time.monotonic()
    try:
        while not app.done:
            show(pygame, surface, app.draw())
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    app.quit()
                elif event.type == pygame.KEYDOWN and event.key in keys:
                    app.key(keys[event.key])
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    app.tap(event.pos)
            app.tick()
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
            next_t = max(next_t, time.monotonic())
    finally:
        pygame.quit()
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="Tune the colour tests on the robot's display.")
    p.add_argument("recording", nargs="?", help="a folder made by tools/record.py "
                                                "(default: the live front camera)")
    args = p.parse_args(argv)
    if args.recording:
        frames = RecordingFrames(args.recording)
    else:
        say(TAG, "Starting the front camera (stop main.py first: only one program can use it) "
                 f"and locking its exposure after {config.CAM_WARMUP_S:.0f} s, as main.py does.")
        frames = LiveFrames(config)
        try:
            frames.start()
        except Exception as e:
            try:
                frames.stop()
            except Exception:
                pass
            raise SystemExit(f"[{TAG}] The camera did not open ({e}). Is main.py or another "
                             "camera program still running?")
    app = TuneApp(config, frames)
    try:
        if not run(app, config):
            say(TAG, "No display. Run this on the Pi with its screen attached (over SSH the "
                     "screen is used automatically).")
    except KeyboardInterrupt:
        app.quit()
    finally:
        frames.stop()


if __name__ == "__main__":
    main()
