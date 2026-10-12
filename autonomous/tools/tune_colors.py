"""Tune the yellow and road colour thresholds on recorded frames.

Run from the autonomous/ folder on the Pi desktop:
    python -m tools.tune_colors recordings/front_20261011_140000

Two windows: "colors" shows four panels (frame | yellow mask | road mask |
lane result) and "controls" holds the sliders. The masks are made by the real
perception/color.py code with the slider values, so what you see is what the
robot will see. The lane panel needs FRONT_BANDS (tools/calibrate.py).

Keys (in the "colors" window):
    n / p      next / previous frame (or use the frame slider)
    click      print H, S, V of the clicked pixel in the terminal
    s          print the current values
    q or Esc   quit and print the values to paste into config.py
"""
import argparse
import os
import sys
import types

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from perception.color import make_front_masks
from perception.front_lane import process_front_lane
from tools.debug_draw import draw_lane, lane_text, put_lines
from tools.recording import load_frame, read_index

WINDOW = "colors"
CONTROLS = "controls"

# (config name, slider maximum)
SLIDERS = (
    ("YELLOW_H_MIN", 179), ("YELLOW_H_MAX", 179),
    ("YELLOW_S_MIN", 255), ("YELLOW_V_MIN", 255),
    ("ROAD_S_MAX", 255), ("ROAD_V_T_MIN", 255), ("ROAD_V_T_MAX", 255),
)


def config_with(overrides):
    # A copy of every config value, with the slider values swapped in.
    values = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_controls(frame_count):
    cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("frame", CONTROLS, 0, max(frame_count - 1, 1), lambda v: None)
    for name, maximum in SLIDERS:
        cv2.createTrackbar(name, CONTROLS, int(getattr(config, name)), maximum, lambda v: None)


def read_sliders():
    return {name: cv2.getTrackbarPos(name, CONTROLS) for name, _ in SLIDERS}


def tint(view, mask, colour):
    # The view with mask pixels painted in colour, half transparent.
    out = view.copy()
    out[mask > 0] = (0.5 * out[mask > 0] + 0.5 * np.array(colour)).astype(np.uint8)
    return out


def render(frame, cfg):
    # The 2x2 grid for one frame with the given thresholds.
    masks = make_front_masks(frame, cfg)
    yellow = put_lines(tint(masks.bgr, masks.yellow, (255, 0, 255)), ["yellow"])
    road = put_lines(tint(masks.bgr, masks.road, (255, 0, 0)), ["road"])
    if cfg.FRONT_BANDS is None:
        lane_panel = put_lines(masks.bgr.copy(), ["lane: run tools/calibrate.py first"])
    else:
        lane = process_front_lane(masks, cfg, frame.t, frame.seq)
        lane_panel = put_lines(draw_lane(masks.bgr, lane, cfg.FRONT_BANDS), lane_text(lane))
    frame_panel = put_lines(masks.bgr.copy(), [f"frame {frame.seq}"])
    return np.vstack([np.hstack([frame_panel, yellow]), np.hstack([road, lane_panel])]), masks


def print_values(values):
    print("\n# Paste into config.py (colour tests)")
    for name, _ in SLIDERS:
        print(f"{name} = {values[name]}")


def on_click(event, x, y, flags, state):
    # Prints HSV of the clicked pixel (any of the four panels).
    if event != cv2.EVENT_LBUTTONDOWN or state.get("hsv") is None:
        return
    hsv = state["hsv"]
    h, w = hsv.shape[:2]
    px, py = x % w, y % h
    hue, sat, val = hsv[py, px]
    print(f"pixel ({px}, {py}): H {hue}  S {sat}  V {val}")


def step_frame(key, index, count):
    # n / p keys move the frame slider.
    if key == ord("n"):
        index = min(index + 1, count - 1)
    elif key == ord("p"):
        index = max(index - 1, 0)
    cv2.setTrackbarPos("frame", CONTROLS, index)


def run(rec_dir):
    rows = read_index(rec_dir)
    if not rows:
        raise SystemExit(f"No frames in {rec_dir}")
    make_controls(len(rows))
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    state = {"hsv": None}
    cv2.setMouseCallback(WINDOW, on_click, state)
    cache = {}
    while True:
        index = min(cv2.getTrackbarPos("frame", CONTROLS), len(rows) - 1)
        if index not in cache:
            cache = {index: load_frame(rec_dir, rows[index])}
        values = read_sliders()
        grid, masks = render(cache[index], config_with(values))
        state["hsv"] = masks.hsv
        cv2.imshow(WINDOW, grid)
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            print_values(values)
        step_frame(key, index, len(rows))
    cv2.destroyAllWindows()
    print_values(values)


def main():
    p = argparse.ArgumentParser(description="Tune yellow and road thresholds on a recording.")
    p.add_argument("recording", help="folder made by tools/record.py")
    run(p.parse_args().recording)


if __name__ == "__main__":
    main()
