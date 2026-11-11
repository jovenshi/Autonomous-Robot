"""Tune the yellow, road, white and red colour thresholds on recorded frames.

Run from the autonomous/ folder on the Pi desktop:
    python -m tools.tune_colors recordings/front_20261011_140000

Two windows: "colors" shows six panels (frame | yellow | white on top, road
| lane result | red below) and "controls" holds the sliders. White and red
feed sign reading: the sign panel must be white and its black box must not,
and the PAUSE octagon must be red. The masks are made by the real
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

# (config name, slider maximum, slider steps per unit): the config value is
# slider position / steps, so WHITE_V_REL (a ratio) moves in steps of 0.01.
SLIDERS = (
    ("YELLOW_H_MIN", 179, 1), ("YELLOW_H_MAX", 179, 1),
    ("YELLOW_S_MIN", 255, 1), ("YELLOW_V_MIN", 255, 1),
    ("ROAD_S_MAX", 255, 1), ("ROAD_V_T_MIN", 255, 1), ("ROAD_V_T_MAX", 255, 1),
    ("WHITE_S_MAX", 255, 1), ("WHITE_V_REL", 100, 100), ("WHITE_V_PCT", 100, 1),
    ("RED_H_LOW_MAX", 179, 1), ("RED_H_HIGH_MIN", 179, 1), ("RED_S_MIN", 255, 1),
)


def config_with(overrides):
    # A copy of every config value, with the slider values swapped in.
    values = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_controls(frame_count):
    cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("frame", CONTROLS, 0, max(frame_count - 1, 1), lambda v: None)
    for name, maximum, steps in SLIDERS:
        cv2.createTrackbar(name, CONTROLS, int(round(getattr(config, name) * steps)), maximum,
                           lambda v: None)


def read_sliders():
    return {name: slider_value(cv2.getTrackbarPos(name, CONTROLS), steps)
            for name, _, steps in SLIDERS}


def slider_value(position, steps):
    # Slider position -> config value: an int, or a ratio when steps > 1.
    return position if steps == 1 else position / steps


def tint(view, mask, colour):
    # The view with mask pixels painted in colour, half transparent.
    out = view.copy()
    out[mask > 0] = (0.5 * out[mask > 0] + 0.5 * np.array(colour)).astype(np.uint8)
    return out


def render(frame, cfg):
    # The 2x3 grid for one frame with the given thresholds.
    masks = make_front_masks(frame, cfg, with_signs=True)
    yellow = put_lines(tint(masks.bgr, masks.yellow, (255, 0, 255)), ["yellow"])
    road = put_lines(tint(masks.bgr, masks.road, (255, 0, 0)), ["road"])
    white = put_lines(tint(masks.bgr, masks.white, (0, 255, 255)), ["white (sign panel)"])
    red = put_lines(tint(masks.bgr, masks.red, (255, 255, 0)), ["red (PAUSE)"])
    if cfg.FRONT_BANDS is None:
        lane_panel = put_lines(masks.bgr.copy(), ["lane: run tools/calibrate.py first"])
    else:
        lane = process_front_lane(masks, cfg, frame.t, frame.seq)
        lane_panel = put_lines(draw_lane(masks.bgr, lane, cfg.FRONT_BANDS), lane_text(lane))
    frame_panel = put_lines(masks.bgr.copy(), [f"frame {frame.seq}"])
    grid = np.vstack([np.hstack([frame_panel, yellow, white]), np.hstack([road, lane_panel, red])])
    return grid, masks


def print_values(values):
    print("\n# Paste into config.py (colour tests)")
    for name, _, _ in SLIDERS:
        print(f"{name} = {values[name]}")


def on_click(event, x, y, flags, state):
    # Prints HSV of the clicked pixel (any of the six panels).
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
    p = argparse.ArgumentParser(description="Tune the colour thresholds on a recording.")
    p.add_argument("recording", help="folder made by tools/record.py")
    run(p.parse_args().recording)


if __name__ == "__main__":
    main()
