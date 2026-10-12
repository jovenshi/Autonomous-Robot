"""Camera-view annotations and the display's colour theme. All overlay drawing
lives here, so the robot's display and tools/replay.py can show identical
debug views.

TODO: tools/debug_draw.py is to be replaced by this module in a follow-up
task (tools/replay.py and tools/tune_colors.py switch to draw_front_overlay).

Geometry comes in processed-image pixels (the 320 x 240 view each camera
result carries, from the same frame the observations were measured on).
The screen enlarges the view first and passes scale (display px per view
px), so lines and text are drawn sharp at display resolution. Functions draw
in place and accept results with missing observations without raising.

Units on screen: cm for distances, degrees for angles.
"""
import math

import cv2
import numpy as np

from ui.text import draw_text

# Theme (BGR): dark background, one accent per meaning.
BG = (23, 20, 18)
PANEL = (39, 34, 30)
LINE = (72, 66, 60)
TEXT = (235, 235, 235)
MUTED = (153, 144, 138)
GREEN = (80, 185, 63)
RED = (73, 81, 248)
AMBER = (34, 153, 210)
BLUE = (255, 166, 88)
YELLOW = (65, 179, 227)
PURPLE = (255, 140, 188)

STRIP_H = 18                     # view px; status strip along the bottom of a view
STRIP_TEXT = 12                  # view px; its text size


def cm(metres):
    # "+1.2 cm", or "--" when not measured.
    return "--" if metres is None else f"{metres * 100:+.1f} cm"


def deg(radians):
    return "--" if radians is None else f"{math.degrees(radians):+.1f}°"


def draw_front_overlay(img, front, scale=1.0):
    # Robot centreline marker, band centre points, fitted centre line and a
    # status strip (lateral, heading, conf as the front camera measured them).
    _draw_centreline(img, scale)
    lane = front.lane if front is not None else None
    if lane is None:
        return
    for x, y in lane.centre_pts_px:
        cv2.circle(img, _px(x, y, scale), max(3, int(4 * scale)), GREEN, -1, cv2.LINE_AA)
    if lane.line_px is not None:
        (x0, y0), (x1, y1) = lane.line_px
        cv2.line(img, _px(x0, y0, scale), _px(x1, y1, scale), BLUE, max(2, int(2 * scale)),
                 cv2.LINE_AA)
    _draw_strip(img, f"lateral {cm(lane.lateral_m)}    heading {deg(lane.heading_err_rad)}    "
                     f"conf {lane.conf:.2f}", scale)


def draw_side_overlay(img, side, cfg, scale=1.0):
    # Target column (SIDE_TARGET_PX, where the edge sits with the robot
    # centred), the detected edge column and the offset at the camera.
    height = img.shape[0]
    if cfg.SIDE_TARGET_PX is not None:
        x = int(round(cfg.SIDE_TARGET_PX * scale))
        cv2.line(img, (x, 0), (x, height - 1), YELLOW, max(1, int(scale)))
    obs = side.side if side is not None else None
    if obs is None:
        return
    if obs.boundary_x_px is not None:
        x = int(round(obs.boundary_x_px * scale))
        cv2.line(img, (x, 0), (x, height - 1), BLUE, max(2, int(2 * scale)))
    _draw_strip(img, f"offset {cm(obs.offset_m)}    conf {obs.conf:.2f}", scale)


def mark_stale(img, age_s):
    # Greys the view out and shows how old it is.
    grey = np.full_like(img, MUTED)
    cv2.addWeighted(img, 0.35, grey, 0.65, 0, dst=img)
    size = max(14, img.shape[0] // 12)
    draw_text(img, f"STALE  {age_s:.1f} s", (size, size * 2), size, AMBER, bold=True)


def draw_no_data(img, label):
    # Nothing has ever arrived for this panel.
    img[:] = PANEL
    h, w = img.shape[:2]
    size = max(12, h // 14)
    draw_text(img, label, (w // 2, h // 2 - size // 2), size, MUTED, align="centre")
    draw_text(img, "NO DATA", (w // 2, h // 2 + size), size, MUTED, bold=True, align="centre")


def _px(x, y, scale):
    return int(round(x * scale)), int(round(y * scale))


def _draw_centreline(img, scale):
    # The column straight ahead of the robot: perception uses (width - 1) / 2
    # of the view as the robot centreline (perception/front_lane.py, img_cx),
    # so the line sits this far from the marker when lateral error is measured.
    h, w = img.shape[:2]
    cx = int(round((w / scale - 1) / 2 * scale))
    dash = max(6, int(6 * scale))
    for y in range(0, h, 2 * dash):
        cv2.line(img, (cx, y), (cx, min(y + dash, h - 1)), TEXT, 1)
    tip, half = h - int((STRIP_H + 2) * scale), max(7, int(7 * scale))
    cv2.fillPoly(img, [np.array([(cx, tip - 2 * half), (cx - half, tip), (cx + half, tip)])],
                 PURPLE, cv2.LINE_AA)


def _draw_strip(img, text, scale):
    # Darkened strip along the bottom with one line of text.
    h = img.shape[0]
    strip = int(STRIP_H * scale)
    img[h - strip:, :] = (img[h - strip:, :] * 0.25).astype(img.dtype)
    draw_text(img, text, (int(6 * scale), h - int(5 * scale)), int(STRIP_TEXT * scale), TEXT)
