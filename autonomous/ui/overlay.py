"""Camera-view annotations and the display's colour theme. All overlay drawing
lives here, so the robot's display and tools/replay.py can show identical
debug views.

TODO: tools/debug_draw.py is to be replaced by this module in a follow-up
task (tools/replay.py switches to draw_front_overlay; tools/tune_colors.py has).
replay already draws sign annotations with draw_annotations.

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

from perception.signs import normalise_crop, scale_box, sign_label
from robot_types import AnnotationKind
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
SIGN_LABEL_TEXT = 16             # view px; the voted sign's label


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


def draw_annotations(img, annotations, scale=1.0):
    # Draws perception's Annotations (view px) without knowing what they
    # mean: rectangles, lines, points and text, each in its own colour.
    thick = max(1, int(2 * scale))
    for note in annotations:
        pts = [_px(x, y, scale) for x, y in note.pts]
        if note.kind == AnnotationKind.RECT:
            cv2.rectangle(img, pts[0], pts[1], note.colour, thick, cv2.LINE_AA)
        elif note.kind == AnnotationKind.LINE:
            cv2.line(img, pts[0], pts[1], note.colour, thick, cv2.LINE_AA)
        elif note.kind == AnnotationKind.POINT:
            cv2.circle(img, pts[0], max(3, int(4 * scale)), note.colour, -1, cv2.LINE_AA)
        elif note.kind == AnnotationKind.TEXT:
            draw_text(img, note.text, pts[0], int(STRIP_TEXT * scale), note.colour, bold=True)


def draw_sign_overlay(img, sign, vote, cfg, scale=1.0):
    # The sign search region, and the sign's box and label. With a voted
    # sign the box is the voter's steady one, drawn heavy with a large
    # label; otherwise the latest reading's own annotations (a candidate not
    # yet decided, or read as no sign). sign: SignObs or None; vote:
    # SignVote or None.
    notes = sign.annotations if sign is not None else ()
    if vote is None or vote.bbox is None:
        draw_annotations(img, notes, scale)
        return
    draw_annotations(img, notes[:1], scale)            # process_signs puts the ROI first
    x, y, w, h = scale_box(vote.bbox, scale * cfg.FRONT_LANE_SIZE[0] / cfg.FRONT_SIZE[0])
    cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, max(2, int(1.5 * scale)), cv2.LINE_AA)
    size = int(SIGN_LABEL_TEXT * scale)
    below = y < size + 4                               # no room above the box
    draw_text(img, sign_label(vote.sign, cfg), (x, y + h + size if below else y - 6), size,
              GREEN, bold=True)


def draw_sign_panel(img, rect, view, sign, vote, cfg):
    # How the sign was read: the crop the classifier works on (cut from the
    # view, normalised as on the robot), the label, and every class score,
    # best first. Nothing is drawn with no reading at all.
    if sign is None and vote is None:
        return
    x, y, w, h = rect
    img[y:y + h, x:x + w] = (img[y:y + h, x:x + w] * 0.25).astype(img.dtype)
    bbox = vote.bbox if vote is not None and vote.bbox is not None else sign.bbox if sign else None
    if bbox is not None and view is not None:
        bx, by, bw, bh = scale_box(bbox, cfg.FRONT_LANE_SIZE[0] / cfg.FRONT_SIZE[0])
        crop = view[by:by + bh, bx:bx + bw]
        if crop.size:
            side = h - 12
            gray = cv2.resize(normalise_crop(crop, cfg), (side, side), interpolation=cv2.INTER_NEAREST)
            img[y + 6:y + 6 + side, x + 6:x + 6 + side] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tx = x + h + 4
    label = sign_label(vote.sign, cfg) if vote is not None else "reading..."
    draw_text(img, label, (tx, y + 26), 22, GREEN if vote is not None else AMBER, bold=True)
    scores = sorted(sign.scores, key=lambda item: -item[1]) if sign is not None else ()
    for i, (name, score) in enumerate(scores):
        draw_text(img, f"{name.lower()}  {score:+.2f}", (tx, y + 46 + i * 15), 12,
                  TEXT if i == 0 else MUTED)


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
