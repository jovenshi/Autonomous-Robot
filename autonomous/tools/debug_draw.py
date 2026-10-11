"""Temporary lane drawing for the tools, until ui/overlay.py exists.

When the display work starts, draw_front_overlay in ui/overlay.py replaces
this module and tune_colors/replay switch to it.
"""
import cv2

GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
GREY = (160, 160, 160)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_lane(view, lane, bands=None):
    # A copy of view with the band rows, band centre points and fitted line.
    img = view.copy()
    width = img.shape[1]
    centre_x = int(round((width - 1) / 2))
    cv2.line(img, (centre_x, 0), (centre_x, img.shape[0] - 1), GREY, 1)
    for band in bands or ():
        cv2.line(img, (0, band[0]), (width - 1, band[0]), GREY, 1)
        cv2.line(img, (0, band[1]), (width - 1, band[1]), GREY, 1)
    if lane is None:
        return img
    for x, y in lane.centre_pts_px:
        cv2.circle(img, (int(round(x)), int(round(y))), 4, GREEN, -1)
    if lane.line_px is not None:
        (x0, y0), (x1, y1) = lane.line_px
        cv2.line(img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), CYAN, 2)
    return img


def put_lines(img, lines, scale=0.45):
    # Text lines in the top-left corner, outlined so they read on any background.
    for i, text in enumerate(lines):
        org = (5, 15 + i * 16)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)
    return img


def lane_text(lane):
    # Lateral error, heading error and conf as short text lines.
    lat = "--" if lane.lateral_m is None else f"{lane.lateral_m * 1000:+.1f} mm"
    head = "--" if lane.heading_err_rad is None else f"{lane.heading_err_rad * 57.2958:+.1f} deg"
    return [f"lat {lat}", f"head {head}", f"conf {lane.conf:.2f}"]
