"""Panels for the DRIVE and DIAGNOSTIC views, drawn onto regions of the screen.

DRIVE is read at a glance while the robot moves: big state and speed, the
lane gauge (robot against the line), the steering bar, pre-flight in IDLE,
and the tuning chart. DIAGNOSTIC (draw_status) is a label/value table of
every number; its sections stack in a fixed order so later rows (junctions,
turns, signs) append without moving what is there.

Signs as everywhere: steer + = left (the steering bar fills LEFT), lateral
+ = rear axle right of the line (the robot marker sits RIGHT of the line on
the gauge), heading + = pointing right of the road, yaw + = counter-clockwise.

A rect is (x, y, w, h) in the image passed in.
"""
from collections import deque

import cv2
import numpy as np

from robot_types import HealthWarning, State
from ui.overlay import (AMBER, BLUE, GREEN, LINE, MUTED, PANEL, PURPLE, RED, TEXT, YELLOW, cm,
                        deg)
from ui.text import draw_text

STATE_COLOURS = {State.IDLE: MUTED, State.LANE_FOLLOW: GREEN, State.SAFE_STOP: RED}
ROW_H = 25                       # px per DIAGNOSTIC row
ROW_TEXT = 17
TITLE_TEXT = 13


def state_colour(state):
    return STATE_COLOURS.get(state, TEXT)


# ---------------------------------------------------------------- DRIVE

def draw_header(img, rect, packet):
    # State (big, coloured) and stop reason on the left; speed on the right.
    x, y, w, h = rect
    draw_text(img, packet.state.name.replace("_", " "), (x + 24, y + 52), 42,
              state_colour(packet.state), bold=True)
    if packet.stop_reason:
        draw_text(img, packet.stop_reason, (x + 26, y + 80), 20, RED)
    draw_text(img, f"{packet.drive.speed:.2f}", (x + w - 24, y + 52), 42, TEXT, bold=True,
              align="right")
    draw_text(img, "SPEED", (x + w - 24, y + 76), 13, MUTED, bold=True, align="right")


def draw_lane_gauge(img, rect, lane, cfg):
    # The line is the centre tick; the robot marker sits at the lateral error
    # (+/-UI_CHART_LAT_M across the gauge), right of the line when lateral +.
    x, y, w, h = rect
    _label(img, (x, y + 16), "LANE")
    lateral = lane.lateral_err_m if lane is not None else None
    value = cm(lateral) if lateral is not None else "no lane"
    draw_text(img, value, (x + w, y + 34), 34, TEXT if lateral is not None else RED,
              bold=True, align="right")
    if lane is not None:
        draw_text(img, f"heading {deg(lane.heading_err_rad)}    conf {lane.conf:.2f}",
                  (x, y + 40), 17, MUTED)
    bar_y, cx, half = y + 62, x + w // 2, w // 2
    cv2.rectangle(img, (x, bar_y), (x + w, bar_y + 22), PANEL, -1)
    cv2.line(img, (cx, bar_y - 6), (cx, bar_y + 28), YELLOW, 3)
    if lateral is not None:
        mx = int(round(cx + max(-1.0, min(1.0, lateral / cfg.UI_CHART_LAT_M)) * half))
        cv2.rectangle(img, (mx - 9, bar_y - 3), (mx + 9, bar_y + 25), PURPLE, -1, cv2.LINE_AA)


def draw_steer_bar(img, rect, steer, cfg, label=True):
    # Centred at 0; + steer (left) fills to the LEFT. Ticks at +/-STEER_MAX.
    x, y, w, h = rect
    if label:
        _label(img, (x, y - 8), "STEER")
        draw_text(img, f"{steer:+.2f}", (x + w, y - 6), 22, TEXT, bold=True, align="right")
    cx, half = x + w // 2, w // 2
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    end = int(round(cx - max(-1.0, min(1.0, steer)) * half))
    cv2.rectangle(img, (min(cx, end), y + 3), (max(cx, end), y + h - 3), BLUE, -1)
    cv2.line(img, (cx, y - 4), (cx, y + h + 4), TEXT, 1)
    for sign in (1, -1):
        tick = int(round(cx - sign * cfg.STEER_MAX * half))
        cv2.line(img, (tick, y - 4), (tick, y + h + 4), AMBER, 2)
    draw_text(img, "L", (x + 6, y + h - 6), 13, MUTED, bold=True)
    draw_text(img, "R", (x + w - 6, y + h - 6), 13, MUTED, bold=True, align="right")


def draw_preflight(img, rect, checks, size=22):
    # Go/no-go list with drawn ticks and crosses.
    x, y, w, h = rect
    _label(img, (x, y + 16), "PRE-FLIGHT")
    for i, (label, ok) in enumerate(checks):
        row_y = y + 16 + (i + 1) * int(size * 1.7)
        draw_mark(img, (x + 12, row_y - size // 3), ok, size // 2)
        draw_text(img, label, (x + 34, row_y), size, TEXT if ok else RED)


# ---------------------------------------------------------------- DIAGNOSTIC

def draw_status(img, packet, cfg):
    # Every number, as label/value rows in two columns of sections.
    h, w = img.shape[:2]
    left = (("DRIVE", drive_rows(packet)), ("LANE", lane_rows(packet)),
            ("ROAD YAW", road_yaw_rows(packet)))
    right = (("IMU", imu_rows(packet)), ("HEALTH", health_rows(packet)),
             ("PRE-FLIGHT", [(label, "ok" if ok else "NO", GREEN if ok else RED)
                             for label, ok in preflight(packet, cfg)]))
    for col, sections in enumerate((left, right)):
        _draw_table(img, (16 + col * w // 2, 10, w // 2 - 32), sections)


def _draw_table(img, origin, sections):
    x, y, w = origin
    for title, rows in sections:
        y += 22
        _label(img, (x, y), title)
        cv2.line(img, (x, y + 6), (x + w, y + 6), LINE, 1)
        y += 6
        for label, value, colour in rows:
            y += ROW_H
            draw_text(img, label, (x, y), ROW_TEXT, MUTED)
            draw_text(img, value, (x + w, y), ROW_TEXT, colour, bold=True, align="right")
        y += 8


def drive_rows(packet):
    d, m = packet.drive, packet.motor
    rows = [("state", packet.state.name, state_colour(packet.state))]
    if packet.stop_reason:
        rows.append(("stop reason", packet.stop_reason, RED))
    return rows + [("speed", f"{d.speed:.2f}", TEXT), ("steer", f"{d.steer:+.2f}", TEXT),
                   ("motors L / R", f"{m.left:+.2f} / {m.right:+.2f}", TEXT)]


def lane_rows(packet):
    lane = packet.lane
    if lane is None:
        return [("lane", "no data", MUTED)]
    return [("lateral", cm(lane.lateral_err_m), TEXT), ("heading", deg(lane.heading_err_rad), TEXT),
            ("conf", f"{lane.conf:.2f}", TEXT), ("cam gap", cm(lane.cam_gap_m), TEXT)]


def road_yaw_rows(packet):
    # Road direction and the time since the camera last anchored it.
    road = packet.road_yaw
    if road is None:
        return [("road yaw", "no data", MUTED)]
    if not road.valid:
        return [("road yaw", "not anchored", AMBER)]
    return [("road yaw", f"{road.yaw_deg:+.1f}°   {packet.t - road.t_anchor:.1f} s ago", TEXT)]


def imu_rows(packet):
    imu = packet.imu
    if imu is None:
        return [("IMU", "no data", MUTED)]
    colour = TEXT if imu.valid else RED
    return [("yaw", f"{imu.yaw_deg:+.1f}°", colour),
            ("yaw rate", f"{imu.yaw_rate_dps:+.1f} °/s", colour),
            ("valid", "yes" if imu.valid else "NO", GREEN if imu.valid else RED),
            ("gyro calib", f"{imu.calib_gyro} / 3", GREEN if imu.calib_gyro == 3 else AMBER),
            ("drift", f"{imu.drift_dps:+.3f} °/s", colour)]


def health_rows(packet):
    rows = [(f"{name} camera", f"{rate:.1f} fps", TEXT) for name, rate in sorted(packet.fps.items())]
    rows.append(("loop overruns", str(packet.overruns), RED if packet.overruns else TEXT))
    if not packet.warnings:
        return rows + [("warnings", "none", GREEN)]
    return rows + [("warning", w.name, RED) for w in packet.warnings]


# ---------------------------------------------------------------- pre-flight

def preflight(packet, cfg):
    # Go/no-go list from the packet: [(label, ok)]. Informational only;
    # Behaviour decides whether START is accepted.
    imu = packet.imu
    checks = [("IMU valid", imu is not None and imu.valid),
              ("gyro calibrated", imu is not None and imu.calib_gyro == 3)]
    for name, rate in sorted(packet.fps.items()):
        checks.append((f"{name} camera {rate:.0f} fps", rate >= cfg.CAM_FPS_MIN))
    if not packet.fps:
        checks.append(("camera fps", False))
    checks.append(("motor serial", HealthWarning.MOTOR_SERIAL_ERROR not in packet.warnings))
    return checks


def draw_mark(img, centre, ok, size=8):
    # A drawn tick (ok) or cross (not ok).
    cx, cy = centre
    if ok:
        pts = [(cx - size, cy), (cx - size // 3, cy + 2 * size // 3), (cx + size, cy - size)]
        cv2.polylines(img, [_pts(pts)], False, GREEN, 3, cv2.LINE_AA)
    else:
        cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), RED, 3, cv2.LINE_AA)
        cv2.line(img, (cx - size, cy + size), (cx + size, cy - size), RED, 3, cv2.LINE_AA)


# ---------------------------------------------------------------- strip chart

class History:
    """Last UI_CHART_S of (t, lateral_err_m, steer), kept in the UI process."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.points = deque()

    def add(self, packet):
        lateral = packet.lane.lateral_err_m if packet.lane is not None else None
        self.points.append((packet.t, lateral, packet.drive.steer))
        while self.points and packet.t - self.points[0][0] > self.cfg.UI_CHART_S:
            self.points.popleft()

    def segments(self, index):
        # Runs of (t, value) for one column (1 = lateral, 2 = steer), broken
        # where a value is missing or packets were dropped for UI_CHART_GAP_S.
        out, current, last_t = [], [], None
        for point in self.points:
            t, value = point[0], point[index]
            gap = last_t is not None and t - last_t > self.cfg.UI_CHART_GAP_S
            if (value is None or gap) and current:
                out.append(current)
                current = []
            if value is not None:
                current.append((t, value))
            last_t = t
        if current:
            out.append(current)
        return out


def draw_chart(img, rect, history, cfg):
    # Lateral error (purple, +/-UI_CHART_LAT_M) and steer (blue, +/-STEER_MAX)
    # against packet time; the newest sample is at the right edge.
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), PANEL, -1)
    cv2.line(img, (x, y + h // 2), (x + w - 1, y + h // 2), LINE, 1)
    if history.points:
        t_end = history.points[-1][0]
        for index, colour, limit in ((1, PURPLE, cfg.UI_CHART_LAT_M), (2, BLUE, cfg.STEER_MAX)):
            for seg in history.segments(index):
                pts = [_chart_xy(t, v, t_end, limit, rect, cfg) for t, v in seg]
                cv2.polylines(img, [_pts(pts)], False, colour, 2, cv2.LINE_AA)
    draw_text(img, f"LATERAL ±{cfg.UI_CHART_LAT_M * 100:.0f} cm", (x + 10, y + 18), 13, PURPLE,
              bold=True)
    draw_text(img, f"STEER ±{cfg.STEER_MAX:.1f}", (x + 150, y + 18), 13, BLUE, bold=True)
    draw_text(img, f"last {cfg.UI_CHART_S:.0f} s", (x + w - 10, y + 18), 13, MUTED,
              align="right")


def _chart_xy(t, value, t_end, limit, rect, cfg):
    x, y, w, h = rect
    px = x + w - 1 - (t_end - t) / cfg.UI_CHART_S * (w - 1)
    frac = max(-1.0, min(1.0, value / limit))
    return px, y + h / 2 - frac * (h / 2 - 4)


def _label(img, org, text):
    draw_text(img, text, org, TITLE_TEXT, MUTED, bold=True)


def _pts(points):
    return np.array([(int(round(px)), int(round(py))) for px, py in points], dtype=np.int32)
