"""Front camera lane measurement: where the yellow line is, at the rear axle.

The lane image is split into the horizontal bands in FRONT_BANDS, each
    (y_top, y_bottom, forward_m, half_road_px, m_per_px)
with y in FRONT_LANE_SIZE pixels and forward_m measured from the rear axle.
In each band the line's centre is found (yellow pixels, or the road edges
when a dash gap leaves no yellow), converted to metres on the ground, and a
straight line is fitted through the band points.

Ground points are (forward_m, lateral_m) in the robot frame, lateral + = right
of the robot's centreline. The camera is assumed to sit on the centreline, so
image column img_cx (the image centre) is straight ahead.

Sign conventions (robot_types.py):
    lateral error + = rear axle right of the line   -> line is to the axle's left
    heading error + = robot points right of the road -> line runs leftward ahead
"""
import math

import cv2
import numpy as np

from robot_types import FrontLaneObs


def find_yellow_x(yellow_band, min_px):
    # Column centroid of yellow pixels; None if fewer than min_px.
    col_counts = np.count_nonzero(yellow_band, axis=0)
    total = int(col_counts.sum())
    if total < min_px:
        return None
    return float((col_counts * np.arange(col_counts.size)).sum() / total)


def find_road_edges(road_band, seed_x, min_px, cfg):
    # Edges of the continuous road stretch that contains seed_x (the line,
    # or the image centre). Gaps up to ROAD_GAP_MAX_PX are bridged. An edge
    # that reaches the image border is not visible, so it is None.
    rows = road_band.shape[0]
    is_road = np.count_nonzero(road_band, axis=0) >= cfg.ROAD_COL_FRAC * rows
    if not is_road.any():
        return None, None
    is_road = _bridge_gaps(is_road, cfg.ROAD_GAP_MAX_PX)
    start, end = _run_containing(is_road, seed_x)
    if end - start + 1 < min_px:
        return None, None
    left = float(start) if start > 0 else None
    right = float(end) if end < is_road.size - 1 else None
    return left, right


def centre_in_band(yellow_x, left_x, right_x, half_road_px):
    # Yellow x if found; else the midpoint of both edges; else one edge
    # +/- half the road width; else None. Covers the gaps between dashes.
    if yellow_x is not None:
        return yellow_x
    if left_x is not None and right_x is not None:
        return (left_x + right_x) / 2.0
    if left_x is not None:
        return left_x + half_road_px
    if right_x is not None:
        return right_x - half_road_px
    return None


def band_to_ground(x_px, band, img_cx):
    # Pixel column in a band -> (forward_m, lateral_m) from the rear axle.
    _, _, forward_m, _, m_per_px = band
    return forward_m, (x_px - img_cx) * m_per_px


def fit_centre_line(points_m):
    # Least-squares line lateral = slope * forward + intercept;
    # None with fewer than 2 points.
    if len(points_m) < 2:
        return None
    pts = np.asarray(points_m, dtype=np.float64)
    if np.ptp(pts[:, 0]) == 0:
        return None
    slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1)
    return float(slope), float(intercept)


def line_to_errors(fit):
    # Lateral error at forward = 0 (the rear axle) and heading error from
    # the slope, signed per the conventions above.
    slope, intercept = fit
    return -intercept, -math.atan(slope)


def fit_rms(points_m, fit):
    # Root-mean-square distance (m) of the band points from the fitted line.
    slope, intercept = fit
    pts = np.asarray(points_m, dtype=np.float64)
    residuals = pts[:, 1] - (slope * pts[:, 0] + intercept)
    return float(np.sqrt(np.mean(residuals ** 2)))


def process_front_lane(masks, cfg, t, seq):
    # Runs the steps above over every band and builds the observation.
    bands = cfg.FRONT_BANDS
    if bands is None:
        raise ValueError("FRONT_BANDS is not set in config.py: run tools/calibrate.py")
    img_cx = (masks.yellow.shape[1] - 1) / 2.0
    points_m, points_px = [], []
    for band in bands:
        x = _band_centre_x(masks, band, img_cx, cfg)
        if x is None:
            continue
        points_px.append((x, (band[0] + band[1]) / 2.0))
        points_m.append(band_to_ground(x, band, img_cx))

    fit = fit_centre_line(points_m)
    if fit is None:
        return FrontLaneObs(None, None, 0.0, tuple(points_px), None, t, seq)
    lateral_m, heading_err_rad = line_to_errors(fit)
    conf = _confidence(len(points_m), len(bands), fit_rms(points_m, fit), cfg)
    line_px = _line_ends_px(fit, bands, img_cx)
    return FrontLaneObs(lateral_m, heading_err_rad, conf, tuple(points_px), line_px, t, seq)


def _band_centre_x(masks, band, img_cx, cfg):
    # The line's column in one band, from yellow or from the road edges.
    y_top, y_bottom, _, half_road_px, _ = band
    yellow_band = masks.yellow[y_top:y_bottom]
    yellow_x = find_yellow_x(yellow_band, cfg.MIN_LINE_PX)
    road_or_line = cv2.bitwise_or(masks.road[y_top:y_bottom], yellow_band)
    seed_x = yellow_x if yellow_x is not None else img_cx
    left_x, right_x = find_road_edges(road_or_line, seed_x, cfg.MIN_LINE_PX, cfg)
    return centre_in_band(yellow_x, left_x, right_x, half_road_px)


def _confidence(found, total, rms_m, cfg):
    # Fraction of bands found, scaled down as the fit gets worse.
    fit_quality = max(0.0, 1.0 - rms_m / cfg.FRONT_FIT_RMS_MAX_M)
    return (found / total) * fit_quality


def _line_ends_px(fit, bands, img_cx):
    # The fitted line drawn back into pixels at the first and last band.
    slope, intercept = fit
    ends = []
    for band in (bands[0], bands[-1]):
        y_top, y_bottom, forward_m, _, m_per_px = band
        lateral = slope * forward_m + intercept
        ends.append((img_cx + lateral / m_per_px, (y_top + y_bottom) / 2.0))
    return tuple(ends)


def _bridge_gaps(is_road, gap_px):
    # Closes gaps up to gap_px columns wide in a row of booleans.
    row = is_road.astype(np.uint8)[np.newaxis, :]
    kernel = np.ones((1, gap_px + 1), np.uint8)
    closed = cv2.morphologyEx(row, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE)
    return closed[0].astype(bool)


def _run_containing(is_road, seed_x):
    # (start, end) columns of the True run containing seed_x, or of the
    # nearest run if seed_x is not on road.
    padded = np.concatenate(([False], is_road, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    distance = np.maximum(starts - seed_x, 0) + np.maximum(seed_x - ends, 0)
    best = int(np.argmin(distance))
    return int(starts[best]), int(ends[best])