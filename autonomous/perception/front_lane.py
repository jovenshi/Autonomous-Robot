"""Front camera lane measurement: where the yellow line is, at the rear axle.

The lane image is split into the horizontal bands in FRONT_BANDS, each
    (y_top, y_bottom, forward_m, half_road_px, m_per_px)
with y in FRONT_LANE_SIZE pixels and forward_m measured from the rear axle.
In each band the line's centre is found (yellow pixels, or the road edges
when a dash gap leaves no yellow), converted to metres on the ground, and a
straight line is fitted through the band points.

Two guards against stray yellow (the crossing line at a junction, a sign,
a patch beside the road):
  * Search window: with a LaneTracker, each band only looks for yellow
    within LANE_SEARCH_M of where the last confident fit puts the line.
    The window switches off when that fit is older than LANE_SEARCH_MAX_AGE_S,
    so a lost line is searched for across the whole width again.
  * Outlier-resistant fit: points from road edges (dash gaps) count
    LANE_EDGE_WEIGHT as much as yellow points, and the point farthest from
    the fit is dropped while it misses by more than LANE_OUTLIER_M (keeping
    at least LANE_MIN_KEEP points). Dropped bands lower conf like missing ones.

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


class LaneTracker:
    """The last confident fit, for the next frame's search window. One per
    camera thread (perception/pipeline.py); replay makes its own."""

    def __init__(self):
        self.fit = None                  # (slope, intercept) in metres
        self.t = None

    def window_fit(self, t, cfg):
        # The fit to search around, or None (no window) when it is too old.
        if self.fit is None or t - self.t > cfg.LANE_SEARCH_MAX_AGE_S or t < self.t:
            return None
        return self.fit

    def update(self, fit, conf, t, cfg):
        if fit is not None and conf >= cfg.LANE_SEARCH_MIN_CONF:
            self.fit, self.t = fit, t


def fit_robust(points_m, weights, cfg):
    # Weighted line fit, dropping the worst point while it misses by more
    # than LANE_OUTLIER_M and more than LANE_MIN_KEEP points remain. Each
    # point is judged against a fit of the OTHER points: an outlier in the
    # nearest or farthest band pulls an ordinary fit towards itself and
    # would hide. Returns (fit or None, kept: list of indices).
    kept = list(range(len(points_m)))
    while len(kept) > cfg.LANE_MIN_KEEP:
        misses = [_miss_without(points_m, weights, kept, i) for i in kept]
        worst = int(np.argmax(misses))
        if misses[worst] <= cfg.LANE_OUTLIER_M:
            break
        kept.pop(worst)
    return _weighted_fit([points_m[i] for i in kept], [weights[i] for i in kept]), kept


def _miss_without(points_m, weights, kept, i):
    # How far point i lies from the fit of the other kept points.
    others = [j for j in kept if j != i]
    fit = _weighted_fit([points_m[j] for j in others], [weights[j] for j in others])
    if fit is None:
        return 0.0
    forward, lateral = points_m[i]
    return abs(lateral - (fit[0] * forward + fit[1]))


def _weighted_fit(points_m, weights):
    # Least-squares lateral = slope * forward + intercept, each point
    # counting by its weight; None with fewer than 2 distinct distances.
    if len(points_m) < 2:
        return None
    pts = np.asarray(points_m, dtype=np.float64)
    if np.ptp(pts[:, 0]) == 0:
        return None
    slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1, w=np.sqrt(weights))
    return float(slope), float(intercept)


def process_front_lane(masks, cfg, t, seq, tracker=None):
    # Runs the steps above over every band and builds the observation.
    # With a tracker, yellow is only searched for near the last fit.
    bands = cfg.FRONT_BANDS
    if bands is None:
        raise ValueError("FRONT_BANDS is not set in config.py: run tools/calib_ui.py")
    img_cx = (masks.yellow.shape[1] - 1) / 2.0
    window = tracker.window_fit(t, cfg) if tracker is not None else None
    points_m, points_px, weights = [], [], []
    for band in bands:
        predicted = _predicted_x(window, band, img_cx)
        x, from_yellow = _band_centre_x(masks, band, img_cx, cfg, predicted)
        if x is None:
            continue
        points_px.append((x, (band[0] + band[1]) / 2.0))
        points_m.append(band_to_ground(x, band, img_cx))
        weights.append(1.0 if from_yellow else cfg.LANE_EDGE_WEIGHT)

    fit, kept = fit_robust(points_m, weights, cfg)
    if fit is None:
        return FrontLaneObs(None, None, 0.0, tuple(points_px), None, t, seq)
    points_m = [points_m[i] for i in kept]
    lateral_m, heading_err_rad = line_to_errors(fit)
    conf = _confidence(len(points_m), len(bands), fit_rms(points_m, fit), cfg)
    if tracker is not None:
        tracker.update(fit, conf, t, cfg)
    line_px = _line_ends_px(fit, bands, img_cx)
    return FrontLaneObs(lateral_m, heading_err_rad, conf, tuple(points_px[i] for i in kept),
                        line_px, t, seq)


def _predicted_x(fit, band, img_cx):
    # Where a fit puts the line in this band (lane px), or None.
    if fit is None:
        return None
    _, _, forward_m, _, m_per_px = band
    slope, intercept = fit
    return img_cx + (slope * forward_m + intercept) / m_per_px


def _band_centre_x(masks, band, img_cx, cfg, predicted_x=None):
    # The line's column in one band, from yellow or from the road edges;
    # returns (x or None, True if it came from yellow). With predicted_x,
    # yellow outside LANE_SEARCH_M of it is ignored.
    y_top, y_bottom, _, half_road_px, m_per_px = band
    yellow_band = masks.yellow[y_top:y_bottom]
    searched = yellow_band
    if predicted_x is not None:
        half = cfg.LANE_SEARCH_M / m_per_px
        lo, hi = int(max(0, round(predicted_x - half))), int(max(0, round(predicted_x + half) + 1))
        searched = np.zeros_like(yellow_band)
        searched[:, lo:hi] = yellow_band[:, lo:hi]
    yellow_x = find_yellow_x(searched, cfg.MIN_LINE_PX)
    road_or_line = cv2.bitwise_or(masks.road[y_top:y_bottom], yellow_band)
    seed_x = next(x for x in (yellow_x, predicted_x, img_cx) if x is not None)
    seed_x = min(max(seed_x, 0), yellow_band.shape[1] - 1)
    left_x, right_x = find_road_edges(road_or_line, seed_x, cfg.MIN_LINE_PX, cfg)
    return centre_in_band(yellow_x, left_x, right_x, half_road_px), yellow_x is not None


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