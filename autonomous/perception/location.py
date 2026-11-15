"""Front camera evidence of where the junction is (docs/quiz3_plan.md, stage 5).

Two cues, each turned into metres ahead of the rear axle:
  * the crossing line: the cross road's dashed yellow centre line, which runs
    across our road through the junction centre (the junction is a plus: our
    own line runs on through it). Dash gaps are closed along each row and a
    few rows either side are searched, so a dashed line, tilted by the
    robot's yaw, still covers CROSS_LINE_MIN_FRAC of the width (the real
    dashes are short with wide gaps). It must lie on road at least
    CROSS_ROAD_EXTRA_M wider than ours (the cross road): yellow specks along
    the mat's far edge, joined by the closing, have grass either side, and a
    dark mat seam is too thin to pass for it. Only
    yellow within CROSS_LINE_ROAD_PX of road counts, so specks on the grass
    can't be joined into a line along a dark mat seam.
  * the flare: where the road starts to widen into the junction's rounded
    corners, FLARE_TO_CENTRE_M before the centre (as this test finds it:
    a little past where the widening starts). It is the nearest row where
    the road is FLARE_MIN_M wider than the road at the bottom of the view.
    Comparing with the road in view, not with ROAD_WIDTH_M, cancels the few
    centimetres the measured width is off on a straight (edge blur, yaw);
    comparing with the bottom, not with the strips just below, finds a
    gradual rounded corner by how far it has widened, not how sharply.
    The bottom must itself be normal road (at most FLARE_BASE_MAX_M over
    ROAD_WIDTH_M): once the view starts inside a rounded corner, the cross
    road beyond would otherwise read as the flare, 0.3 m too far ahead. A
    flare at or beyond the crossing line seen in the same frame is dropped:
    inside the junction the far corners can read as one (junction_3).

Image rows become metres by interpolating between the band centres of
FRONT_BANDS; rows outside them give None (no extrapolation). The front
camera sees the ground 0.71 m to 1.66 m ahead, so each cue is only seen in
part of the approach; estimation/location.py carries s between fixes.
"""
import cv2
import numpy as np

from perception.color import front_lane_roi
from perception.front_lane import road_span
from robot_types import Annotation, AnnotationKind, LocationEvidence


def row_to_metres(row, cfg):
    # Lane-size image row -> metres ahead of the rear axle, or None outside
    # the band centres.
    rows, forward, _ = _band_table(cfg)
    if row is None or not rows[0] <= row <= rows[-1]:
        return None
    return float(np.interp(row, rows, forward))


def crossing_line_row(yellow_mask, road_mask, cfg):
    # Centre row of the nearest (lowest) run of rows where yellow, with dash
    # gaps closed, covers at least CROSS_LINE_MIN_FRAC of the width, on road
    # CROSS_ROAD_EXTRA_M wider than ours; None if there is none.
    x, top, w, h = front_lane_roi(cfg, yellow_mask.shape[1], yellow_mask.shape[0])
    near = np.ones((cfg.CROSS_LINE_ROAD_PX, cfg.CROSS_LINE_ROAD_PX), np.uint8)
    band = cv2.bitwise_and(yellow_mask[top:top + h], cv2.dilate(road_mask[top:top + h], near))
    closed = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((1, cfg.CROSS_LINE_CLOSE_PX), np.uint8))
    tall = cv2.dilate(closed, np.ones((2 * cfg.CROSS_LINE_TILT_PX + 1, 1), np.uint8))
    covered = np.count_nonzero(tall, axis=1) >= cfg.CROSS_LINE_MIN_FRAC * w
    for start, end in reversed(_runs(covered)):
        row = top + (start + end) // 2
        if _on_wide_road(row, yellow_mask, road_mask, cfg):
            return row
    return None


def _on_wide_road(row, yellow_mask, road_mask, cfg):
    # True if road (or yellow) in the rows CROSS_ROAD_CHECK_PX either side
    # of row is CROSS_ROAD_EXTRA_M wider than ROAD_WIDTH_M, or reaches both
    # image edges (close in, the view itself is narrower than that): the
    # cross road is that wide and that thick; our road alone, or a dark mat
    # seam, is not.
    lo, hi = max(0, row - cfg.CROSS_ROAD_CHECK_PX), row + cfg.CROSS_ROAD_CHECK_PX + 1
    strip = cv2.bitwise_or(road_mask[lo:hi], yellow_mask[lo:hi])
    span = road_span(strip, cfg.MIN_LINE_PX, cfg)
    if span is None:
        return False
    if span[0] == 0 and span[1] == strip.shape[1] - 1:
        return True
    rows, _, m_per_px = _band_table(cfg)
    width_m = (span[1] - span[0] + 1) * float(np.interp(row, rows, m_per_px))
    return width_m >= cfg.ROAD_WIDTH_M + cfg.CROSS_ROAD_EXTRA_M


def road_widening(road_mask, yellow_mask, cfg):
    # [(bottom row, extra width in m or None)] per FLARE_STRIP_PX strip,
    # nearest strip first. Extra width = the widest stretch of road (our
    # line bridged) minus ROAD_WIDTH_M; None where there is no road. Where
    # the road runs off the image edge the width is a lower bound, which can
    # only under-report widening.
    rows, _, m_per_px = _band_table(cfg)
    # Speckle holes (the road's texture under bright light, junction_1) are
    # closed first, so a strip's width doesn't jump with them.
    closing = np.ones((cfg.FLARE_CLOSE_PX, cfg.FLARE_CLOSE_PX), np.uint8)
    road_or_line = cv2.morphologyEx(cv2.bitwise_or(road_mask, yellow_mask), cv2.MORPH_CLOSE, closing)
    out = []
    step = cfg.FLARE_STRIP_PX
    for bottom in range(int(rows[-1]), int(rows[0]) + step - 1, -step):
        span = road_span(road_or_line[bottom - step + 1:bottom + 1], cfg.MIN_LINE_PX, cfg)
        if span is None:
            out.append((bottom, None))
            continue
        width_m = (span[1] - span[0] + 1) * float(np.interp(bottom, rows, m_per_px))
        out.append((bottom, width_m - cfg.ROAD_WIDTH_M))
    return out


def flare_row(widening, cfg):
    # The bottom row of the nearest run of FLARE_WIDE_STRIPS strips that are
    # all FLARE_MIN_M wider than the road at the bottom of the view (the
    # median of the FLARE_NORMAL_STRIPS nearest strips); None if the bottom
    # is not normal road or nothing beyond is that much wider.
    extra = [e for _, e in widening]
    n_wide, n_normal = cfg.FLARE_WIDE_STRIPS, cfg.FLARE_NORMAL_STRIPS
    if None in extra[:n_normal]:
        return None
    base = float(np.median(extra[:n_normal]))
    if base > cfg.FLARE_BASE_MAX_M:
        return None
    for i in range(n_normal, len(widening) - n_wide + 1):
        above = extra[i:i + n_wide]
        if None not in above and min(above) - base >= cfg.FLARE_MIN_M:
            return widening[i][0]
    return None


def process_location(masks, cfg, t, seq):
    # FrontMasks -> LocationEvidence, with a line and a label for each cue
    # seen. side_surface stays None until the side camera's box (stage 8).
    cross_row = crossing_line_row(masks.yellow, masks.road, cfg)
    flare = flare_row(road_widening(masks.road, masks.yellow, cfg), cfg)
    cross_m, flare_m = row_to_metres(cross_row, cfg), row_to_metres(flare, cfg)
    if cross_m is not None and flare_m is not None and flare_m >= cross_m:
        flare, flare_m = None, None      # the flare is always before the centre: a false one
    width = masks.yellow.shape[1]
    notes = _cue_notes(cross_row, cross_m, "cross", cfg.CROSS_LINE_COLOUR, width)
    notes += _cue_notes(flare, flare_m, "flare", cfg.FLARE_COLOUR, width)
    return LocationEvidence(crossing_dist_m=cross_m, flare_dist_m=flare_m, side_surface=None,
                            annotations=tuple(notes), t=t, seq=seq)


def _cue_notes(row, dist_m, name, colour, width):
    # A full-width line at the cue's row and its distance, or nothing.
    if row is None or dist_m is None:
        return []
    return [Annotation(AnnotationKind.LINE, ((0, row), (width - 1, row)), "", colour),
            Annotation(AnnotationKind.TEXT, ((4, max(12, row - 4)),), f"{name} {dist_m:.2f} m",
                       colour)]


def _band_table(cfg):
    # Band centre rows (ascending), their forward_m and m_per_px.
    bands = sorted(cfg.FRONT_BANDS, key=lambda b: b[0])
    rows = np.array([(b[0] + b[1]) / 2.0 for b in bands])
    return rows, np.array([b[2] for b in bands]), np.array([b[4] for b in bands])


def _runs(flags):
    # [(start, end)] of the True runs in a 1-D bool array, top to bottom.
    padded = np.concatenate(([False], flags, [False])).astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1))
