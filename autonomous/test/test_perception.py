"""Tests for perception/color.py, front_lane.py and pipeline.py on synthetic
road images (green grass, black road, yellow centre line).

Run from the robot/ folder:  python -m pytest tests
"""
import math
import sys

import cv2
import numpy as np

import os

import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402
from perception import color, front_lane
from perception.pipeline import process_front_frame
from robot_types import Frame, FrontResult

W, H = 320, 240
CX = (W - 1) / 2.0
M_PER_PX = 0.002
GREEN = (40, 170, 40)        # BGR
ROAD = (25, 25, 25)
YELLOW = (0, 210, 230)
FLOOR = (15, 15, 15)

# (y_top, y_bottom, forward_m, half_road_px, m_per_px); constant scale keeps maths exact
BANDS = [(40, 70, 0.60, 100, M_PER_PX), (80, 110, 0.50, 100, M_PER_PX),
         (120, 150, 0.40, 100, M_PER_PX), (160, 190, 0.30, 100, M_PER_PX),
         (200, 230, 0.20, 100, M_PER_PX)]


@pytest.fixture(autouse=True)
def bands(monkeypatch):
    monkeypatch.setattr(config, "FRONT_BANDS", BANDS)
    monkeypatch.setattr(config, "FRONT_LANE_SIZE", (W, H))


def scene(line_x=lambda y: CX, half_road=100, line_w=8, gap_rows=None, dark_floor=False):
    """Lane-size BGR image: road of half width half_road centred on the line."""
    img = np.full((H, W, 3), GREEN, np.uint8)
    if dark_floor:
        img[:, :20] = FLOOR
    for y in range(H):
        c = line_x(y)
        img[y, max(0, int(round(c - half_road))):int(round(c + half_road))] = ROAD
        if gap_rows is None or not gap_rows[0] <= y < gap_rows[1]:
            x0 = int(round(c - line_w / 2 + 0.5))
            img[y, x0:x0 + line_w] = YELLOW
    return img


def lane_obs(img):
    frame = Frame(image=img, t=1.0, seq=7, cam_id="front")
    masks = color.make_front_masks(frame, config)
    return front_lane.process_front_lane(masks, config, frame.t, frame.seq)


# ---------------------------------------------------------------- color.py

def hsv_of(bgr):
    return cv2.cvtColor(np.full((4, 4, 3), bgr, np.uint8), cv2.COLOR_BGR2HSV)


def test_yellow_mask():
    assert color.mask_yellow(hsv_of(YELLOW), config).all()
    assert not color.mask_yellow(hsv_of(GREEN), config).any()
    assert not color.mask_yellow(hsv_of((180, 200, 200)), config).any()   # pale, unsaturated


def test_green_mask_ignores_brightness_and_darkness():
    assert color.mask_green(np.full((4, 4, 3), (40, 170, 40), np.uint8), config).all()
    assert color.mask_green(np.full((4, 4, 3), (20, 85, 20), np.uint8), config).all()  # shadow
    assert not color.mask_green(np.full((4, 4, 3), (10, 30, 10), np.uint8), config).any()  # too dark
    assert not color.mask_green(np.full((4, 4, 3), ROAD, np.uint8), config).any()


def test_road_mask_is_dark_and_unsaturated():
    img = scene()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    road = color.mask_road(hsv, (0, 0, W, H), config)
    assert road[120, 100] == 255           # road
    assert road[120, 10] == 0              # grass
    assert road[120, int(CX)] == 0         # yellow line


def test_white_threshold_is_relative():
    hsv = np.zeros((10, 10, 3), np.uint8)
    hsv[:5, :, 2] = 120                    # dim white sign in a dim frame
    hsv[5:, :, 2] = 40
    assert color.mask_white(hsv, config)[:5].all()
    assert not color.mask_white(hsv, config)[5:].any()


def test_red_mask_wraps_around_zero():
    assert color.mask_red(hsv_of((0, 0, 200)), config).all()           # hue ~0
    assert color.mask_red(hsv_of((60, 0, 200)), config).all()          # hue ~175
    assert not color.mask_red(hsv_of(YELLOW), config).any()


def test_sign_masks_only_when_asked():
    frame = Frame(scene(), 0.0, 1, "front")
    assert color.make_front_masks(frame, config).white is None
    masks = color.make_front_masks(frame, config, with_signs=True)
    assert masks.white is not None and masks.red is not None


def test_front_masks_downscale_and_scale():
    big = cv2.resize(scene(), (640, 480), interpolation=cv2.INTER_NEAREST)
    masks = color.make_front_masks(Frame(big, 0.0, 1, "front"), config)
    assert masks.bgr.shape == (H, W, 3) and masks.yellow.shape == (H, W)
    assert masks.scale == 2.0


def test_lane_roi_from_bands():
    assert color.front_lane_roi(config, W, H) == (0, 40, W, 190)


# ---------------------------------------------------------------- front_lane.py helpers

def test_find_yellow_x_centroid_and_minimum():
    band = np.zeros((10, 50), np.uint8)
    band[:, 20:24] = 255
    assert front_lane.find_yellow_x(band, 30) == pytest.approx(21.5)
    assert front_lane.find_yellow_x(band[:2, :], 30) is None      # 8 px < 30


def test_road_edges_ignore_dark_floor_beyond_grass():
    band = np.zeros((10, 100), np.uint8)
    band[:, 0:10] = 255          # dark floor at the image edge
    band[:, 30:70] = 255         # road
    assert front_lane.find_road_edges(band, 50, 5, config) == (30.0, 69.0)


def test_road_edges_bridge_line_gap_and_hide_border_edges():
    band = np.zeros((10, 100), np.uint8)
    band[:, 0:46] = 255
    band[:, 52:80] = 255         # 6 px gap (no line drawn) is bridged
    assert front_lane.find_road_edges(band, 49, 5, config) == (None, 79.0)


def test_centre_in_band_fallbacks():
    assert front_lane.centre_in_band(50, 10, 90, 40) == 50
    assert front_lane.centre_in_band(None, 10, 90, 40) == 50
    assert front_lane.centre_in_band(None, 10, None, 40) == 50
    assert front_lane.centre_in_band(None, None, 90, 40) == 50
    assert front_lane.centre_in_band(None, None, None, 40) is None


def test_line_to_errors_signs():
    # line 0.05 m to the right of the axle, running further right ahead
    lateral, heading = front_lane.line_to_errors((0.1, 0.05))
    assert lateral == pytest.approx(-0.05)      # axle is left of the line
    assert heading == pytest.approx(-math.atan(0.1))   # robot points left of the road


def test_fit_needs_two_distinct_points():
    assert front_lane.fit_centre_line([(0.2, 0.0)]) is None
    assert front_lane.fit_centre_line([(0.2, 0.0), (0.2, 0.1)]) is None


# ---------------------------------------------------------------- process_front_lane

def test_centred_and_straight():
    obs = lane_obs(scene())
    assert obs.lateral_m == pytest.approx(0.0, abs=0.002)
    assert obs.heading_err_rad == pytest.approx(0.0, abs=0.01)
    assert obs.conf > 0.9
    assert len(obs.centre_pts_px) == 5 and obs.seq == 7


def test_robot_left_of_line_gives_negative_lateral():
    obs = lane_obs(scene(line_x=lambda y: CX + 20))     # line 20 px right = 0.04 m
    assert obs.lateral_m == pytest.approx(-0.04, abs=0.003)
    assert obs.heading_err_rad == pytest.approx(0.0, abs=0.01)


def test_robot_pointing_right_gives_positive_heading():
    # line drifts left further ahead (smaller y = further ahead)
    obs = lane_obs(scene(line_x=lambda y: CX - 0.5 * (240 - y)))
    assert obs.heading_err_rad > 0.05


def test_dash_gap_uses_road_edges():
    gap = (BANDS[2][0] - 5, BANDS[2][1] + 5)
    obs = lane_obs(scene(gap_rows=gap))
    assert len(obs.centre_pts_px) == 5
    assert obs.centre_pts_px[2][0] == pytest.approx(CX, abs=1.5)
    assert obs.lateral_m == pytest.approx(0.0, abs=0.003)


def test_dark_floor_does_not_move_the_centre():
    obs = lane_obs(scene(line_x=lambda y: CX + 30, dark_floor=True, gap_rows=(0, H)))
    assert obs.lateral_m == pytest.approx(-0.06, abs=0.004)


def test_no_road_gives_zero_conf():
    obs = lane_obs(np.full((H, W, 3), GREEN, np.uint8))
    assert obs.lateral_m is None and obs.heading_err_rad is None and obs.conf == 0.0


def test_missing_bands_raise(monkeypatch):
    monkeypatch.setattr(config, "FRONT_BANDS", None)
    with pytest.raises(ValueError):
        lane_obs(scene())


# ---------------------------------------------------------------- search window and robust fit

def band_rows(i):
    return BANDS[i][0], BANDS[i][1]


def with_blob(img, rows, x0, x1):
    out = img.copy()
    out[rows[0]:rows[1], x0:x1] = YELLOW
    return out


def tracked_obs(img, tracker, t):
    frame = Frame(image=img, t=t, seq=1, cam_id="front")
    return front_lane.process_front_lane(color.make_front_masks(frame, config), config, t, 1,
                                         tracker)


def test_stray_yellow_in_one_band_is_dropped():
    # A yellow patch beside the line in band 3 drags that band's centroid
    # about 25 px (5 cm) off; the fit drops it and stays on the line.
    obs = lane_obs(with_blob(scene(), band_rows(2), int(CX) + 40, int(CX) + 70))
    assert len(obs.centre_pts_px) == 4
    assert obs.lateral_m == pytest.approx(0.0, abs=0.002)
    assert obs.conf < 0.9                                  # a dropped band costs conf


def test_crossing_line_is_ignored_inside_the_window():
    # Line 4 cm left of the axle; a crossing line spans the whole far band.
    line = lambda y: CX + 20
    crossing = with_blob(scene(line_x=line), band_rows(0), 0, W)
    untracked = lane_obs(crossing)
    assert len(untracked.centre_pts_px) == 4               # far band dropped as an outlier
    tracker = front_lane.LaneTracker()
    tracked_obs(scene(line_x=line), tracker, 1.0)          # a clean frame sets the window
    obs = tracked_obs(crossing, tracker, 1.03)
    assert len(obs.centre_pts_px) == 5                     # windowed centroid stays on the line
    assert obs.lateral_m == pytest.approx(-0.04, abs=0.003)


def test_line_outside_the_window_falls_back_to_road_edges():
    tracker = front_lane.LaneTracker()
    tracked_obs(scene(), tracker, 1.0)
    obs = tracked_obs(scene(line_x=lambda y: CX + 50), tracker, 1.03)   # jumped 10 cm
    assert obs.lateral_m == pytest.approx(-0.10, abs=0.006)


def test_window_needs_a_recent_confident_fit():
    tracker = front_lane.LaneTracker()
    assert tracker.window_fit(1.0, config) is None
    tracker.update((0.0, 0.01), config.LANE_SEARCH_MIN_CONF - 0.01, 1.0, config)
    assert tracker.window_fit(1.0, config) is None         # not confident enough
    tracker.update((0.0, 0.01), 0.9, 1.0, config)
    assert tracker.window_fit(1.0 + config.LANE_SEARCH_MAX_AGE_S, config) == (0.0, 0.01)
    assert tracker.window_fit(1.01 + config.LANE_SEARCH_MAX_AGE_S, config) is None


def test_fit_robust_weights_and_keeps_a_minimum():
    pts = [(0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.05)]
    fit, kept = front_lane.fit_robust(pts, [1, 1, 1, 1], config)
    assert kept == [0, 1, 2] and fit == pytest.approx((0.0, 0.0), abs=1e-9)
    fit, kept = front_lane.fit_robust(pts[:3], [1, 1, 1], config)
    assert kept == [0, 1, 2]                               # never below LANE_MIN_KEEP
    even = front_lane.fit_robust([(0.2, 0.0), (0.4, 0.01)], [1, 1], config)[0]
    light = front_lane.fit_robust([(0.2, 0.0), (0.3, 0.01), (0.4, 0.0)], [1, 0.01, 1], config)[0]
    assert even[0] == pytest.approx(0.05) and light[1] == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------- pipeline.py

def test_process_front_frame():
    big = cv2.resize(scene(line_x=lambda y: CX + 20), (640, 480), interpolation=cv2.INTER_NEAREST)
    result = process_front_frame(Frame(big, 3.5, 42, "front"), signs_enabled=False, cfg=config)
    assert isinstance(result, FrontResult)
    assert result.view.shape == (H, W, 3)
    assert (result.t, result.seq) == (3.5, 42)
    assert result.inter is None and result.sign is None
    assert result.lane.lateral_m == pytest.approx(-0.04, abs=0.003)
    assert 0 < result.mean_v < 255

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))