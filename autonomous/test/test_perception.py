"""Tests for perception/color.py, front_lane.py and pipeline.py on synthetic
road images (green grass, black road, yellow centre line).

Run from the robot/ folder:  python -m pytest tests
"""
import math
import sys
import types

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
    assert front_lane.find_yellow_x(band, 30, 10) == pytest.approx(21.5)
    assert front_lane.find_yellow_x(band[:2, :], 30, 10) is None      # 8 px < 30


def test_road_span_bridges_line_and_picks_widest():
    road = np.zeros((10, 150), np.uint8)
    road[:, 30:50] = 255                                             # road, line at 50-59,
    road[:, 60:80] = 255                                             # then road again
    road[:, 120:126] = 255                                           # a small dark patch, far off
    assert front_lane.road_span(road, 30, config) == (30, 79)
    assert front_lane.road_span(np.zeros((10, 100), np.uint8), 30, config) is None


def test_band_centre_ignores_yellow_grass_beside_road():
    masks = types.SimpleNamespace(yellow=np.zeros((20, 200), np.uint8),
                                  road=np.zeros((20, 200), np.uint8))
    masks.road[:, 60:140] = 255
    masks.road[:, 98:104] = 0
    masks.yellow[:, 98:104] = 255                                    # the line, on the road
    masks.yellow[:, 0:50] = 255                                      # grass passing the yellow test
    band = (0, 20, 1.0, 40, 0.002)
    assert front_lane._band_centre_x(masks, band, 99.5, config) == pytest.approx(100.5)


def test_find_yellow_x_ignores_stray_specks():
    band = np.zeros((10, 50), np.uint8)
    band[:, 20:24] = 255
    band[2, 45:48] = 255                                             # specks far from the line
    band[7, 2] = 255
    assert front_lane.find_yellow_x(band, 30, 5) == pytest.approx(21.5)


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


def test_drop_outliers_removes_far_point_only():
    pts = [(0.7, 0.0), (0.8, 0.0), (0.9, 0.0), (1.0, 0.3), (1.1, 0.0)]
    px = [(i, i) for i in range(5)]
    fit, kept, kept_px = front_lane.drop_outliers(
        front_lane.fit_centre_line(pts), pts, px, 0.05)
    assert len(kept) == 4 and (1.0, 0.3) not in kept and (3, 3) not in kept_px
    assert fit[0] == pytest.approx(0.0, abs=1e-9) and fit[1] == pytest.approx(0.0, abs=1e-9)
    # A far point with leverage pulls the plain fit onto itself; still dropped.
    lever = [(0.7, 0.0), (0.8, 0.0), (0.9, 0.0), (1.7, -0.4)]
    fit, kept, _ = front_lane.drop_outliers(front_lane.fit_centre_line(lever), lever, px[:4], 0.05)
    assert (1.7, -0.4) not in kept and fit[0] == pytest.approx(0.0, abs=1e-9)
    clean = [(0.7, 0.01), (0.8, 0.0), (0.9, -0.01)]
    assert front_lane.drop_outliers(front_lane.fit_centre_line(clean), clean, px[:3], 0.05)[1] == clean


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