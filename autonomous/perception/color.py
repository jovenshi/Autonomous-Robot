"""The only place colour tests are applied.

All masks are uint8 images, 255 where the test passes and 0 elsewhere.
OpenCV HSV ranges: H 0-179, S and V 0-255.
A region (roi) is (x, y, w, h) in pixels of the image it applies to.

Tests rely on hue, saturation and colour ratios rather than raw brightness,
so they hold up as the room light changes.
"""
import cv2
import numpy as np

from robot_types import FrontMasks


def mask_yellow(hsv, cfg):
    # Hue range plus minimum saturation; the brightness bound is kept loose.
    lower = (cfg.YELLOW_H_MIN, cfg.YELLOW_S_MIN, cfg.YELLOW_V_MIN)
    upper = (cfg.YELLOW_H_MAX, 255, 255)
    return cv2.inRange(hsv, lower, upper)


def mask_green(img, cfg):
    # Green chromaticity G / (R+G+B) above GREEN_CHROM_MIN, ignoring pixels
    # too dark to judge. Written without division: G > k * (R+G+B).
    total = img.sum(axis=2, dtype=np.int32)
    green = img[:, :, 1].astype(np.int32)
    passed = (green > cfg.GREEN_CHROM_MIN * total) & (total >= cfg.DARK_SUM_MIN)
    return passed.astype(np.uint8) * 255


def mask_road(hsv, roi, cfg):
    # Dark and unsaturated. The darkness threshold comes from Otsu on the
    # region's V channel each frame, clamped to ROAD_V_T_MIN-ROAD_V_T_MAX.
    x, y, w, h = roi
    v_roi = np.ascontiguousarray(hsv[y:y + h, x:x + w, 2])
    otsu, _ = cv2.threshold(v_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_max = int(min(max(otsu, cfg.ROAD_V_T_MIN), cfg.ROAD_V_T_MAX))
    return cv2.inRange(hsv, (0, 0, 0), (179, cfg.ROAD_S_MAX, v_max))


def mask_white(hsv, cfg):
    # Low saturation and brightness above WHITE_V_REL x the frame's
    # WHITE_V_PCT percentile (relative, not absolute).
    v_ref = np.percentile(hsv[:, :, 2], cfg.WHITE_V_PCT)
    v_min = int(cfg.WHITE_V_REL * v_ref)
    return cv2.inRange(hsv, (0, 0, v_min), (179, cfg.WHITE_S_MAX, 255))


def mask_red(hsv, cfg):
    # Two hue ranges, because red wraps around 0/179, plus minimum saturation.
    low = cv2.inRange(hsv, (0, cfg.RED_S_MIN, 0), (cfg.RED_H_LOW_MAX, 255, 255))
    high = cv2.inRange(hsv, (cfg.RED_H_HIGH_MIN, cfg.RED_S_MIN, 0), (179, 255, 255))
    return cv2.bitwise_or(low, high)


def roi_brightness(hsv, roi):
    # Mean V in the region, for exposure management and health checks.
    x, y, w, h = roi
    return float(hsv[y:y + h, x:x + w, 2].mean())


def front_lane_roi(cfg, width, height):
    # The rows covered by FRONT_BANDS, full width; the whole image until
    # the bands are calibrated.
    if cfg.FRONT_BANDS is None:
        return (0, 0, width, height)
    top = min(band[0] for band in cfg.FRONT_BANDS)
    bottom = max(band[1] for band in cfg.FRONT_BANDS)
    return (0, top, width, bottom - top)


def make_front_masks(frame, cfg, with_signs=False):
    # Downscale to FRONT_LANE_SIZE, convert once, build the lane masks.
    # The white and red masks are built only when with_signs is set.
    small = cv2.resize(frame.image, tuple(cfg.FRONT_LANE_SIZE), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    height, width = small.shape[:2]
    roi = front_lane_roi(cfg, width, height)
    return FrontMasks(
        bgr=small,
        hsv=hsv,
        yellow=mask_yellow(hsv, cfg),
        road=mask_road(hsv, roi, cfg),
        white=mask_white(hsv, cfg) if with_signs else None,
        red=mask_red(hsv, cfg) if with_signs else None,
        scale=frame.image.shape[1] / width,
        mean_v=roi_brightness(hsv, roi),
    )


def make_side_masks(frame, cfg):
    # Same for the side camera: returns (road, green).
    hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
    height, width = frame.image.shape[:2]
    road = mask_road(hsv, (0, 0, width, height), cfg)
    return road, mask_green(frame.image, cfg)