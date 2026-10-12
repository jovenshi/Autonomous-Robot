"""Anti-aliased text on NumPy (BGR) images, in the UI_FONT / UI_FONT_BOLD fonts.

OpenCV's built-in fonts are hard to read at a glance; this renders each
string once with Pillow into an alpha mask (cached) and blends it in place.
org is (x, baseline y), like cv2.putText.
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config


@lru_cache(maxsize=32)
def _font(size, bold):
    return ImageFont.truetype(config.UI_FONT_BOLD if bold else config.UI_FONT, size)


@lru_cache(maxsize=2048)
def _mask(text, size, bold):
    # (alpha 0-1 float array, x offset, y offset from the baseline origin)
    font = _font(size, bold)
    left, top, right, bottom = font.getbbox(text, anchor="ls")
    img = Image.new("L", (max(1, right - left), max(1, bottom - top)))
    ImageDraw.Draw(img).text((-left, -top), text, font=font, fill=255, anchor="ls")
    return np.asarray(img, dtype=np.float32)[..., None] / 255.0, left, top


def text_width(text, size, bold=False):
    left, _, right, _ = _font(size, bold).getbbox(text, anchor="ls")
    return right - left


def draw_text(img, text, org, size, colour, bold=False, align="left"):
    # Draws text at org (x, baseline); align 'left', 'centre' or 'right' about x.
    if not text:
        return
    alpha, left, top = _mask(text, size, bold)
    h, w = alpha.shape[:2]
    x = org[0] + left - {"left": 0, "centre": w // 2, "right": w}[align]
    y = org[1] + top
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    region = img[y0:y1, x0:x1]
    region[:] = (region * (1.0 - a) + np.array(colour, np.float32) * a).astype(np.uint8)
