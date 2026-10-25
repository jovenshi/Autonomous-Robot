"""Drawing and touch helpers for guided tool screens on the 7" display.

Used by tools/calib_ui.py; meant for other step-by-step tools too (the
steering response test, the IMU checks). Same look as the driving display
(ui/overlay.py colours, ui/text.py fonts, the 160 px control column on the
right), so the screens feel like one system.

Screens are drawn as BGR NumPy images at DISPLAY_SIZE. Anything tappable is
recorded as a Zone while drawing; the screen's tap handler looks them up.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from ui.overlay import AMBER, GREEN, LINE, MUTED, PANEL, RED, TEXT
from ui.panels import draw_mark
from ui.text import draw_text, text_width

COLUMN_W = 160                           # px; control column on the right
PRIMARY = (40, 95, 36)                   # BGR; the green "do this next" button
DISABLED = (33, 30, 27)
LEVEL_COLOURS = {"ok": GREEN, "warn": AMBER, "bad": RED, "info": TEXT}


@dataclass
class Button:
    name: str                            # what the screen's handler receives
    label: str
    sub: str = ""                        # small line under the label
    primary: bool = False                # green: what Enter presses
    enabled: bool = True


@dataclass
class Zone:
    rect: tuple                          # (x, y, w, h) in screen px
    name: str


def column_rects(display_size, count=4):
    # Equal button slots down the right-hand column.
    w, h = display_size
    slot = h // count
    return [(w - COLUMN_W, i * slot, COLUMN_W, slot) for i in range(count)]


def draw_column(img, buttons, display_size):
    # Draws up to four buttons down the column; returns their Zones.
    zones = []
    x0 = display_size[0] - COLUMN_W
    img[:, x0:] = (img[:, x0:] * 0.7).astype(img.dtype)
    for button, rect in zip(buttons, column_rects(display_size)):
        if button is None:
            continue
        draw_button(img, rect, button)
        if button.enabled:
            zones.append(Zone(rect, button.name))
    return zones


def draw_button(img, rect, button):
    x, y, w, h = rect
    fill = PRIMARY if button.primary and button.enabled else (PANEL if button.enabled else DISABLED)
    cv2.rectangle(img, (x + 6, y + 6), (x + w - 7, y + h - 7), fill, -1, cv2.LINE_AA)
    colour = TEXT if button.enabled else MUTED
    size = 26 if text_width(button.label, 26, True) < w - 24 else 20
    cy = y + h // 2 + (0 if button.sub else size // 3)
    draw_text(img, button.label, (x + w // 2, cy), size, colour, bold=True, align="centre")
    if button.sub:
        draw_text(img, button.sub, (x + w // 2, cy + 22), 14, MUTED, align="centre")


def hit(zones, pos):
    # The name of the zone under pos, or None (later zones win).
    px, py = pos
    for zone in reversed(zones):
        x, y, w, h = zone.rect
        if x <= px < x + w and y <= py < y + h:
            return zone.name
    return None


def wrap(text, size, width, bold=False):
    # Lines of text that fit width at this size.
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and text_width(trial, size, bold) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


def clip_text(text, size, width, bold=False):
    # text shortened with "..." until it fits width.
    if text_width(text, size, bold) <= width:
        return text
    while text and text_width(text + "...", size, bold) > width:
        text = text[:-1]
    return text.rstrip() + "..."


def draw_paragraph(img, text, org, width, size=18, colour=TEXT, bold=False, gap=1.35):
    # Wrapped text from org (x, first baseline); returns the next baseline y.
    x, y = org
    for line in wrap(text, size, width, bold):
        draw_text(img, line, (x, y), size, colour, bold=bold)
        y += int(size * gap)
    return y


def draw_steps(img, steps, org, width, size=18):
    # Numbered instructions, each wrapped under its number; returns next y.
    x, y = org
    for i, step in enumerate(steps, start=1):
        draw_text(img, f"{i}.", (x, y), size, MUTED, bold=True)
        y = draw_paragraph(img, step, (x + 28, y), width - 28, size) + size // 2
    return y


def draw_level_line(img, level, text, org, width, size=16):
    # One report line with its mark (tick, cross or "!"); returns next y.
    x, y = org
    if level in ("ok", "bad"):
        draw_mark(img, (x + 8, y - size // 3), level == "ok", size // 3 + 1)
    elif level == "warn":
        draw_text(img, "!", (x + 8, y), size + 2, AMBER, bold=True, align="centre")
    return draw_paragraph(img, text, (x + 24, y), width - 24, size, LEVEL_COLOURS[level], gap=1.25)


def draw_banner(img, text, colour=TEXT, top=True, size=18):
    # A darkened strip across the top (or bottom) of img with one line of text.
    h, w = img.shape[:2]
    strip = size + 18
    y0, y1 = (0, strip) if top else (h - strip, h)
    img[y0:y1] = (img[y0:y1] * 0.25).astype(img.dtype)
    draw_text(img, text, (12, y1 - 12), size, colour, bold=top)


def draw_adjuster(img, rect, label, value_text, name, steps):
    # "label  [-big] [-small]  value  [+small] [+big]"; returns Zones named
    # f"{name}:{delta}" for each step button. steps: ((label, delta), ...).
    x, y, w, h = rect
    draw_text(img, label, (x, y + 18), 16, MUTED)
    by, bh = y + 26, h - 26
    bw = 64
    zones = []
    left = [s for s in steps if s[1] < 0]
    right = [s for s in steps if s[1] > 0]
    bx = x
    for text, delta in left:
        zones.append(_small_button(img, (bx, by, bw, bh), text, f"{name}:{delta}"))
        bx += bw + 6
    value_w = w - 2 * len(right) * (bw + 6) - (bx - x)
    cv2.rectangle(img, (bx, by), (bx + value_w, by + bh), LINE, 1, cv2.LINE_AA)
    draw_text(img, value_text, (bx + value_w // 2, by + bh // 2 + 9), 26, TEXT, bold=True,
              align="centre")
    bx += value_w + 6
    for text, delta in right:
        zones.append(_small_button(img, (bx, by, bw, bh), text, f"{name}:{delta}"))
        bx += bw + 6
    return zones


def _small_button(img, rect, text, name):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1, cv2.LINE_AA)
    draw_text(img, text, (x + w // 2, y + h // 2 + 7), 18, TEXT, bold=True, align="centre")
    return Zone(rect, name)


def fit_image(image, size):
    # image scaled to fit size (w, h), letterboxed; returns (canvas, scale, (ox, oy)).
    w, h = size
    scale = min(w / image.shape[1], h / image.shape[0])
    nw, nh = int(image.shape[1] * scale), int(image.shape[0] * scale)
    canvas = np.zeros((h, w, 3), np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, scale, (ox, oy)


def say(prefix, text, width=78):
    # A console note: "[prefix] text", wrapped, continuation lines indented.
    words, lines, line = text.split(), [], ""
    lead = f"[{prefix}] "
    for word in words:
        trial = f"{line} {word}".strip()
        if line and len(lead) + len(trial) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    print(lead + ("\n" + " " * len(lead)).join(lines), flush=True)
