"""Tools menu on the robot's 7" touch display: one place to start everything.

    python -m tools.menu            (from the autonomous/ folder; over SSH it
                                     opens on the robot's screen)

Tiles start the driving program and each tool, so nobody has to remember
the commands. The header shows what needs doing (camera check due, steering
not measured) and the matching tile is highlighted green.

Tools that use the screen (driving, calibration, steering test) take it over
and the menu comes back when they finish. Terminal tools (IMU check,
recording) keep the menu on screen with a STOP button; their output is in
the terminal. Changes a tool APPLYs to config.py show up when the menu
returns.
"""
import argparse
import glob
import importlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from tools.calib_status import check_status  # noqa: E402
from tools.guide_draw import (Button, Zone, clip_text, draw_column,  # noqa: E402
                              draw_level_line, hit, say)
from ui.overlay import AMBER, BG, GREEN, LINE, MUTED, PANEL, TEXT  # noqa: E402
from ui.text import draw_text  # noqa: E402

TAG = "menu"
MAIN_W = 640
AUTONOMOUS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Tool:
    key: str
    label: str
    sub: str
    args: list                           # after `python`, run from autonomous/
    screen: bool                         # True: takes over the display


def tools_list(cfg):
    # The tiles, in order. Tune colours needs a recording to open.
    recordings = sorted(glob.glob(os.path.join(AUTONOMOUS_DIR, "recordings", "front_*")))
    items = [
        Tool("drive", "DRIVE", "main.py", ["main.py"], cfg.UI_ENABLED),
        Tool("check", "CAMERA CHECK", "2 min, before driving",
             ["-m", "tools.calib_ui", "--check"], True),
        Tool("calibrate", "CALIBRATE", "board session, 15 min", ["-m", "tools.calib_ui"], True),
        Tool("steer", "STEERING TEST", "wheels move, 10 min", ["-m", "tools.steer_test"], True),
        Tool("imu", "IMU CHECK", "sign check, terminal", ["-m", "tools.imu_drift_test"], False),
        Tool("record", "RECORD", "20 s of camera, terminal",
             ["-m", "tools.record", "--seconds", "20"], False),
        Tool("lens", "LENS STEP", "optional", ["-m", "tools.calib_ui", "--lens"], True),
    ]
    if recordings:
        items.append(Tool("colours", "TUNE COLOURS", os.path.basename(recordings[-1]),
                          ["-m", "tools.tune_colors", recordings[-1]], True))
    else:
        items.append(Tool("demo", "UI DEMO", "display only, 30 s",
                          ["-m", "tools.ui_demo", "--seconds", "30"], True))
    return items


def status_lines(cfg):
    # (level, text, key of the tool that fixes it) for the header.
    ok, text = check_status(cfg)
    lines = [("ok" if ok else "warn", text, None if ok else
              ("calibrate" if cfg.FRONT_GROUND_H is None else "check"))]
    if cfg.STEER_YAW_DPS_PER_UNIT is None:
        lines.append(("warn", "steering response: not measured yet", "steer"))
    else:
        lines.append(("ok", "steering response: measured", None))
    return lines


class MenuApp:
    """Tiles and the running-tool screen. The display loop runs screen tools
    (it must give up the display); terminal tools run here as children."""

    def __init__(self, cfg, popen=subprocess.Popen):
        self.cfg, self.popen = cfg, popen
        self.size = tuple(cfg.DISPLAY_SIZE)
        self.zones, self.done, self.page = [], False, "tiles"
        self.child, self.child_tool, self.pending, self.flash = None, None, None, ""
        self.refresh()

    def refresh(self):
        # Re-reads config.py (a tool may have APPLYed new values) and the status.
        if hasattr(self.cfg, "__spec__"):
            self.cfg = importlib.reload(self.cfg)
        self.tools = tools_list(self.cfg)
        self.status = status_lines(self.cfg)

    def draw(self):
        img = np.empty((self.size[1], self.size[0], 3), np.uint8)
        img[:] = BG
        self.zones = []
        main = img[:, :MAIN_W]
        if self.page == "tiles":
            self._draw_tiles(main)
            buttons = [None, None, None, Button("quit", "QUIT")]
        else:
            self._draw_running(main)
            buttons = [Button("stop", "STOP", "Ctrl-C to it", primary=True), None, None, None]
        self.zones += draw_column(img, buttons, self.size)
        return img

    def _draw_tiles(self, img):
        draw_text(img, "Robot tools", (16, 34), 26, TEXT, bold=True)
        y = 60
        for level, text, _ in self.status:
            y = draw_level_line(img, level, text, (16, y), MAIN_W - 32, 16) + 2
        if self.flash:
            draw_text(img, clip_text(self.flash, 14, MAIN_W - 32), (16, y + 2), 14, MUTED)
        wanted = {key for _, _, key in self.status if key}
        top, cols, gap = 128, 2, 8
        tile_w = (MAIN_W - 16 * 2 - gap) // cols
        tile_h = (img.shape[0] - top - 12 - gap * 3) // 4
        for i, tool in enumerate(self.tools):
            x = 16 + (i % cols) * (tile_w + gap)
            ty = top + (i // cols) * (tile_h + gap)
            _tile(img, (x, ty, tile_w, tile_h), tool, tool.key in wanted)
            self.zones.append(Zone((x, ty, tile_w, tile_h), f"tool:{tool.key}"))

    def _draw_running(self, img):
        tool = self.child_tool
        draw_text(img, f"Running: {tool.label}", (20, 60), 28, AMBER, bold=True)
        draw_text(img, "Output is in the terminal.", (20, 100), 18, TEXT)
        draw_text(img, "STOP sends it Ctrl-C (motors stop first in every tool).", (20, 130), 16,
                  MUTED)
        draw_text(img, "python " + " ".join(tool.args), (20, 170), 14, MUTED)

    def tap(self, pos):
        name = hit(self.zones, pos)
        if name is not None:
            self.press(name)

    def press(self, name):
        if name == "quit":
            self.done = True
        elif name == "stop" and self.child is not None:
            self.child.send_signal(signal.SIGINT)
            say(TAG, f"Sent Ctrl-C to {self.child_tool.label}.")
        elif name.startswith("tool:") and self.page == "tiles":
            tool = next(t for t in self.tools if t.key == name[5:])
            say(TAG, f"Starting {tool.label}: python {' '.join(tool.args)}")
            if tool.screen:
                self.pending = tool              # the display loop runs it
            else:
                self.child, self.child_tool = self._start(tool), tool
                self.page = "running"

    def key(self, name):
        if name == "q" and self.page == "tiles":
            self.done = True
        elif name in ("escape", "space") and self.page == "running":
            self.press("stop")

    def tick(self):
        if self.child is not None and self.child.poll() is not None:
            self.finished(self.child_tool, self.child.returncode)
            self.child = None
            self.page = "tiles"

    def finished(self, tool, code):
        self.refresh()
        self.flash = f"{tool.label} finished" + ("" if code in (0, -signal.SIGINT, 130)
                                                 else f" (exit code {code})")
        say(TAG, self.flash)

    def _start(self, tool):
        return self.popen([sys.executable] + tool.args, cwd=AUTONOMOUS_DIR)

    def run_screen_tool(self, tool):
        # Blocking: the display loop has released the screen.
        code = subprocess.call([sys.executable] + tool.args, cwd=AUTONOMOUS_DIR)
        self.finished(tool, code)


def _tile(img, rect, tool, wanted):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1, cv2.LINE_AA)
    if wanted:
        cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 3, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), LINE, 1, cv2.LINE_AA)
    draw_text(img, tool.label, (x + 14, y + h // 2 + 2), 22, GREEN if wanted else TEXT, bold=True)
    draw_text(img, clip_text(tool.sub, 14, w - 28), (x + 14, y + h // 2 + 24), 14, MUTED)


# ---------------------------------------------------------------- display loop

KEYS = {"K_ESCAPE": "escape", "K_q": "q", "K_SPACE": "space"}


def run(app, cfg):
    # Menu loop; gives the display up while a screen tool runs. Returns
    # False if no display opened.
    from ui.screen import _set_display_env, open_display, show
    _set_display_env()
    while not app.done:
        display = open_display(cfg)
        if display is None:
            return False
        pygame, surface = display
        keys = {getattr(pygame, k): v for k, v in KEYS.items()}
        try:
            while not app.done and app.pending is None:
                show(pygame, surface, app.draw())
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        app.done = True
                    elif event.type == pygame.KEYDOWN and event.key in keys:
                        app.key(keys[event.key])
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        app.tap(event.pos)
                app.tick()
                time.sleep(0.1)
        finally:
            pygame.quit()
        if app.pending is not None:
            tool, app.pending = app.pending, None
            app.run_screen_tool(tool)
    return True


def main(argv=None):
    argparse.ArgumentParser(description="Tools menu on the robot's display.").parse_args(argv)
    app = MenuApp(config)
    for level, text, _ in app.status:
        say(TAG, text)
    try:
        if not run(app, config):
            say(TAG, "No display: run this on the Pi with its screen attached.")
    except KeyboardInterrupt:
        pass
    finally:
        if app.child is not None and app.child.poll() is None:
            app.child.send_signal(signal.SIGINT)
            app.child.wait(5)


if __name__ == "__main__":
    main()
