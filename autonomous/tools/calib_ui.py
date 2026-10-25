"""Guided front camera calibration on the robot's 7" touch display.

Step-by-step screens with the live camera view; the terminal prints a short
note at every step. The maths is in tools/calibrate.py; this file guides.

Stop main.py first (it owns the camera), then from the autonomous/ folder:

    python -m tools.calib_ui             # full board session, about 15 minutes
    python -m tools.calib_ui --check     # 30-second pre-session check
    python -m tools.calib_ui --lens      # optional lens step (if the fit suggests it)
    python -m tools.calib_ui --resume calibration/front_20261025_140000

What you need: the robot on the course mat with the camera in its driving
mount, a tape measure, tape, and the printed chessboard (9 x 6 inner corners).

Board session, screen by screen:
    WELCOME   setup checklist; set the measured square size and road width
    PLACE     live view; put the board at the spot shown; CAPTURE when found
    CONFIRM   enter where the board really is (tape reading); KEEP or RETAKE
    RESULTS   per-board error with hints, the check photo verdict, new bands;
              APPLY writes config.py (config.py.bak keeps the old one)
Photos and positions are saved as you go (calibration/front_<time>/), so a
quit or crash loses nothing: --resume carries on, or refits a finished session.

Controls: tap the buttons, or Enter = the green button, Esc = back/retake,
q = quit. On CONFIRM, the arrow keys nudge the values (Shift = 1 cm).
"""
import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from robot_types import LatestValue  # noqa: E402
from tools import calibrate, config_writer  # noqa: E402
from tools.guide_draw import (Button, Zone, clip_text, draw_adjuster,  # noqa: E402
                              draw_banner, draw_column, draw_level_line, draw_paragraph,
                              draw_steps, fit_image, hit, say)
from ui.panels import draw_mark  # noqa: E402
from ui.overlay import AMBER, BG, GREEN, LINE, MUTED, PURPLE, RED, TEXT  # noqa: E402
from ui.text import draw_text  # noqa: E402

TAG = "calib"
MAIN_W = 640                             # px; content area left of the button column
NEAR_STEPS = (("-1cm", -0.01), ("-1mm", -0.001), ("+1mm", 0.001), ("+1cm", 0.01))
SQUARE_STEPS = (("-1", -0.001), ("-.1", -0.0001), ("+.1", 0.0001), ("+1", 0.001))
ROAD_STEPS = (("-1", -0.01), ("-.5", -0.005), ("+.5", 0.005), ("+1", 0.01))
BACK_NAMES = ("back", "retake", "again")
FLASH_S = 3.0                            # s; a short status note stays this long

# Printed once when each screen opens: the small things worth knowing there.
TIPS = {
    "welcome": "Board session. Keep the robot and camera still from now until APPLY: if either "
               "moves, the photos no longer agree. Measure one printed square with a ruler "
               "(printers rescale) and set it on the screen.",
    "place": "Put the board's NEAREST row of inner corners (where four squares meet) on the "
             "tape reading shown, square to the tape. Flat, fully in view, no glare. "
             "If it can't be fully seen there, slide it until it is and enter where it really is.",
    "confirm": "Read the tape at the nearest row of inner corners, and how far the board's "
               "centre sits left or right of the tape. A few mm matter: these numbers ARE the "
               "calibration.",
    "check_photo": "This is the CHECK photo: it is left out of the fit and used to verify it. "
                   "Before moving the board, put two pieces of tape at its near corners: "
                   "that is your spot for the 30-second check before each session.",
    "results": "Each board is scored against a fit made without it, so a mis-measured spot "
               "stands out with the direction it seems off. Tap a row, then RETAKE to redo it.",
    "applied": "Saved. Before each session: python -m tools.calib_ui --check (board on the "
               "taped marks). If it fails, the camera has moved: run the full session again.",
    "check": "Pre-session check: put the board exactly on the taped marks. The purple dots "
             "show where its corners should appear if the camera has not moved.",
    "lens_intro": "Lens step (optional). Hold the board 40-80 cm from the camera (the focus "
                  "is fixed for driving), tilted 20-45 degrees in different directions, in the "
                  "centre, edges and corners of the view. Every photo must show the whole "
                  "board, sharp.",
    "lens_result": "Lens values only help if they move band points by a millimetre or more. "
                   "After APPLY, refit the board session so the bands use them.",
}


@dataclass
class BoardView:
    image: np.ndarray                    # full frame, BGR
    corners: object                      # (N, 2) frame px from the fast detector, or None
    t: float


class LiveBoard:
    """The front camera in its own thread, looking for the board in every frame."""

    def __init__(self, cfg, source=None):
        from hal.camera import CameraStream, CsiSource
        self.cfg = cfg
        self.slot = LatestValue()
        source = source or CsiSource(cfg.FRONT_CAM_NUM, cfg.FRONT_SIZE, cfg)
        self.stream = CameraStream(source, self._detect, self.slot, "front", cfg)

    def _detect(self, frame):
        corners = calibrate.find_board_corners(frame.image, self.cfg.CALIB_PATTERN, fast=True)
        return BoardView(frame.image, corners, frame.t)

    def start(self):
        self.stream.start()              # auto-exposure stays on: fine for a paper board

    def stop(self):
        self.stream.stop()

    def latest(self):
        return self.slot.get()


class CalibApp:
    """All screens and their logic. draw() -> image, tap()/key() -> actions,
    tick() runs slow work (the fit) after its screen has been shown once."""

    def __init__(self, cfg, camera, mode="session", folder=None):
        self.cfg, self.camera, self.mode = cfg, camera, mode
        self.size = tuple(cfg.DISPLAY_SIZE)
        self.folder, self.rows = folder, []
        self.square_m, self.road_m = cfg.CALIB_SQUARE_M, cfg.ROAD_WIDTH_M
        self.spot, self.capture, self.entry = 0, None, None
        self.result, self.selected, self.message = None, None, ""
        self.retaking = False
        self.lens_sets, self.lens_folder, self.lens_result = [], None, None
        self.check_result, self.last_corners, self.applied_backup = None, None, None
        self.zones, self.work, self.page_frames, self.done = [], None, 0, False
        self.flash, self.flash_t, self.tips_shown = "", 0.0, set()
        self.page = None
        self._open_first_page(folder)

    # ---------------------------------------------------------------- framework

    def _open_first_page(self, folder):
        if self.mode == "check":
            if self.cfg.FRONT_GROUND_H is None or self.cfg.CALIB_CHECK_NEAR_M is None:
                self.go("message", "No calibration to check yet: run the full board session "
                                   "first (python -m tools.calib_ui).")
            else:
                self.go("check")
        elif self.mode == "lens":
            self.go("lens_intro")
        elif folder is not None:
            self._resume(folder)
        else:
            self.go("welcome")

    def go(self, page, message=""):
        self.page, self.page_frames, self.message, self.flash = page, 0, message, ""
        tip_name = "check_photo" if page == "confirm" and self._spot_role() == "check" else page
        tip = TIPS.get(tip_name) if tip_name not in self.tips_shown else None
        self.tips_shown.add(tip_name)
        if page == "place":
            near, lat, role = self.cfg.CALIB_POSITIONS[self.spot]
            say(TAG, f"Spot {self.spot + 1} of {len(self.cfg.CALIB_POSITIONS)} ({role}): nearest "
                     f"corner row at {near * 100:.1f} cm on the tape, centre {_side(lat)}.")
        if tip:
            say(TAG, tip)
        if message:
            say(TAG, message)

    def draw(self):
        img = np.empty((self.size[1], self.size[0], 3), np.uint8)
        img[:] = BG
        self.zones = []
        getattr(self, f"_draw_{self.page}")(img[:, :MAIN_W])
        self.zones += draw_column(img, self.buttons(), self.size)
        self.page_frames += 1
        return img

    def buttons(self):
        return getattr(self, f"_buttons_{self.page}")()

    def press(self, name):
        # A button or zone by name; ignored if not on screen or disabled.
        for button in self.buttons():
            if button is not None and button.name == name:
                if button.enabled:
                    getattr(self, f"_on_{self.page}")(name)
                return
        if ":" in name:                  # zones drawn in the content area
            getattr(self, f"_on_{self.page}")(name)

    def tap(self, pos):
        name = hit(self.zones, pos)
        if name is not None:
            self.press(name)

    def key(self, name, shift=False):
        if name == "q":
            self.quit()
        elif name == "enter":
            primary = [b for b in self.buttons() if b is not None and b.primary and b.enabled]
            if primary:
                self.press(primary[0].name)
        elif name == "escape":
            names = [b.name for b in self.buttons() if b is not None]
            for back in BACK_NAMES:
                if back in names:
                    self.press(back)
                    return
        elif self.page == "confirm" and name in ("up", "down", "left", "right"):
            step = 0.01 if shift else 0.001
            field = "near" if name in ("up", "down") else "lat"
            self.press(f"{field}:{step if name in ('up', 'right') else -step}")

    def tick(self):
        if self.work is not None and self.page_frames > 0:
            work, self.work = self.work, None
            work()

    def quit(self):
        if self.folder and self.rows and self.page != "applied":
            say(TAG, f"Photos kept in {self.folder}; carry on with: "
                     f"python -m tools.calib_ui --resume {self.folder}")
        self.done = True

    # ---------------------------------------------------------------- WELCOME

    def _draw_welcome(self, img):
        draw_text(img, "Front camera calibration", (20, 44), 30, TEXT, bold=True)
        y = draw_steps(img, [
            "Robot on the course mat, camera in its driving mount. Keep both still until APPLY.",
            "Mark the rear-axle midpoint on the mat. Tape a tape measure forward from it along "
            "the robot's centreline.",
            f"You will place the board at {len(self.cfg.CALIB_POSITIONS)} spots. Each time: its "
            "nearest row of inner corners on the tape reading shown, square to the tape.",
            "Measure one printed square with a ruler and set it below, and the road width.",
        ], (20, 84), MAIN_W - 40, 17)
        y = max(y, 300)
        self.zones += draw_adjuster(img, (20, y, MAIN_W - 40, 76), "Printed square size",
                                    f"{self.square_m * 1000:.1f} mm", "square", SQUARE_STEPS)
        self.zones += draw_adjuster(img, (20, y + 86, MAIN_W - 40, 76), "Road width, edge to edge",
                                    f"{self.road_m * 100:.1f} cm", "road", ROAD_STEPS)

    def _buttons_welcome(self):
        return [Button("start", "START", "board session", primary=True),
                Button("lens", "LENS", "optional step"), None, Button("quit", "QUIT")]

    def _on_welcome(self, name):
        if name.startswith("square:"):
            self.square_m = max(0.005, round(self.square_m + float(name[7:]), 4))
        elif name.startswith("road:"):
            self.road_m = max(0.05, round(self.road_m + float(name[5:]), 3))
        elif name == "start":
            self.folder = calibrate.new_session_dir(self.cfg)
            calibrate.write_session_meta(self.folder, self.cfg, self.square_m, self.road_m)
            say(TAG, f"Session folder: {self.folder}")
            self.spot = 0
            self.go("place")
        elif name == "lens":
            self.mode = "lens"
            self.go("lens_intro")
        elif name == "quit":
            self.quit()

    # ---------------------------------------------------------------- PLACE

    def _draw_place(self, img):
        near, lat, role = self.cfg.CALIB_POSITIONS[self.spot]
        view = self.camera.latest()
        found = self._draw_live(img, view)
        expected = calibrate.expected_corners(self.cfg, near, lat, view.image.shape) \
            if view is not None else None
        if expected is not None:
            _draw_outline(img, expected, self.cfg.CALIB_PATTERN, img.shape[1] / view.image.shape[1],
                          PURPLE)
        kept = sum(r["role"] == "fit" for r in self.rows)
        draw_banner(img, f"Spot {self.spot + 1}/{len(self.cfg.CALIB_POSITIONS)}  {role.upper()}:  "
                         f"near row {near * 100:.1f} cm,  centre {_side(lat)}", size=17)
        draw_banner(img, self._live_status(view, found, f"{kept} fit photos kept"),
                    GREEN if found else AMBER, top=False, size=16)

    def _buttons_place(self):
        found = self._board_found()
        enough = sum(r["role"] == "fit" for r in self.rows) >= self.cfg.CALIB_MIN_FIT_BOARDS
        return [Button("capture", "CAPTURE", "board found" if found else "no board",
                       primary=True, enabled=found),
                Button("skip", "SKIP", "this spot"),
                Button("finish", "FINISH", "fit now" if enough else
                       f"needs {self.cfg.CALIB_MIN_FIT_BOARDS} fit", enabled=enough),
                Button("quit", "QUIT", "photos are kept")]

    def _on_place(self, name):
        if name == "capture":
            self._take_capture()
        elif name == "skip":
            say(TAG, f"Skipped spot {self.spot + 1}.")
            self._next_spot(skip=True)
        elif name == "finish":
            self.go("fitting")
            self.work = self._run_fit
        elif name == "quit":
            self.quit()

    def _take_capture(self):
        # Full (not fast) detection on the newest frame; then CONFIRM.
        view = self.camera.latest()
        corners = calibrate.find_board_corners(view.image, self.cfg.CALIB_PATTERN) \
            if view is not None else None
        if corners is None:
            self._flash("Lost the board: hold still and try again")
            return
        near, lat, _ = self.cfg.CALIB_POSITIONS[self.spot]
        existing = self._row_for_spot(self.spot)
        if existing is not None:
            near, lat = existing["near_m"], existing["lateral_m"]
        self.capture = BoardView(view.image.copy(), corners, view.t)
        self.entry = [near, lat]
        self.go("confirm")

    # ---------------------------------------------------------------- CONFIRM

    def _draw_confirm(self, img):
        canvas, scale, (ox, oy) = fit_image(self.capture.image, (400, 300))
        _draw_corners(canvas, self.capture.corners * scale + (ox, oy), GREEN)
        img[12:312, 12:412] = canvas
        role = self._spot_role()
        text = ("CHECK photo: left out of the fit, used to verify it. Tape its two near "
                "corners on the mat before moving it." if role == "check" else
                "Read the tape at the nearest row of inner corners, and how far the board's "
                "centre is left or right of the tape.")
        draw_text(img, "Where is it?", (424, 40), 22, TEXT, bold=True)
        draw_paragraph(img, text, (424, 70), MAIN_W - 436, 16, AMBER if role == "check" else TEXT)
        near, lat = self.entry
        self.zones += draw_adjuster(img, (12, 318, MAIN_W - 24, 76),
                                    "Nearest corner row (tape reading)",
                                    f"{near * 100:.1f} cm", "near", NEAR_STEPS)
        self.zones += draw_adjuster(img, (12, 400, MAIN_W - 24, 76),
                                    "Board centre, right (+) of the tape",
                                    f"{lat * 100:+.1f} cm", "lat", NEAR_STEPS)

    def _buttons_confirm(self):
        return [Button("keep", "KEEP", "save photo", primary=True),
                Button("retake", "RETAKE"), None, Button("quit", "QUIT", "photos are kept")]

    def _on_confirm(self, name):
        if name.startswith(("near:", "lat:")):
            field, delta = name.split(":")
            i = 0 if field == "near" else 1
            self.entry[i] = round(self.entry[i] + float(delta), 4)
        elif name == "keep":
            self._keep_capture()
        elif name == "retake":
            self.go("place")
        elif name == "quit":
            self.quit()

    def _keep_capture(self):
        image_name = _spot_image(self.spot)
        cv2.imwrite(os.path.join(self.folder, image_name), self.capture.image)
        role = self._spot_role()
        self.rows = [r for r in self.rows if r["image"] != image_name]
        self.rows.append({"image": image_name, "near_m": self.entry[0],
                          "lateral_m": self.entry[1], "role": role})
        self.rows.sort(key=lambda r: r["image"])
        calibrate.write_positions(self.folder, self.rows)
        say(TAG, f"Kept {image_name}: {self.entry[0] * 100:.1f} cm, "
                 f"{_side(self.entry[1])} ({role}).")
        self._next_spot()

    def _next_spot(self, skip=False):
        # The next spot without a photo; the fit once every spot has one (or
        # straight back to the fit after a RETAKE from the results).
        if self.retaking and not skip:
            self.retaking = False
            self.go("fitting")
            self.work = self._run_fit
            return
        n = len(self.cfg.CALIB_POSITIONS)
        todo = [i for i in list(range(self.spot + 1, n)) + list(range(0, self.spot + 1))
                if self._row_for_spot(i) is None and not (skip and i == self.spot)]
        if todo:
            self.spot = todo[0]
            self.go("place")
        elif sum(r["role"] == "fit" for r in self.rows) >= self.cfg.CALIB_MIN_FIT_BOARDS:
            self.go("fitting")
            self.work = self._run_fit
        else:
            self.spot = (self.spot + 1) % n
            self.go("place")
            self._flash("Not enough fit photos yet: every spot left is one you skipped")

    # ---------------------------------------------------------------- FITTING and RESULTS

    def _draw_fitting(self, img):
        draw_text(img, "Fitting...", (MAIN_W // 2, 220), 34, TEXT, bold=True, align="centre")
        draw_text(img, f"{len(self.rows)} photos", (MAIN_W // 2, 260), 18, MUTED, align="centre")

    def _buttons_fitting(self):
        return [None, None, None, None]

    def _on_fitting(self, name):
        pass

    def _run_fit(self):
        try:
            self.result = calibrate.fit_session(self.folder, self.cfg)
        except (ValueError, FileNotFoundError) as e:
            self.go("message", f"The fit stopped: {e}")
            return
        lines = calibrate.report_lines(self.result, self.cfg)
        for level, text in lines + [("info", t) for t in calibrate.band_lines(self.result)]:
            say(TAG, ("" if level == "info" else level.upper() + ": ") + text)
        flagged = [b["name"] for b in self.result["boards"]
                   if calibrate.board_note(b, self.cfg)[0] != "ok"]
        self.selected = flagged[0] if flagged else None
        self.go("results")

    def _draw_results(self, img):
        r = self.result
        draw_text(img, f"Fit: {r['rms_m'] * 1000:.1f} mm RMS", (16, 34), 24, TEXT, bold=True)
        draw_text(img, "tap a row, then RETAKE to redo it", (MAIN_W - 16, 34), 14, MUTED,
                  align="right")
        y = 64
        for b in r["boards"]:
            level, note = calibrate.board_note(b, self.cfg)
            if b["name"] == self.selected:
                cv2.rectangle(img, (8, y - 17), (MAIN_W - 8, y + 6), LINE, -1)
            err = f"{b['rms_m'] * 1000:.1f} mm" if b["found"] else "--"
            _draw_level_mark(img, level, (20, y - 5))
            draw_text(img, f"{b['name'][6:8]}  {b['role']}", (36, y), 15, TEXT)
            draw_text(img, f"{b['near_m'] * 100:.1f} cm  {b['lateral_m'] * 100:+.1f} cm",
                      (130, y), 15, TEXT)
            draw_text(img, err, (300, y), 15, TEXT, bold=level != "ok")
            draw_text(img, clip_text(note.split(":")[0], 14, MAIN_W - 380), (372, y), 14,
                      MUTED if level == "ok" else AMBER if level == "warn" else RED)
            self.zones.append(Zone((8, y - 17, MAIN_W - 16, 23), f"row:{b['name']}"))
            y += 23
        y += 6
        for level, text in calibrate.report_lines(r, self.cfg)[len(r["boards"]) + 1:]:
            y = draw_level_line(img, level, text, (12, y), MAIN_W - 24, 15) + 2
        for text in calibrate.band_lines(r):
            if y > img.shape[0] - 8:
                break
            draw_text(img, text, (36, y), 14, MUTED)
            y += 18

    def _buttons_results(self):
        passed = self.result["check"] is None or self.result["check"]["passed"]
        return [Button("apply", "APPLY", "write config.py" if passed else "check failed!",
                       primary=passed),
                Button("retake", "RETAKE", self.selected[6:8] if self.selected else "tap a row",
                       enabled=self.selected is not None),
                Button("preview", "BANDS", "show on image"), Button("quit", "QUIT")]

    def _on_results(self, name):
        if name.startswith("row:"):
            self.selected = name[4:]
        elif name == "apply":
            self.applied_backup = self._apply(calibrate.fit_config_values(self.result))
            self.go("applied")
        elif name == "retake" and self.selected:
            self.spot = int(self.selected[6:8]) - 1
            self.retaking = True
            self.go("place")
        elif name == "preview":
            self.go("preview")
        elif name == "quit":
            self.quit()

    def _draw_preview(self, img):
        preview = cv2.imread(os.path.join(self.folder, calibrate.PREVIEW_PNG), cv2.IMREAD_COLOR)
        if preview is not None:
            img[:] = fit_image(preview, (img.shape[1], img.shape[0]))[0]
        draw_banner(img, "Board corners (red) and the bands (green) on the first photo",
                    size=16)

    def _buttons_preview(self):
        return [Button("back", "BACK", primary=True), None, None, None]

    def _on_preview(self, name):
        self.go("results")

    def _draw_applied(self, img):
        check = self.result.get("check")
        draw_text(img, "Saved to config.py", (20, 50), 30, GREEN, bold=True)
        lines = [f"The old file is kept as {os.path.basename(self.applied_backup)}."]
        if check:
            lines.append(f"Check spot (tape it if you haven't): nearest corner row "
                         f"{check['near_m'] * 100:.1f} cm, centre {_side(check['lateral_m'])}.")
        lines.append("Before each session: python -m tools.calib_ui --check")
        hint = self.result.get("lens_hint")
        if hint and hint["suggest"] and not self.result["lens_used"]:
            lines.append("Edge boards fit worse than central ones: the optional lens step "
                         "(python -m tools.calib_ui --lens) may help.")
        y = 96
        for line in lines:
            y = draw_paragraph(img, line, (20, y), MAIN_W - 40, 18) + 10

    def _buttons_applied(self):
        return [Button("done", "DONE", primary=True), None, None, None]

    def _on_applied(self, name):
        self.done = True

    # ---------------------------------------------------------------- CHECK (pre-session)

    def _draw_check(self, img):
        view = self.camera.latest()
        found = self._draw_live(img, view)
        near, lat = self.cfg.CALIB_CHECK_NEAR_M, self.cfg.CALIB_CHECK_LATERAL_M
        if view is not None:
            expected = calibrate.expected_corners(self.cfg, near, lat, view.image.shape)
            scale = img.shape[1] / view.image.shape[1]
            for x, y in expected * scale:
                cv2.circle(img, (int(x), int(y)), 3, PURPLE, -1, cv2.LINE_AA)
        draw_banner(img, f"Board on the taped marks: near row {near * 100:.1f} cm, centre "
                         f"{_side(lat)}", size=17)
        draw_banner(img, self._live_status(view, found, "purple = where it should be", "CHECK"),
                    GREEN if found else AMBER, top=False, size=16)

    def _buttons_check(self):
        found = self._board_found()
        return [Button("check", "CHECK", "board found" if found else "no board", primary=True,
                       enabled=found), None, None, Button("quit", "QUIT")]

    def _on_check(self, name):
        if name == "quit":
            self.quit()
            return
        view = self.camera.latest()
        try:
            self.check_result = calibrate.check_photo(view.image, self.cfg)
        except ValueError as e:
            self._flash(str(e))
            return
        for level, text in calibrate.check_lines(self.check_result, self.cfg):
            say(TAG, text)
        self.capture = BoardView(view.image.copy(), None, view.t)
        self.go("check_result")

    def _draw_check_result(self, img):
        passed = self.check_result["passed"]
        draw_text(img, "PASS" if passed else "FAIL", (MAIN_W // 2, 90), 64,
                  GREEN if passed else RED, bold=True, align="centre")
        y = 140
        for level, text in calibrate.check_lines(self.check_result, self.cfg):
            y = draw_level_line(img, level, text, (40, y), MAIN_W - 80, 18) + 8
        canvas, scale, (ox, oy) = fit_image(self.capture.image, (320, 240))
        lane = self.check_result["grid"].reshape(-1, 2)
        frame_px = lane * self.capture.image.shape[1] / self.cfg.FRONT_LANE_SIZE[0]
        _draw_corners(canvas, frame_px * scale + (ox, oy), GREEN if passed else RED)
        img[img.shape[0] - 250:img.shape[0] - 10, (MAIN_W - 320) // 2:(MAIN_W + 320) // 2] = canvas

    def _buttons_check_result(self):
        return [Button("done", "DONE", primary=self.check_result["passed"]),
                Button("again", "AGAIN", primary=not self.check_result["passed"]), None, None]

    def _on_check_result(self, name):
        if name == "again":
            self.go("check")
        else:
            self.done = True

    # ---------------------------------------------------------------- LENS (optional)

    def _draw_lens_intro(self, img):
        draw_text(img, "Lens step (optional)", (20, 44), 30, TEXT, bold=True)
        draw_steps(img, [
            "Only worth doing if the board fit says edge boards fit much worse than central ones.",
            "Hold the board 40-80 cm from the camera (sharp at the driving focus), tilted "
            "20-45 degrees, a different way each photo.",
            "Cover the centre, edges and corners of the view: the grid shows what is covered.",
            f"Take at least {self.cfg.CALIB_LENS_MIN_PHOTOS} sharp photos with the whole board in "
            "view, then COMPUTE.",
            "Afterwards, refit the board session so the bands use the lens values.",
        ], (20, 90), MAIN_W - 40, 18)

    def _buttons_lens_intro(self):
        return [Button("start", "START", primary=True), None, None, Button("quit", "QUIT")]

    def _on_lens_intro(self, name):
        if name == "start":
            self.lens_folder = calibrate.new_session_dir(self.cfg, "lens")
            say(TAG, f"Lens photos go to {self.lens_folder}")
            self.go("lens")
        else:
            self.quit()

    def _draw_lens(self, img):
        view = self.camera.latest()
        found = self._draw_live(img, view)
        cells = self._lens_cells(view.image.shape if view is not None else None)
        h, w = img.shape[:2]
        for r in range(3):
            for c in range(3):
                x0, y0, x1, y1 = c * w // 3, r * h // 3, (c + 1) * w // 3, (r + 1) * h // 3
                if (r, c) in cells:
                    patch = img[y0:y1, x0:x1]
                    patch[:] = (patch * 0.7 + np.array(GREEN) * 0.3).astype(np.uint8)
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), LINE, 1)
        new = found and self._lens_new(view)
        draw_banner(img, f"Photos {len(self.lens_sets)}/{self.cfg.CALIB_LENS_MIN_PHOTOS}   "
                         f"cells {len(cells)}/9 (need {self.cfg.CALIB_LENS_MIN_CELLS})", size=17)
        status = self._current_flash() or ("Board found: CAPTURE" if new else
                                  "Too like an earlier photo: move or tilt it" if found else
                                  "No board: the whole board must be in view")
        draw_banner(img, status, GREEN if new else AMBER, top=False, size=16)

    def _buttons_lens(self):
        view = self.camera.latest()
        new = self._board_found() and self._lens_new(view)
        cells = self._lens_cells(view.image.shape if view else None)
        ready = (len(self.lens_sets) >= self.cfg.CALIB_LENS_MIN_PHOTOS and
                 len(cells) >= self.cfg.CALIB_LENS_MIN_CELLS)
        return [Button("capture", "CAPTURE", primary=new and not ready, enabled=new),
                Button("compute", "COMPUTE", primary=ready, enabled=ready), None,
                Button("quit", "QUIT")]

    def _on_lens(self, name):
        if name == "capture":
            view = self.camera.latest()
            corners = calibrate.find_board_corners(view.image, self.cfg.CALIB_PATTERN)
            if corners is None:
                self._flash("Lost the board: hold still and try again")
                return
            self.lens_sets.append(corners)
            name_png = f"lens_{len(self.lens_sets):02d}.png"
            cv2.imwrite(os.path.join(self.lens_folder, name_png), view.image)
            say(TAG, f"Lens photo {len(self.lens_sets)} saved ({name_png}).")
        elif name == "compute":
            self._compute_lens()
        else:
            self.quit()

    def _compute_lens(self):
        h, w = self.camera.latest().image.shape[:2]
        lens, rms = calibrate.calibrate_lens(self.lens_sets, self.cfg.CALIB_PATTERN,
                                             self.cfg.CALIB_SQUARE_M, (w, h))
        self.lens_result = {"lens": lens, "rms_px": rms,
                            "impact_mm": calibrate.lens_impact_mm(lens, self.cfg)}
        say(TAG, f"Lens fit {rms:.2f} px RMS; band measurements change by up to "
                 f"{self.lens_result['impact_mm'] or 0:.1f} mm.")
        self.go("lens_result")

    def _draw_lens_result(self, img):
        res, cfg = self.lens_result, self.cfg
        draw_text(img, "Lens result", (20, 44), 30, TEXT, bold=True)
        rms_ok = res["rms_px"] <= cfg.CALIB_LENS_RMS_WARN_PX
        impact = res["impact_mm"]
        y = draw_level_line(img, "ok" if rms_ok else "warn",
                            f"Fit error {res['rms_px']:.2f} px RMS" +
                            ("" if rms_ok else ": high, retake blurry or bent photos"),
                            (20, 96), MAIN_W - 40, 18) + 8
        if impact is not None:
            worth = impact >= cfg.CALIB_LENS_NEGLIGIBLE_MM
            y = draw_level_line(img, "info", f"Correction changes band measurements by up to "
                                f"{impact:.1f} mm: " + ("worth applying." if worth else
                                                        "negligible, no need to apply."),
                                (20, y), MAIN_W - 40, 18) + 8
        draw_paragraph(img, "After APPLY, refit the board session (REFIT) so the bands use "
                            "the lens values.", (20, y + 10), MAIN_W - 40, 18, MUTED)

    def _buttons_lens_result(self):
        return [Button("apply", "APPLY", "write config.py", primary=True),
                Button("discard", "DISCARD"), None, None]

    def _on_lens_result(self, name):
        if name == "apply":
            self._apply(calibrate.lens_config_values(self.lens_result["lens"]))
            latest = latest_session(self.cfg)
            if latest is None:
                self.go("message", "Lens values saved. Now run the board session: "
                                   "python -m tools.calib_ui")
            else:
                self.go("message", f"Lens values saved. REFIT the last board session "
                                   f"({os.path.basename(latest)}) so the bands use them.")
                self.folder = latest
        else:
            self.done = True

    # ---------------------------------------------------------------- MESSAGE

    def _draw_message(self, img):
        draw_paragraph(img, self.message, (30, 120), MAIN_W - 60, 22)

    def _buttons_message(self):
        if self.mode == "lens" and self.folder and self.lens_result:
            return [Button("refit", "REFIT", "board session", primary=True), None, None,
                    Button("quit", "QUIT")]
        return [Button("quit", "OK", primary=True), None, None, None]

    def _on_message(self, name):
        if name == "refit":
            self._resume(self.folder)
        else:
            self.quit()

    # ---------------------------------------------------------------- shared

    def _resume(self, folder):
        # Carry on a session: the first spot without a photo, or the fit.
        self.folder = folder
        meta, self.rows = calibrate.read_session(folder)
        self.square_m, self.road_m = meta["square_m"], meta["road_width_m"]
        todo = [i for i in range(len(self.cfg.CALIB_POSITIONS)) if self._row_for_spot(i) is None]
        say(TAG, f"Session {folder}: {len(self.rows)} photos kept.")
        if todo:
            self.spot = todo[0]
            self.go("place")
        else:
            self.go("fitting")
            self.work = self._run_fit

    def _apply(self, values):
        # Writes config.py (with a backup) and updates the running config too.
        backup = config_writer.write_values(config_writer.config_path(), values)
        for name, value in values.items():
            setattr(self.cfg, name, value)
        say(TAG, f"Wrote {', '.join(values)} to config.py (backup {os.path.basename(backup)}).")
        return backup

    def _draw_live(self, img, view):
        # The camera view with the detected corners; True if a board is found.
        if view is None:
            draw_text(img, "waiting for the camera...", (img.shape[1] // 2, img.shape[0] // 2),
                      22, MUTED, align="centre")
            return False
        canvas, scale, (ox, oy) = fit_image(view.image, (img.shape[1], img.shape[0]))
        img[:] = canvas
        if view.corners is not None:
            _draw_outline(img, view.corners * scale + (ox, oy), self.cfg.CALIB_PATTERN, 1.0, GREEN)
            _draw_corners(img, view.corners * scale + (ox, oy), GREEN)
        return view.corners is not None

    def _flash(self, text):
        # A short note in the status strip (and the terminal).
        self.flash, self.flash_t = text, time.monotonic()
        say(TAG, text)

    def _current_flash(self):
        return self.flash if self.flash and time.monotonic() - self.flash_t < FLASH_S else ""

    def _live_status(self, view, found, extra, action="CAPTURE"):
        if self._current_flash():
            return self._current_flash()
        if view is None:
            return "Camera starting..."
        if not found:
            return "No board: whole board in view, flat, no glare"
        still = self.last_corners is not None and self.last_corners.shape == view.corners.shape \
            and np.abs(view.corners - self.last_corners).mean() < self.cfg.CALIB_STILL_PX
        self.last_corners = view.corners
        return f"Board found{'' if still else ' (hold still)'}: {action}   ·   {extra}"

    def _board_found(self):
        view = self.camera.latest()
        return view is not None and view.corners is not None

    def _lens_cells(self, shape):
        if shape is None:
            return set()
        size = (shape[1], shape[0])
        return set().union(*[calibrate.lens_cells(c, size) for c in self.lens_sets]) \
            if self.lens_sets else set()

    def _lens_new(self, view):
        return calibrate.is_new_view(view.corners, self.lens_sets, self.cfg.CALIB_LENS_NEW_PX)

    def _spot_role(self):
        return self.cfg.CALIB_POSITIONS[self.spot][2]

    def _row_for_spot(self, i):
        return next((r for r in self.rows if r["image"] == _spot_image(i)), None)


# ---------------------------------------------------------------- small helpers

def _spot_image(i):
    return f"board_{i + 1:02d}.png"


def _side(lateral_m):
    # "on the tape", "7.0 cm right", "7.0 cm left".
    if abs(lateral_m) < 0.0005:
        return "on the tape"
    return f"{abs(lateral_m) * 100:.1f} cm {'right' if lateral_m > 0 else 'left'}"


def _draw_level_mark(img, level, centre):
    # Tick, cross or "!" for a report level ("info" draws nothing).
    if level in ("ok", "bad"):
        draw_mark(img, centre, level == "ok", 6)
    elif level == "warn":
        draw_text(img, "!", (centre[0], centre[1] + 7), 18, AMBER, bold=True, align="centre")


def _draw_corners(img, corners, colour):
    for x, y in np.asarray(corners).reshape(-1, 2):
        cv2.circle(img, (int(round(x)), int(round(y))), 3, colour, -1, cv2.LINE_AA)


def _draw_outline(img, corners, pattern, scale, colour):
    # The quadrilateral through the four outer inner-corners.
    cols, rows = pattern
    pts = np.asarray(corners, np.float64).reshape(-1, 2) * scale
    if len(pts) != cols * rows:
        return
    grid = pts.reshape(rows, cols, 2)
    quad = np.array([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]], np.int32)
    cv2.polylines(img, [quad], True, colour, 2, cv2.LINE_AA)


def latest_session(cfg):
    # The newest board session folder with photos, or None.
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cfg.CALIB_DIR)
    folders = sorted(glob.glob(os.path.join(root, "front_*", calibrate.POSITIONS_CSV)))
    return os.path.dirname(folders[-1]) if folders else None


# ---------------------------------------------------------------- display loop

KEYS = {"K_RETURN": "enter", "K_KP_ENTER": "enter", "K_ESCAPE": "escape", "K_q": "q",
        "K_UP": "up", "K_DOWN": "down", "K_LEFT": "left", "K_RIGHT": "right"}


def run(app, cfg):
    # Shows the app full screen until it is done; returns False if no display.
    from ui.screen import _set_display_env, open_display, show
    _set_display_env()
    display = open_display(cfg)
    if display is None:
        return False
    pygame, surface = display
    keys = {getattr(pygame, k): v for k, v in KEYS.items()}
    period, next_t = 1.0 / cfg.CALIB_UI_HZ, time.monotonic()
    try:
        while not app.done:
            show(pygame, surface, app.draw())
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    app.quit()
                elif event.type == pygame.KEYDOWN and event.key in keys:
                    app.key(keys[event.key], bool(event.mod & pygame.KMOD_SHIFT))
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    app.tap(event.pos)
            app.tick()
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
            next_t = max(next_t, time.monotonic())
    finally:
        pygame.quit()
    return True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Guided front camera calibration on the display.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="30-second pre-session check")
    group.add_argument("--lens", action="store_true", help="optional lens distortion step")
    group.add_argument("--resume", metavar="FOLDER", help="carry on (or refit) a board session")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    mode = "check" if args.check else "lens" if args.lens else "session"
    say(TAG, "Starting the front camera (stop main.py first: only one program can use it).")
    camera = LiveBoard(config)
    try:
        camera.start()
    except Exception as e:
        raise SystemExit(f"[{TAG}] The camera did not open ({e}). Is main.py or another "
                         "camera program still running?")
    try:
        app = CalibApp(config, camera, mode, args.resume)
        if not run(app, config):
            say(TAG, "No display. Run this on the Pi with its screen attached (over SSH the "
                     "screen is used automatically). Without a screen: take board photos with "
                     "tools/record.py and use: python -m tools.calibrate fit <folder>")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
