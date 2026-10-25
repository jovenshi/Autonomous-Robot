"""Steering response test, guided on the robot's 7" touch display.

Measures how the robot really turns, so the controller, the sim and the
lane filter stop relying on guesses: turn gain per speed, the steering
deadband, the straight-line drift (motor trim), dead time and motor lag,
and forward speed. The analysis is tools/steer_report.py.

THE WHEELS MOVE. Clear floor: about 3 m ahead and 1 m either side. Stop
main.py first (it owns the Sabertooth), then from the autonomous/ folder:

    python -m tools.steer_test

Screens:
    WELCOME   safety and setup; waits for the IMU to calibrate
    READY     what the next run does and the space it needs; GO / SKIP
    RUNNING   live yaw rate; STOP (or Space / Esc / Enter) stops the wheels
    DISTANCE  after a straight run: tape the distance from the start mark
    RESULTS   per-speed gain, deadband, trim, speed; APPLY writes config.py
    LATENCY   camera latency from the newest main.py loop log (see below)

Runs: pivot steps (turning on the spot, left and right) first, then for
each speed in STEER_TEST_SPEEDS a straight run and one short left/right run
per size in STEER_TEST_STEERS. A run that is stopped is thrown away and can
be run again. Data goes to calibration/steer_<time>/ after every run.

Camera latency: run main.py with the robot in front of the line (IDLE,
LOOP_LOG_ENABLED on), twist it left and right by hand for 10 s, quit, then
tap LATENCY here (or: python -m tools.steer_report latency).

Keys: Enter = the green button, Esc = back/stop, Space = stop, q = quit.
"""
import argparse
import os
import sys
import threading
import time
import types

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from control.mixer import mix  # noqa: E402
from robot_types import DriveCmd  # noqa: E402
from tools import calibrate, config_writer, steer_report  # noqa: E402
from tools.guide_draw import (Button, draw_adjuster, draw_banner, draw_column,  # noqa: E402
                              draw_level_line, draw_paragraph, draw_steps, say)
from ui.overlay import AMBER, BG, BLUE, GREEN, LINE, MUTED, TEXT, YELLOW  # noqa: E402
from ui.text import draw_text  # noqa: E402

TAG = "steer"
MAIN_W = 640
PLOT_S = 6.0                             # s of yaw rate shown while running
IMU_STALE_S = 0.2                        # s; an older IMU sample stops a run
DIST_STEPS = (("-10", -0.10), ("-1", -0.01), ("+1", 0.01), ("+10", 0.10))
GUESS_MPS = 1.0                          # m/s per unit, for space hints before it's measured


def raw_motor(speed, steer, cfg):
    # Wheel commands with NO trim or deadband lift: the test measures the
    # robot as it is, not as mix() corrects it.
    raw = types.SimpleNamespace(STEER_GAIN=cfg.STEER_GAIN, STEER_TRIM=None, STEER_DEADBAND=0.0,
                                STEER_DEADBAND_RAMP=1.0)
    return mix(DriveCmd(speed=speed, steer=steer), raw)


class ImuLog:
    """Every IMU sample, read from the ImuReader's slot faster than it updates."""

    def __init__(self, slot, cfg):
        self.slot, self.cfg = slot, cfg
        self.samples = []                # (t, yaw_rate_dps, yaw_deg)
        self._stop = threading.Event()
        self._thread = None

    def poll(self):
        sample = self.slot.get()
        new = sample is not None and (not self.samples or sample.t > self.samples[-1][0])
        if new and sample.valid:
            self.samples.append((sample.t, sample.yaw_rate_dps, sample.yaw_deg))

    def latest(self):
        return self.slot.get()

    def since(self, t0):
        return [s for s in self.samples[-2000:] if s[0] >= t0]

    def start(self):
        def loop():
            while not self._stop.is_set():
                self.poll()
                time.sleep(1.0 / self.cfg.STEER_TEST_IMU_POLL_HZ)
        self._thread = threading.Thread(target=loop, name="imu_log", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


class Runner:
    """Plays one run's schedule on the motors, feeding the Sabertooth every
    loop, and stops at once on stop() or if the IMU fails."""

    def __init__(self, saber, imu_log, cfg, clock=time.monotonic, sleep=time.sleep,
                 on_tick=None, drift_request=None):
        self.saber, self.imu_log, self.cfg = saber, imu_log, cfg
        self.clock, self.sleep, self.on_tick = clock, sleep, on_tick
        self.drift_request = drift_request
        self.segments, self.reason, self.done, self.t_start = [], None, True, None
        self.current = None              # the piece being driven now (for the live plot)
        self._stop = False

    def start(self, run, background=True):
        self.segments, self.reason, self.done, self._stop = [], None, False, False
        self.t_start = self.clock()
        if background:
            threading.Thread(target=self.run, args=(run,), name="steer_run", daemon=True).start()
        else:
            self.run(run)

    def stop(self):
        self._stop = True

    def run(self, run):
        if self.drift_request is not None:
            self.drift_request.clear()                # never measure drift while moving
        try:
            for kind, seconds, speed, steer in run.schedule:
                seg = {"run": run.name, "kind": kind, "speed": speed, "steer": steer,
                       "t0": self.clock()}
                self.current = seg
                self._hold(speed, steer, seg["t0"] + seconds)
                seg["t1"] = self.clock()
                self.current = None
                self.segments.append(seg)
                if self.reason:
                    break
        finally:
            self.saber.stop()
            if self.drift_request is not None:
                self.drift_request.set()
            self.done = True

    def _hold(self, speed, steer, until):
        period = 1.0 / self.cfg.CONTROL_HZ
        motor = raw_motor(speed, steer, self.cfg)
        while self.clock() < until:
            self.reason = "stopped" if self._stop else self._imu_problem()
            if self.reason:
                return
            self.saber.set(motor)
            if self.on_tick is not None:
                self.on_tick()
            self.sleep(period)

    def _imu_problem(self):
        sample = self.imu_log.latest()
        if sample is None or not sample.valid:
            return "the IMU stopped giving valid readings"
        if self.clock() - sample.t > IMU_STALE_S:
            return "the IMU readings went stale"
        return None


class SteerApp:
    """All screens and their logic: draw() -> image, tap()/key() -> actions,
    tick() notices a finished run."""

    def __init__(self, cfg, runner, imu_log, folder=None, background=True, clock=time.monotonic):
        self.cfg, self.runner, self.imu_log = cfg, runner, imu_log
        self.background, self.clock, self.t_open = background, clock, clock()
        self.size = tuple(cfg.DISPLAY_SIZE)
        self.runs = steer_report.plan_runs(cfg)
        self.index, self.folder = 0, folder
        self.segments, self.distances, self.done_runs = [], {}, set()
        self.distance_m, self.result, self.latency, self.latency_error = 1.0, None, None, ""
        self.applied, self.flash, self.message = None, "", ""
        self.zones, self.done, self.page = [], False, None
        self.go("welcome")

    # ---------------------------------------------------------------- framework

    def go(self, page, message=""):
        self.page, self.message = page, message
        tip = {"welcome": "The wheels WILL move. Clear floor: about 3 m ahead, 1 m either side. "
                          "STOP on the screen, Space or Esc stops them at once; the Sabertooth "
                          "also stops by itself if this program dies.",
               "results": "Gain = how fast the robot turns per unit of steer. Trim = the steer "
                          "that makes it drive straight. APPLY writes them to config.py.",
               "latency": "Camera latency comes from a main.py loop log: robot in front of the "
                          "line in IDLE, twisted left and right by hand for 10 s."}.get(page)
        if page == "ready":
            run = self.runs[self.index]
            say(TAG, f"Run {self.index + 1} of {len(self.runs)}: {run.name}. {self._run_hint(run)}")
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
        return img

    def buttons(self):
        return getattr(self, f"_buttons_{self.page}")()

    def press(self, name):
        for button in self.buttons():
            if button is not None and button.name == name:
                if button.enabled:
                    getattr(self, f"_on_{self.page}")(name)
                return
        if ":" in name:
            getattr(self, f"_on_{self.page}")(name)

    def tap(self, pos):
        from tools.guide_draw import hit
        name = hit(self.zones, pos)
        if name is not None:
            self.press(name)

    def key(self, name, shift=False):
        if self.page == "running" and name in ("space", "escape", "enter", "q"):
            self.press("stop")
        elif name == "q":
            self.quit()
        elif name == "enter":
            primary = [b for b in self.buttons() if b is not None and b.primary and b.enabled]
            if primary:
                self.press(primary[0].name)
        elif name == "escape":
            names = [b.name for b in self.buttons() if b is not None]
            for back in ("back", "redo", "skip"):
                if back in names:
                    self.press(back)
                    return

    def tick(self):
        if self.page == "running" and self.runner.done:
            self._run_finished()

    def quit(self):
        self.runner.stop()
        if self.folder and self.segments:
            say(TAG, f"Data kept in {self.folder} (python -m tools.steer_report {self.folder})")
        self.done = True

    # ---------------------------------------------------------------- WELCOME

    def _draw_welcome(self, img):
        minutes = sum(r.seconds() for r in self.runs) / 60 + len(self.runs) * 0.4
        draw_text(img, "Steering response test", (20, 44), 30, TEXT, bold=True)
        y = draw_steps(img, [
            "Robot on the floor with about 3 m clear ahead and 1 m either side; battery charged.",
            "The wheels WILL move. STOP (or Space / Esc) stops them at once.",
            f"{len(self.runs)} runs, about {minutes:.0f} min: turning on the spot first, then "
            "for each speed a straight run and short left/right steers.",
            "Have a tape measure ready: after each straight run you measure how far it went.",
        ], (20, 84), MAIN_W - 40, 17)
        ok, text = self._imu_status()
        draw_level_line(img, "ok" if ok else "warn", text, (20, max(y, 330) + 20), MAIN_W - 40, 18)

    def _buttons_welcome(self):
        ok, _ = self._imu_status()
        return [Button("start", "START", "IMU ready" if ok else "waiting for IMU",
                       primary=True, enabled=ok),
                Button("latency", "LATENCY", "from a loop log"), None, Button("quit", "QUIT")]

    def _on_welcome(self, name):
        if name == "start":
            self.folder = self.folder or calibrate.new_session_dir(self.cfg, "steer")
            say(TAG, f"Data folder: {self.folder}")
            self.go("ready")
        elif name == "latency":
            self._run_latency()
        else:
            self.quit()

    def _imu_status(self):
        sample = self.imu_log.latest()
        if sample is None or not sample.valid:
            return False, "IMU: no valid readings yet"
        if sample.calib_gyro < 3:
            waited = self.clock() - self.t_open
            if waited < self.cfg.IMU_CALIB_WAIT_S:
                return False, f"IMU: gyro calibrating ({sample.calib_gyro}/3); keep the robot still"
            return True, "IMU: gyro not fully calibrated; going ahead (results may be noisier)"
        return True, "IMU: calibrated"

    # ---------------------------------------------------------------- READY

    def _run_hint(self, run):
        mps = self.cfg.SPEED_MPS_PER_UNIT or GUESS_MPS
        if run.kind == "pivot":
            return f"Turns on the spot, left and right, {len(self.cfg.STEER_TEST_STEERS)} sizes. " \
                   "Needs 0.5 m clear all round."
        if run.kind == "straight":
            return (f"Drives straight for {run.seconds():.0f} s. Mark the floor at the rear axle "
                    f"first. About {run.distance_m(mps):.1f} m clear ahead.")
        return (f"Drives straight briefly, then steers left and right ({run.steer:.2f}). About "
                f"{run.distance_m(mps):.1f} m clear ahead, 0.5 m either side.")

    def _draw_ready(self, img):
        run = self.runs[self.index]
        draw_text(img, f"Run {self.index + 1} of {len(self.runs)}", (20, 44), 20, MUTED)
        draw_text(img, run.name, (20, 84), 32, TEXT, bold=True)
        y = draw_paragraph(img, self._run_hint(run), (20, 130), MAIN_W - 40, 20)
        draw_paragraph(img, "Put the robot at the start, pointing at the open floor, then GO.",
                       (20, y + 16), MAIN_W - 40, 18, MUTED)
        done = len(self.done_runs)
        draw_banner(img, self.flash or f"{done} of {len(self.runs)} runs done",
                    GREEN if self.flash else MUTED, top=False, size=16)

    def _buttons_ready(self):
        return [Button("go", "GO", "wheels move", primary=True),
                Button("skip", "SKIP", "this run"),
                Button("results", "RESULTS", "analyse now", enabled=bool(self.segments)),
                Button("quit", "QUIT")]

    def _on_ready(self, name):
        if name == "go":
            self.flash = ""
            self.page = "running"
            say(TAG, f"Running {self.runs[self.index].name}...")
            self.runner.start(self.runs[self.index], self.background)
            if not self.background:
                self.tick()
        elif name == "skip":
            self._next_run()
        elif name == "results":
            self._analyse()
        else:
            self.quit()

    def _next_run(self):
        later = [i for i in range(self.index + 1, len(self.runs)) if i not in self.done_runs]
        if later:
            self.index = later[0]
            self.go("ready")
        else:
            self._analyse()

    # ---------------------------------------------------------------- RUNNING

    def _draw_running(self, img):
        run = self.runs[self.index]
        draw_text(img, f"RUNNING: {run.name}", (20, 40), 26, AMBER, bold=True)
        draw_text(img, "STOP, Space or Esc stops the wheels", (20, 68), 16, MUTED)
        t0 = self.runner.t_start or 0.0
        samples = self.imu_log.since(max(t0, self.clock() - PLOT_S))
        pieces = self.runner.segments + ([self.runner.current] if self.runner.current else [])
        _draw_plot(img, (20, 90, MAIN_W - 40, 360), samples, pieces, self.clock())

    def _buttons_running(self):
        return [Button("stop", "STOP", "wheels", primary=True), None, None, None]

    def _on_running(self, name):
        self.runner.stop()

    def _run_finished(self):
        run = self.runs[self.index]
        if self.runner.reason:
            self.flash = f"Run stopped ({self.runner.reason}); nothing kept. GO to run it again."
            self.go("ready", self.flash)
            return
        self.segments += self.runner.segments
        self.done_runs.add(self.index)
        self._save()
        if run.kind == "straight":
            self.distance_m = round(run.distance_m(self.cfg.SPEED_MPS_PER_UNIT or GUESS_MPS), 2)
            self.go("distance")
        else:
            self.flash = self._run_summary(run)
            say(TAG, self.flash)
            self._next_run()

    def _run_summary(self, run):
        # "left 0.20: +38 deg/s, right: -36 deg/s" from the run just done.
        t, rate = self._arrays()
        segs = [s for s in self.segments if s["run"] == run.name]
        parts = []
        for i, seg in enumerate(segs):
            if seg["kind"] == "step":
                base = steer_report.baseline_before(t, rate, segs, i)
                resp = steer_report.step_response(t, rate, seg, base, self.cfg)
                if resp is not None:
                    parts.append(f"{seg['steer']:+.2f}: {resp['change']:+.0f}")
        return f"{run.name} done; deg/s per step " + ("  ".join(parts) if parts else "--")

    # ---------------------------------------------------------------- DISTANCE

    def _draw_distance(self, img):
        run = self.runs[self.index]
        draw_text(img, "How far did it go?", (20, 50), 30, TEXT, bold=True)
        draw_paragraph(img, f"Measure from the start mark to the rear axle now ({run.name}, "
                            f"{run.seconds():.0f} s). This sets the forward speed.",
                       (20, 90), MAIN_W - 40, 18)
        self.zones += draw_adjuster(img, (20, 200, MAIN_W - 40, 86), "Distance travelled",
                                    f"{self.distance_m * 100:.0f} cm", "dist", DIST_STEPS)

    def _buttons_distance(self):
        return [Button("keep", "KEEP", primary=True), Button("redo", "REDO", "run again"),
                Button("skip", "SKIP", "no distance"), None]

    def _on_distance(self, name):
        run = self.runs[self.index]
        if name.startswith("dist:"):
            self.distance_m = max(0.0, round(self.distance_m + float(name[5:]), 2))
        elif name == "keep":
            self.distances[run.speed] = self.distance_m
            self._save()
            say(TAG, f"{run.name}: {self.distance_m * 100:.0f} cm in {run.seconds():.0f} s.")
            self._next_run()
        elif name == "redo":
            self.segments = [s for s in self.segments if s["run"] != run.name]
            self.done_runs.discard(self.index)
            self.go("ready")
        else:
            self._next_run()

    # ---------------------------------------------------------------- RESULTS

    def _analyse(self):
        t, rate = self._arrays()
        self.result = steer_report.analyse(t, rate, self.segments, self.distances, self.cfg)
        for level, text in steer_report.report_lines(self.result, self.cfg):
            say(TAG, ("" if level in ("info", "ok") else level.upper() + ": ") + text)
        self.go("results")

    def _draw_results(self, img):
        draw_text(img, "Steering response", (16, 36), 26, TEXT, bold=True)
        y = 70
        for level, text in steer_report.report_lines(self.result, self.cfg):
            y = draw_level_line(img, level, text, (12, y), MAIN_W - 24, 16) + 4
        values = steer_report.config_values(self.result)
        y += 8
        draw_text(img, "APPLY writes:", (16, y), 15, MUTED, bold=True)
        draw_paragraph(img, ", ".join(values) if values else "nothing measured yet",
                       (16, y + 20), MAIN_W - 32, 14, MUTED)

    def _buttons_results(self):
        values = steer_report.config_values(self.result)
        return [Button("apply", "APPLY", "write config.py", primary=bool(values),
                       enabled=bool(values)),
                Button("back", "BACK", "more runs"), None, Button("quit", "QUIT")]

    def _on_results(self, name):
        if name == "apply":
            self.applied = self._apply(steer_report.config_values(self.result))
            self.go("applied")
        elif name == "back":
            todo = [i for i in range(len(self.runs)) if i not in self.done_runs]
            self.index = todo[0] if todo else 0
            self.go("ready")
        else:
            self.quit()

    def _draw_applied(self, img):
        draw_text(img, "Saved to config.py", (20, 50), 30, GREEN, bold=True)
        y = draw_paragraph(img, "Written: " + ", ".join(self.applied) + ". The old file is kept "
                                "as config.py.bak.", (20, 96), MAIN_W - 40, 18) + 12
        for line in ("mix() now cancels the measured drift and steering deadband.",
                     "Next: python -m tools.lane_sim, then start the robot 5 cm off the line "
                     "with logging on and compare: python -m tools.lane_sim --log <that log>.",
                     "Tune KP_LAT, KH, KD_YAW in the sim once it agrees with the robot."):
            y = draw_paragraph(img, line, (20, y), MAIN_W - 40, 18) + 10

    def _buttons_applied(self):
        return [Button("done", "DONE", primary=True), None, None, None]

    def _on_applied(self, name):
        self.quit()

    # ---------------------------------------------------------------- LATENCY

    def _run_latency(self):
        path = steer_report.newest_loop_log(self.cfg)
        self.latency, self.latency_error = None, ""
        try:
            if path is None:
                raise ValueError(f"no loop logs in {self.cfg.LOOP_LOG_DIR}/ yet")
            self.latency = steer_report.latency_from_log(path, self.cfg)
            self.latency["log"] = os.path.basename(path)
            say(TAG, f"Camera latency {self.latency['latency_s'] * 1000:.0f} ms from "
                     f"{self.latency['log']}.")
        except (ValueError, KeyError, FileNotFoundError) as e:
            self.latency_error = str(e)
        self.go("latency", self.latency_error)

    def _draw_latency(self, img):
        draw_text(img, "Camera latency", (20, 44), 30, TEXT, bold=True)
        y = draw_steps(img, [
            "Run main.py with the robot in front of the line (IDLE, loop log on).",
            "Twist it left and right by hand, about 30 degrees each way, once a second, "
            "for 10 s. Keep the line in view.",
            "Quit main.py, then come back here: the newest loop log is used.",
        ], (20, 84), MAIN_W - 40, 17) + 10
        if self.latency:
            lat = self.latency
            draw_level_line(img, "ok", f"{lat['latency_s'] * 1000:.0f} ms (correlation "
                                       f"{lat['corr']:.2f}, {lat['seconds']:.0f} s of "
                                       f"{lat['log']})", (20, y), MAIN_W - 40, 20)
        elif self.latency_error:
            draw_level_line(img, "warn", self.latency_error, (20, y), MAIN_W - 40, 18)

    def _buttons_latency(self):
        return [Button("apply", "APPLY", "CAMERA_LATENCY_S", primary=bool(self.latency),
                       enabled=bool(self.latency)),
                Button("again", "AGAIN", "newest log"), Button("back", "BACK"), None]

    def _on_latency(self, name):
        if name == "apply":
            self._apply({"CAMERA_LATENCY_S": round(self.latency["latency_s"], 3)})
            self.latency = None
            self.go("welcome", "Camera latency saved.")
        elif name == "again":
            self._run_latency()
        else:
            self.go("welcome")

    # ---------------------------------------------------------------- shared

    def _arrays(self):
        samples = self.imu_log.samples
        return (np.array([s[0] for s in samples]), np.array([s[1] for s in samples]))

    def _save(self):
        steer_report.write_samples(self.folder, self.imu_log.samples)
        steer_report.write_json(self.folder, steer_report.SEGMENTS_JSON, self.segments)
        steer_report.write_json(self.folder, steer_report.DISTANCES_JSON,
                                {str(k): v for k, v in self.distances.items()})

    def _apply(self, values):
        backup = config_writer.write_values(config_writer.config_path(), values)
        for name, value in values.items():
            setattr(self.cfg, name, value)
        say(TAG, f"Wrote {', '.join(values)} to config.py (backup {os.path.basename(backup)}).")
        return list(values)


def _draw_plot(img, rect, samples, segments, now):
    # Yaw rate over the last PLOT_S seconds, with the steer commanded.
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), LINE, 1)
    mid = y + h // 2
    cv2.line(img, (x, mid), (x + w, mid), LINE, 1)
    rates = [abs(s[1]) for s in samples]
    limit = max(60.0, max(rates) * 1.2 if rates else 0.0)
    draw_text(img, f"+{limit:.0f} deg/s (left)", (x + 6, y + 18), 14, MUTED)
    draw_text(img, f"-{limit:.0f}", (x + 6, y + h - 6), 14, MUTED)

    def px(t, value, scale):
        return int(x + w - (now - t) / PLOT_S * w), int(mid - value / scale * (h / 2))

    for seg in segments:
        if seg["steer"] and seg.get("t1", now) > now - PLOT_S:
            a, b = (px(seg["t0"], seg["steer"] * 100, limit),
                    px(seg.get("t1", now), seg["steer"] * 100, limit))
            cv2.line(img, (max(a[0], x), a[1]), (max(b[0], x), b[1]), YELLOW, 2)
    pts = np.array([px(t, r, limit) for t, r, _ in samples], np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, BLUE, 2, cv2.LINE_AA)
    draw_text(img, "yaw rate", (x + w - 90, y + 18), 14, BLUE)
    draw_text(img, "steer (x100)", (x + w - 90, y + 36), 14, YELLOW)


# ---------------------------------------------------------------- display loop

KEYS = {"K_RETURN": "enter", "K_KP_ENTER": "enter", "K_ESCAPE": "escape", "K_q": "q",
        "K_SPACE": "space"}


def run_display(app, cfg):
    # Shows the app full screen until it is done; returns False if no display.
    from ui.screen import _set_display_env, open_display, show
    _set_display_env()
    display = open_display(cfg)
    if display is None:
        return False
    pygame, surface = display
    keys = {getattr(pygame, k): v for k, v in KEYS.items()}
    period = 1.0 / cfg.CALIB_UI_HZ
    try:
        while not app.done:
            show(pygame, surface, app.draw())
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    app.quit()
                elif event.type == pygame.KEYDOWN and event.key in keys:
                    app.key(keys[event.key])
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    app.tap(event.pos)
            app.tick()
            time.sleep(period)
    finally:
        pygame.quit()
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Guided steering response test (wheels move).")
    parser.parse_args(argv)
    from hal.imu import ImuReader
    from hal.sabertooth import SabertoothDriver
    from robot_types import LatestValue

    say(TAG, "Opening the IMU and the Sabertooth (stop main.py first).")
    slot, drift_request = LatestValue(), threading.Event()
    drift_request.set()                              # measure drift while the robot is still
    reader = ImuReader(slot, drift_request, config)
    saber = SabertoothDriver(config.SABER_PORT, config.SABER_BAUD, config.SABER_ADDRESS, config)
    saber.open()
    if saber.serial_error:
        raise SystemExit(f"[{TAG}] The Sabertooth port did not open: is main.py still running?")
    imu_log = ImuLog(slot, config)
    try:
        reader.start()
        imu_log.start()
        runner = Runner(saber, imu_log, config, drift_request=drift_request)
        app = SteerApp(config, runner, imu_log)
        if not run_display(app, config):
            say(TAG, "No display: run this on the Pi with its screen attached.")
    finally:
        saber.stop()                                 # motors first
        saber.close()
        imu_log.stop()
        reader.close()


if __name__ == "__main__":
    main()
