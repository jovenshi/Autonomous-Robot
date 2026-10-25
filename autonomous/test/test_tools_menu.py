"""Tests for tools/menu.py, tools/calib_status.py and the camera line in the
pre-flight list. No display (the loop test uses SDL's dummy driver), no
child processes (a fake Popen).

Run from the autonomous/ folder:  python -m pytest test/test_tools_menu.py
"""
import os
import signal
import sys
import types

import numpy as np
import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from robot_types import ImuSample  # noqa: E402
from test_ui import bare_packet  # noqa: E402
from tools import calib_status, menu  # noqa: E402
from ui.panels import preflight  # noqa: E402

NOW = 1_800_000_000.0


def ns(tmp_path, **overrides):
    values = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    values.update(CALIB_DIR=str(tmp_path / "calibration"), FRONT_GROUND_H=np.eye(3).tolist(),
                  STEER_YAW_DPS_PER_UNIT={0.5: 150.0}, DISPLAY_SIZE=(800, 480))
    values.update(overrides)
    return types.SimpleNamespace(**values)


# ---------------------------------------------------------------- calib_status

def test_status_texts(tmp_path):
    cfg = ns(tmp_path)
    assert calib_status.check_status(ns(tmp_path, FRONT_GROUND_H=None), NOW) == \
        (False, "camera: not calibrated with calib_ui yet")
    assert calib_status.check_status(cfg, NOW) == (False, "camera check: not run yet")
    calib_status.record_check(cfg, True, 0.002, "check", now=NOW - 1800)
    assert calib_status.check_status(cfg, NOW) == (True, "camera checked 30 min ago")
    assert "Camera: camera checked" in calib_status.startup_line(cfg, NOW)
    too_old = NOW + cfg.CALIB_CHECK_MAX_AGE_H * 3600
    ok, text = calib_status.check_status(cfg, too_old)
    assert not ok and text.startswith("camera checked")
    assert "--check" in calib_status.startup_line(cfg, too_old)
    calib_status.record_check(cfg, False, 0.02, "check", now=NOW)
    assert calib_status.check_status(cfg, NOW + 7200) == (False, "camera check FAILED 2 h ago")


def test_unreadable_status_counts_as_not_run(tmp_path):
    cfg = ns(tmp_path)
    os.makedirs(cfg.CALIB_DIR)
    open(calib_status.status_path(cfg), "w").write("{broken")
    assert calib_status.check_status(cfg, NOW)[0] is False


def test_preflight_shows_the_camera_line_only_when_given():
    packet = bare_packet(imu=ImuSample(0.0, 0.0, True, 3, 0.0, 1.0), fps={"front": 30.0})
    assert len(preflight(packet, config)) == 4
    checks = dict(preflight(packet, config, (False, "camera check: not run yet")))
    assert checks["camera check: not run yet"] is False


# ---------------------------------------------------------------- menu

class FakeChild:
    def __init__(self, args, cwd):
        self.args, self.cwd, self.code, self.signals = args, cwd, None, []

    def poll(self):
        return self.code

    @property
    def returncode(self):
        return self.code

    def send_signal(self, sig):
        self.signals.append(sig)
        self.code = -sig


def test_tiles_and_highlighted_fixes(tmp_path):
    app = menu.MenuApp(ns(tmp_path, STEER_YAW_DPS_PER_UNIT=None))
    assert {t.key for t in app.tools} >= {"drive", "check", "calibrate", "steer", "imu", "record"}
    wanted = {key for _, _, key in app.status if key}
    assert wanted == {"check", "steer"}
    assert app.draw().shape == (480, 800, 3)
    assert {z.name for z in app.zones} >= {"tool:drive", "tool:steer", "quit"}


def test_terminal_tool_runs_with_stop_then_returns(tmp_path):
    children = []
    app = menu.MenuApp(ns(tmp_path), popen=lambda args, cwd: children.append(FakeChild(args, cwd))
                       or children[-1])
    app.draw()
    tile = next(z for z in app.zones if z.name == "tool:imu")
    app.tap((tile.rect[0] + 5, tile.rect[1] + 5))
    assert app.page == "running" and children[0].args[1:] == ["-m", "tools.imu_drift_test"]
    assert app.draw().shape == (480, 800, 3)
    app.key("space")                                    # STOP sends Ctrl-C
    assert children[0].signals == [signal.SIGINT]
    app.tick()
    assert app.page == "tiles" and app.flash == "IMU CHECK finished"


def test_screen_tool_is_left_to_the_display_loop(tmp_path):
    app = menu.MenuApp(ns(tmp_path))
    app.press("tool:calibrate")
    assert app.pending.args == ["-m", "tools.calib_ui"] and app.page == "tiles"


def test_crashed_tool_shows_its_exit_code(tmp_path):
    app = menu.MenuApp(ns(tmp_path))
    app.finished(app.tools[0], 1)
    assert app.flash == "DRIVE finished (exit code 1)"


def test_display_loop_quits(tmp_path, monkeypatch):
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = menu.MenuApp(ns(tmp_path))
    draws = []

    def draw():
        draws.append(1)
        app.done = len(draws) >= 2
        return np.zeros((480, 800, 3), np.uint8)

    app.draw = draw
    assert menu.run(app, ns(tmp_path)) and len(draws) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
