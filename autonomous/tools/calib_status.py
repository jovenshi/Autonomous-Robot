"""Has the front camera been checked recently? One small JSON file says.

Written by the pre-session check (tools/calib_ui.py --check, or
tools/calibrate.py check) and by APPLY at the end of a board session (its
check photo). Read by main.py at start-up, by the driving display's
pre-flight list and by the tools menu, so a bumped camera is caught
before the robot drives on it.

The file lives at CALIB_DIR/last_check.json:
    {"time": unix seconds, "passed": bool, "rms_m": float or null, "kind": "check" | "session"}
"""
import json
import os
import time

STATUS_FILE = "last_check.json"


def status_path(cfg):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cfg.CALIB_DIR)
    return os.path.join(root, STATUS_FILE)


def record_check(cfg, passed, rms_m, kind, now=None):
    # Remembers the newest check result (written atomically).
    path = status_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"time": time.time() if now is None else now, "passed": bool(passed),
                   "rms_m": rms_m, "kind": kind}, f)
    os.replace(tmp, path)


def read_check(cfg):
    # The stored result dict, or None (never checked, or unreadable).
    try:
        with open(status_path(cfg)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def check_status(cfg, now=None):
    # (ok, short text) for start-up checks: ok only after a calib_ui
    # calibration, with a PASS within CALIB_CHECK_MAX_AGE_H hours.
    if cfg.FRONT_GROUND_H is None:
        return False, "camera: not calibrated with calib_ui yet"
    last = read_check(cfg)
    if last is None:
        return False, "camera check: not run yet"
    hours = ((time.time() if now is None else now) - last["time"]) / 3600
    age = f"{hours * 60:.0f} min" if hours < 1 else f"{hours:.0f} h" if hours < 48 \
        else f"{hours / 24:.0f} days"
    if not last["passed"]:
        return False, f"camera check FAILED {age} ago"
    if hours > cfg.CALIB_CHECK_MAX_AGE_H:
        return False, f"camera checked {age} ago"
    return True, f"camera checked {age} ago"


def startup_line(cfg, now=None):
    # One line for main.py's terminal at start-up.
    ok, text = check_status(cfg, now)
    if ok:
        return f"Camera: {text}"
    return (f"Warning: {text}. Two minutes before driving: python -m tools.calib_ui --check "
            "(or the tools menu: python -m tools.menu)")
