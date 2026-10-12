"""Feed synthetic UiPackets to the display, with no robot or sensors attached.

Run from the autonomous/ folder (over SSH it opens on the Pi's screen):
    python -m tools.ui_demo                 # until Ctrl-C
    python -m tools.ui_demo --seconds 60

One cycle, repeated:
    IDLE         gyro calibrating for the first seconds (pre-flight shows X, then OK)
    LANE_FOLLOW  lateral error swinging, LANE_CONF_LOW toggling, steer following
    SAFE_STOP    "lane lost": no lane, and no new front views, so the view goes STALE
The side camera never sends anything, so its panel shows NO DATA.
START / STOP taps (or the space bar) are printed here as main would receive them.
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from control.mixer import mix
from robot_types import (DriveCmd, FrontLaneObs, FrontResult, HealthWarning, ImuSample,
                         LaneEstimate, State, UiPacket)
from ui.link import read_commands, send, start_ui, stop_ui

# Demo script timing (not robot tunables).
PHASES = ((State.IDLE, 6.0), (State.LANE_FOLLOW, 12.0), (State.SAFE_STOP, 5.0))
CALIBRATING_S = 3.0              # s of IDLE with calib_gyro < 3
WEAVE_HZ = 0.4                   # lateral error oscillation
WEAVE_M = 0.04                   # m amplitude
WARN_TOGGLE_S = 2.0              # LANE_CONF_LOW on/off period


def phase_at(elapsed):
    # (state, seconds into that phase) for a time into the repeating cycle.
    cycle = sum(seconds for _, seconds in PHASES)
    into = elapsed % cycle
    for state, seconds in PHASES:
        if into < seconds:
            return state, into
        into -= seconds
    return PHASES[-1][0], 0.0


def demo_packet(elapsed, now, loop, cfg):
    # One synthetic packet; views every UI_IMAGE_EVERY loops (none in SAFE_STOP).
    state, into = phase_at(elapsed)
    lateral = WEAVE_M * math.sin(2 * math.pi * WEAVE_HZ * elapsed)
    heading = 0.15 * math.cos(2 * math.pi * WEAVE_HZ * elapsed)
    lane = (LaneEstimate(None, None, 0.0, None, now) if state == State.SAFE_STOP
            else LaneEstimate(lateral, heading, 0.9, None, now))
    steer = 0.0
    if state == State.LANE_FOLLOW:
        steer = max(-cfg.STEER_MAX, min(cfg.STEER_MAX, cfg.KP_LAT * lateral + cfg.KH * heading))
    drive = DriveCmd(speed=cfg.CRUISE_SPEED if state == State.LANE_FOLLOW else 0.0, steer=steer)
    calib = 1 if state == State.IDLE and into < CALIBRATING_S else 3
    warnings = ()
    if state == State.LANE_FOLLOW and int(into / WARN_TOGGLE_S) % 2:
        warnings = (HealthWarning.LANE_CONF_LOW,)
    with_view = loop % cfg.UI_IMAGE_EVERY == 0 and state != State.SAFE_STOP
    return UiPacket(
        t=now, state=state, stop_reason="lane lost" if state == State.SAFE_STOP else None,
        drive=drive, motor=mix(drive, cfg), lane=lane, road_yaw=None,
        imu=ImuSample(20.0 * math.sin(0.3 * elapsed), 8.0 * steer / cfg.STEER_MAX if steer else 0.0,
                      True, calib, 0.0, now),
        sign_vote=None, fps={"front": 30.0}, overruns=0, warnings=warnings,
        front=synthetic_front(now, loop, lateral, heading, cfg) if with_view else None, side=None,
    )


def synthetic_front(now, seq, lateral_m, heading_rad, cfg):
    # A grey road with a yellow line, offset by lateral_m and slanted by
    # heading, with band centre points and the fitted line to match.
    w, h = cfg.FRONT_LANE_SIZE
    img = np.full((h, w, 3), (40, 90, 40), np.uint8)
    cv2.rectangle(img, (w // 6, 0), (w - w // 6, h - 1), (110, 110, 110), -1)
    cx = (w - 1) / 2.0
    px_per_m, slope = 600.0, math.tan(heading_rad) * 0.5

    def line_x(y):
        return cx - lateral_m * px_per_m + (h - y) * slope

    cv2.line(img, (int(line_x(0)), 0), (int(line_x(h - 1)), h - 1), (0, 220, 230), 6)
    ys = [(top + bottom) / 2.0 for top, bottom, *_ in cfg.FRONT_BANDS] if cfg.FRONT_BANDS \
        else [h * f for f in (0.55, 0.65, 0.75, 0.85, 0.95)]
    points = tuple((line_x(y), y) for y in ys)
    obs = FrontLaneObs(lateral_m, heading_rad, 0.9, points, (points[0], points[-1]), now, seq)
    return FrontResult(img, obs, None, None, 110.0, now, seq)


def main():
    p = argparse.ArgumentParser(description="Synthetic packets to the robot display.")
    p.add_argument("--seconds", type=float, default=None, help="stop after this long")
    seconds = p.parse_args().seconds
    process, packet_q, cmd_q = start_ui(config)
    start, loop, period = time.monotonic(), 0, 1.0 / config.CONTROL_HZ
    try:
        while seconds is None or time.monotonic() - start < seconds:
            now = time.monotonic()
            send(packet_q, demo_packet(now - start, now, loop, config))
            for command in read_commands(cmd_q):
                print(f"UI command: {command.name}")
            loop += 1
            time.sleep(max(0.0, period - (time.monotonic() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        stop_ui(process, config)


if __name__ == "__main__":
    main()
