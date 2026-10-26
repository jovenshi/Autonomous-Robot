"""Replay a recording through the front pipeline, fusion and controller, with
the motors stubbed: nothing is sent to the Sabertooth.

Each frame goes through the same code as the robot in LANE_FOLLOW:
process_front_frame -> fuse_lane -> drive_for_state -> mix, using the
recording's own timestamps. One line per frame is printed; a window shows
the lane drawing (needs the Pi desktop; --no-window to only print).

Run from the robot/ folder:
    python -m tools.replay recordings/front_20261011_140000
    python -m tools.replay recordings/front_20261011_140000 --speed 0 --no-window

Keys: space pause/resume, n next frame while paused, q or Esc quit.
"""
import argparse
import os
import sys
import time

import cv2

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from control.command import drive_for_state
from control.lane_controller import LaneController
from control.mixer import mix
from estimation.lane_fusion import fuse_lane
from perception.front_lane import LaneTracker
from perception.pipeline import process_front_frame
from robot_types import State
from tools.debug_draw import draw_lane, lane_text, put_lines
from tools.recording import iter_frames

WINDOW = "replay"


def replay_step(frame, lane_ctrl, cfg, tracker=None):
    # One frame through the robot's LANE_FOLLOW path; returns everything shown.
    # tracker: a LaneTracker kept across frames, so the search window works
    # as it does on the robot.
    result = process_front_frame(frame, False, cfg, tracker)
    lane = fuse_lane(result, None, frame.t, cfg)
    drive = drive_for_state(State.LANE_FOLLOW, lane, None, None, lane_ctrl, result, cfg)
    return result, lane, drive, mix(drive, cfg)


def format_line(frame, result, drive, motor):
    obs = result.lane
    lat = "    --   " if obs.lateral_m is None else f"{obs.lateral_m * 1000:+7.1f}mm"
    head = "   --   " if obs.heading_err_rad is None else f"{obs.heading_err_rad * 57.2958:+6.1f}deg"
    return (f"seq {frame.seq:6d}  t {frame.t:9.3f}  lat {lat}  head {head}  "
            f"conf {obs.conf:.2f}  speed {drive.speed:.2f}  steer {drive.steer:+.2f}  "
            f"L {motor.left:+.2f}  R {motor.right:+.2f}")


def show(result, drive, motor, cfg):
    img = draw_lane(result.view, result.lane, cfg.FRONT_BANDS)
    lines = lane_text(result.lane) + [f"steer {drive.steer:+.2f}  speed {drive.speed:.2f}",
                                      f"L {motor.left:+.2f}  R {motor.right:+.2f}"]
    cv2.imshow(WINDOW, cv2.resize(put_lines(img, lines), None, fx=2, fy=2,
                                  interpolation=cv2.INTER_NEAREST))


def handle_keys(paused, frame_wait_ms):
    # Returns (quit, paused). While paused, waits for n, space or quit.
    while True:
        key = cv2.waitKey(0 if paused else max(1, frame_wait_ms)) & 0xFF
        if key in (ord("q"), 27):
            return True, paused
        if key == ord(" "):
            paused = not paused
            if not paused:
                return False, paused
        elif not paused or key == ord("n"):
            return False, paused


def run(rec_dir, speed, window, cfg):
    if cfg.FRONT_BANDS is None:
        raise SystemExit("FRONT_BANDS is not set in config.py: run python -m tools.calib_ui first")
    lane_ctrl, tracker = LaneController(cfg), LaneTracker()
    prev_t, paused = None, False
    for frame in iter_frames(rec_dir):
        result, _, drive, motor = replay_step(frame, lane_ctrl, cfg, tracker)
        print(format_line(frame, result, drive, motor))
        gap_s = 0.0 if prev_t is None or speed <= 0 else (frame.t - prev_t) / speed
        prev_t = frame.t
        if window:
            show(result, drive, motor, cfg)
            quit_now, paused = handle_keys(paused, int(gap_s * 1000))
            if quit_now:
                break
        elif gap_s > 0:
            time.sleep(gap_s)
    if window:
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Replay a recording with the motors stubbed.")
    p.add_argument("recording", help="folder made by tools/record.py")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed (1 = real time, 0 = as fast as possible)")
    p.add_argument("--no-window", action="store_true", help="print only")
    args = p.parse_args()
    run(args.recording, args.speed, not args.no_window, config)


if __name__ == "__main__":
    main()
