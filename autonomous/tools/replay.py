"""Replay a recording through the front pipeline, fusion and controller, with
the motors stubbed: nothing is sent to the Sabertooth.

Each frame goes through the same code as the robot in LANE_FOLLOW:
process_front_frame -> fuse_lane -> drive_for_state -> mix, using the
recording's own timestamps. Every frame is also read for signs
(process_signs with the model in SIGN_MODEL_PATH; with no model yet, only
the candidates are shown). One line per frame is printed; a window shows the
lane drawing, the sign search region and candidate box, and beside it the
normalised crop with its label and class scores (needs the Pi desktop;
--no-window to only print).

Run from the robot/ folder:
    python -m tools.replay recordings/front_20261011_140000
    python -m tools.replay recordings/front_20261011_140000 --speed 0 --no-window
    python -m tools.replay recordings/signs_left_1 --model models/sign_svm.npz

Keys: space pause/resume, n next frame while paused, q or Esc quit.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from control.command import drive_for_state
from control.lane_controller import LaneController
from control.mixer import mix
from estimation.lane_fusion import fuse_lane
from perception.color import make_front_masks
from perception.front_lane import LaneTracker
from perception.pipeline import process_front_frame
from perception.signs import crop_sign, load_sign_model, normalise_crop, process_signs, sign_label
from robot_types import State
from tools.debug_draw import draw_lane, lane_text, put_lines
from tools.recording import iter_frames
from ui.overlay import draw_annotations

WINDOW = "replay"


def replay_step(frame, lane_ctrl, cfg, tracker=None):
    # One frame through the robot's LANE_FOLLOW path; returns everything shown.
    # tracker: a LaneTracker kept across frames, so the search window works
    # as it does on the robot.
    result = process_front_frame(frame, False, cfg, tracker)
    lane = fuse_lane(result, None, frame.t, cfg)
    drive = drive_for_state(State.LANE_FOLLOW, lane, None, None, lane_ctrl, result, cfg)
    return result, lane, drive, mix(drive, cfg)


def sign_step(frame, model, cfg):
    # One frame through the robot's sign reader -> (SignObs, normalised crop
    # or None). The crop is cut again from the reported box for display.
    obs = process_signs(frame, make_front_masks(frame, cfg, with_signs=True), model, cfg)
    crop = None if obs.bbox is None else normalise_crop(crop_sign(frame, obs.bbox, 1.0), cfg)
    return obs, crop


def format_line(frame, result, drive, motor, sign=None, cfg=config):
    obs = result.lane
    lat = "    --   " if obs.lateral_m is None else f"{obs.lateral_m * 1000:+7.1f}mm"
    head = "   --   " if obs.heading_err_rad is None else f"{obs.heading_err_rad * 57.2958:+6.1f}deg"
    line = (f"seq {frame.seq:6d}  t {frame.t:9.3f}  lat {lat}  head {head}  "
            f"conf {obs.conf:.2f}  speed {drive.speed:.2f}  steer {drive.steer:+.2f}  "
            f"L {motor.left:+.2f}  R {motor.right:+.2f}")
    if sign is not None:
        box = "no candidate" if sign.bbox is None else f"margin {sign.score:5.2f}"
        line += f"  sign {sign_label(sign.sign, cfg):<8} {box}"
    return line


def sign_panel(sign, crop, size, cfg):
    # Square panel: the normalised crop enlarged, with the label and every
    # class score (best first).
    panel = np.zeros((size, size, 3), np.uint8)
    if crop is not None:
        panel[:] = cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST),
                                cv2.COLOR_GRAY2BGR)
    lines = [f"sign: {sign_label(sign.sign, cfg)}" if crop is not None else "no candidate"]
    lines += [f"{name.lower():<9}{score:+6.2f}"
              for name, score in sorted(sign.scores, key=lambda item: -item[1])]
    return put_lines(panel, lines)


def show(result, drive, motor, cfg, sign=None, crop=None):
    img = draw_lane(result.view, result.lane, cfg.FRONT_BANDS)
    if sign is not None:
        draw_annotations(img, sign.annotations)
    lines = lane_text(result.lane) + [f"steer {drive.steer:+.2f}  speed {drive.speed:.2f}",
                                      f"L {motor.left:+.2f}  R {motor.right:+.2f}"]
    img = put_lines(img, lines)
    if sign is not None:
        img = np.hstack([img, sign_panel(sign, crop, img.shape[0], cfg)])
    cv2.imshow(WINDOW, cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


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


def run(rec_dir, speed, window, cfg, model=None):
    # model: a SignModel, or None to show sign candidates only.
    if cfg.FRONT_BANDS is None:
        raise SystemExit("FRONT_BANDS is not set in config.py: run python -m tools.calib_ui first")
    lane_ctrl, tracker = LaneController(cfg), LaneTracker()
    prev_t, paused = None, False
    for frame in iter_frames(rec_dir):
        result, _, drive, motor = replay_step(frame, lane_ctrl, cfg, tracker)
        sign, crop = sign_step(frame, model, cfg)
        print(format_line(frame, result, drive, motor, sign, cfg))
        gap_s = 0.0 if prev_t is None or speed <= 0 else (frame.t - prev_t) / speed
        prev_t = frame.t
        if window:
            show(result, drive, motor, cfg, sign, crop)
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
    p.add_argument("--model", default=config.SIGN_MODEL_PATH, help="sign model (.npz)")
    args = p.parse_args()
    run(args.recording, args.speed, not args.no_window, config, open_model(args.model, config))


def open_model(path, cfg):
    # The sign model, or None (candidates only) with a note when it can't be read.
    try:
        return load_sign_model(path, cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"{e}\nShowing sign candidates only.")
        return None


if __name__ == "__main__":
    main()
