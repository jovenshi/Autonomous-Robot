"""Replay a recording through the front pipeline, fusion and controller, with
the motors stubbed: nothing is sent to the Sabertooth.

Each frame goes through the same code as the robot in LANE_FOLLOW:
process_front_frame -> fuse_lane -> drive_for_state -> mix, using the
recording's own timestamps. Every frame is also read for signs
(process_signs with the model in SIGN_MODEL_PATH; with no model yet, only
the candidates are shown). One line per frame is printed; a window shows the
lane drawing, the sign search region and candidate box, and beside it the
normalised crop with its label and class scores (needs the Pi desktop;
--no-window to only print). The junction cues (crossing line, flare) are
drawn with their distances, and a LocationTracker gives s and the zone; a
pushed recording has no speed command, so s only moves on camera fixes. At
the end, the frames that saw both cues give the measured FLARE_TO_CENTRE_M.

Run from the robot/ folder:
    python -m tools.replay recordings/front_20261011_140000
    python -m tools.replay recordings/front_20261011_140000 --speed 0 --no-window
    python -m tools.replay recordings/signs_left_1 --model models/sign_svm.npz
    python -m tools.replay recordings/signs_left_1 --save replay_out/signs_left_1 --step 15
    python -m tools.replay recordings/junction_1 --speed 0 --no-window

With no desktop (SSH, VS Code remote), --save writes the same picture for
every --step-th frame as a JPEG in that folder: open the folder in the
editor to look.

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
from estimation.location import LocationTracker
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
    obs = process_signs(frame, make_front_masks(frame, cfg), model, cfg)
    crop = None if obs.bbox is None else normalise_crop(crop_sign(frame, obs.bbox), cfg)
    return obs, crop


def location_text(result, loc):
    # 'cross 1.23 flare --   s -1.23 crossing READ_SIGN'
    ev = result.location
    cue = lambda d: "  -- " if d is None else f"{d:5.2f}"
    s = "  --  " if loc.dist_to_centre_m is None else f"{-loc.dist_to_centre_m:+6.2f}"
    return (f"cross {cue(ev.crossing_dist_m)} flare {cue(ev.flare_dist_m)}  "
            f"s {s} {loc.source:<14} {loc.zone.name}")


def flare_offsets(results):
    # crossing - flare distance on every frame that saw both: the measured
    # FLARE_TO_CENTRE_M (as the flare test finds it).
    return [r.location.crossing_dist_m - r.location.flare_dist_m for r in results
            if r.location.crossing_dist_m is not None and r.location.flare_dist_m is not None]


def print_flare_summary(offsets, cfg):
    if not offsets:
        print("No frame saw both the crossing line and the flare.")
        return
    print(f"Flare to centre over {len(offsets)} frames: median {np.median(offsets):.3f} m "
          f"(min {min(offsets):.3f}, max {max(offsets):.3f}); "
          f"FLARE_TO_CENTRE_M is {cfg.FLARE_TO_CENTRE_M}")


def format_line(frame, result, drive, motor, sign=None, cfg=config, loc=None):
    obs = result.lane
    lat = "    --   " if obs.lateral_m is None else f"{obs.lateral_m * 1000:+7.1f}mm"
    head = "   --   " if obs.heading_err_rad is None else f"{obs.heading_err_rad * 57.2958:+6.1f}deg"
    line = (f"seq {frame.seq:6d}  t {frame.t:9.3f}  lat {lat}  head {head}  "
            f"conf {obs.conf:.2f}  speed {drive.speed:.2f}  steer {drive.steer:+.2f}  "
            f"L {motor.left:+.2f}  R {motor.right:+.2f}")
    if sign is not None:
        box = "no candidate" if sign.bbox is None else f"margin {sign.score:5.2f}"
        line += f"  sign {sign_label(sign.sign, cfg):<8} {box}"
    if loc is not None:
        line += "  " + location_text(result, loc)
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


def compose(result, drive, motor, cfg, sign=None, crop=None, loc=None):
    # The picture for one frame: lane drawing with the location and sign
    # annotations, and the sign panel beside it. Twice the view size.
    img = draw_lane(result.view, result.lane, cfg.FRONT_BANDS)
    if result.location is not None:
        draw_annotations(img, result.location.annotations)
    if sign is not None:
        draw_annotations(img, sign.annotations)
    lines = lane_text(result.lane) + [f"steer {drive.steer:+.2f}  speed {drive.speed:.2f}",
                                      f"L {motor.left:+.2f}  R {motor.right:+.2f}"]
    if loc is not None:
        lines.append(f"s {'--' if loc.dist_to_centre_m is None else f'{-loc.dist_to_centre_m:+.2f}'}"
                     f"  {loc.zone.name}")
    img = put_lines(img, lines)
    if sign is not None:
        img = np.hstack([img, sign_panel(sign, crop, img.shape[0], cfg)])
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)


def save_picture(save_dir, frame, picture):
    # <save_dir>/frame_<seq>.jpg
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"frame_{frame.seq:06d}.jpg"), picture)


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


def run(rec_dir, speed, window, cfg, model=None, save_dir=None, step=1):
    # model: a SignModel, or None to show sign candidates only. save_dir:
    # folder for one picture per frame shown (no desktop needed). step: use
    # every step-th frame.
    if cfg.FRONT_BANDS is None:
        raise SystemExit("FRONT_BANDS is not set in config.py: run python -m tools.calib_ui first")
    lane_ctrl, tracker, locator = LaneController(cfg), LaneTracker(), LocationTracker(cfg)
    prev_t, paused, offsets = None, False, []
    for frame in iter_frames(rec_dir, step, skip_missing=True):
        result, _, drive, motor = replay_step(frame, lane_ctrl, cfg, tracker)
        sign, crop = sign_step(frame, model, cfg)
        loc = locator.update(result.location, 0.0, frame.t)     # pushed: no speed command
        offsets += flare_offsets([result])
        print(format_line(frame, result, drive, motor, sign, cfg, loc))
        gap_s = 0.0 if prev_t is None or speed <= 0 else (frame.t - prev_t) / speed
        prev_t = frame.t
        if save_dir is not None:
            save_picture(save_dir, frame, compose(result, drive, motor, cfg, sign, crop, loc))
        if window:
            cv2.imshow(WINDOW, compose(result, drive, motor, cfg, sign, crop, loc))
            quit_now, paused = handle_keys(paused, int(gap_s * 1000))
            if quit_now:
                break
        elif gap_s > 0:
            time.sleep(gap_s)
    print_flare_summary(offsets, cfg)
    if window:
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Replay a recording with the motors stubbed.")
    p.add_argument("recording", help="folder made by tools/record.py")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed (1 = real time, 0 = as fast as possible)")
    p.add_argument("--no-window", action="store_true", help="print only")
    p.add_argument("--model", default=config.SIGN_MODEL_PATH, help="sign model (.npz)")
    p.add_argument("--save", default=None, metavar="FOLDER",
                   help="write a picture per frame here instead of opening a window")
    p.add_argument("--step", type=int, default=1, help="use every n-th frame")
    args = p.parse_args()
    window = not args.no_window and args.save is None
    run(args.recording, 0.0 if args.save else args.speed, window, config,
        open_model(args.model, config), args.save, args.step)


def open_model(path, cfg):
    # The sign model, or None (candidates only) with a note when it can't be read.
    try:
        return load_sign_model(path, cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"{e}\nShowing sign candidates only.")
        return None


if __name__ == "__main__":
    main()
