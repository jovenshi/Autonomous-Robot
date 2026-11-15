"""Measure the side camera's fixes (SIDE_FIXES_M) on two-camera recordings.

For each recording made with `python -m tools.record --side`, the front
frames go through the robot's own pipeline and LocationTracker (pushed by
hand: no speed command), which gives the rear axle's s at every moment. Each
side frame (same clock) gets the s at its time, and its SIDE_BOX is classed
as on the robot. Every settled surface change (SIDE_SURFACE_FRAMES in a row)
is printed with its s; at the end, the median s of each change across the
recordings, next to SIDE_FIXES_M. Changes far from the junction (mat seams,
shadows) are printed too, to show what SIDE_FIX_GATE_M must reject.

Run from the autonomous/ folder:
    python -m tools.side_fixes recordings/junction_side_1 recordings/junction_side_2
    python -m tools.side_fixes recordings/junction_side_1 --save replay_out/side_1

--save writes each side frame with the box, its class and s as a JPEG, to
choose SIDE_BOX by eye.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from estimation.location import LocationTracker  # noqa: E402
from perception.front_lane import LaneTracker  # noqa: E402
from perception.pipeline import process_front_frame  # noqa: E402
from perception.side_surface import process_side_frame  # noqa: E402
from tools.recording import iter_frames  # noqa: E402
from ui.overlay import draw_annotations  # noqa: E402


def front_track(rec_dir, cfg):
    # (times, s) of every front frame during a junction pass.
    lane, tracker, ts, ss = LaneTracker(), LocationTracker(cfg), [], []
    for frame in iter_frames(rec_dir, 1, skip_missing=True):
        obs = tracker.update(process_front_frame(frame, False, cfg, lane).location, 0.0, frame.t)
        if obs.dist_to_centre_m is not None:
            ts.append(frame.t)
            ss.append(-obs.dist_to_centre_m)
    return np.array(ts), np.array(ss)


def side_changes(rec_dir, cfg, save_dir=None):
    # [(new surface name, s)] for every settled change of the side box,
    # with s from the front camera; side frames outside the pass are skipped.
    ts, ss = front_track(rec_dir, cfg)
    settled, run, changes = None, (None, 0), []
    for frame in iter_frames(os.path.join(rec_dir, "side"), 1, skip_missing=True):
        if ts.size == 0 or not ts[0] <= frame.t <= ts[-1]:
            continue
        s = float(np.interp(frame.t, ts, ss))
        result = process_side_frame(frame, cfg)
        if save_dir is not None:
            save_picture(save_dir, frame, result, s)
        if result.surface is None:
            run = (None, 0)                  # mixed: the run is broken, as on the robot
            continue
        run = (result.surface, run[1] + 1 if result.surface == run[0] else 1)
        if run[1] >= cfg.SIDE_SURFACE_FRAMES and result.surface != settled:
            if settled is not None:
                changes.append((result.surface.name, s))
            settled = result.surface
    return changes


def save_picture(save_dir, frame, result, s):
    os.makedirs(save_dir, exist_ok=True)
    img = frame.image.copy()
    draw_annotations(img, result.annotations)
    cv2.putText(img, f"s {s:+.3f}", (4, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(save_dir, f"side_{frame.seq:06d}.jpg"), img)


def near_junction(changes, cfg):
    # The changes within SIDE_FIX_GATE_M of where SIDE_FIXES_M expects them:
    # {surface name: [s, ...]}.
    out = {}
    for name, s in changes:
        expected = cfg.SIDE_FIXES_M.get(name)
        if expected is not None and abs(s - expected) <= cfg.SIDE_FIX_GATE_M:
            out.setdefault(name, []).append(s)
    return out


def summary_lines(per_recording, cfg):
    # One line per fix: median, spread and the value in config.py.
    lines = []
    for name, expected in cfg.SIDE_FIXES_M.items():
        values = [s for found in per_recording for s in found.get(name, [])[:1]]
        if not values:
            lines.append(f"{name}: not seen near {expected:+.2f}")
            continue
        lines.append(f"{name}: median {np.median(values):+.3f} m over {len(values)} recordings "
                     f"({min(values):+.3f} to {max(values):+.3f}); SIDE_FIXES_M has {expected:+.2f}")
    return lines


def main():
    p = argparse.ArgumentParser(description="Measure the side camera's fixes.")
    p.add_argument("recordings", nargs="+", help="folders made by tools/record.py --side")
    p.add_argument("--save", default=None, metavar="FOLDER", help="write annotated side frames")
    args = p.parse_args()
    per_recording = []
    for rec in args.recordings:
        changes = side_changes(rec, config, args.save)
        print(f"{rec}: " + ", ".join(f"{name} at {s:+.3f}" for name, s in changes))
        per_recording.append(near_junction(changes, config))
    for line in summary_lines(per_recording, config):
        print(line)


if __name__ == "__main__":
    main()
