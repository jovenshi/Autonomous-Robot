"""Measure the side camera's edges (SIDE_EDGES_M) on two-camera recordings.

The side camera's band (SIDE_BOX) crosses SIDE_EDGE_FRAC road going into
the junction (ENTRY) and coming out (EXIT); estimation/location.py fixes the
robot's position there. This tool finds those crossings in recordings made
with `python -m tools.record --side` and prints the rear axle's s at each.

s comes from one of two places:
  * the robot's own estimate (default): the front frames go through the
    pipeline and LocationTracker, pushed by hand (no speed command), with
    no side camera (so the side fixes don't measure themselves);
  * a tape (--tape START): the robot was held still at tape-measured spots
    every SIDE_TAPE_STEP_M from START (rear axle to the junction centre,
    negative before it). The holds are found from the front camera's dash
    odometry (still for SIDE_TAPE_HOLD_S), and s between them is the
    odometry stretched to fit the tape. Use --skip for holds that were not
    tape spots (a wheel locking up, a stumble), counted from 0.

The default answers "where does the robot think the edges are"; --tape
answers "where are they really", which is what SIDE_EDGES_M needs.

Run from the autonomous/ folder:
    python -m tools.side_fixes recordings/truth_2 --tape -1.10
    python -m tools.side_fixes recordings/truth_1 --tape -1.10 --skip 2
    python -m tools.side_fixes recordings/junction_side_1 --save replay_out/side_1

--save writes each side frame with the band, its road fraction and s.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from estimation.location import LocationTracker, band_edge, dash_travel  # noqa: E402
from perception.front_lane import LaneTracker  # noqa: E402
from perception.pipeline import process_front_frame  # noqa: E402
from perception.side_surface import process_side_frame  # noqa: E402
from tools.recording import iter_frames  # noqa: E402
from ui.overlay import draw_annotations  # noqa: E402


def front_track(rec_dir, cfg):
    # (times, s) of every front frame during a junction pass, from the
    # robot's own estimate.
    lane, tracker, ts, ss = LaneTracker(), LocationTracker(cfg), [], []
    for frame in iter_frames(rec_dir, 1, skip_missing=True):
        obs = tracker.update(process_front_frame(frame, False, cfg, lane).location, 0.0, frame.t)
        if obs.dist_to_centre_m is not None:
            ts.append(frame.t)
            ss.append(-obs.dist_to_centre_m)
    return np.array(ts), np.array(ss)


def odometry(rec_dir, cfg):
    # (times, metres travelled) of every front frame, from dash travel.
    lane, prev, total, ts, odo = LaneTracker(), None, 0.0, [], []
    for frame in iter_frames(rec_dir, 1, skip_missing=True):
        ev = process_front_frame(frame, False, cfg, lane).location
        step = dash_travel(prev.dash_dists_m, ev.dash_dists_m, cfg) if prev is not None else None
        total, prev = total + (step or 0.0), ev
        ts.append(frame.t)
        odo.append(total)
    return np.array(ts), np.array(odo)


def find_holds(ts, odo, cfg):
    # Median odometry of each stretch still (within SIDE_TAPE_STILL_M) for
    # at least SIDE_TAPE_HOLD_S.
    holds, i = [], 0
    while i < len(ts):
        j = i
        while j + 1 < len(ts) and abs(odo[j + 1] - odo[i]) < cfg.SIDE_TAPE_STILL_M:
            j += 1
        if ts[j] - ts[i] >= cfg.SIDE_TAPE_HOLD_S:
            holds.append(float(np.median(odo[i:j + 1])))
            i = j + 1
        else:
            i += 1
    return holds


def tape_track(rec_dir, cfg, start, skip=()):
    # (times, s) of every front frame from the tape spots: the holds (minus
    # skip) are at start, start + SIDE_TAPE_STEP_M, ...; odometry between.
    ts, odo = odometry(rec_dir, cfg)
    holds = [h for i, h in enumerate(find_holds(ts, odo, cfg)) if i not in set(skip)]
    if len(holds) < 2:
        return np.array([]), np.array([])
    tape = start + cfg.SIDE_TAPE_STEP_M * np.arange(len(holds))
    inside = (odo >= holds[0]) & (odo <= holds[-1])
    return ts[inside], np.interp(odo[inside], holds, tape)


def side_edges(rec_dir, track, cfg, save_dir=None):
    # [(edge name, s)] for every band crossing while the track is known.
    ts, ss = track
    prev, edges = None, []
    for frame in iter_frames(os.path.join(rec_dir, "side"), 1, skip_missing=True):
        if ts.size == 0 or not ts[0] <= frame.t <= ts[-1]:
            continue
        result = process_side_frame(frame, cfg)
        if save_dir is not None:
            save_picture(save_dir, frame, result, float(np.interp(frame.t, ts, ss)))
        cur = (frame.t, result.road_frac)
        edge = band_edge(prev, cur, cfg) if prev is not None else None
        if edge is not None:
            edges.append((edge[0], float(np.interp(edge[1], ts, ss))))
        prev = cur
    return edges


def save_picture(save_dir, frame, result, s):
    os.makedirs(save_dir, exist_ok=True)
    img = frame.image.copy()
    draw_annotations(img, result.annotations)
    cv2.putText(img, f"s {s:+.3f}", (4, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(save_dir, f"side_{frame.seq:06d}.jpg"), img)


def near_expected(edges, cfg):
    # The edges within SIDE_FIX_GATE_M of SIDE_EDGES_M: {name: [s, ...]}.
    out = {}
    for name, s in edges:
        expected = cfg.SIDE_EDGES_M.get(name)
        if expected is not None and abs(s - expected) <= cfg.SIDE_FIX_GATE_M:
            out.setdefault(name, []).append(s)
    return out


def summary_lines(per_recording, cfg):
    # One line per edge: median, spread and the value in config.py.
    lines = []
    for name, expected in cfg.SIDE_EDGES_M.items():
        values = [s for found in per_recording for s in found.get(name, [])[:1]]
        if not values:
            lines.append(f"{name}: not seen near {expected:+.2f}")
            continue
        lines.append(f"{name}: median {np.median(values):+.3f} m over {len(values)} recordings "
                     f"({min(values):+.3f} to {max(values):+.3f}); SIDE_EDGES_M has {expected:+.2f}")
    return lines


def main():
    p = argparse.ArgumentParser(description="Measure the side camera's edges.")
    p.add_argument("recordings", nargs="+", help="folders made by tools/record.py --side")
    p.add_argument("--tape", type=float, default=None, metavar="START",
                   help="held at tape spots from START (m, negative before the centre)")
    p.add_argument("--skip", default="", help="comma-separated holds that were not tape spots")
    p.add_argument("--save", default=None, metavar="FOLDER", help="write annotated side frames")
    args = p.parse_args()
    skip = [int(x) for x in args.skip.split(",") if x]
    per_recording = []
    for rec in args.recordings:
        track = (tape_track(rec, config, args.tape, skip) if args.tape is not None
                 else front_track(rec, config))
        edges = side_edges(rec, track, config, args.save)
        print(f"{rec}: " + (", ".join(f"{name} at {s:+.3f}" for name, s in edges) or "no edges"))
        per_recording.append(near_expected(edges, config))
    for line in summary_lines(per_recording, config):
        print(line)


if __name__ == "__main__":
    main()
