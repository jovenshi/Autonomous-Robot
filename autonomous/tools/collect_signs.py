"""Save sign candidate crops from a recording, for sorting and training.

Run from the autonomous/ folder:
    python -m tools.collect_signs recordings/signs_left_1
    python -m tools.collect_signs recordings/signs_left_1 --all --step 2
    python -m tools.collect_signs recordings/signs_left_1 --out signs_data/left
    python -m tools.collect_signs recordings/signs_none_1 --random 2 --out signs_data/none

Every frame goes through the robot's own candidate code (sign_boxes,
pick_sign, crop_sign), and the crop is saved
as a colour PNG in SIGNS_DATA_DIR/unsorted/, named <recording>_<seq>.png
(with _c<n> for each candidate under --all). The crops are saved before
normalising: tools/train_signs.py augments them and then calls the same
normalise_crop as the robot.

Then sort: move each PNG into pause/, left/, right/, straight/ or none/
under SIGNS_DATA_DIR. Every false candidate belongs in none/. A recording
of one known sign can go straight to its folder with --out; look through
the folder afterwards and move anything that is not that sign to none/.

--random N also saves N sign-sized patches from random places in each frame.
Use it on recordings with no sign in view, into none/: the candidate finder
rarely fires there, and the classifier still needs to see what is not a sign. Keep the
recording name at the start of each file name: train_signs.py holds out
whole recordings by it.
"""
import argparse
import os
import sys

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from perception.signs import crop_sign, pick_sign, scale_box, sign_boxes, sign_roi
from tools.recording import iter_frames

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # autonomous/


def frame_crops(frame, cfg, every_candidate=False):
    # [(suffix, crop)] for one frame: the picked candidate, or all of them.
    boxes = sign_boxes(frame, frame.image.shape[1] / cfg.FRONT_LANE_SIZE[0], cfg)
    chosen = boxes if every_candidate else [pick_sign(boxes)] if boxes else []
    return [(f"_c{i}" if every_candidate else "", crop_sign(frame, box))
            for i, box in enumerate(chosen)]


def random_crops(frame, count, rng, cfg):
    # [(suffix, crop)]: count patches of a sign's size and shape from random
    # places inside SIGN_ROI. Sizes run from the smallest sign box accepted
    # up to half the search region's width.
    scale = frame.image.shape[1] / cfg.FRONT_LANE_SIZE[0]
    rx, ry, rw, rh = scale_box(sign_roi(cfg, *cfg.FRONT_LANE_SIZE), scale)
    out = []
    for i in range(count):
        aspect = rng.uniform(cfg.SIGN_ASPECT_MIN, cfg.SIGN_ASPECT_MAX)
        w = int(rng.uniform((cfg.SIGN_MIN_AREA_PX * aspect) ** 0.5, rw / 2))
        h = min(int(w / aspect), rh)
        x = rx + int(rng.integers(0, rw - w + 1))
        y = ry + int(rng.integers(0, rh - h + 1))
        out.append((f"_r{i}", crop_sign(frame, (x, y, w, h))))
    return out


def collect(rec_dir, out_dir, cfg, every_candidate=False, step=1, random_count=0, seed=0):
    # Writes the crops of a whole recording; returns (frames read, crops saved).
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(rec_dir))
    rng = np.random.default_rng(seed)
    frames = saved = 0
    for frame in iter_frames(rec_dir, step, skip_missing=True):
        frames += 1
        crops = frame_crops(frame, cfg, every_candidate) + random_crops(frame, random_count, rng, cfg)
        for suffix, crop in crops:
            cv2.imwrite(os.path.join(out_dir, f"{name}_{frame.seq:06d}{suffix}.png"), crop)
            saved += 1
    return frames, saved


def main():
    p = argparse.ArgumentParser(description="Save sign candidate crops from a recording.")
    p.add_argument("recording", help="folder made by tools/record.py")
    p.add_argument("--out", default=os.path.join(_HERE, config.SIGNS_DATA_DIR, "unsorted"),
                   help="folder for the crops (default SIGNS_DATA_DIR/unsorted)")
    p.add_argument("--all", action="store_true", help="every candidate, not only the biggest")
    p.add_argument("--step", type=int, default=1, help="use every n-th frame")
    p.add_argument("--random", type=int, default=0, metavar="N",
                   help="also save N random background patches per frame (for none/)")
    args = p.parse_args()
    frames, saved = collect(args.recording, args.out, config, args.all, args.step, args.random)
    print(f"{frames} frames, {saved} crops saved to {args.out}")


if __name__ == "__main__":
    main()
