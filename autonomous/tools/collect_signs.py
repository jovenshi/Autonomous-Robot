"""Save sign candidate crops from a recording, for sorting and training.

Run from the autonomous/ folder:
    python -m tools.collect_signs recordings/signs_left_1
    python -m tools.collect_signs recordings/signs_left_1 --all --step 2

Every frame goes through the robot's own candidate code (make_front_masks,
find_sign_candidates, pick_sign, pad_box, crop_sign), and the crop is saved
as a colour PNG in SIGNS_DATA_DIR/unsorted/, named <recording>_<seq>.png
(with _c<n> for each candidate under --all). The crops are saved before
normalising: tools/train_signs.py augments them and then calls the same
normalise_crop as the robot.

Then sort: move each PNG into pause/, left/, right/, straight/ or none/
under SIGNS_DATA_DIR. Every false candidate belongs in none/. Keep the
recording name at the start of each file name: train_signs.py holds out
whole recordings by it.
"""
import argparse
import os
import sys

import cv2

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from perception.color import make_front_masks
from perception.signs import crop_sign, find_sign_candidates, pad_box, pick_sign
from tools.recording import iter_frames

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # autonomous/


def frame_crops(frame, cfg, every_candidate=False):
    # [(suffix, crop)] for one frame: the picked candidate, or all of them.
    masks = make_front_masks(frame, cfg, with_signs=True)
    candidates = find_sign_candidates(masks.white, masks.road, cfg)
    chosen = candidates if every_candidate else [pick_sign(candidates)] if candidates else []
    height, width = masks.white.shape[:2]
    out = []
    for i, box in enumerate(chosen):
        padded = pad_box(box, cfg.SIGN_BBOX_PAD_FRAC, width, height)
        out.append((f"_c{i}" if every_candidate else "", crop_sign(frame, padded, masks.scale)))
    return out


def collect(rec_dir, out_dir, cfg, every_candidate=False, step=1):
    # Writes the crops of a whole recording; returns (frames read, crops saved).
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(rec_dir))
    frames = saved = 0
    for frame in iter_frames(rec_dir, step):
        frames += 1
        for suffix, crop in frame_crops(frame, cfg, every_candidate):
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
    args = p.parse_args()
    frames, saved = collect(args.recording, args.out, config, args.all, args.step)
    print(f"{frames} frames, {saved} crops saved to {args.out}")


if __name__ == "__main__":
    main()
