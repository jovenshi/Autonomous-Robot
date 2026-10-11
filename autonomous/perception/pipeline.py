"""Camera thread entry points: one call per frame, one result object out.

process_side_frame is added with perception/side_lane.py; junction detection
and sign reading are added with intersection.py and signs.py.
"""
from perception.color import make_front_masks
from perception.front_lane import process_front_lane
from robot_types import FrontResult


def process_front_frame(frame, signs_enabled, cfg):
    # Front thread entry: masks once, then lane. Junction and sign results
    # are None until those modules exist; signs_enabled is accepted now so
    # the signature (and build_system's partial) won't change later.
    masks = make_front_masks(frame, cfg, with_signs=False)
    lane = process_front_lane(masks, cfg, frame.t, frame.seq)
    return FrontResult(
        view=masks.bgr,              # shared with the display; draw on a copy
        lane=lane,
        inter=None,
        sign=None,
        mean_v=masks.mean_v,
        t=frame.t,
        seq=frame.seq,
    )