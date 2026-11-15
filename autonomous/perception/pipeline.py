"""Camera thread entry points: one call per frame, one result object out.

process_side_frame is added with perception/side_lane.py. Location evidence
and signs are read here for now (docs/quiz3_plan.md: they move to the vision
worker in stage 7).
"""
from perception.color import make_front_masks
from perception.front_lane import process_front_lane
from perception.location import process_location
from perception.signs import process_signs
from robot_types import FrontResult


class SignPacer:
    # Lets sign reading run at SIGN_HZ, not on every frame, and keeps the
    # newest reading so the frames in between still carry it. One per
    # camera thread, like LaneTracker.
    def __init__(self):
        self._last_t = None
        self.latest = None             # newest SignObs, or None while signs are off

    def due(self, t, cfg):
        # True (and the clock restarts) when 1 / SIGN_HZ has passed since
        # the last sign frame, to within half a camera frame: at 30 fps and
        # 10 Hz that is every third frame, whatever the timestamps' jitter.
        if self._last_t is not None and t - self._last_t < 1.0 / cfg.SIGN_HZ - 0.5 / cfg.CAM_FPS:
            return False
        self._last_t = t
        return True


def _read_sign(frame, masks, signs_enabled, sign_model, pacer, cfg):
    # The frame's SignObs. Between sign frames the newest reading is passed
    # on again (its seq and t say which frame it came from): the slot holds
    # one result, so a reading made on one frame in three would otherwise be
    # overwritten before the control loop or the display saw it.
    if not signs_enabled:
        if pacer is not None:
            pacer.latest = None
        return None
    if pacer is None:
        return process_signs(frame, masks, sign_model, cfg)
    if pacer.due(frame.t, cfg):
        pacer.latest = process_signs(frame, masks, sign_model, cfg)
    return pacer.latest


def process_front_frame(frame, signs_enabled, cfg, tracker=None, sign_model=None, pacer=None):
    # Front thread entry: masks once, then lane, location evidence and
    # (when signs_enabled, at SIGN_HZ) the sign reader.
    # tracker: the camera thread's LaneTracker (search window), or None.
    # sign_model: SignModel, or None to report candidates only.
    # pacer: the camera thread's SignPacer, or None to read every frame.
    masks = make_front_masks(frame, cfg, with_signs=False)
    lane = process_front_lane(masks, cfg, frame.t, frame.seq, tracker)
    sign = _read_sign(frame, masks, signs_enabled, sign_model, pacer, cfg)
    return FrontResult(
        view=masks.bgr,              # shared with the display; draw on a copy
        lane=lane,
        location=process_location(masks, cfg, frame.t, frame.seq),
        sign=sign,
        mean_v=masks.mean_v,
        t=frame.t,
        seq=frame.seq,
    )