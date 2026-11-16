"""Camera thread entry points: one call per frame, one result object out.

process_side_frame is added with perception/side_lane.py. Location evidence
and signs (process_vision) are read in the front camera thread, or, with
VISION_WORKER_ENABLED, in the vision worker process (perception/worker.py),
which calls the same process_vision on the frames the camera thread shares.
"""
from perception.color import make_front_masks
from perception.front_lane import process_front_lane
from perception.location import process_location
from perception.signs import process_signs
from robot_types import FrontResult, VisionResult


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


def process_vision(frame, line_px, signs_enabled, cfg, sign_model=None, pacer=None, masks=None):
    # Location evidence and (when signs_enabled, at SIGN_HZ) the sign
    # reading for one frame -> VisionResult. line_px: the lane's fitted line
    # (for the dashes), or None. masks: the frame's FrontMasks if already
    # made (camera thread); the worker makes its own.
    if masks is None:
        masks = make_front_masks(frame, cfg, with_signs=False)
    return VisionResult(
        location=process_location(masks, cfg, frame.t, frame.seq, line_px),
        sign=_read_sign(frame, masks, signs_enabled, sign_model, pacer, cfg),
        t=frame.t, seq=frame.seq)


def process_front_frame(frame, signs_enabled, cfg, tracker=None, sign_model=None, pacer=None,
                        with_vision=True):
    # Front thread entry: masks once, then lane, then (with_vision) location
    # evidence and the sign reader. with_vision False when the vision
    # worker does that work: location and sign are then None.
    # tracker: the camera thread's LaneTracker (search window), or None.
    # sign_model: SignModel, or None to report candidates only.
    # pacer: the camera thread's SignPacer, or None to read every frame.
    masks = make_front_masks(frame, cfg, with_signs=False)
    lane = process_front_lane(masks, cfg, frame.t, frame.seq, tracker)
    vision = (process_vision(frame, lane.line_px, signs_enabled, cfg, sign_model, pacer, masks)
              if with_vision else None)
    return FrontResult(
        view=masks.bgr,              # shared with the display; draw on a copy
        lane=lane,
        location=vision.location if vision is not None else None,
        sign=vision.sign if vision is not None else None,
        mean_v=masks.mean_v,
        t=frame.t,
        seq=frame.seq,
    )