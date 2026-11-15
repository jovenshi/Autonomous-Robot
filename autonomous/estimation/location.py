"""Where the robot is along one junction pass (docs/quiz3_plan.md, stage 5).

s is the rear axle's distance along the road from the junction centre,
negative before it. On each new camera frame s is set from the best source:
  1. the front camera's crossing-line distance: s = -crossing_dist_m
  2. the front camera's flare distance: s = -(flare_dist_m + FLARE_TO_CENTRE_M)
  3. dash odometry: s += how far our own line's dashes slid towards the
     robot since the last frame (dash_travel). It measures real motion, so
     battery charge, stops and restarts don't matter.
  4. learned speed: s += learned m/s per unit x the speed commanded since
     the last frame, when the dashes can't be matched. The speed is learned
     from dash travel over the last LOCATION_SPEED_FRAMES frames that were
     driven (not stopped), so it follows the battery.
  otherwise s holds ('held').
(The side camera's surface changes are added in stage 8, between 2 and 3.)

Motion is only counted when a camera frame arrives, from the frame before:
counting it on every control loop as well would count it twice.

Within a pass the robot only goes forward, so a camera fix that puts s more
than LOCATION_BACK_MAX_M behind where it already is is ignored: inside the
junction the far corners can read as a flare when the crossing line is
missed (junction_3), and the mat's far end can too (junction_1).

A pass starts after LOCATION_START_FRAMES camera frames in a row with a fix,
so one stray frame can't start one, and ends once s passes
LOCATION_PASS_END_M; the tracker then waits for the next junction. Outside
a pass the zone is CONTINUE and s is None. conf falls with the distance
taken on the learned speed since the last camera fix (dash travel is
measured, so it doesn't lower conf).

The zone is where s + ZONE_REF_X_M falls among ZONE_BOUNDS_M, and only moves
forward within a pass: a camera fix that puts s slightly back cannot step
the zone back. READ_SIGN starts earlier than its bound when the sign is cut
off by the edge of the front camera's view: a sign last seen within
SIGN_CUT_EDGE_PX of the edge, then no sign for SIGN_CUT_READINGS sign
readings in a row (the sign finder drops an outline the view cuts; one
missed reading while the sign is still whole doesn't count). A sign that
drops out mid-view doesn't count.
"""
import bisect
from collections import deque

import numpy as np

from estimation.lane_fusion import is_fresh
from robot_types import LocationObs, Zone

# Zone for each position among ZONE_BOUNDS_M: before the first bound, then
# one per bound.
_ZONE_ORDER = (Zone.CONTINUE, Zone.READ_SIGN, Zone.PAUSE, Zone.ENTERING, Zone.TURN,
               Zone.EXITING, Zone.CONTINUE)


def zone_position(s, cfg):
    # Index into _ZONE_ORDER for a position s.
    return bisect.bisect_right(cfg.ZONE_BOUNDS_M, s + cfg.ZONE_REF_X_M)


def camera_fix(evidence, cfg):
    # (s, source) from one frame's LocationEvidence, or None.
    if evidence.crossing_dist_m is not None:
        return -evidence.crossing_dist_m, "crossing"
    if evidence.flare_dist_m is not None:
        return -(evidence.flare_dist_m + cfg.FLARE_TO_CENTRE_M), "flare"
    return None


def dash_travel(prev, cur, cfg):
    # Metres moved forward between two frames' dash positions (m ahead,
    # either order), or None. Every pairing within -DASH_MAX_BACK_M to
    # DASH_MAX_STEP_M is a candidate shift; the one most dashes agree on
    # (within DASH_MATCH_TOL_M; DASH_MIN_MATCHES at least) wins, and the
    # median of its pairings is returned.
    shifts = [p - c for p in prev for c in cur]
    best = []
    for d in shifts:
        if not -cfg.DASH_MAX_BACK_M <= d <= cfg.DASH_MAX_STEP_M:
            continue
        agree = [x for x in shifts if abs(x - d) <= cfg.DASH_MATCH_TOL_M]
        if len(agree) > len(best):
            best = agree
    if len(best) < cfg.DASH_MIN_MATCHES:
        return None
    return float(np.median(best))


class LocationTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self._learn = deque(maxlen=cfg.LOCATION_SPEED_FRAMES)   # (metres, speed command x s)
        self._prev = None                # last frame's LocationEvidence
        self._cmd_s = 0.0                # speed command x s since that frame
        self._last_t = None
        self.reset()

    def reset(self):
        # Back to waiting for a junction (the learned speed is kept).
        self.s = None
        self.source = "none"
        self.position = 0                # index into _ZONE_ORDER; never decreases in a pass
        self.guessed_m = 0.0             # metres on the learned speed since the last camera fix
        self.fix_run = 0
        self.sign_cut = False            # the sign has left the front camera's view
        self._sign_at_edge = False       # the last sign box seen touched the view's edge
        self._sign_gone = 0              # sign readings in a row with no box since then
        self._sign_seq = None
        self._obs = None

    def update(self, evidence, speed_cmd, now, sign=None):
        # evidence: the newest LocationEvidence (or None); speed_cmd: the
        # speed commanded last loop (0-1); sign: the newest SignObs (or None,
        # also while sign reading is off). Returns the LocationObs.
        self._note_sign(sign)
        if self._last_t is not None:
            self._cmd_s += speed_cmd * max(0.0, now - self._last_t)
        self._last_t = now
        if self._is_new_frame(evidence, now):
            self._on_frame(evidence)
        self._obs = self._observation(now)
        return self._obs

    def get(self):
        return self._obs

    def _note_sign(self, sign):
        # Sets sign_cut when a sign last seen at the view's edge is gone.
        if sign is None or sign.seq == self._sign_seq:
            return
        self._sign_seq = sign.seq
        if sign.bbox is not None:
            x, _, w, _ = sign.bbox
            edge = self.cfg.SIGN_CUT_EDGE_PX
            self._sign_at_edge = x <= edge or x + w >= self.cfg.FRONT_SIZE[0] - edge
            self._sign_gone = 0
        elif self._sign_at_edge:
            self._sign_gone += 1
            if self._sign_gone >= self.cfg.SIGN_CUT_READINGS:
                self.sign_cut, self._sign_at_edge = True, False

    def learned_speed(self):
        # m/s per unit of speed command from recent dash travel, or None.
        cmd_s = sum(c for _, c in self._learn)
        return sum(m for m, _ in self._learn) / cmd_s if cmd_s > 0 else None

    def _is_new_frame(self, evidence, now):
        return (is_fresh(evidence, now, self.cfg)
                and (self._prev is None or evidence.seq != self._prev.seq))

    def _on_frame(self, evidence):
        # Motion since the last frame, then any camera fix, then the pass end.
        moved, how = self._motion(evidence)
        self._prev, self._cmd_s = evidence, 0.0
        fix = camera_fix(evidence, self.cfg)
        self.fix_run = self.fix_run + 1 if fix is not None else 0
        if self.s is None:
            if fix is not None and self.fix_run >= self.cfg.LOCATION_START_FRAMES:
                self.s, self.source = fix
            return
        self.s += moved
        if fix is not None and fix[0] >= self.s - self.cfg.LOCATION_BACK_MAX_M:
            self.s, self.source, self.guessed_m = fix[0], fix[1], 0.0
        else:
            self.source = how
            self.guessed_m += moved if how == "speed" else 0.0
        if self.s >= self.cfg.LOCATION_PASS_END_M:
            self.reset()

    def _motion(self, evidence):
        # (metres moved since the last frame, 'dashes' | 'speed' | 'held');
        # dash travel also teaches the speed when the robot was driven.
        if self._prev is None:
            return 0.0, "held"
        moved = dash_travel(self._prev.dash_dists_m, evidence.dash_dists_m, self.cfg)
        if moved is not None:
            dt = evidence.t - self._prev.t
            if dt > 0 and self._cmd_s / dt >= self.cfg.LOCATION_SPEED_MIN_CMD:
                self._learn.append((moved, self._cmd_s))
            return moved, "dashes"
        speed = self.learned_speed()
        if speed is None:
            return 0.0, "held"
        return speed * self._cmd_s, "speed"

    def _observation(self, now):
        if self.s is None:
            return LocationObs(Zone.CONTINUE, False, False, False, None, 0.0, "none", now)
        self.position = max(self.position, zone_position(self.s, self.cfg),
                            1 if self.sign_cut else 0)
        pos = self.position
        conf = max(0.0, 1.0 - self.guessed_m / self.cfg.LOCATION_DR_CONF_M)
        return LocationObs(zone=_ZONE_ORDER[pos], sign_read_ready=pos >= 1,
                           pause_point_ready=pos >= 2, turn_point_ready=pos >= 4,
                           dist_to_centre_m=-self.s, conf=conf, source=self.source, t=now)
