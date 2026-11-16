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
The side camera adds two fixes (stage 8) where the front camera can't see
the junction any more and the dashes undercount (2-6 cm by the centre on
truth_1 and truth_2): the road fraction of its band (SideResult.road_frac)
crossing SIDE_EDGE_FRAC upwards (ENTRY, the corner's grass leaving its view)
and downwards (EXIT, the far side's grass coming in), at the s in
SIDE_EDGES_M. The crossing's time is interpolated between the two side
frames either side of it, and the estimate at that time between the front
frames around it (_s_at), so neither camera's frame rate nor the time since
the edge counts. The difference is the estimate's error there; with
SIDE_FIXES_ENABLED it is added to s, otherwise it is only reported
(LocationObs.side_edge / side_error_m). Each edge is used once per pass, and
one further than SIDE_FIX_GATE_M from the estimate is ignored (a mat seam or
a shadow crossing the band).

Motion is only counted when a camera frame arrives, from the frame before:
counting it on every control loop as well would count it twice.

Within a pass the robot only goes forward, so a crossing-line fix that puts
s more than LOCATION_BACK_MAX_M behind where it already is is ignored. The
flare is trusted less: a flare fix is ignored once the crossing line has
fixed s in the pass (the line is seen over the same stretch and is more
precise), and a flare fix never moves s back at all. Inside the junction
the far corners can read as a flare when the crossing line is missed
(junction_3), and the mat's far end can too (junction_1); on the robot
(run_20261115_175843) false flares 2-8 cm behind s, each within
LOCATION_BACK_MAX_M, ratcheted s back 25 cm in a second on the approach and
made every zone from PAUSE on late.

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


def band_edge(prev, cur, cfg):
    # (edge name, time) if the band's road fraction crossed SIDE_EDGE_FRAC
    # between two side frames given as (t, road_frac), else None. ENTRY
    # going up (the corner's grass leaving the view), EXIT going down; the
    # time is interpolated between the frames.
    (t0, f0), (t1, f1), edge = prev, cur, cfg.SIDE_EDGE_FRAC
    if (f0 < edge) == (f1 < edge):
        return None
    return ("ENTRY" if f1 > f0 else "EXIT"), t0 + (edge - f0) / (f1 - f0) * (t1 - t0)


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
        self._side_seq = None
        self._side_prev = None           # (t, road_frac) of the last side frame
        self.reset()

    def reset(self):
        # Back to waiting for a junction (the learned speed is kept).
        self.s = None
        self.source = "none"
        self.position = 0                # index into _ZONE_ORDER; never decreases in a pass
        self.guessed_m = 0.0             # metres on the learned speed since the last camera fix
        self.fix_run = 0
        self.crossing_seen = False       # the crossing line has fixed s this pass
        self.sign_cut = False            # the sign has left the front camera's view
        self._sign_at_edge = False       # the last sign box seen touched the view's edge
        self._sign_gone = 0              # sign readings in a row with no box since then
        self._sign_seq = None
        self._history = deque(maxlen=self.cfg.LOCATION_HISTORY_FRAMES)   # (front frame t, s)
        self._side_used = set()          # side edges used this pass
        self.side_edge, self.side_error_m = "", None   # the latest one and the estimate's error there
        self._obs = None

    def update(self, evidence, speed_cmd, now, sign=None, side=None):
        # evidence: the newest LocationEvidence (or None); speed_cmd: the
        # speed commanded last loop (0-1); sign: the newest SignObs (or None,
        # also while sign reading is off); side: the newest SideResult (or
        # None without a side camera). Returns the LocationObs.
        self._note_sign(sign)
        if self._last_t is not None:
            self._cmd_s += speed_cmd * max(0.0, now - self._last_t)
        self._last_t = now
        if self._is_new_frame(evidence, now):
            self._on_frame(evidence)
        self._apply_side_edge(self._side_edge(side, now))
        self._obs = self._observation(now)
        return self._obs

    def _side_edge(self, side, now):
        # (edge name, time) if the band's road fraction crossed
        # SIDE_EDGE_FRAC between the last side frame and this one, else None.
        if not is_fresh(side, now, self.cfg) or side.seq == self._side_seq or side.road_frac is None:
            return None
        self._side_seq = side.seq
        prev, self._side_prev = self._side_prev, (side.t, side.road_frac)
        return band_edge(prev, self._side_prev, self.cfg) if prev is not None else None

    def _apply_side_edge(self, edge):
        # Measures the estimate's error at a side edge and, with
        # SIDE_FIXES_ENABLED, corrects s by it (see the docstring).
        if edge is None or self.s is None or edge[0] in self._side_used:
            return
        then = self._s_at(edge[1])
        if then is None:
            return
        error = self.cfg.SIDE_EDGES_M[edge[0]] - then
        if abs(error) > self.cfg.SIDE_FIX_GATE_M:
            return
        self._side_used.add(edge[0])
        self.side_edge, self.side_error_m = edge[0], error
        if self.cfg.SIDE_FIXES_ENABLED:
            self.s += error
            self.source, self.guessed_m = "side", 0.0
            self._history = deque(((t, s + error) for t, s in self._history), self._history.maxlen)

    def _s_at(self, t):
        # The estimate at time t: interpolated between the front frames
        # around it, carried on at the last frames' rate up to
        # SIDE_EDGE_MAX_GAP_S after the newest. None if t is before the
        # history, or the front frames around it are further apart than
        # SIDE_EDGE_MAX_GAP_S (the front camera stalled: straight_3 has 3 s
        # gaps, over which the estimate is unknown).
        if len(self._history) < 2 or t < self._history[0][0]:
            return None
        ts, ss = zip(*self._history)
        gap = self.cfg.SIDE_EDGE_MAX_GAP_S
        if t > ts[-1]:
            if t - ts[-1] > gap or ts[-1] - ts[-2] > gap or ts[-1] <= ts[-2]:
                return None
            return ss[-1] + (ss[-1] - ss[-2]) / (ts[-1] - ts[-2]) * (t - ts[-1])
        i = bisect.bisect_left(ts, t)
        if i > 0 and ts[i] - ts[i - 1] > gap:
            return None
        return float(np.interp(t, ts, ss))

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
                self.crossing_seen = fix[1] == "crossing"
                self._history.append((evidence.t, self.s))
            return
        self.s += moved
        if self._accepts(fix):
            self.s, self.source, self.guessed_m = fix[0], fix[1], 0.0
            self.crossing_seen = self.crossing_seen or fix[1] == "crossing"
        else:
            self.source = how
            self.guessed_m += moved if how == "speed" else 0.0
        self._history.append((evidence.t, self.s))
        if self.s >= self.cfg.LOCATION_PASS_END_M:
            self.reset()

    def _accepts(self, fix):
        # Whether a camera fix replaces s (see the docstring).
        if fix is None:
            return False
        if fix[1] == "flare":
            return not self.crossing_seen and fix[0] >= self.s
        return fix[0] >= self.s - self.cfg.LOCATION_BACK_MAX_M

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
                           dist_to_centre_m=-self.s, conf=conf, source=self.source, t=now,
                           side_edge=self.side_edge, side_error_m=self.side_error_m)
