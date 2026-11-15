"""Where the robot is along one junction pass (docs/quiz3_plan.md, stage 5).

s is the rear axle's distance along the road from the junction centre,
negative before it. Each update sets s from the best source available:
  1. the front camera's crossing-line distance: s = -crossing_dist_m
  2. the front camera's flare distance: s = -(flare_dist_m + FLARE_TO_CENTRE_M)
  3. dead reckoning: s += speed command x SPEED_MPS_PER_UNIT x dt
     (s holds still while SPEED_MPS_PER_UNIT is unmeasured)
A camera fix is moved on by dead reckoning for the frame's age. Within a
pass the robot only goes forward, so a fix that puts s more than
LOCATION_BACK_MAX_M behind where it already is is ignored: inside the
junction the far corners can read as a flare when the crossing line is
missed (junction_3), and the mat's far end can too (junction_1).
(The side camera's surface changes are added in stage 8.)

A pass starts after LOCATION_START_FRAMES camera frames in a row with a fix,
so one stray frame can't start one, and ends once s passes
LOCATION_PASS_END_M; the tracker then waits for the next junction. Outside
a pass the zone is CONTINUE and s is None.

The zone is where s + ZONE_REF_X_M falls among ZONE_BOUNDS_M, and only moves
forward within a pass: a camera fix that puts s slightly back cannot step
the zone back.
"""
import bisect

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


class LocationTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        # Back to waiting for a junction.
        self.s = None
        self.source = "none"
        self.position = 0                # index into _ZONE_ORDER; never decreases in a pass
        self.dr_m = 0.0                  # metres dead-reckoned since the last camera fix
        self.fix_run = 0
        self._last_seq = None
        self._last_t = None
        self._obs = None

    def update(self, evidence, speed_cmd, now):
        # evidence: the newest LocationEvidence (or None); speed_cmd: the
        # speed commanded last loop (0-1). Returns the LocationObs.
        mps = self._speed_mps(speed_cmd)
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        fix = self._new_fix(evidence, now)
        if fix is not None and self.s is not None and fix[0] < self.s - self.cfg.LOCATION_BACK_MAX_M:
            fix = None                   # behind the robot: a false cue
        if fix is not None and self.s is not None:
            self.s, self.source, self.dr_m = fix[0] + mps * (now - evidence.t), fix[1], 0.0
        elif self.s is not None:
            self.s += mps * dt
            self.dr_m += abs(mps * dt)
            self.source = "dead_reckoning"
        elif fix is not None and self.fix_run >= self.cfg.LOCATION_START_FRAMES:
            self.s, self.source = fix[0] + mps * (now - evidence.t), fix[1]
        self._end_pass_if_past()
        self._obs = self._observation(now)
        return self._obs

    def get(self):
        return self._obs

    def _new_fix(self, evidence, now):
        # (s, source) from a frame not used before, counting frames in a row
        # with a fix; None for a repeated, stale or empty frame.
        if not is_fresh(evidence, now, self.cfg) or evidence.seq == self._last_seq:
            return None
        self._last_seq = evidence.seq
        fix = camera_fix(evidence, self.cfg)
        self.fix_run = self.fix_run + 1 if fix is not None else 0
        return fix

    def _speed_mps(self, speed_cmd):
        if self.cfg.SPEED_MPS_PER_UNIT is None:
            return 0.0
        return speed_cmd * self.cfg.SPEED_MPS_PER_UNIT

    def _end_pass_if_past(self):
        if self.s is not None and self.s >= self.cfg.LOCATION_PASS_END_M:
            self.reset()

    def _observation(self, now):
        if self.s is None:
            return LocationObs(Zone.CONTINUE, False, False, False, None, 0.0, "none", now)
        self.position = max(self.position, zone_position(self.s, self.cfg))
        pos = self.position
        conf = max(0.0, 1.0 - self.dr_m / self.cfg.LOCATION_DR_CONF_M)
        return LocationObs(zone=_ZONE_ORDER[pos], sign_read_ready=pos >= 1,
                           pause_point_ready=pos >= 2, turn_point_ready=pos >= 4,
                           dist_to_centre_m=-self.s, conf=conf, source=self.source, t=now)
