# Junction plan: DECIDE, STOP_HOLD, TURN, CROSS, REACQUIRE

**Status:** not built (saved 2026-10-11, side camera added 2026-10-25). These
states still go to `Behaviour._not_built` and raise. Current goal is straight
driving in LANE_FOLLOW with the IMU; pick this up when junction detection exists.

**Needs first:** junction detection filling `FrontResult.inter`, the side
camera's opening flag (`SideLaneObs.opening`), and a sign voter. The side
camera's lane fusion (design doc, "Side camera in decisions", build steps 1-4)
should be working before junctions start.

**Already in place:** `turn_steer`, `turn_done`, `vision_turn_steer`,
`hold_steer` (control/turn_controller.py); `RoadYawTracker.after_turn`;
`choose_action`; stale IMU -> SAFE_STOP in TURN and CROSS only;
`drift_request` set in IDLE and STOP_HOLD.

## Side camera roles at junctions

The front camera **arms** (APPROACH), the side camera **triggers** (DECIDE).
No single camera frame decides anything:

- `opening` counts only in APPROACH, and only after `SIDE_OPENING_FRAMES`
  frames in a row. A shadow or a worn patch of green in LANE_FOLLOW is ignored.
- The edge returning counts only in CROSS, and only after
  `SIDE_EDGE_BACK_FRAMES` frames in a row.
- Behaviour keeps both counters; perception still reports one bool per frame.

## Flow

```
LANE_FOLLOW -> APPROACH -> DECIDE -> STOP_HOLD -> DECIDE (stop served)
                                  -> TURN  (LEFT / RIGHT) -> REACQUIRE -> LANE_FOLLOW
                                  -> CROSS (STRAIGHT)     -> REACQUIRE -> LANE_FOLLOW
                                  -> SAFE_STOP (RIGHT voted, no opening seen; proposed)
```

## DECIDE

- Entered when APPROACH ends on a position fix. The fix that ended APPROACH
  is the **advance cue**: `opening` (side camera saw the road leave to the
  right for `SIDE_OPENING_FRAMES` in a row) or `line` (front camera saw the
  crossing line).
- `_enter(DECIDE)` records the cue and `opening_seen` (bool); TURN reads the cue.
- `choose_action(sign, stop_done, cfg)`: STOP -> STOP_HOLD, LEFT/RIGHT ->
  TURN, STRAIGHT -> CROSS.
- **Right-turn check (proposed):** action RIGHT with `opening_seen` False
  -> SAFE_STOP with reason "RIGHT sign, no right road". Either the sign was
  misread or the junction has no right branch; pivoting would drive into the
  grass.

## STOP_HOLD

- Motors 0 for `STOP_HOLD_S`, then back to DECIDE with the stop marked served.
- `drift_request` is set on entry (already wired), so one drift window
  (`IMU_DRIFT_SETTLE_S + IMU_STILL_S < STOP_HOLD_S`) runs during the stop.

## TURN

Direction `d`: +1 left, -1 right.

1. **Advance:** `hold_steer` on the road heading at `APPROACH_SPEED` for
   `ADVANCE_AFTER_OPENING_S` (cue `opening`, counted from the first
   debounced opening frame) or `ADVANCE_AFTER_LINE_S` (cue `line`).
2. **Pivot:** `turn_steer(yaw, target)` at `TURN_SPEED` until
   `turn_done(yaw, target, front_conf)`.
3. **Finish:** `road_yaw.after_turn(d)`, then REACQUIRE.

**Turn target**, chosen at TURN entry:
- road_yaw valid -> `road_yaw + d * TURN_ANGLE_DEG`
- road_yaw never anchored, IMU valid -> `IMU yaw at TURN entry + d * TURN_ANGLE_DEG`
- IMU invalid (at entry or mid-turn) -> `vision_turn_steer(front, d)` until
  it reports done. Used **only** when the IMU is invalid.

`TURN_TIMEOUT_S` -> SAFE_STOP (proposed, not yet confirmed).

## CROSS

- `hold_steer` on the road heading at `CROSS_SPEED`. With no valid IMU,
  steer 0.
- **Exit to REACQUIRE**, in order of preference:
  1. the right edge returns after the opening (side camera boundary found
     again) for `SIDE_EDGE_BACK_FRAMES` frames in a row. Preferred: the side
     camera can't lock onto the crossing road's line.
  2. front lane conf >= `REACQUIRE_CONF` once past the junction. Fallback
     for junctions with no right road (cue `line`).
- `CROSS_TIMEOUT_S` -> REACQUIRE (not SAFE_STOP).

**Risk (fallback exit only):** mid-junction the front camera can lock onto
the *crossing road's* yellow line, which is perpendicular to the robot. A
high-conf lane there would end CROSS early and steer the robot into the
crossing road. Guard the conf exit with "past the junction" (e.g. the
crossing line has passed below `STOP_LINE_Y_FRAC`, or a minimum
time/distance), and/or require a small heading error before accepting it.

## REACQUIRE

- Lane following at `REACQUIRE_SPEED` (`target_speed` already handles this).
- Exit to LANE_FOLLOW once lane conf >= `REACQUIRE_CONF` for
  `REACQUIRE_FRAMES` loops in a row.
- On exit: reset the sign voter, clear `signs_enabled` and clear
  `exposure_hold`.
- Lane-lost safety applies (REACQUIRE is in `_LANE_STATES`).
- `SideBias` doesn't learn here (LANE_FOLLOW only), so a skewed side reading
  just after a turn can't pull the learned offset.

## New config (also in the design doc)

| Variable | Start value | Used by |
| --- | --- | --- |
| `SIDE_OPENING_FRAMES` | 3 | `_on_approach` |
| `SIDE_EDGE_BACK_FRAMES` | 3 | `_on_cross` |

## Open questions

- `TURN_TIMEOUT_S`: SAFE_STOP, or REACQUIRE like CROSS?
- How "past the junction" is measured for the CROSS conf exit (now only
  needed when there is no right road).
- Speed during the vision-only turn fallback (`TURN_SPEED` as for the pivot?).
- RIGHT voted, no opening seen: SAFE_STOP (proposed), or trust the sign and
  turn on the crossing-line fix?
