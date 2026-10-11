"""Speed per state, slowed by low lane confidence or large heading error.

Lane-following states: base x conf factor x heading factor, where
    conf factor    = SPEED_MIN_SCALE at conf 0 rising to 1 at conf 1
    heading factor = 1 at zero error falling to SPEED_MIN_SCALE at
                     SPEED_HEADING_SLOW_RAD and beyond
CROSS keeps its base speed (the lane is expected to vanish there).
TURN speed depends on its phase and is set by drive_for_state.
"""
from robot_types import State

_LANE_STATES = (State.LANE_FOLLOW, State.APPROACH, State.REACQUIRE)


def target_speed(state, est, cfg):
    base = _base_speed(state, cfg)
    if state not in _LANE_STATES:
        return base
    return base * _conf_factor(est, cfg) * _heading_factor(est, cfg)


def _base_speed(state, cfg):
    return {
        State.LANE_FOLLOW: cfg.CRUISE_SPEED,
        State.APPROACH: cfg.APPROACH_SPEED,
        State.REACQUIRE: cfg.REACQUIRE_SPEED,
        State.CROSS: cfg.CROSS_SPEED,
    }.get(state, 0.0)


def _conf_factor(est, cfg):
    conf = est.conf if est is not None else 0.0
    return cfg.SPEED_MIN_SCALE + (1.0 - cfg.SPEED_MIN_SCALE) * conf


def _heading_factor(est, cfg):
    if est is None or est.heading_err_rad is None:
        return 1.0
    fraction = min(1.0, abs(est.heading_err_rad) / cfg.SPEED_HEADING_SLOW_RAD)
    return 1.0 - (1.0 - cfg.SPEED_MIN_SCALE) * fraction
