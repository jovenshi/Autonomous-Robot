"""Chooses the steering source and speed for the current state.

TURN and CROSS steer from the Behaviour's yaw target and turn phase, which
are wired in once behaviour/state_machine.py exists; until then they raise.
"""
from control.speed_governor import target_speed
from estimation.road_yaw import off_road_deg
from robot_types import DriveCmd, State

_LANE_STATES = (State.LANE_FOLLOW, State.APPROACH, State.REACQUIRE)
_STOPPED_STATES = (State.IDLE, State.DECIDE, State.STOP_HOLD, State.SAFE_STOP)


def drive_for_state(state, est, imu, behaviour, lane_ctrl, front, cfg, road_yaw=None):
    if state in _STOPPED_STATES:
        return DriveCmd(speed=0.0, steer=0.0)
    off = off_road_deg(imu, road_yaw)
    if state in _LANE_STATES and off is not None and abs(off) > cfg.RECOVER_DEG:
        # Knocked more than RECOVER_DEG off the road's direction: pivot back
        # in place (keeps the robot over the line), then lane control re-centres.
        lane_ctrl.reset()
        steer = max(cfg.TURN_MIN_STEER, min(cfg.STEER_MAX, cfg.TURN_KP * abs(off)))
        return DriveCmd(speed=0.0, steer=steer if off > 0 else -steer)
    if state in _LANE_STATES:
        speed = target_speed(state, est, cfg)
        steer = lane_ctrl.update(est, _yaw_rate(imu), 1.0 / cfg.CONTROL_HZ, speed)
        return DriveCmd(speed=speed, steer=_arc_limit(steer, speed, cfg))
    raise NotImplementedError(f"drive_for_state: {state.name} is wired with state_machine.py")


def _arc_limit(steer, speed, cfg):
    # Caps steer to a fraction of speed so the inner wheel slows but keeps
    # rolling forward: the robot arcs back to centre instead of pivoting.
    limit = cfg.LANE_STEER_SPEED_RATIO * speed
    return max(-limit, min(limit, steer))


def _yaw_rate(imu):
    # IMU yaw rate for damping; 0 when there is no valid IMU (no damping).
    if imu is None or not imu.valid:
        return 0.0
    return imu.yaw_rate_dps
