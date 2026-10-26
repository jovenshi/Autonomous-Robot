"""Behaviour state machine: decides who steers and how fast.

step() runs once per control loop: commands first (START, STOP), then
safety, then the current state's handler. drive_for_state() turns the
resulting state into a DriveCmd.

Working now: IDLE, LANE_FOLLOW, SAFE_STOP (straight driving).
APPROACH, DECIDE, STOP_HOLD, TURN, CROSS and REACQUIRE raise until their
inputs exist (intersection.py, signs.py, sign_voter.py, the IMU reader).
choose_action is complete, ready for DECIDE.
"""
from estimation.road_yaw import off_road_deg
from robot_types import Action, SignType, State, UiCommand

# States where the robot is moving and must keep seeing the lane.
_LANE_STATES = (State.LANE_FOLLOW, State.APPROACH, State.REACQUIRE)
_MOVING_STATES = _LANE_STATES + (State.TURN, State.CROSS)
# States that steer from IMU yaw: a stale IMU stops the robot only here.
_IMU_STEER_STATES = (State.TURN, State.CROSS)
# States where the robot stands still long enough to measure gyro drift.
_STILL_STATES = (State.IDLE, State.STOP_HOLD)


class Behaviour:
    def __init__(self, cfg, signs_enabled=None, exposure_hold=None, drift_request=None):
        # signs_enabled, exposure_hold: threading.Event flags read by the
        # camera threads; drift_request: read by the IMU thread. None when
        # not wired (tests, tools).
        self.cfg = cfg
        self.signs_enabled = signs_enabled
        self.exposure_hold = exposure_hold
        self.drift_request = drift_request
        self.state = None
        self.state_t = 0.0                 # time the current state was entered
        self.stop_reason = None            # why SAFE_STOP was entered
        self._start_requested = False
        self._lane_seen_t = 0.0
        self._recover_t = None             # when the current pivot back to the road began
        self._enter(State.IDLE, 0.0)

    def step(self, front, side, lane, road_yaw, sign, imu, commands, motor_error, now):
        # commands: list of UiCommand; motor_error: SabertoothDriver.serial_error.
        if self._apply_commands(commands, now):
            return self.state
        if lane is not None and lane.lateral_err_m is not None:
            self._lane_seen_t = now
        if self._recovering(imu, road_yaw, now):
            # Pivoting back to the road (control/command.py): the line is
            # expected to be out of view, for up to RECOVER_TIMEOUT_S.
            if self._recover_t is None:
                self._recover_t = now
            if now - self._recover_t > self.cfg.RECOVER_TIMEOUT_S:
                self._enter(State.SAFE_STOP, now, "could not turn back to the road")
                return self.state
            self._lane_seen_t = now
        else:
            self._recover_t = None
        if self.state in _MOVING_STATES:
            reason = self._check_safety(lane, imu, motor_error, now)
            if reason is not None:
                self._enter(State.SAFE_STOP, now, reason)
                return self.state
        next_state = self._handler(self.state)(front, side, lane, road_yaw, sign, imu, now)
        if next_state != self.state:
            self._enter(next_state, now)
        return self.state

    def _apply_commands(self, commands, now):
        # STOP from any state -> IDLE. START in SAFE_STOP -> IDLE; START in
        # IDLE is remembered for _on_idle. True if the state changed here.
        if UiCommand.STOP in commands and self.state != State.IDLE:
            self._enter(State.IDLE, now)
            return True
        if UiCommand.START in commands:
            if self.state == State.SAFE_STOP:
                self._enter(State.IDLE, now)
                return True
            if self.state == State.IDLE:
                self._start_requested = True
        return False

    def _recovering(self, imu, road_yaw, now):
        # True while knocked more than RECOVER_DEG off the road's direction
        # in a lane state, with a fresh IMU to steer back by.
        if self.state not in _LANE_STATES or imu is None or now - imu.t > self.cfg.MAX_OBS_AGE_S:
            return False
        off = off_road_deg(imu, road_yaw)
        return off is not None and abs(off) > self.cfg.RECOVER_DEG

    def _check_safety(self, lane, imu, motor_error, now):
        # Reason to stop, or None.
        if motor_error:
            return "motor serial error"
        if (self.state in _IMU_STEER_STATES and imu is not None
                and now - imu.t > self.cfg.MAX_OBS_AGE_S):
            return "IMU data stale"
        if self.state in _LANE_STATES and now - self._lane_seen_t > self.cfg.LANE_LOST_TIMEOUT_S:
            return "lane lost"
        return None

    def _handler(self, state):
        return {
            State.IDLE: self._on_idle,
            State.LANE_FOLLOW: self._on_lane_follow,
            State.SAFE_STOP: self._on_safe_stop,
        }.get(state, self._not_built)

    def _on_idle(self, front, side, lane, road_yaw, sign, imu, now):
        # After START: go once the gyro is calibrated, at once if there is no
        # IMU, or after IMU_CALIB_WAIT_S with a warning.
        if not self._start_requested:
            return State.IDLE
        if imu is None or imu.calib_gyro == 3:
            return State.LANE_FOLLOW
        if now - self.state_t >= self.cfg.IMU_CALIB_WAIT_S:
            print("Warning: starting with the IMU gyro not calibrated")
            return State.LANE_FOLLOW
        return State.IDLE

    def _on_lane_follow(self, front, side, lane, road_yaw, sign, imu, now):
        inter = front.inter if front is not None else None
        if inter is not None and inter.approach_score >= self.cfg.APPROACH_ON:
            return State.APPROACH
        return State.LANE_FOLLOW

    def _on_safe_stop(self, front, side, lane, road_yaw, sign, imu, now):
        # Leaves only on a manual START (handled in _apply_commands).
        return State.SAFE_STOP

    def _not_built(self, *args):
        raise NotImplementedError(
            f"Behaviour: {self.state.name} is built with the junction work")

    def _enter(self, state, now, reason=None):
        # Records entry time and reason; sets the camera and IMU thread flags.
        self.state = state
        self.state_t = now
        self.stop_reason = reason if state == State.SAFE_STOP else None
        if state == State.IDLE:
            self._start_requested = False
        if state == State.LANE_FOLLOW:
            self._lane_seen_t = now        # fresh grace period on start
        self._set_flag(self.exposure_hold, state != State.LANE_FOLLOW)
        self._set_flag(self.drift_request, state in _STILL_STATES)
        if state == State.APPROACH:
            self._set_flag(self.signs_enabled, True)
        elif state in (State.LANE_FOLLOW, State.IDLE, State.SAFE_STOP):
            self._set_flag(self.signs_enabled, False)

    @staticmethod
    def _set_flag(flag, on):
        if flag is None:
            return
        if on:
            flag.set()
        else:
            flag.clear()


def choose_action(sign, stop_done, cfg):
    # Sign -> action. A STOP already served, or no sign -> DEFAULT_ACTION.
    if sign == SignType.STOP and not stop_done:
        return Action.STOP
    if sign in (SignType.LEFT, SignType.RIGHT, SignType.STRAIGHT):
        return Action[sign.name]
    return Action[cfg.DEFAULT_ACTION]
