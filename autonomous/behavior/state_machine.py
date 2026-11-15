"""Behaviour state machine: decides who steers and how fast.

step() runs once per control loop: commands first (START, STOP), then
safety, then the current state's handler. drive_for_state() turns the
resulting state into a DriveCmd.

Working now: IDLE, LANE_FOLLOW, SAFE_STOP (straight driving), and for the
intersection quiz (RunMode.QUIZ_INTERSECTION, docs/quiz3_plan.md stage 6)
APPROACH, CROSS and REACQUIRE, driven by the LocationObs:
    LANE_FOLLOW -> APPROACH in READ_SIGN and PAUSE
                -> CROSS (hold_steer on road_yaw) in ENTERING, TURN, EXITING
                -> REACQUIRE in CONTINUE after the junction
                -> IDLE once QUIZ_EXIT_M past the centre.
At every zone change while moving, `paused` is set for ZONE_PAUSE_S
(drive_for_state stops the robot) so the judge can check the zone. In DRIVE
mode the robot stays in LANE_FOLLOW as before; QUIZ_SIGNS stays in IDLE.
DECIDE, STOP_HOLD and TURN raise (out of scope for Quiz 3).
choose_action is complete, ready for DECIDE.
"""
from estimation.road_yaw import off_road_deg
from robot_types import Action, RunMode, SignType, State, UiCommand, Zone

_MODE_COMMANDS = {UiCommand.MODE_DRIVE: RunMode.DRIVE,
                  UiCommand.MODE_INTERSECTION: RunMode.QUIZ_INTERSECTION,
                  UiCommand.MODE_SIGNS: RunMode.QUIZ_SIGNS}

# States where the robot is moving and must keep seeing the lane.
_LANE_STATES = (State.LANE_FOLLOW, State.APPROACH, State.REACQUIRE)
_MOVING_STATES = _LANE_STATES + (State.TURN, State.CROSS)
# States that steer from IMU yaw: a stale IMU stops the robot only here.
_IMU_STEER_STATES = (State.TURN, State.CROSS)
# States where the robot stands still long enough to measure gyro drift.
_STILL_STATES = (State.IDLE, State.STOP_HOLD)
# Intersection quiz: the state each zone of a pass calls for.
_APPROACH_ZONES = (Zone.READ_SIGN, Zone.PAUSE)
_CROSS_ZONES = (Zone.ENTERING, Zone.TURN, Zone.EXITING)


class Behaviour:
    def __init__(self, cfg, signs_enabled=None, exposure_hold=None, drift_request=None):
        # signs_enabled, exposure_hold: threading.Event flags read by the
        # camera threads; drift_request: read by the IMU thread. None when
        # not wired (tests, tools).
        self.cfg = cfg
        self.signs_enabled = signs_enabled
        self.exposure_hold = exposure_hold
        self.drift_request = drift_request
        self.mode = RunMode[cfg.START_MODE]
        self.state = None
        self.state_t = 0.0                 # time the current state was entered
        self.stop_reason = None            # why SAFE_STOP was entered
        self._start_requested = False
        self._lane_seen_t = 0.0
        self._recover_t = None             # when the current pivot back to the road began
        self.location = None               # newest LocationObs (intersection quiz only)
        self.paused = False                # judging pause: drive_for_state stops the robot
        self._zone = None                  # zone last seen, for the judging pause
        self._pause_until = None
        self._last_step_t = None
        self._moving_s = 0.0               # time in this state not spent paused
        self._reacquire_run = 0            # loops in a row with lane conf >= REACQUIRE_CONF
        self._enter(State.IDLE, 0.0)

    def step(self, front, side, lane, road_yaw, sign, imu, commands, motor_error, now,
             location=None):
        # commands: list of UiCommand; motor_error: SabertoothDriver.serial_error;
        # location: LocationObs (used in QUIZ_INTERSECTION only), or None.
        self.location = location if self.mode == RunMode.QUIZ_INTERSECTION else None
        self._update_pause(now)
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
        # IDLE is remembered for _on_idle, except in QUIZ_SIGNS, where the
        # robot never moves. A mode command is obeyed only in IDLE. True if
        # the state changed here.
        if UiCommand.STOP in commands and self.state != State.IDLE:
            self._enter(State.IDLE, now)
            return True
        self._apply_mode(commands)
        if UiCommand.START in commands:
            if self.state == State.SAFE_STOP:
                self._enter(State.IDLE, now)
                return True
            if self.state == State.IDLE and self.mode != RunMode.QUIZ_SIGNS:
                self._start_requested = True
        return False

    def _apply_mode(self, commands):
        # The last mode command wins; ignored outside IDLE.
        if self.state != State.IDLE:
            return
        for command in commands:
            if command in _MODE_COMMANDS and _MODE_COMMANDS[command] != self.mode:
                self.mode = _MODE_COMMANDS[command]
                self._start_requested = False
                self._set_sign_flag()

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

    def _update_pause(self, now):
        # A judging pause of ZONE_PAUSE_S at every zone change while moving,
        # and the time in this state not spent paused (for CROSS_TIMEOUT_S).
        # The time since the last step counts as driving only if the robot
        # was not paused during it.
        if self._last_step_t is not None and not self.paused:
            self._moving_s += max(0.0, now - self._last_step_t)
        self._last_step_t = now
        zone = self.location.zone if self.location is not None else None
        if (self.state in _MOVING_STATES and zone is not None and self._zone is not None
                and zone != self._zone):
            self._pause_until = now + self.cfg.ZONE_PAUSE_S
        self._zone = zone
        self.paused = self._pause_until is not None and now < self._pause_until

    def _quiz_target(self):
        # The state the intersection quiz calls for at this location, or
        # None (not in the quiz, or no junction pass under way).
        loc = self.location
        if loc is None or loc.dist_to_centre_m is None:
            return None
        if loc.turn_point_ready and loc.dist_to_centre_m <= -self.cfg.QUIZ_EXIT_M:
            return State.IDLE
        if loc.zone in _APPROACH_ZONES:
            return State.APPROACH
        if loc.zone in _CROSS_ZONES:
            return State.CROSS
        if loc.turn_point_ready:
            return State.REACQUIRE
        return None

    def _handler(self, state):
        return {
            State.IDLE: self._on_idle,
            State.LANE_FOLLOW: self._on_lane_follow,
            State.APPROACH: self._on_approach,
            State.CROSS: self._on_cross,
            State.REACQUIRE: self._on_reacquire,
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
        # Intersection quiz: on to APPROACH (or straight to CROSS, or IDLE
        # when the run is over) as the location calls for. After REACQUIRE
        # the robot may be back here before QUIZ_EXIT_M.
        target = self._quiz_target()
        if target in (State.APPROACH, State.CROSS, State.IDLE):
            return target
        return State.LANE_FOLLOW

    def _on_approach(self, front, side, lane, road_yaw, sign, imu, now):
        # Lane control at APPROACH_SPEED through READ_SIGN and PAUSE; CROSS
        # from ENTERING. Back to LANE_FOLLOW if the pass is lost.
        target = self._quiz_target()
        return State.LANE_FOLLOW if target is None else target

    def _on_cross(self, front, side, lane, road_yaw, sign, imu, now):
        # hold_steer on road_yaw from ENTERING to the end of EXITING, then
        # REACQUIRE. After CROSS_TIMEOUT_S of driving (pauses not counted),
        # or if the pass is lost, REACQUIRE: the lane controller takes over.
        target = self._quiz_target()
        if target in (State.REACQUIRE, State.IDLE):
            return target
        if target is None or self._moving_s > self.cfg.CROSS_TIMEOUT_S:
            return State.REACQUIRE
        return State.CROSS

    def _on_reacquire(self, front, side, lane, road_yaw, sign, imu, now):
        # Lane control at REACQUIRE_SPEED; IDLE once QUIZ_EXIT_M past the
        # centre, LANE_FOLLOW after REACQUIRE_FRAMES loops in a row with
        # lane conf >= REACQUIRE_CONF.
        if self._quiz_target() == State.IDLE:
            return State.IDLE
        confident = lane is not None and lane.conf >= self.cfg.REACQUIRE_CONF
        self._reacquire_run = self._reacquire_run + 1 if confident else 0
        if self._reacquire_run >= self.cfg.REACQUIRE_FRAMES:
            return State.LANE_FOLLOW
        return State.REACQUIRE

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
        self._moving_s, self._reacquire_run = 0.0, 0
        if state == State.IDLE:
            self._start_requested = False
        if state not in _MOVING_STATES:
            self._pause_until, self.paused = None, False
        if state in (State.LANE_FOLLOW, State.REACQUIRE):
            self._lane_seen_t = now        # fresh grace period on start and after CROSS
        self._set_flag(self.exposure_hold, state != State.LANE_FOLLOW)
        self._set_flag(self.drift_request, state in _STILL_STATES)
        self._set_sign_flag()

    def _set_sign_flag(self):
        # Signs are read all the time in QUIZ_SIGNS; otherwise from APPROACH
        # until the junction is left (LANE_FOLLOW, IDLE or SAFE_STOP).
        if self.mode == RunMode.QUIZ_SIGNS or self.state == State.APPROACH:
            self._set_flag(self.signs_enabled, True)
        elif self.state in (State.LANE_FOLLOW, State.IDLE, State.SAFE_STOP):
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
    # Sign -> action. A PAUSE already served, or no sign -> DEFAULT_ACTION.
    if sign == SignType.PAUSE and not stop_done:
        return Action.PAUSE
    if sign in (SignType.LEFT, SignType.RIGHT, SignType.STRAIGHT):
        return Action[sign.name]
    return Action[cfg.DEFAULT_ACTION]
