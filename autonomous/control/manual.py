"""Driving by hand while stopped: PS5 triggers and left stick -> DriveCmd and MotorCmd.

The driving itself is manual/manual_drive.py, shared with
manual/robot_controller.py, so the robot handles the same in both programs:
R2 = throttle, L2 = reverse, left stick X = steering, ramps, and IMU heading
hold with the stick released.

Only in MANUAL_STATES (the robot is stopped: IDLE, or SAFE_STOP), so in every
mode, and all the time in Sign Reading, where the robot never drives itself.
The readings come from the display process (ui/gamepad.py) at UI_HZ; one older
than MANUAL_INPUT_TIMEOUT_S (controller lost, display stopped) stops the
motors at once, without a ramp.
"""
from robot_types import DriveCmd, MotorCmd, State
from ui.course import course_module

MANUAL_STATES = (State.IDLE, State.SAFE_STOP)


class ManualControl:
    def __init__(self, cfg, drive=None):
        self.cfg = cfg
        self.drive = drive if drive is not None else _manual_drive_module().ManualDrive(cfg)
        self._mix = _manual_drive_module().mix
        self.latest = None               # newest ManualInput

    def take(self, manual_input):
        # The newest reading this loop (None: nothing new, keep the last).
        if manual_input is not None:
            self.latest = manual_input

    def step(self, state, imu, now):
        # -> (DriveCmd, MotorCmd) while driven by hand, else None (the
        # autonomous command stands). drive.speed: + forward, - reverse;
        # DriveCmd.steer + = left as everywhere in autonomous/.
        inp = self.latest
        if (not self.cfg.MANUAL_DRIVE_ENABLED or state not in MANUAL_STATES or inp is None
                or now - inp.t > self.cfg.MANUAL_INPUT_TIMEOUT_S):
            self.drive.reset()
            return None
        speed, turn = self.drive.update(inp.r2, inp.l2, inp.stick_x, imu, 1.0 / self.cfg.CONTROL_HZ)
        if not self.drive.moving:
            return None
        left, right = self._mix(speed, turn, self.cfg)
        return DriveCmd(speed=speed, steer=-turn), MotorCmd(left=left, right=right)


def _manual_drive_module():
    return course_module("manual.manual_drive")
