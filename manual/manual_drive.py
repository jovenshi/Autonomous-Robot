##########################################################################
# Manual driving logic for ND Robotics Course, shared by robot_controller.py
# and the autonomous program (which drives by hand while stopped).
# No pygame or hardware here: triggers and stick in, speed and turn out.
##########################################################################
import math

# Throttle ramps in units per second, so they feel the same at any loop rate
ACCEL_RATE = 1.0   # speeding up (lower = gentler)
DECEL_RATE = 2.0    # slowing down or reversing

# Steering. Full stick gives PIVOT_TURN at low speed (spin in place when stopped), rising
# to cfg.LANE_STEER_SPEED_RATIO x speed once that is larger, capped at cfg.STEER_MAX.
# The limit only ever grows with speed, so throttle changes never yank the steering.
PIVOT_TURN = 0.4        # full-stick turn when stopped or slow
STICK_DEADZONE = 0.1    # matches ps5_controller; stick travel is rescaled from here

# Heading hold: with the stick released, the IMU gently keeps the robot on the heading
# it had when it stopped rotating (cfg.HOLD_KP on yaw error). Set False to turn it off.
HEADING_HOLD = True
HOLD_MIN_SPEED = 0.1    # below this speed the hold is off (no creeping pivots)
HOLD_SETTLE_DPS = 10    # deg/s; the target heading is taken once rotation drops below this

TURN_ZERO = 1e-3        # |turn| below this snaps to 0, so the smoothing settles at rest


def ramp(current, target, dt):
    # Move current toward target at ACCEL_RATE or DECEL_RATE per second.
    if abs(target) > abs(current) and (target * current >= 0):
        step = ACCEL_RATE * dt   # speeding up in the same direction
    else:
        step = DECEL_RATE * dt   # slowing down or reversing
    if target > current:
        return min(current + step, target)
    return max(current - step, target)


def turn_limit(speed, cfg):
    # Largest |turn| allowed at this speed; never shrinks as speed rises.
    return min(cfg.STEER_MAX, max(PIVOT_TURN, cfg.LANE_STEER_SPEED_RATIO * abs(speed)))


def mix(speed, turn, cfg):
    # Speed and turn (+ = right) -> (left, right) wheel commands, scaled to fit -1..1.
    left = speed + cfg.STEER_GAIN * turn
    right = speed - cfg.STEER_GAIN * turn
    biggest = max(abs(left), abs(right), 1.0)
    return left / biggest, right / biggest


class ManualDrive:
    """Racing controls: R2 = throttle, L2 = reverse, left stick X = steering."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.speed = 0.0
        self.turn = 0.0          # + = right
        self.hold_yaw = None     # deg; heading being held, None when not holding

    def reset(self):
        self.speed, self.turn, self.hold_yaw = 0.0, 0.0, None

    @property
    def moving(self):
        return self.speed != 0.0 or self.turn != 0.0

    def update(self, r2, l2, stick_x, imu_sample, dt):
        # r2, l2: trigger 0-1; stick_x: left stick -1 (left) to 1 (right), 0 when
        # released; imu_sample: ImuSample or None. -> (speed, turn), turn + = right.
        cfg = self.cfg
        self.speed = ramp(self.speed, r2 - l2, dt)
        limit = turn_limit(self.speed, cfg)
        if stick_x != 0:
            # Rescale past the deadzone so steering starts from 0 instead of jumping to 0.1
            stick = math.copysign(max(0.0, abs(stick_x) - STICK_DEADZONE) / (1 - STICK_DEADZONE),
                                  stick_x)
            target_turn = stick * limit
            self.hold_yaw = None
        else:
            target_turn = self._heading_hold_turn(imu_sample)
        target_turn = max(-limit, min(limit, target_turn))
        # Same low-pass filter as the autonomous LaneController
        self.turn = cfg.STEER_SMOOTH * self.turn + (1 - cfg.STEER_SMOOTH) * target_turn
        if abs(self.turn) < TURN_ZERO and target_turn == 0:
            self.turn = 0.0
        return self.speed, self.turn

    def _heading_hold_turn(self, imu_sample):
        # Turn (+ = right) that keeps the robot on hold_yaw; 0 when the hold is off.
        if (not HEADING_HOLD or imu_sample is None or not imu_sample.valid
                or abs(self.speed) < HOLD_MIN_SPEED):
            self.hold_yaw = None
            return 0.0
        if self.hold_yaw is None or abs(imu_sample.yaw_rate_dps) > HOLD_SETTLE_DPS:
            self.hold_yaw = imu_sample.yaw_deg   # still rotating from the last turn: let it coast
        # Same as the autonomous hold_steer; IMU yaw + = left, so negate for + = right
        return -self.cfg.HOLD_KP * (self.hold_yaw - imu_sample.yaw_deg)
