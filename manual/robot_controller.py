##########################################################################
# Main Robot Controller for ND Robotics Course
# 5-8-2026
# Professor McLaughlin
##########################################################################
import math
import os
import sys
import pygame
import time

# Add the code/ folder so "from autonomous.robot_types import ..." works here and in sabertooth.py,
# and autonomous/ so the shared config and IMU reader import the same way they do there
CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(CODE_DIR)
sys.path.append(os.path.join(CODE_DIR, "autonomous"))

import config as cfg    # autonomous/config.py: Sabertooth, IMU and steering values tuned there
from autonomous.robot_types import MotorCmd
from hal.imu import ImuReader
from robot_types import LatestValue
from sabertooth import SabertoothDriver
from ps5_controller import PS5_Controller

# Throttle ramps in units per second, so they feel the same at any loop rate
ACCEL_RATE = 1.25   # speeding up (lower = gentler)
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

current_speed = 0.0
current_turn = 0.0
hold_yaw = None         # deg; heading being held, None when not holding
last_debug_time = 0.0

def ramp(current, target, dt):
    # Move current toward target at ACCEL_RATE or DECEL_RATE per second.
    if abs(target) > abs(current) and (target * current >= 0):
        step = ACCEL_RATE * dt   # speeding up in the same direction
    else:
        step = DECEL_RATE * dt   # slowing down or reversing
    if target > current:
        return min(current + step, target)
    return max(current - step, target)

def turn_limit(speed):
    # Largest |turn| allowed at this speed; never shrinks as speed rises.
    return min(cfg.STEER_MAX, max(PIVOT_TURN, cfg.LANE_STEER_SPEED_RATIO * abs(speed)))

def heading_hold_turn(imu_sample, speed):
    # Turn (+ = right) that keeps the robot on hold_yaw; 0 when the hold is off.
    global hold_yaw
    if (not HEADING_HOLD or imu_sample is None or not imu_sample.valid
            or abs(speed) < HOLD_MIN_SPEED):
        hold_yaw = None
        return 0.0
    if hold_yaw is None or abs(imu_sample.yaw_rate_dps) > HOLD_SETTLE_DPS:
        hold_yaw = imu_sample.yaw_deg   # still rotating from the last turn: let it coast
    # Same as the autonomous hold_steer; IMU yaw + = left, so negate for + = right
    return -cfg.HOLD_KP * (hold_yaw - imu_sample.yaw_deg)

def move_robot(saber, control_request, imu_sample, dt):
    # Racing controls: R2 = throttle, L2 = reverse, left stick X = steering
    global current_speed, current_turn, hold_yaw
    target_speed = control_request["reqR2Value"] - control_request["reqL2Value"]
    current_speed = ramp(current_speed, target_speed, dt)
    limit = turn_limit(current_speed)

    if control_request["reqLeftJoyMade"] and control_request["reqLeftJoyXValue"] != 0:
        stick = control_request["reqLeftJoyXValue"] / 127
        # Rescale past the deadzone so steering starts from 0 instead of jumping to 0.1
        stick = math.copysign(max(0.0, abs(stick) - STICK_DEADZONE) / (1 - STICK_DEADZONE), stick)
        target_turn = stick * limit
        hold_yaw = None
    else:
        target_turn = heading_hold_turn(imu_sample, current_speed)
    target_turn = max(-limit, min(limit, target_turn))

    # Same low-pass filter as the autonomous LaneController
    current_turn = cfg.STEER_SMOOTH * current_turn + (1 - cfg.STEER_SMOOTH) * target_turn

    # Mix speed and turn into left/right wheel commands (+ turn = right)
    left = current_speed + cfg.STEER_GAIN * current_turn
    right = current_speed - cfg.STEER_GAIN * current_turn
    biggest = max(abs(left), abs(right), 1.0)
    cmd = MotorCmd(left=left / biggest, right=right / biggest)
    saber.set(cmd)

    global last_debug_time
    if (cmd.left or cmd.right) and time.time() - last_debug_time >= 0.25:
        hold = f"{hold_yaw:+.1f}" if hold_yaw is not None else "off"
        yaw = f"{imu_sample.yaw_deg:+.1f}" if imu_sample is not None and imu_sample.valid else "n/a"
        print(f"R2 {control_request['reqR2Value']:.2f} L2 {control_request['reqL2Value']:.2f} "
              f"-> left {cmd.left:+.2f} right {cmd.right:+.2f} turn {current_turn:+.2f} "
              f"yaw {yaw} hold {hold} serial_error {saber.serial_error}")
        last_debug_time = time.time()

def main():
    saber = None
    imu = None
    try:
        # Initialize the PS5 Controller class
        ps5 = PS5_Controller()
        ps5.initialize_controller()

        # Initialize Sabertooth motor controller
        saber = SabertoothDriver(cfg.SABER_PORT, cfg.SABER_BAUD, cfg.SABER_ADDRESS, cfg)
        saber.open()

        # IMU for heading hold; if it fails to open it reads invalid and the hold stays off
        imu_slot = LatestValue()
        imu = ImuReader(imu_slot, None, cfg)
        imu.start()

        ps5_last_check_time = time.time()
        ps5_loop_interval = 0.02  # 20ms interval

        motor_controller_last_check_time = time.time()
        motor_controller_loop_interval = 1.0 / cfg.CONTROL_HZ  # same rate as the autonomous loop

        while True:
            current_time = time.time()

            # Check PS5 controller state
            if current_time - ps5_last_check_time >= ps5_loop_interval:
                pygame.event.pump()
                ps5.check_controls()
                ps5_last_check_time = current_time

            # Example: Use Arrow Up as a function call trigger
            if ps5.control_request["reqArrowUp"]:
                print("This should call by function tied to Arrow Up")

            # Update motors every interval (also keeps the Sabertooth auto-stop from firing)
            if current_time - motor_controller_last_check_time >= motor_controller_loop_interval:
                move_robot(saber, ps5.control_request, imu_slot.get(), motor_controller_loop_interval)
                motor_controller_last_check_time = current_time

            # Reset PS5 request variables for next loop
            if ps5.control_request["reqMade"]:
                ps5.reset_controller_state()

            # Provide a brief sleep to allow worker threads to catch up to main loop
            time.sleep(.001)

    except KeyboardInterrupt:
        print("Exiting program...")

    finally:
        pygame.joystick.quit()
        pygame.quit()
        if saber is not None:
            saber.close()
        if imu is not None:
            imu.close()
        print("PS5 controller disconnected.")

if __name__ == "__main__":
    main()
