##########################################################################
# Main Robot Controller for ND Robotics Course
# 5-8-2026
# Professor McLaughlin
##########################################################################
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
from manual_drive import ManualDrive, mix

# Driving logic (ramps, steering limits, heading hold) lives in manual_drive.py, shared
# with the autonomous program, which uses it to drive by hand while stopped.
drive = None            # ManualDrive, made in main()
last_debug_time = 0.0

def move_robot(saber, control_request, imu_sample, dt):
    # Racing controls: R2 = throttle, L2 = reverse, left stick X = steering
    stick = 0.0
    if control_request["reqLeftJoyMade"]:
        stick = control_request["reqLeftJoyXValue"] / 127
    speed, turn = drive.update(control_request["reqR2Value"], control_request["reqL2Value"],
                               stick, imu_sample, dt)
    left, right = mix(speed, turn, cfg)
    cmd = MotorCmd(left=left, right=right)
    saber.set(cmd)

    global last_debug_time
    if (cmd.left or cmd.right) and time.time() - last_debug_time >= 0.25:
        hold = f"{drive.hold_yaw:+.1f}" if drive.hold_yaw is not None else "off"
        yaw = f"{imu_sample.yaw_deg:+.1f}" if imu_sample is not None and imu_sample.valid else "n/a"
        print(f"R2 {control_request['reqR2Value']:.2f} L2 {control_request['reqL2Value']:.2f} "
              f"-> left {cmd.left:+.2f} right {cmd.right:+.2f} turn {turn:+.2f} "
              f"yaw {yaw} hold {hold} serial_error {saber.serial_error}")
        last_debug_time = time.time()

def main():
    global drive
    saber = None
    imu = None
    drive = ManualDrive(cfg)
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
