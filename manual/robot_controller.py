##########################################################################
# Main Robot Controller for ND Robotics Course
# 5-8-2026
# Professor McLaughlin
##########################################################################
import os
import sys
import pygame
import time

# Add the code/ folder so "from autonomous.robot_types import ..." works here and in sabertooth.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from autonomous.robot_types import MotorCmd
from sabertooth import SabertoothDriver
from ps5_controller import PS5_Controller

class SaberConfig:
    # Settings the SabertoothDriver reads from cfg
    SABER_READ_TIMEOUT_S = 0.1
    SABER_STARTUP_WAIT_S = 2
    SABER_TIMEOUT_MS = 200      # motors stop if no command for this long
    SABER_HW_DEADBAND = 5
    LEFT_CHANNEL = 2
    RIGHT_CHANNEL = 1
    LEFT_INVERT = True          # flip if a wheel spins the wrong way
    RIGHT_INVERT = True
    MOTOR_DEADBAND = 0.15       # 0-1; lifts small commands so the wheels still move

SABER_PORT = '/dev/ttyAMA0'
SABER_BAUD = 9600
SABER_ADDRESS = 128

ACCEL_STEP = 0.05   # max change per update while speeding up (lower = gentler)
DECEL_STEP = 0.08   # max change per update while slowing down
TURN_STEP = 0.25    # max steering change per update; much faster than throttle so steering feels direct

# Curvature steering: turn grows with speed so the stick sets how tight the curve is,
# not how fast the robot spins. MIN_TURN keeps a strong turn at low speed and stopped.
MIN_TURN = 0.5          # full-stick turn when stopped or slow (spin in place)
CURVATURE_GAIN = 0.8    # full-stick turn per unit of speed; takes over above MIN_TURN / CURVATURE_GAIN speed

current_speed = 0.0
current_turn = 0.0
last_debug_time = 0.0

def ramp(current, target, accel_step=ACCEL_STEP, decel_step=DECEL_STEP):
    # Move current toward target, limited to accel_step or decel_step per update.
    if abs(target) > abs(current) and (target * current >= 0):
        step = accel_step   # speeding up in the same direction
    else:
        step = decel_step   # slowing down or reversing
    if target > current:
        return min(current + step, target)
    return max(current - step, target)

def move_robot(saber, control_request):
    # Racing controls: R2 = throttle, L2 = reverse, left stick X = steering
    global current_speed, current_turn
    target_speed = control_request["reqR2Value"] - control_request["reqL2Value"]
    current_speed = ramp(current_speed, target_speed)

    if control_request["reqLeftJoyMade"]:
        steer = control_request["reqLeftJoyXValue"] / 127
        target_turn = steer * max(MIN_TURN, abs(current_speed) * CURVATURE_GAIN)
    else:
        target_turn = 0.0   # stick released: straighten out
    current_turn = ramp(current_turn, target_turn, TURN_STEP, TURN_STEP)

    # Mix speed and turn into left/right wheel commands (+ turn = right)
    left = current_speed + current_turn
    right = current_speed - current_turn
    biggest = max(abs(left), abs(right), 1.0)
    cmd = MotorCmd(left=left / biggest, right=right / biggest)
    saber.set(cmd)

    global last_debug_time
    if (cmd.left or cmd.right) and time.time() - last_debug_time >= 0.25:
        print(f"R2 {control_request['reqR2Value']:.2f} L2 {control_request['reqL2Value']:.2f} "
              f"-> left {cmd.left:+.2f} right {cmd.right:+.2f} serial_error {saber.serial_error}")
        last_debug_time = time.time()

def main():
    saber = None
    try:
        # Initialize the PS5 Controller class
        ps5 = PS5_Controller()
        ps5.initialize_controller()

        # Initialize Sabertooth motor controller
        saber = SabertoothDriver(SABER_PORT, SABER_BAUD, SABER_ADDRESS, SaberConfig)
        saber.open()

        ps5_last_check_time = time.time()
        ps5_loop_interval = 0.02  # 20ms interval

        motor_controller_last_check_time = time.time()
        motor_controller_loop_interval = 0.04  # 40ms interval

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
                move_robot(saber, ps5.control_request)
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
        print("PS5 controller disconnected.")

if __name__ == "__main__":
    main()
