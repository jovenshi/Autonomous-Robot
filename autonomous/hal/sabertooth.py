################################################################
# Sabertooth motor driver for the autonomous robot
# Independent-motor version of the ND Robotics course Sabertooth class
# (Professor McLaughlin, 3-7-2025): same packet format, send_command,
# set_auto_stop, set_deadband and map_integer, driven by MotorCmd.
################################################################
"""Packetized serial driver for the Sabertooth, one command per motor.

Packets are 4 bytes: address, command, value 0-127, 7-bit checksum.
control_step must call set() every loop: with the auto-stop timeout set, the
Sabertooth stops both motors if no command arrives for SABER_TIMEOUT_MS.

Writes go straight to the port without flush(); the OS sends the bytes, so
the control loop never waits on the serial line.
"""
import math
import time

import serial

from robot_types import MotorCmd

# Packetized serial command numbers (protocol facts from the Sabertooth
# manual, not tunables, so they live here rather than in config.py).
CMD_M1_FORWARD = 0
CMD_M1_BACKWARD = 1
CMD_M2_FORWARD = 4
CMD_M2_BACKWARD = 5
CMD_SERIAL_TIMEOUT = 14
CMD_DEADBAND = 17
MAX_VALUE = 127

# channel -> (forward command, backward command)
_CHANNEL_COMMANDS = {
    1: (CMD_M1_FORWARD, CMD_M1_BACKWARD),
    2: (CMD_M2_FORWARD, CMD_M2_BACKWARD),
}


class SabertoothDriver:
    def __init__(self, port, baud, address, cfg):
        self.port = port
        self.baud = baud
        self.address = address
        self.cfg = cfg
        self.ser = None
        self.serial_error = False   # latched; read by health checks -> MOTOR_SERIAL_ERROR

    def open(self):
        # Opens the port, waits for the controller, then sets auto-stop and deadband.
        try:
            self.ser = serial.Serial(self.port, baudrate=self.baud,
                                     timeout=self.cfg.SABER_READ_TIMEOUT_S)
        except (serial.SerialException, OSError) as e:
            print(f"Serial Port Error: {e}")
            self.serial_error = True
            return

        time.sleep(self.cfg.SABER_STARTUP_WAIT_S)

        self.set_auto_stop(self.cfg.SABER_TIMEOUT_MS)
        self.set_deadband(self.cfg.SABER_HW_DEADBAND)

    def set(self, cmd: MotorCmd):
        # Sends one packet per motor. Call every control loop, even if unchanged.
        motors = (
            (cmd.left, self.cfg.LEFT_CHANNEL, self.cfg.LEFT_INVERT),
            (cmd.right, self.cfg.RIGHT_CHANNEL, self.cfg.RIGHT_INVERT),
        )
        for x, channel, invert in motors:
            if invert:
                x = -x
            self._write(self._side_packet(self._with_deadband(x), channel))

    def stop(self):
        # Speed 0 to both motors; called by shutdown and SAFE_STOP.
        self.send_command(CMD_M1_FORWARD, 0)
        self.send_command(CMD_M2_FORWARD, 0)

    def send_command(self, command, value):
        # Sends a properly formatted packetized serial command to the Sabertooth.
        self._write(self._packet(command, value))

    def set_auto_stop(self, timeout_ms):
        # Sets the serial timeout period: how long the driver waits without a
        # command before shutting off. 100 ms units, range 0-12700 ms; 0 disables.
        value = min(max(int(timeout_ms / 100), 0), MAX_VALUE)
        self.send_command(CMD_SERIAL_TIMEOUT, value)

    def set_deadband(self, deadband_value):
        # Sets the Sabertooth's own deadband. Default 3; 0 resets to default.
        # This setting persists through a power cycle.
        if not (0 <= deadband_value <= MAX_VALUE):
            raise ValueError("Deadband value must be between 0 and 127.")
        self.send_command(CMD_DEADBAND, deadband_value)

    def close(self):
        # Stops the motors, then releases the port.
        self.stop()
        try:
            if self.ser is not None:
                self.ser.close()
        except (serial.SerialException, OSError):
            pass
        self.ser = None

    def _side_packet(self, x, channel):
        # x in -1..1 -> forward or backward command for that motor channel.
        x = max(-1.0, min(1.0, x))
        forward, backward = _CHANNEL_COMMANDS[channel]
        command = forward if x >= 0 else backward
        value = self.map_integer(abs(x), 0, 1, 0, MAX_VALUE)
        return self._packet(command, value)

    def _packet(self, command, value):
        # address, command, value, (address + command + value) & 0x7F
        if not (0 <= value <= MAX_VALUE):
            raise ValueError("Value must be between 0 and 127")
        address_byte = int(self.address)
        command_byte = int(command)
        data_byte = int(value)
        checksum = (address_byte + command_byte + data_byte) & 0x7F
        return bytes([address_byte, command_byte, data_byte, checksum])

    def _with_deadband(self, x):
        # Clamps to -1..1, then maps non-zero commands from (0, 1] onto
        # (MOTOR_DEADBAND, 1] so small commands still move the wheels.
        x = max(-1.0, min(1.0, x))
        if x == 0:
            return 0.0
        lift = self.cfg.MOTOR_DEADBAND
        return math.copysign(lift + (1.0 - lift) * abs(x), x)

    def _write(self, packet):
        # Writes without flush(); a serial failure latches serial_error
        # instead of raising, so the control loop can go to SAFE_STOP.
        if self.ser is None:
            self.serial_error = True
            return
        try:
            self.ser.write(packet)
        except (serial.SerialException, OSError):
            self.serial_error = True

    @staticmethod
    def map_integer(value, old_min, old_max, new_min, new_max):
        if old_max - old_min == 0:
            raise ValueError("Old range cannot be zero")
        return round(new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min))