"""Tests for hal/sabertooth.py with the serial port faked.

Run with:  python test/test_sabertooth.py   (or: python -m pytest)
"""
import os
import sys

import pytest
import serial

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from hal import sabertooth
from hal.sabertooth import SabertoothDriver
from robot_types import MotorCmd


class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.writes = []
        self.fail = False
        self.closed = False

    def write(self, data):
        if self.fail:
            raise serial.SerialException("unplugged")
        self.writes.append(bytes(data))

    def close(self):
        self.closed = True


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(sabertooth.serial, "Serial", FakeSerial)
    monkeypatch.setattr(sabertooth.time, "sleep", lambda s: None)
    # Packets below assume the default wiring, whatever the robot's config says.
    for name, value in (("LEFT_CHANNEL", 1), ("RIGHT_CHANNEL", 2),
                        ("LEFT_INVERT", False), ("RIGHT_INVERT", False)):
        monkeypatch.setattr(config, name, value)
    d = SabertoothDriver(config.SABER_PORT, config.SABER_BAUD, config.SABER_ADDRESS, config)
    d.open()
    d.ser.writes.clear()
    return d


def pkt(command, value, address=128):
    return bytes([address, command, value, (address + command + value) & 0x7F])


def test_packet_checksum(driver):
    assert driver._packet(0, 64) == bytes([128, 0, 64, 64])
    assert driver._packet(4, 127) == bytes([128, 4, 127, 3])


def test_packet_rejects_bad_value(driver):
    with pytest.raises(ValueError):
        driver._packet(0, 128)


def test_open_sets_auto_stop_and_deadband(monkeypatch):
    monkeypatch.setattr(sabertooth.serial, "Serial", FakeSerial)
    waits = []
    monkeypatch.setattr(sabertooth.time, "sleep", waits.append)
    d = SabertoothDriver(config.SABER_PORT, config.SABER_BAUD, config.SABER_ADDRESS, config)
    d.open()
    assert waits == [config.SABER_STARTUP_WAIT_S]
    assert d.ser.writes == [pkt(14, 2), pkt(17, config.SABER_HW_DEADBAND)]


def test_open_failure_latches_error(monkeypatch):
    def boom(*a, **k):
        raise serial.SerialException("no port")
    monkeypatch.setattr(sabertooth.serial, "Serial", boom)
    d = SabertoothDriver("/dev/none", 9600, 128, config)
    d.open()
    assert d.serial_error
    d.set(MotorCmd(0.5, 0.5))   # must not raise
    assert d.serial_error


def test_full_forward(driver):
    driver.set(MotorCmd(1.0, 1.0))
    assert driver.ser.writes == [pkt(0, 127), pkt(4, 127)]


def test_reverse_and_zero(driver):
    driver.set(MotorCmd(-1.0, 0.0))
    assert driver.ser.writes == [pkt(1, 127), pkt(4, 0)]


def test_out_of_range_is_clamped(driver):
    driver.set(MotorCmd(2.5, -3.0))
    assert driver.ser.writes == [pkt(0, 127), pkt(5, 127)]


def test_small_command_is_lifted_above_deadband(driver):
    driver.set(MotorCmd(0.01, -0.01))
    left, right = driver.ser.writes
    floor = round(config.MOTOR_DEADBAND * 127)
    assert left[1] == 0 and left[2] >= floor
    assert right[1] == 5 and right[2] >= floor


def test_invert_flag_flips_direction(driver, monkeypatch):
    monkeypatch.setattr(config, "LEFT_INVERT", True)
    driver.set(MotorCmd(1.0, 1.0))
    assert driver.ser.writes == [pkt(1, 127), pkt(4, 127)]


def test_stop_sends_zero_to_both(driver):
    driver.stop()
    assert driver.ser.writes == [pkt(0, 0), pkt(4, 0)]


def test_write_failure_latches_error_without_raising(driver):
    driver.ser.fail = True
    driver.set(MotorCmd(0.5, 0.5))
    assert driver.serial_error


def test_close_stops_then_releases(driver):
    fake = driver.ser
    driver.close()
    assert fake.writes == [pkt(0, 0), pkt(4, 0)]
    assert fake.closed and driver.ser is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
