"""Tests for hal/imu.py with a fake BNO055 (no Pi or sensor needed).

Most tests call ImuReader._step(t) directly with chosen times, so they are
exact and need no thread. The thread itself is tested at the end.

Run with:  python -m pytest test/test_imu.py
"""
import math
import os
import sys
import threading
import time
import types

import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from hal.imu import Bno055, ImuReader
from robot_types import LatestValue

DT = 1.0 / config.IMU_HZ


class FakeSensor:
    """BNO055 stand-in. Set .heading, .gyro_dps and .calib, or .error to raise."""

    def __init__(self, heading=0.0, gyro_dps=0.0, calib=3):
        self.heading, self.gyro_dps, self.calib = heading, gyro_dps, calib
        self.error = None
        self.mode = None

    @property
    def euler(self):
        if self.error is not None:
            raise self.error
        return (self.heading, 0.0, 0.0)

    @property
    def gyro(self):
        g = None if self.gyro_dps is None else math.radians(self.gyro_dps)
        return (0.0, 0.0, g)

    @property
    def calibration_status(self):
        return (0, self.calib, 0, 0)


class FakeDevice:
    def __init__(self, sensor=None, close_error=None):
        self.sensor = sensor or FakeSensor()
        self.closed = 0
        self.close_error = close_error

    def close(self):
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def make_reader(sensor=None, drift_request=None):
    dev = FakeDevice(sensor)
    reader = ImuReader(LatestValue(), drift_request, config, device_factory=lambda a, c: dev)
    reader._open()
    return reader, dev.sensor


def feed(reader, sensor, headings, t0=0.0, dt=DT):
    # One _step per heading, dt apart; returns the time after the last step.
    t = t0
    for h in headings:
        sensor.heading = h
        reader._step(t)
        t += dt
    return t


def yaw(reader):
    return reader.slot.get().yaw_deg


# ---------------------------------------------------------------- sign and unwrap

def test_first_reading_is_zero():
    reader, sensor = make_reader(FakeSensor(heading=123.0))
    reader._step(0.0)
    s = reader.slot.get()
    assert s.yaw_deg == 0.0 and s.valid and s.calib_gyro == 3 and s.t == 0.0


def test_clockwise_heading_is_negative_yaw():
    # Heading +10 (clockwise, right) -> yaw -10 with IMU_YAW_SIGN = -1.
    reader, sensor = make_reader()
    assert config.IMU_YAW_SIGN == -1
    feed(reader, sensor, [100.0, 105.0, 110.0])
    assert yaw(reader) == pytest.approx(-10.0)


def test_sign_constant_flips_yaw(monkeypatch):
    monkeypatch.setattr(config, "IMU_YAW_SIGN", 1)
    reader, sensor = make_reader()
    feed(reader, sensor, [100.0, 105.0, 110.0])
    assert yaw(reader) == pytest.approx(10.0)


def test_gyro_rate_sign():
    # Counter-clockwise gyro z (+) -> yaw_rate + (left) with IMU_GYRO_SIGN = 1.
    reader, sensor = make_reader(FakeSensor(gyro_dps=20.0))
    reader._step(0.0)
    assert reader.slot.get().yaw_rate_dps == pytest.approx(config.IMU_GYRO_SIGN * 20.0)


def test_unwrap_clockwise_across_zero():
    # 350 -> 10 is 20 deg clockwise, not 340 counter-clockwise.
    reader, sensor = make_reader()
    feed(reader, sensor, [350.0, 355.0, 359.0, 0.0, 5.0, 10.0])
    assert yaw(reader) == pytest.approx(config.IMU_YAW_SIGN * 20.0)


def test_unwrap_counter_clockwise_across_zero():
    reader, sensor = make_reader()
    feed(reader, sensor, [10.0, 5.0, 1.0, 0.0, 359.0, 355.0, 350.0])
    assert yaw(reader) == pytest.approx(config.IMU_YAW_SIGN * -20.0)


@pytest.mark.parametrize("direction", [1, -1])
def test_full_spin_reaches_360_without_wrapping(direction):
    # 10 deg every 0.05 s = 200 deg/s, under IMU_MAX_RATE_DPS.
    reader, sensor = make_reader()
    headings = [(30.0 + direction * 10.0 * i) % 360.0 for i in range(37)]
    feed(reader, sensor, headings, dt=0.05)
    assert yaw(reader) == pytest.approx(config.IMU_YAW_SIGN * direction * 360.0)


def test_two_spins_keep_counting():
    reader, sensor = make_reader()
    feed(reader, sensor, [(10.0 * i) % 360.0 for i in range(73)], dt=0.05)
    assert yaw(reader) == pytest.approx(config.IMU_YAW_SIGN * 720.0)


# ---------------------------------------------------------------- rejects

def test_spike_is_rejected_and_last_good_value_held():
    reader, sensor = make_reader()
    t = feed(reader, sensor, [0.0, 1.0])
    held = reader.slot.get()
    sensor.heading = 90.0                           # 89 deg in one tick: far above the limit
    reader._step(t)
    assert reader.slot.get() is held
    sensor.heading = 2.0                            # back on track
    reader._step(t + DT)
    assert reader.slot.get().valid and yaw(reader) == pytest.approx(config.IMU_YAW_SIGN * 2.0)


def test_rate_limit_uses_time_since_last_good_read():
    # 10 deg in one tick (500 deg/s) is too fast; after 1 s it is not.
    reader, sensor = make_reader()
    feed(reader, sensor, [0.0])
    sensor.heading = 10.0
    reader._step(DT)
    assert yaw(reader) == 0.0
    reader._step(1.0)
    assert yaw(reader) == pytest.approx(config.IMU_YAW_SIGN * 10.0)


def test_invalid_after_max_rejects_then_recovers():
    reader, sensor = make_reader()
    t = feed(reader, sensor, [0.0, 1.0])
    good = reader.slot.get()
    sensor.heading = None
    for _ in range(config.IMU_MAX_REJECTS - 1):
        reader._step(t)
        t += DT
    assert reader.slot.get() is good                # still holding the last good sample
    reader._step(t)
    bad = reader.slot.get()
    assert not bad.valid and bad.yaw_deg == good.yaw_deg and bad.t == t
    t += DT
    sensor.heading = 1.5
    reader._step(t)
    assert reader.slot.get().valid and yaw(reader) == pytest.approx(config.IMU_YAW_SIGN * 1.5)


@pytest.mark.parametrize("field,value", [
    ("heading", None), ("heading", float("nan")), ("heading", 400.0), ("heading", -5.0),
    ("gyro_dps", None), ("gyro_dps", float("inf")), ("gyro_dps", config.IMU_MAX_RATE_DPS + 1),
])
def test_garbage_reads_are_rejected(field, value):
    reader, sensor = make_reader()
    feed(reader, sensor, [0.0])
    good = reader.slot.get()
    setattr(sensor, field, value)
    reader._step(DT)
    assert reader.slot.get() is good


def test_sensor_exception_is_a_reject_and_counted(capsys):
    reader, sensor = make_reader()
    feed(reader, sensor, [0.0])
    sensor.error = OSError("I2C remote I/O error")
    t = DT
    for _ in range(config.IMU_MAX_REJECTS):
        reader._step(t)
        t += DT
    assert not reader.slot.get().valid
    assert reader.error_count == config.IMU_MAX_REJECTS
    assert capsys.readouterr().out.count("IMU read error") == 1
    sensor.error = None
    reader._step(t)
    assert reader.slot.get().valid


def test_sensor_that_never_opens_publishes_invalid(capsys):
    def broken(address, cfg):
        raise ValueError("No I2C device at address: 0x28")
    reader = ImuReader(LatestValue(), None, config, device_factory=broken)
    reader._open()
    for i in range(config.IMU_MAX_REJECTS):
        reader._step(i * DT)
    s = reader.slot.get()
    assert not s.valid and s.yaw_deg == 0.0 and s.calib_gyro == 0
    assert "IMU did not open" in capsys.readouterr().out


# ---------------------------------------------------------------- drift

# Time from setting drift_request until the first window has closed.
FIRST_WINDOW_S = config.IMU_DRIFT_SETTLE_S + config.IMU_STILL_S + 2 * DT
# A drift rate inside IMU_DRIFT_MAX_DPS, so it is stored.
CREEP = config.IMU_DRIFT_MAX_DPS / 2


def creep(reader, sensor, t0, seconds, rate_dps, yaw0=0.0):
    # Feeds a yaw that creeps from yaw0 at rate_dps (robot sign) for
    # `seconds`, one read per DT. Returns (next t, next yaw).
    n = int(round(seconds / DT))
    yaws = [yaw0 + rate_dps * i * DT for i in range(n)]
    feed(reader, sensor, [(config.IMU_YAW_SIGN * y) % 360.0 for y in yaws], t0=t0)
    return t0 + n * DT, yaw0 + rate_dps * n * DT


def requested():
    request = threading.Event()
    request.set()
    return request


def test_drift_window_fits_inside_stop_hold():
    assert config.IMU_DRIFT_SETTLE_S + config.IMU_STILL_S < config.STOP_HOLD_S


def test_drift_is_stored_during_one_stop_hold():
    # drift_request set for exactly STOP_HOLD_S, as Behaviour does in STOP_HOLD.
    request = requested()
    reader, sensor = make_reader(drift_request=request)
    t, y = creep(reader, sensor, 0.0, config.STOP_HOLD_S, CREEP)
    request.clear()
    creep(reader, sensor, t, 1.0, CREEP, yaw0=y)
    assert reader.slot.get().drift_dps == pytest.approx(CREEP, rel=1e-6)


def test_settling_motion_is_left_out_of_the_window():
    # The robot rocks 3 deg in the first IMU_DRIFT_SETTLE_S, then creeps at CREEP.
    request = requested()
    reader, sensor = make_reader(drift_request=request)
    settle_rate = 3.0 / config.IMU_DRIFT_SETTLE_S
    t, y = creep(reader, sensor, 0.0, config.IMU_DRIFT_SETTLE_S, settle_rate)
    creep(reader, sensor, t, config.IMU_STILL_S + 2 * DT, CREEP, yaw0=y)
    assert reader.slot.get().drift_dps == pytest.approx(CREEP, rel=1e-6)


def test_no_drift_before_settle_plus_window():
    request = requested()
    reader, sensor = make_reader(drift_request=request)
    creep(reader, sensor, 0.0, config.IMU_DRIFT_SETTLE_S + config.IMU_STILL_S - 2 * DT, CREEP)
    assert reader.slot.get().drift_dps == 0.0


def test_settle_restarts_on_each_request():
    # A request cleared during settling and set again waits the full settle.
    request = requested()
    reader, sensor = make_reader(drift_request=request)
    t, y = creep(reader, sensor, 0.0, config.IMU_DRIFT_SETTLE_S / 2, CREEP)
    request.clear()
    t, y = creep(reader, sensor, t, DT, CREEP, yaw0=y)
    request.set()
    creep(reader, sensor, t, config.IMU_DRIFT_SETTLE_S + config.IMU_STILL_S - 2 * DT, CREEP, yaw0=y)
    assert reader.slot.get().drift_dps == 0.0


def test_drift_is_subtracted_after_storing():
    request = requested()
    reader, sensor = make_reader(drift_request=request)
    t, y = creep(reader, sensor, 0.0, FIRST_WINDOW_S, CREEP)
    request.clear()
    yaw_before = yaw(reader)
    # 10 s more of the same creep: the correction cancels it.
    creep(reader, sensor, t, 10.0, CREEP, yaw0=y)
    assert yaw(reader) == pytest.approx(yaw_before, abs=1e-6)


def test_drift_corrects_yaw_rate():
    reader, sensor = make_reader(drift_request=requested())
    creep(reader, sensor, 0.0, FIRST_WINDOW_S, CREEP)
    assert reader.slot.get().yaw_rate_dps == pytest.approx(-CREEP, rel=1e-6)


def test_drift_above_max_is_not_stored(capsys):
    reader, sensor = make_reader(drift_request=requested())
    creep(reader, sensor, 0.0, FIRST_WINDOW_S, config.IMU_DRIFT_MAX_DPS * 2)
    assert reader.slot.get().drift_dps == 0.0
    assert "rejected" in capsys.readouterr().out


def test_cleared_request_drops_unfinished_window():
    request = requested()
    reader, sensor = make_reader(drift_request=request)
    t, y = creep(reader, sensor, 0.0, config.IMU_DRIFT_SETTLE_S + config.IMU_STILL_S / 2, CREEP)
    request.clear()
    creep(reader, sensor, t, FIRST_WINDOW_S, CREEP, yaw0=y)
    assert reader.slot.get().drift_dps == 0.0


def test_no_drift_measured_without_request():
    reader, sensor = make_reader()
    creep(reader, sensor, 0.0, FIRST_WINDOW_S * 2, CREEP)
    assert reader.slot.get().drift_dps == 0.0


# ---------------------------------------------------------------- thread and close

def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_thread_publishes_and_close_releases_device():
    dev = FakeDevice(FakeSensor(heading=42.0))
    opened_on = []

    def factory(address, cfg):
        opened_on.append(threading.current_thread().name)
        assert address == config.IMU_I2C_ADDR
        return dev

    reader = ImuReader(LatestValue(), None, config, device_factory=factory)
    reader.start()
    assert wait_for(lambda: reader.slot.get() is not None)
    s = reader.slot.get()
    assert s.valid and s.yaw_deg == 0.0
    reader.close()
    assert not reader._thread.is_alive()
    assert dev.closed == 1 and opened_on == ["imu"]


def test_close_never_raises():
    dev = FakeDevice(close_error=OSError("bus gone"))
    reader = ImuReader(LatestValue(), None, config, device_factory=lambda a, c: dev)
    reader.start()
    assert wait_for(lambda: reader.slot.get() is not None)
    reader.close()
    assert dev.closed == 1
    ImuReader(LatestValue(), None, config).close()  # never started


# ---------------------------------------------------------------- Bno055 (fake library)

class FakeI2C:
    def __init__(self, scl, sda):
        self.pins = (scl, sda)
        self.deinit_calls = 0

    def deinit(self):
        self.deinit_calls += 1


class FakeBNO055_I2C:
    def __init__(self, i2c, address):
        self.i2c, self.address = i2c, address
        self.modes = []

    @property
    def mode(self):
        return self.modes[-1]

    @mode.setter
    def mode(self, value):
        self.modes.append(value)


@pytest.fixture
def blinka(monkeypatch):
    monkeypatch.setitem(sys.modules, "board", types.SimpleNamespace(SCL="SCL", SDA="SDA"))
    monkeypatch.setitem(sys.modules, "busio", types.SimpleNamespace(I2C=FakeI2C))
    monkeypatch.setitem(sys.modules, "adafruit_bno055", types.SimpleNamespace(
        BNO055_I2C=FakeBNO055_I2C, IMUPLUS_MODE=0x08, CONFIG_MODE=0x00))
    monkeypatch.setattr(config, "IMU_MODE_SETTLE_S", 0.0)


def test_bno055_opens_in_imuplus_mode(blinka):
    dev = Bno055(0x28, config)
    assert dev.i2c.pins == ("SCL", "SDA")
    assert dev.sensor.address == 0x28 and dev.sensor.modes == [0x08]


def test_bno055_close_sets_config_mode_and_deinits(blinka):
    dev = Bno055(0x28, config)
    dev.close()
    assert dev.sensor.modes == [0x08, 0x00] and dev.i2c.deinit_calls == 1


class FailingCloseBNO055(FakeBNO055_I2C):
    # Accepts the first mode write (open), raises on the next (close).
    @FakeBNO055_I2C.mode.setter
    def mode(self, value):
        if self.modes:
            raise OSError("I2C error")
        self.modes.append(value)


def test_bno055_close_deinits_even_if_mode_write_fails(blinka):
    sys.modules["adafruit_bno055"].BNO055_I2C = FailingCloseBNO055
    dev = Bno055(0x28, config)
    dev.close()
    assert dev.i2c.deinit_calls == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
