"""BNO055 IMU reader: a thread at IMU_HZ that publishes ImuSample.

The sensor runs in IMUPLUS mode (gyro + accel fusion, magnetometer off), set
up with the same library calls as the course IMUDevice (imu_controller.py,
Professor McLaughlin). All I2C traffic, including open and close, happens on
the IMU thread.

yaw_deg is relative to the first good reading, unwrapped (it keeps counting
past +/-180 and 360) and drift-corrected; + = counter-clockwise (left). The
BNO055 heading grows clockwise, so IMU_YAW_SIGN flips it. Gyro z already
reads + counter-clockwise, so it has its own sign, IMU_GYRO_SIGN.

A read that is None, garbage, or implies a rate above IMU_MAX_RATE_DPS is
rejected and nothing is published, so the slot keeps the last good sample.
After IMU_MAX_REJECTS rejects in a row, valid=False is published every tick
until a good read comes back.

Drift is measured while the drift_request Event is set (by Behaviour, when
the robot stands still): after IMU_DRIFT_SETTLE_S, the slope of the
uncorrected yaw over each IMU_STILL_S window. A slope under IMU_DRIFT_MAX_DPS
is stored and subtracted from then on. Clearing the Event drops an unfinished
window. SETTLE + STILL < STOP_HOLD_S, so one window fits in a stop (checked in
test/test_imu.py). The control loop never waits for it.
"""
import math
import threading
import time
import traceback

from robot_types import ImuSample


class Bno055:
    """Opens the BNO055 on the Pi's I2C pins, as the course IMUDevice does."""

    def __init__(self, address, cfg):
        # imported here so tests and laptops don't need Blinka
        import adafruit_bno055
        import board
        import busio

        self.cfg = cfg
        self._lib = adafruit_bno055
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(self.i2c, address=address)
        self.sensor.mode = adafruit_bno055.IMUPLUS_MODE
        time.sleep(cfg.IMU_MODE_SETTLE_S)

    def close(self):
        # CONFIG_MODE, then release the bus; each step runs even if one fails.
        try:
            self.sensor.mode = self._lib.CONFIG_MODE
            time.sleep(self.cfg.IMU_MODE_SETTLE_S)
        except Exception:
            pass
        try:
            self.i2c.deinit()
        except Exception:
            pass


class ImuReader:
    def __init__(self, slot, drift_request, cfg, device_factory=Bno055):
        # slot: LatestValue[ImuSample]. drift_request: threading.Event, or None.
        # device_factory(address, cfg) -> object with .sensor and .close();
        # tests pass a fake.
        self.slot = slot
        self.drift_request = drift_request
        self.cfg = cfg
        self.device_factory = device_factory
        self.error_count = 0             # reads that raised
        self._dev = None
        self._stop = threading.Event()
        self._thread = None
        self._h0 = None                  # first good heading = yaw 0
        self._yaw_raw = 0.0              # last good yaw, before drift correction
        self._t_good = None              # time of the last good read
        self._correction = 0.0           # accumulated drift_dps x time
        self._drift_dps = 0.0
        self._drift_set_t = None         # first good read with drift_request set
        self._drift_start = None         # (t, yaw_raw) of the open drift window
        self._calib = 0
        self._rejects = 0

    def start(self):
        # Starts the thread; it opens the sensor in IMUPLUS mode itself.
        self._thread = threading.Thread(target=self._loop, name="imu", daemon=True)
        self._thread.start()

    def close(self):
        # Stops the thread, which sets CONFIG_MODE and releases I2C on its way
        # out. Never raises. Called by shutdown after the motors stop.
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=self.cfg.IMU_JOIN_TIMEOUT_S)
                if self._thread.is_alive():
                    print("IMU thread did not stop in time")
        except Exception as e:
            print(f"IMU close failed: {e}")

    def read_raw(self):
        # The only function that reads the sensor.
        # -> (heading_deg, gyro_z_dps, calib_gyro) in the sensor's own sign,
        # or None on any failure.
        if self._dev is None:
            return None
        try:
            sensor = self._dev.sensor
            heading = sensor.euler[0]
            gyro_z = sensor.gyro[2]                  # rad/s
            calib = sensor.calibration_status[1]     # (sys, gyro, accel, mag)
            if heading is None or gyro_z is None:
                return None
            return float(heading), math.degrees(gyro_z), int(calib)
        except Exception:
            self.error_count += 1
            if self.error_count == 1:
                print("IMU read error (further errors counted only):")
                traceback.print_exc()
            return None

    # ---------------------------------------------------------------- IMU thread

    def _loop(self):
        # Fixed rate at IMU_HZ; an overrun restarts the schedule from now.
        self._open()
        try:
            period = 1.0 / self.cfg.IMU_HZ
            next_t = time.monotonic()
            while not self._stop.is_set():
                self._step(time.monotonic())
                next_t += period
                wait = next_t - time.monotonic()
                if wait > 0:
                    self._stop.wait(wait)
                else:
                    next_t = time.monotonic()
        finally:
            self._close_device()

    def _open(self):
        # A sensor that fails to open reads as None every tick, so valid=False
        # is published after IMU_MAX_REJECTS ticks.
        try:
            self._dev = self.device_factory(self.cfg.IMU_I2C_ADDR, self.cfg)
        except Exception:
            print("IMU did not open; publishing valid=False:")
            traceback.print_exc()
            self._dev = None

    def _close_device(self):
        if self._dev is None:
            return
        try:
            self._dev.close()
        except Exception as e:
            print(f"IMU device close failed: {e}")
        self._dev = None

    def _step(self, now):
        # One tick: read, check, then publish or count a reject.
        raw = self.read_raw()
        if raw is not None and self._sane(raw):
            heading, gyro_dps, calib = raw
            self._calib = calib
            if self._h0 is None:
                self._h0, self._t_good = heading, now
            new_yaw = self._to_robot_yaw(heading, self._yaw_raw)
            if self._plausible(new_yaw, self._yaw_raw, now - self._t_good):
                self._accept(new_yaw, gyro_dps, now)
                return
        self._reject(now)

    def _sane(self, raw):
        # Finite heading inside 0-360 and a gyro rate the robot can make.
        heading, gyro_dps, _ = raw
        return (math.isfinite(heading) and 0.0 <= heading <= 360.0
                and math.isfinite(gyro_dps) and abs(gyro_dps) <= self.cfg.IMU_MAX_RATE_DPS)

    def _to_robot_yaw(self, heading_deg, prev_yaw):
        # Unwrapped yaw (+ = left) from a 0-360 heading: prev_yaw plus the
        # shortest step from the heading prev_yaw stands for.
        sign = self.cfg.IMU_YAW_SIGN
        prev_heading = self._h0 + sign * prev_yaw
        step = (heading_deg - prev_heading + 180.0) % 360.0 - 180.0
        return prev_yaw + sign * step

    def _plausible(self, new_yaw, prev_yaw, dt):
        # dt is the time since the last good read, so a run of rejects
        # widens the window rather than locking the reader out.
        if dt <= 0:
            return new_yaw == prev_yaw
        return abs(new_yaw - prev_yaw) / dt <= self.cfg.IMU_MAX_RATE_DPS

    def _accept(self, yaw_raw, gyro_dps, now):
        self._correction += self._drift_dps * (now - self._t_good)
        self._yaw_raw, self._t_good, self._rejects = yaw_raw, now, 0
        self._track_drift(now)
        rate = self.cfg.IMU_GYRO_SIGN * gyro_dps - self._drift_dps
        self.slot.set(ImuSample(yaw_deg=yaw_raw - self._correction, yaw_rate_dps=rate,
                                valid=True, calib_gyro=self._calib,
                                drift_dps=self._drift_dps, t=now))

    def _reject(self, now):
        # Holds the last good value; valid=False once rejects reach the limit.
        self._rejects += 1
        if self._rejects < self.cfg.IMU_MAX_REJECTS:
            return
        self.slot.set(ImuSample(yaw_deg=self._yaw_raw - self._correction, yaw_rate_dps=0.0,
                                valid=False, calib_gyro=self._calib,
                                drift_dps=self._drift_dps, t=now))

    def _track_drift(self, now):
        # Back-to-back IMU_STILL_S windows while drift_request is set, the
        # first opening IMU_DRIFT_SETTLE_S after it was set (robot settling).
        if self.drift_request is None or not self.drift_request.is_set():
            self._drift_set_t = self._drift_start = None
            return
        if self._drift_set_t is None:
            self._drift_set_t = now
        if now - self._drift_set_t < self.cfg.IMU_DRIFT_SETTLE_S:
            return
        if self._drift_start is None:
            self._drift_start = (now, self._yaw_raw)
            return
        t0, y0 = self._drift_start
        if now - t0 < self.cfg.IMU_STILL_S:
            return
        measured = (self._yaw_raw - y0) / (now - t0)
        if abs(measured) < self.cfg.IMU_DRIFT_MAX_DPS:
            self._drift_dps = measured
        else:
            print(f"IMU drift {measured:+.3f} deg/s rejected (limit {self.cfg.IMU_DRIFT_MAX_DPS})")
        self._drift_start = (now, self._yaw_raw)
