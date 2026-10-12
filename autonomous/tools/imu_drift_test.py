"""IMU bench check: sign check and drift measurement, through ImuReader.

Three runs, in this order, from the autonomous/ folder. Stop main.py first
(Ctrl-C in its terminal) so nothing else drives the motors. Ctrl-C ends any
run and prints a summary line.

Each output line (DRIFT_TEST_PRINT_HZ per second):
    yaw      deg since the run started; + = robot turned left
    rate     gyro yaw rate, deg/s; + = turning left
    dyaw/dt  yaw change per second since the previous line
    valid    1 = good reads, 0 = the IMU is failing
    calib    gyro calibration, 0-3; 3 = calibrated
    drift    drift stored so far, deg/s (subtracted from yaw)

----------------------------------------------------------------------------
RUN 1: SIGN CHECK (motors off, about 1 minute)

    python -m tools.imu_drift_test

 1. Put the robot on the floor or bench. Wheels may touch the ground; the
    motors are not used.
 2. Leave it untouched until calib shows 3 (a few seconds). yaw and rate
    should sit near 0.
 3. Turn the robot LEFT (counter-clockwise, seen from above) by hand, about
    90 deg, steadily over 1-2 s.
    Expect: yaw rises to about +90; rate and dyaw/dt are both positive while
    it turns; no line says SIGN MISMATCH.
 4. Turn it back RIGHT to where it started.
    Expect: yaw returns to about 0; rate and dyaw/dt both negative.
 5. Turn it LEFT one full circle (360 deg).
    Expect: yaw keeps counting up to about +360 (it must not jump back to
    -180 or 0 on the way).
 6. Ctrl-C.

 If yaw FALLS when you turn left:  set IMU_YAW_SIGN = 1 in config.py, rerun.
 If yaw is right but SIGN MISMATCH appears while turning:  flip
 IMU_GYRO_SIGN (1 <-> -1) in config.py, rerun.
 Repeat until a left turn gives rising yaw and no mismatch lines.

----------------------------------------------------------------------------
RUN 2: DRIFT, MOTORS OFF (1 minute)

    python -m tools.imu_drift_test --still --seconds 60

 1. Robot on the floor, motors off. Start the run and step away: do not
    touch the robot or bump the table for the whole minute.
 2. The first drift value appears after about IMU_DRIFT_SETTLE_S +
    IMU_STILL_S (2.5 s), then a new one every IMU_STILL_S (2 s).
 3. Watch the drift column: it should settle to a small, steady number.
    A line "IMU drift ... rejected" means one window was over
    IMU_DRIFT_MAX_DPS (usually the robot was bumped).
 4. Write down: the drift values you see, and the summary line at the end
    ("corrected yaw moved ..." should be close to 0 deg).

----------------------------------------------------------------------------
RUN 3: DRIFT, MOTORS RUNNING (1 minute; ROBOT ON A STAND)

    python -m tools.imu_drift_test --motors --seconds 60

 1. Put the robot on its stand, all wheels clear of the ground and of
    anything else. Keep hands clear.
 2. Start the run: both wheels spin forward at DRIFT_TEST_SPEED for the
    whole minute. Ctrl-C stops them at once.
 3. Do not touch the robot. Watch the drift column as in run 2.
 4. Write down the drift values and the summary line, and compare them
    with run 2. Motor vibration can make the drift larger.

----------------------------------------------------------------------------
AFTERWARDS

    From summaries, set in config.py:
    IMU_DRIFT_WARN_DPS  a little above the largest drift seen in runs 2-3
    IMU_DRIFT_MAX_DPS   above IMU_DRIFT_WARN_DPS, so normal drift is kept
                        but a bumped measurement is thrown away
"""
import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from hal.imu import ImuReader
from hal.sabertooth import SabertoothDriver
from robot_types import LatestValue, MotorCmd


@dataclass
class Run:
    first: object = None                 # first ImuSample printed
    last: object = None                  # last ImuSample printed
    mismatches: int = 0


def parse_args():
    p = argparse.ArgumentParser(description="IMU bench check through ImuReader.")
    p.add_argument("--still", action="store_true", help="measure drift with the motors off")
    p.add_argument("--motors", action="store_true",
                   help="measure drift with the wheels spinning (robot on a stand)")
    p.add_argument("--seconds", type=float, default=None, help="stop after this long")
    return p.parse_args()


def yaw_change_dps(sample, prev):
    # Yaw change per second between two printed samples, or None.
    if prev is None or sample.t <= prev.t:
        return None
    return (sample.yaw_deg - prev.yaw_deg) / (sample.t - prev.t)


def sign_mismatch(rate_dps, change_dps, cfg):
    # True when both rates are clearly non-zero and point opposite ways.
    if change_dps is None:
        return False
    floor = cfg.DRIFT_TEST_SIGN_MIN_DPS
    if abs(rate_dps) < floor or abs(change_dps) < floor:
        return False
    return (rate_dps > 0) != (change_dps > 0)


def status_line(sample, change_dps, mismatch):
    change = f"{change_dps:+7.2f}" if change_dps is not None else "     --"
    flag = "  SIGN MISMATCH (IMU_YAW_SIGN vs IMU_GYRO_SIGN)" if mismatch else ""
    return (f"yaw {sample.yaw_deg:+9.2f} deg  rate {sample.yaw_rate_dps:+7.2f}  "
            f"dyaw/dt {change}  valid {int(sample.valid)}  calib {sample.calib_gyro}  "
            f"drift {sample.drift_dps:+.3f} deg/s{flag}")


def report(slot, saber, run, seconds, cfg):
    # Prints a line per tick; keeps the Sabertooth fed when spinning.
    period = 1.0 / cfg.DRIFT_TEST_PRINT_HZ
    start = time.monotonic()
    while seconds is None or time.monotonic() - start < seconds:
        if saber is not None:
            saber.set(MotorCmd(left=cfg.DRIFT_TEST_SPEED, right=cfg.DRIFT_TEST_SPEED))
        sample = slot.get()
        if sample is not None:
            change = yaw_change_dps(sample, run.last)
            mismatch = sign_mismatch(sample.yaw_rate_dps, change, cfg)
            run.mismatches += mismatch
            print(status_line(sample, change, mismatch))
            run.first = run.first or sample
            run.last = sample
        time.sleep(period)


def open_saber(cfg):
    saber = SabertoothDriver(cfg.SABER_PORT, cfg.SABER_BAUD, cfg.SABER_ADDRESS, cfg)
    saber.open()
    if saber.serial_error:
        raise SystemExit("Sabertooth port did not open")
    print(f"Wheels at {cfg.DRIFT_TEST_SPEED}: robot must be on a stand")
    return saber


def summary(run):
    if run.first is None or run.last is None or run.last.t <= run.first.t:
        return "No IMU samples received"
    span = run.last.t - run.first.t
    moved = run.last.yaw_deg - run.first.yaw_deg
    return (f"Stored drift {run.last.drift_dps:+.3f} deg/s; corrected yaw moved {moved:+.2f} deg "
            f"in {span:.1f} s ({moved / span:+.3f} deg/s); sign mismatches {run.mismatches}")


def main():
    args = parse_args()
    slot = LatestValue()
    drift_request = threading.Event()
    if args.still or args.motors:
        drift_request.set()
    reader = ImuReader(slot, drift_request, config)
    run, saber = Run(), None
    try:
        if args.motors:
            saber = open_saber(config)
        reader.start()
        report(slot, saber, run, args.seconds, config)
    except KeyboardInterrupt:
        pass
    finally:
        if saber is not None:                   # motors first
            saber.stop()
            saber.close()
        reader.close()
        print(summary(run))


if __name__ == "__main__":
    main()
