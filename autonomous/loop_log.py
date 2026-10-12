"""Per-loop CSV log, for comparing tuning runs (e.g. KD_YAW on vs off).

One file per run in LOOP_LOG_DIR, named by start time. Line 1 is a comment
with the gains in use, line 2 the column names, then one row per control
loop. Rows go through Python's file buffer with no flush per row, so the
control loop never waits on the SD card; close() (called by shutdown)
flushes the rest.

An empty cell means "not measured this loop". yaw_deg and yaw_rate_dps come
from the fresh IMU sample (empty when missing or stale); the controller used
yaw_rate_dps for damping only when imu_valid is 1.

tools/weave_report.py reads these files.
"""
import csv
import datetime
import os

COLUMNS = ("t", "state", "lateral_err_m", "heading_err_rad", "lane_conf",
           "yaw_deg", "yaw_rate_dps", "imu_valid", "steer", "speed")
GAINS = ("KD_YAW", "KP_LAT", "KH", "CRUISE_SPEED")


def gains_line(cfg):
    # "# KD_YAW=0.004 KP_LAT=4.0 KH=0.8 CRUISE_SPEED=0.7"
    return "# " + " ".join(f"{name}={getattr(cfg, name)}" for name in GAINS)


class LoopLog:
    def __init__(self, directory, cfg, start=None):
        # start: datetime used for the file name; now by default.
        start = start or datetime.datetime.now()
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, start.strftime("run_%Y%m%d_%H%M%S.csv"))
        self._file = open(self.path, "w", newline="")
        self._file.write(gains_line(cfg) + "\n")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)

    def write(self, now, state, lane, imu, drive):
        # lane: LaneEstimate; imu: fresh ImuSample or None; drive: DriveCmd.
        self._writer.writerow((
            f"{now:.4f}", state.name,
            _cell(lane.lateral_err_m), _cell(lane.heading_err_rad), _cell(lane.conf),
            _cell(imu.yaw_deg if imu is not None else None),
            _cell(imu.yaw_rate_dps if imu is not None else None),
            int(imu is not None and imu.valid),
            _cell(drive.steer), _cell(drive.speed),
        ))

    def close(self):
        # Flushes and closes; safe to call twice.
        if self._file is not None:
            self._file.close()
            self._file = None


def _cell(x):
    return "" if x is None else f"{x:.6g}"
