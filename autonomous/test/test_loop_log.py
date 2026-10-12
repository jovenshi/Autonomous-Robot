"""Tests for loop_log.py (per-loop CSV log). No hardware.

Run with:  python -m pytest test/test_loop_log.py
"""
import csv
import datetime
import os
import sys

import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from loop_log import COLUMNS, LoopLog
from robot_types import DriveCmd, ImuSample, LaneEstimate, State

START = datetime.datetime(2026, 10, 11, 15, 30, 5)


def lane(lateral=0.012, heading=-0.03, conf=0.9):
    return LaneEstimate(lateral, heading, conf, None, 0.0)


def imu(yaw=4.5, rate=-12.25, valid=True):
    return ImuSample(yaw, rate, valid, 3, 0.0, 0.0)


def read(path):
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    return lines[0], list(csv.reader(lines[1:]))


def test_file_named_by_start_time_with_gains_then_header(tmp_path):
    log = LoopLog(str(tmp_path / "logs"), config, start=START)
    log.close()
    assert log.path == str(tmp_path / "logs" / "run_20261011_153005.csv")
    first, rows = read(log.path)
    assert first == (f"# KD_YAW={config.KD_YAW} KP_LAT={config.KP_LAT} "
                     f"KH={config.KH} CRUISE_SPEED={config.CRUISE_SPEED}")
    assert tuple(rows[0]) == COLUMNS


def test_rows_hold_the_loop_values(tmp_path):
    log = LoopLog(str(tmp_path), config, start=START)
    log.write(12.3456789, State.LANE_FOLLOW, lane(), imu(), DriveCmd(speed=0.7, steer=-0.125))
    log.close()
    _, rows = read(log.path)
    assert rows[1] == ["12.3457", "LANE_FOLLOW", "0.012", "-0.03", "0.9",
                       "4.5", "-12.25", "1", "-0.125", "0.7"]


def test_missing_values_are_empty_cells(tmp_path):
    log = LoopLog(str(tmp_path), config, start=START)
    log.write(1.0, State.LANE_FOLLOW, lane(lateral=None, heading=None, conf=0.0), None,
              DriveCmd(speed=0.3, steer=0.0))
    log.write(1.1, State.LANE_FOLLOW, lane(), imu(valid=False), DriveCmd(speed=0.3, steer=0.0))
    log.close()
    _, rows = read(log.path)
    assert rows[1] == ["1.0000", "LANE_FOLLOW", "", "", "0", "", "", "0", "0", "0.3"]
    assert rows[2][7] == "0"                       # invalid sample: imu_valid 0


def test_rows_are_buffered_until_close(tmp_path):
    # No flush per row: nothing reaches the file until close().
    log = LoopLog(str(tmp_path), config, start=START)
    for i in range(30):
        log.write(i / 30, State.LANE_FOLLOW, lane(), imu(), DriveCmd(speed=0.7, steer=0.1))
    assert os.path.getsize(log.path) == 0
    log.close()
    _, rows = read(log.path)
    assert len(rows) == 31
    log.close()                                    # second close is harmless


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
