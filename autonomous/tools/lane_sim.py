"""Closed-loop lane-following simulation with the robot's measured response.

Runs the real LaneController, drive_for_state and mix() against a model of
the robot built from the steering test's numbers (tools/steer_test.py):
turn gain per speed, steering deadband, straight-line drift (cancelled by
STEER_TRIM, as on the robot), dead time, motor lag, camera latency and
forward speed. Unmeasured values fall back to SIM_DEFAULTS, and the report
says which.

Tune KP_LAT, KH, KD_YAW here, then check against the robot (from autonomous/):

    python -m tools.lane_sim                         # 5 cm offset start at CRUISE_SPEED
    python -m tools.lane_sim --start-m 0.05 --speed 0.6 --log logs/run_20261025_170000.csv

With --log, the first LANE_FOLLOW stretch of that run (start the robot about
5 cm off the line) is scored the same way: peak overshoot and settling time
should agree within about 20% before the sim is trusted for tuning.
"""
import argparse
import csv
import math
import os
import sys
import types
from collections import deque

import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from control.command import drive_for_state  # noqa: E402
from control.lane_controller import LaneController  # noqa: E402
from control.mixer import mix, steer_trim  # noqa: E402
from robot_types import ImuSample, LaneEstimate, State  # noqa: E402

# Used for anything the steering test hasn't measured yet (guesses).
SIM_DEFAULTS = {"STEER_YAW_DPS_PER_UNIT": {0.0: 150.0}, "MOTOR_DEAD_TIME_S": 0.03,
                "MOTOR_LAG_S": 0.10, "CAMERA_LATENCY_S": 0.06, "SPEED_MPS_PER_UNIT": 1.0}
SETTLE_M = 0.01                          # m; settled once |lateral| stays below this


def sim_values(cfg):
    # The model's numbers: measured where set, else SIM_DEFAULTS; also the
    # names that fell back to a default.
    values, guessed = {}, []
    for name, default in SIM_DEFAULTS.items():
        values[name] = getattr(cfg, name, None)
        if values[name] is None:
            values[name] = default
            guessed.append(name)
    return values, guessed


def interp_speed(table, speed):
    # A {speed: value} table at this speed (flat beyond the ends).
    speeds = sorted(table)
    return float(np.interp(speed, speeds, [table[s] for s in speeds]))


class Plant:
    """The robot's yaw response to motor commands, and its motion on the floor."""

    def __init__(self, cfg, dt):
        self.cfg, self.dt = cfg, dt
        self.values, self.guessed = sim_values(cfg)
        delay = int(round(self.values["MOTOR_DEAD_TIME_S"] / dt))
        self._queue = deque([0.0] * max(delay, 0))
        self.rate = 0.0                  # deg/s, + = left
        self.heading = 0.0               # rad, + = pointing right of the road
        self.lateral = 0.0               # m, + = axle right of the line

    def target_rate(self, motor, speed):
        # Steady yaw rate these wheel commands would give: the steering part
        # of the command past the robot's deadband, times the turn gain, plus
        # the robot's own drift (what STEER_TRIM was measured to cancel).
        cfg = self.cfg
        steer = (motor.right - motor.left) / (2 * cfg.STEER_GAIN)
        past = max(0.0, abs(steer) - cfg.STEER_DEADBAND)
        gain = interp_speed(self.values["STEER_YAW_DPS_PER_UNIT"], speed)
        drift = -steer_trim(speed, cfg) * gain
        return math.copysign(past, steer) * gain + drift

    def step(self, motor, speed):
        # Advances one dt; returns the yaw rate the gyro would read.
        self._queue.append(self.target_rate(motor, speed))
        target = self._queue.popleft()
        lag = max(self.values["MOTOR_LAG_S"], 1e-6)
        self.rate += (target - self.rate) * min(1.0, self.dt / lag)
        self.heading -= math.radians(self.rate) * self.dt
        mps = speed * self.values["SPEED_MPS_PER_UNIT"]
        self.lateral += mps * math.sin(self.heading) * self.dt
        return self.rate


def simulate(cfg, start_m=0.05, speed=None, seconds=6.0):
    # Closed loop from start_m off the line, square to the road, held at a
    # fixed speed command. Returns (t, lateral_m, steer) arrays and the plant.
    dt = 1.0 / cfg.CONTROL_HZ
    run_cfg = types.SimpleNamespace(**{n: getattr(cfg, n) for n in dir(cfg) if n.isupper()})
    run_cfg.CRUISE_SPEED = speed if speed is not None else cfg.CRUISE_SPEED
    run_cfg.SPEED_MIN_SCALE = 1.0                    # hold the speed fixed for a clean test
    plant = Plant(run_cfg, dt)
    plant.lateral = start_m
    latency = int(round(plant.values["CAMERA_LATENCY_S"] / dt))
    seen = deque([(start_m, 0.0)] * (latency + 1), maxlen=latency + 1)
    ctrl, rate = LaneController(run_cfg), 0.0
    ts, lats, steers = [], [], []
    for k in range(int(seconds / dt)):
        lateral, heading = seen[0]
        est = LaneEstimate(lateral, heading, 1.0, None, k * dt)
        imu = ImuSample(0.0, rate, True, 3, 0.0, k * dt)
        drive = drive_for_state(State.LANE_FOLLOW, est, imu, None, ctrl, None, run_cfg)
        rate = plant.step(mix(drive, run_cfg), drive.speed)
        seen.append((plant.lateral, plant.heading))
        ts.append(k * dt)
        lats.append(plant.lateral)
        steers.append(drive.steer)
    return np.array(ts), np.array(lats), np.array(steers), plant


def score(t, lateral, start_m=None):
    # Peak overshoot (m, on the far side of the line) and settling time (s,
    # until |lateral| stays under SETTLE_M); None when it never settles.
    start = lateral[0] if start_m is None else start_m
    side = np.sign(start) if start else 1.0
    overshoot = float(max(0.0, -(side * lateral).min()))
    outside = np.flatnonzero(np.abs(lateral) >= SETTLE_M)
    if outside.size == 0:
        settle = 0.0
    elif outside[-1] == len(lateral) - 1:
        settle = None
    else:
        settle = float(t[outside[-1] + 1] - t[0])
    return {"overshoot_m": overshoot, "settle_s": settle, "start_m": float(start)}


def log_segment(path):
    # (t, lateral) of the first LANE_FOLLOW stretch in a loop log.
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    t, lat, started = [], [], False
    for row in rows:
        follow = row["state"] == "LANE_FOLLOW"
        if follow and row["lateral_err_m"]:
            started = True
            t.append(float(row["t"]))
            lat.append(float(row["lateral_err_m"]))
        elif started and not follow:
            break
    if len(t) < 10:
        raise ValueError("no LANE_FOLLOW stretch with lane measurements in that log")
    return np.array(t), np.array(lat)


def compare_lines(sim, real):
    # Side by side, with the difference as a share of the robot's value.
    lines = []
    for key, label, unit, scale in (("overshoot_m", "overshoot", "mm", 1000),
                                    ("settle_s", "settling", "s", 1)):
        s, r = sim[key], real[key]
        fmt = (lambda x: "never" if x is None else f"{x * scale:.1f} {unit}")
        diff = (f"  ({round((s - r) / r * 100) + 0:+d}%)" if s is not None and r not in (None, 0)
                else "")
        lines.append(f"{label:10s} sim {fmt(s):>10s}   robot {fmt(r):>10s}{diff}")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description="Lane-following simulation (see the docstring).")
    p.add_argument("--start-m", type=float, default=0.05, help="starting offset from the line")
    p.add_argument("--speed", type=float, default=None, help="speed command (default CRUISE_SPEED)")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--log", help="a loop log to compare with (first LANE_FOLLOW stretch)")
    args = p.parse_args(argv)
    t, lat, steer, plant = simulate(config, args.start_m, args.speed, args.seconds)
    if plant.guessed:
        print("Not measured yet, using guesses: " + ", ".join(plant.guessed)
              + "  (run python -m tools.steer_test)")
    sim = score(t, lat)
    print(f"Sim: start {args.start_m * 100:.1f} cm, peak steer {np.abs(steer).max():.2f}")
    if args.log:
        lt, llat = log_segment(args.log)
        print("\n".join(compare_lines(sim, score(lt, llat))))
    else:
        print("\n".join(line.split("   robot")[0] for line in compare_lines(sim, sim)))


if __name__ == "__main__":
    main()
