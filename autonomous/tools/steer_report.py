"""Steering response analysis: turns a steering test log into config values.

tools/steer_test.py (the guided screen) records the runs and calls these
functions; the same analysis runs from the command line (from autonomous/):

    python -m tools.steer_report calibration/steer_20261025_160000 [--apply]
    python -m tools.steer_report latency [logs/run_20261025_161500.csv] [--apply]

What a test folder holds:
    samples.csv     t, yaw_rate_dps, yaw_deg     every IMU sample during the runs
    segments.json   one entry per commanded piece of a run:
                    {"run", "kind", "speed", "steer", "t0", "t1"}
                    kind: settle | lead | step | straight
    distances.json  {"<speed>": metres} measured after each straight run

What comes out (design doc, "Steering response test"):
    turn gain       deg/s per unit steer at each speed (steady rate / steer)
    deadband        the steer below which the robot doesn't turn, per speed
    dead time, lag  command -> first movement, and the response's time constant
    bias, trim      yaw rate with steer 0 on a straight, and the steer that cancels it
    forward speed   m/s per unit of speed command, from the straight runs
    camera latency  from a main.py loop log: delay of the camera heading behind the gyro

Plant model used throughout: change in yaw rate = gain x (|steer| - deadband),
signed, for |steer| above the deadband.
"""
import argparse
import csv
import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402

SAMPLES_CSV = "samples.csv"
SEGMENTS_JSON = "segments.json"
DISTANCES_JSON = "distances.json"
BASELINE_S = 0.3                         # s at the end of a steer-0 piece used as the baseline
SPIN_UP_S = 1.0                          # s skipped at the start of a straight run (bias)
TIMING_FRACTIONS = (0.1, 0.63)           # first movement, and one time constant


# ---------------------------------------------------------------- the runs

@dataclass
class Run:
    name: str
    kind: str                            # 'pivot' | 'straight' | 'pair'
    speed: float
    steer: float                         # pair size; 0 for pivot and straight
    schedule: list                       # [(kind, seconds, speed, steer)]

    def seconds(self):
        return sum(piece[1] for piece in self.schedule)

    def distance_m(self, mps_per_unit=1.0):
        # Rough ground covered, for the "space needed" hint.
        return sum(piece[1] * piece[2] for piece in self.schedule) * mps_per_unit


def plan_runs(cfg):
    # Pivot steps first (robot stays in place), then for each driving speed
    # a straight run (speed, trim) and one run per steer size (left, right).
    settle, hold = cfg.STEER_TEST_SETTLE_S, cfg.STEER_TEST_HOLD_S
    pivot = [("settle", settle, 0.0, 0.0)]
    for s in cfg.STEER_TEST_STEERS:
        pivot += [("step", hold, 0.0, s), ("settle", settle, 0.0, 0.0),
                  ("step", hold, 0.0, -s), ("settle", settle, 0.0, 0.0)]
    runs = [Run("pivot", "pivot", 0.0, 0.0, pivot)]
    for v in cfg.STEER_TEST_SPEEDS:
        runs.append(Run(f"straight {v:.1f}", "straight", v, 0.0,
                        [("straight", cfg.STEER_TEST_STRAIGHT_S, v, 0.0)]))
        for s in cfg.STEER_TEST_STEERS:
            runs.append(Run(f"steer {s:.2f} at {v:.1f}", "pair", v, s,
                            [("lead", cfg.STEER_TEST_LEAD_S, v, 0.0), ("step", hold, v, s),
                             ("step", hold, v, -s)]))
    return runs


# ---------------------------------------------------------------- files

def write_samples(folder, samples):
    # samples: list of (t, yaw_rate_dps, yaw_deg).
    tmp = os.path.join(folder, SAMPLES_CSV + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("t", "yaw_rate_dps", "yaw_deg"))
        writer.writerows((f"{t:.4f}", f"{r:.3f}", f"{y:.3f}") for t, r, y in samples)
    os.replace(tmp, os.path.join(folder, SAMPLES_CSV))


def write_json(folder, name, value):
    tmp = os.path.join(folder, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=1)
    os.replace(tmp, os.path.join(folder, name))


def read_folder(folder):
    # (t array, rate array, segments, distances {speed: m}).
    with open(os.path.join(folder, SAMPLES_CSV), newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    rate = np.array([float(r["yaw_rate_dps"]) for r in rows])
    with open(os.path.join(folder, SEGMENTS_JSON)) as f:
        segments = json.load(f)
    distances = {}
    path = os.path.join(folder, DISTANCES_JSON)
    if os.path.exists(path):
        with open(path) as f:
            distances = {float(k): v for k, v in json.load(f).items()}
    return t, rate, segments, distances


# ---------------------------------------------------------------- one step

def window_mean(t, rate, t0, t1):
    # Mean yaw rate over [t0, t1], or None with no samples there.
    sel = (t >= t0) & (t <= t1)
    return float(rate[sel].mean()) if sel.any() else None


def step_response(t, rate, seg, baseline, cfg):
    # The steady change in yaw rate a step caused, and when it started
    # moving (dead time) and its time constant (lag), when measurable.
    steady = window_mean(t, rate, seg["t1"] - cfg.STEER_TEST_STEADY_S, seg["t1"])
    if steady is None or baseline is None:
        return None
    change = steady - baseline
    out = {"speed": seg["speed"], "steer": seg["steer"], "change": change,
           "dead_s": None, "lag_s": None}
    if abs(change) >= cfg.STEER_TEST_MOVE_DPS:
        sel = (t >= seg["t0"]) & (t <= seg["t1"])
        moved = (rate[sel] - baseline) / change
        times = []
        for fraction in TIMING_FRACTIONS:
            hit = np.flatnonzero(moved >= fraction)
            times.append(float(t[sel][hit[0]] - seg["t0"]) if hit.size else None)
        if None not in times:
            out["dead_s"], out["lag_s"] = times[0], max(0.0, times[1] - times[0])
    return out


def baseline_before(t, rate, segments, i):
    # Mean rate at the end of the nearest steer-0 piece before segment i
    # in the same run.
    for seg in reversed(segments[:i]):
        if seg["run"] != segments[i]["run"]:
            return None
        if seg["steer"] == 0:
            return window_mean(t, rate, seg["t1"] - min(BASELINE_S, seg["t1"] - seg["t0"]),
                               seg["t1"])
    return None


# ---------------------------------------------------------------- per speed

def gain_and_deadband(steps, cfg):
    # Fit |change| = gain * (|steer| - deadband) over the steps that moved.
    # With only one size moving, the deadband is bounded by the sizes that
    # didn't move. Returns (signed gain, deadband, +1 or -1 direction).
    moved = [(abs(s["steer"]), abs(s["change"])) for s in steps
             if abs(s["change"]) >= cfg.STEER_TEST_MOVE_DPS]
    still = [abs(s["steer"]) for s in steps if abs(s["change"]) < cfg.STEER_TEST_MOVE_DPS]
    signs = [np.sign(s["change"]) * np.sign(s["steer"]) for s in steps
             if abs(s["change"]) >= cfg.STEER_TEST_MOVE_DPS]
    if not moved:
        return None, max(still) if still else None, 1.0
    steer = np.array([m[0] for m in moved])
    change = np.array([m[1] for m in moved])
    if np.ptp(steer) > 0:                            # two sizes or more: fit the line
        gain, intercept = np.polyfit(steer, change, 1)
        deadband = min(max(0.0, -intercept / gain), float(steer.min())) if gain > 0 else 0.0
    else:                                            # one size: bounded by the still ones
        gain = float(change.mean() / steer.mean())
        deadband = min(max(still), float(steer.min())) if still else 0.0
    direction = 1.0 if np.mean(signs) >= 0 else -1.0
    return float(gain) * direction, float(deadband), direction


def asymmetry(steps, cfg):
    # (left gain - right gain) / mean gain, from steps that moved.
    left = [abs(s["change"] / s["steer"]) for s in steps
            if s["steer"] > 0 and abs(s["change"]) >= cfg.STEER_TEST_MOVE_DPS]
    right = [abs(s["change"] / s["steer"]) for s in steps
             if s["steer"] < 0 and abs(s["change"]) >= cfg.STEER_TEST_MOVE_DPS]
    if not left or not right:
        return None
    gl, gr = np.mean(left), np.mean(right)
    return float((gl - gr) / ((gl + gr) / 2))


# ---------------------------------------------------------------- the analysis

def analyse(t, rate, segments, distances, cfg):
    # Everything the test measures, as a result dict (see report_lines).
    steps = []
    for i, seg in enumerate(segments):
        if seg["kind"] == "step":
            resp = step_response(t, rate, seg, baseline_before(t, rate, segments, i), cfg)
            if resp is not None:
                resp["from_zero"] = segments[i - 1]["steer"] == 0 if i else False
                steps.append(resp)
    timed = [s for s in steps if s["from_zero"] and s["dead_s"] is not None]
    dead = float(np.median([s["dead_s"] for s in timed])) if timed else None
    lag = float(np.median([s["lag_s"] for s in timed])) if timed else None
    speeds = {}
    for v in sorted({s["speed"] for s in steps} | {seg["speed"] for seg in segments
                                                   if seg["kind"] == "straight"}):
        speeds[v] = _speed_result(t, rate, segments, steps, distances, v, dead, lag, cfg)
    result = {"speeds": speeds, "dead_s": dead, "lag_s": lag, "steps": steps}
    result["mps_per_unit"] = _mps_per_unit(speeds)
    pivot = speeds.get(0.0)
    result["turn_min_steer"] = (round(pivot["deadband"] + cfg.STEER_TEST_TURN_MARGIN, 3)
                                if pivot and pivot["deadband"] is not None else None)
    driving = [r["deadband"] for v, r in speeds.items() if v > 0 and r["deadband"] is not None]
    result["deadband"] = round(float(np.mean(driving)), 3) if driving else None
    return result


def _speed_result(t, rate, segments, steps, distances, v, dead, lag, cfg):
    here = [s for s in steps if s["speed"] == v]
    gain, deadband, direction = gain_and_deadband(here, cfg)
    out = {"gain": gain, "deadband": deadband, "asym": asymmetry(here, cfg),
           "bias_dps": None, "trim": None, "mps": None, "reversed": direction < 0}
    straight = [seg for seg in segments if seg["kind"] == "straight" and seg["speed"] == v]
    if straight:
        seg = straight[-1]
        out["bias_dps"] = window_mean(t, rate, seg["t0"] + SPIN_UP_S, seg["t1"])
        if out["bias_dps"] is not None and gain:
            out["trim"] = -out["bias_dps"] / gain
        if v in distances:
            moving_s = (seg["t1"] - seg["t0"]) - (dead or 0.0) - (lag or 0.0)
            out["mps"] = distances[v] / moving_s if moving_s > 0 else None
    return out


def _mps_per_unit(speeds):
    # Least-squares slope through 0 of measured m/s against speed command.
    pairs = [(v, r["mps"]) for v, r in speeds.items() if v > 0 and r["mps"]]
    if not pairs:
        return None
    return float(sum(v * m for v, m in pairs) / sum(v * v for v, _ in pairs))


def analyse_folder(folder, cfg):
    return analyse(*read_folder(folder), cfg)


# ---------------------------------------------------------------- what it means

def report_lines(result, cfg):
    # (level, text) lines for the screen and the console.
    lines = []
    for v, r in result["speeds"].items():
        name = "pivot (speed 0)" if v == 0 else f"speed {v:.1f}"
        if r["reversed"]:
            lines.append(("bad", f"{name}: steer + turned RIGHT. Check LEFT/RIGHT_INVERT and "
                                 "the channels, or IMU_GYRO_SIGN, before using any of this"))
        parts = [f"{abs(r['gain']):.0f} deg/s per unit steer" if r["gain"] else "no turning seen"]
        if r["deadband"] is not None:
            parts.append(f"deadband {r['deadband']:.3f}")
        if r["trim"] is not None:
            parts.append(f"drifts {r['bias_dps']:+.1f} deg/s, trim {r['trim']:+.3f}")
        if r["mps"] is not None:
            parts.append(f"{r['mps']:.2f} m/s")
        level = "ok" if r["gain"] and not r["reversed"] else "warn"
        lines.append((level, f"{name}: " + ", ".join(parts)))
        if r["asym"] is not None and abs(r["asym"]) > cfg.STEER_TEST_ASYM_WARN:
            side = "left" if r["asym"] > 0 else "right"
            lines.append(("warn", f"{name}: turns {abs(r['asym']) * 100:.0f}% harder to the "
                                  f"{side}"))
    if result["dead_s"] is not None:
        lines.append(("info", f"response: starts after {result['dead_s'] * 1000:.0f} ms, "
                              f"time constant {result['lag_s'] * 1000:.0f} ms"))
    if result["mps_per_unit"] is not None:
        lines.append(("info", f"forward speed {result['mps_per_unit']:.2f} m/s per unit command"))
    if result["turn_min_steer"] is not None:
        pivot = result["turn_min_steer"] - cfg.STEER_TEST_TURN_MARGIN
        lines.append(("info", f"pivots need steer {pivot:.3f}"
                              f"; TURN_MIN_STEER -> {result['turn_min_steer']:.3f} (was "
                              f"{cfg.TURN_MIN_STEER})"))
    return lines


def config_values(result):
    # The config.py values APPLY writes (only what was measured).
    speeds = result["speeds"]
    values = {}
    gains = {round(v, 3): round(abs(r["gain"]), 2) for v, r in speeds.items() if r["gain"]}
    if gains and not any(r["reversed"] for r in speeds.values()):
        values["STEER_YAW_DPS_PER_UNIT"] = gains
        trims = {round(v, 3): round(r["trim"], 4) for v, r in speeds.items()
                 if v > 0 and r["trim"] is not None}
        if trims:
            values["STEER_TRIM"] = trims
        if result["deadband"] is not None:
            values["STEER_DEADBAND"] = result["deadband"]
        if result["turn_min_steer"] is not None:
            values["TURN_MIN_STEER"] = result["turn_min_steer"]
    if result["dead_s"] is not None:
        values["MOTOR_DEAD_TIME_S"] = round(result["dead_s"], 3)
        values["MOTOR_LAG_S"] = round(result["lag_s"], 3)
    if result["mps_per_unit"] is not None:
        values["SPEED_MPS_PER_UNIT"] = round(result["mps_per_unit"], 3)
    return values


# ---------------------------------------------------------------- camera latency

def read_loop_log(path):
    # (t, heading_err_rad, yaw_rate_dps) arrays from a main.py loop log,
    # rows where both were measured.
    with open(path, newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    rows = [r for r in csv.DictReader(lines) if r["heading_err_rad"] and r["yaw_rate_dps"]]
    return (np.array([float(r["t"]) for r in rows]),
            np.array([float(r["heading_err_rad"]) for r in rows]),
            np.array([float(r["yaw_rate_dps"]) for r in rows]))


def latency_from_log(path, cfg, grid_hz=200):
    # How far the camera's heading change lags the gyro, from a loop log
    # recorded while the robot was twisted left and right by hand in front
    # of the line. heading = road_yaw - yaw, so -d(heading)/dt follows the
    # yaw rate, delayed by the camera. Returns {latency_s, corr, seconds}.
    t, heading, rate = read_loop_log(path)
    if len(t) < 30:
        raise ValueError("too few rows with both the camera heading and the gyro: was the line "
                         "in view?")
    cam = -np.degrees(np.gradient(heading, t))
    grid = np.arange(t[0], t[-1], 1.0 / grid_hz)
    a, b = np.interp(grid, t, rate), np.interp(grid, t, cam)
    if a.std() < 10:
        raise ValueError("not enough turning: twist the robot left and right by hand (about "
                         "30 deg each way, once a second) for 10 s")
    a, b = a - a.mean(), b - b.mean()
    lags = np.arange(0, int(cfg.STEER_TEST_LATENCY_MAX_S * grid_hz) + 1)
    corr = np.array([np.corrcoef(a[:len(a) - k], b[k:])[0, 1] for k in lags])
    k = int(np.argmax(corr))
    if 0 < k < len(corr) - 1:                         # parabola through the peak
        y0, y1, y2 = corr[k - 1:k + 2]
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    best = float(corr.max())
    if best < 0.5:
        raise ValueError(f"camera and gyro don't agree (correlation {best:.2f}): was the line "
                         "in view the whole time?")
    return {"latency_s": float(k) / grid_hz, "corr": best, "seconds": float(t[-1] - t[0])}


def newest_loop_log(cfg):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        cfg.LOOP_LOG_DIR)
    logs = sorted(glob.glob(os.path.join(root, "run_*.csv")))
    return logs[-1] if logs else None


# ---------------------------------------------------------------- command line

def print_lines(lines):
    marks = {"ok": "  ok ", "warn": "  !! ", "bad": "  XX ", "info": "     "}
    for level, text in lines:
        print(marks[level] + text)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Steering response analysis (see the docstring).")
    p.add_argument("target", help="a steering test folder, or 'latency'")
    p.add_argument("log", nargs="?", help="latency: a loop log (default: the newest)")
    p.add_argument("--apply", action="store_true", help="write the results into config.py")
    return p.parse_args(argv)


def main(argv=None):
    from tools.config_writer import config_path, write_values

    args = parse_args(argv)
    try:
        if args.target == "latency":
            path = args.log or newest_loop_log(config)
            if path is None:
                raise ValueError(f"no loop logs in {config.LOOP_LOG_DIR}/")
            res = latency_from_log(path, config)
            print(f"{os.path.basename(path)}: camera latency {res['latency_s'] * 1000:.0f} ms "
                  f"(correlation {res['corr']:.2f}, {res['seconds']:.0f} s)")
            values = {"CAMERA_LATENCY_S": round(res["latency_s"], 3)}
        else:
            result = analyse_folder(args.target, config)
            print_lines(report_lines(result, config))
            values = config_values(result)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"Steering analysis stopped: {e}")
    if args.apply:
        backup = write_values(config_path(), values)
        print(f"\nWritten to config.py (old file kept as {os.path.basename(backup)})")
    else:
        print("\nAdd --apply to write these into config.py:")
        for name, value in values.items():
            print(f"{name} = {value!r}")


if __name__ == "__main__":
    main()
