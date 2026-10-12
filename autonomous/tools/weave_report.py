"""Compare lane-following runs from loop logs (loop_log.py).

Run from the autonomous/ folder, with one or more logs:
    python -m tools.weave_report logs/run_20261011_150000.csv logs/run_20261011_151500.csv

One line per log, from LANE_FOLLOW rows only:
    the gains from the log's first line,
    lateral error RMS and peak (mm),
    heading error RMS (deg),
    steer sign changes per second (weaving).
Rows with no lateral (or heading) measurement are left out of that average.
Time and sign changes count only between consecutive LANE_FOLLOW rows, so a
stop in the middle of a log adds neither.

Typical use: one run with the current KD_YAW, one with KD_YAW = 0 in
config.py, same course and speed; less weaving shows as fewer sign changes
per second and a smaller lateral RMS.
"""
import argparse
import csv
import math
import os


def read_log(path):
    # -> (gains dict from the '# NAME=value ...' line, list of row dicts)
    with open(path, newline="") as f:
        first = f.readline()
        if first.startswith("#"):
            gains = dict(item.split("=", 1) for item in first[1:].split())
        else:
            gains = {}
            f.seek(0)
        return gains, list(csv.DictReader(f))


def lane_follow_segments(rows):
    # Runs of consecutive LANE_FOLLOW rows.
    segments, current = [], []
    for row in rows:
        if row["state"] == "LANE_FOLLOW":
            current.append(row)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def summarize(rows):
    # Weaving numbers for the LANE_FOLLOW rows of one log; None where there
    # is nothing to measure.
    segments = lane_follow_segments(rows)
    follow = [row for seg in segments for row in seg]
    lateral = _values(follow, "lateral_err_m")
    heading = _values(follow, "heading_err_rad")
    seconds = sum(float(seg[-1]["t"]) - float(seg[0]["t"]) for seg in segments)
    flips = sum(_sign_changes(_values(seg, "steer")) for seg in segments)
    return {
        "rows": len(follow),
        "seconds": seconds,
        "lat_rms_mm": _rms(lateral, 1000.0),
        "lat_peak_mm": max(abs(x) for x in lateral) * 1000.0 if lateral else None,
        "head_rms_deg": _rms(heading, math.degrees(1.0)),
        "flips_per_s": flips / seconds if seconds > 0 else None,
    }


def format_line(name, gains, s):
    gain_text = " ".join(f"{k}={v}" for k, v in gains.items())
    return (f"{name}  {gain_text}  rows {s['rows']}  {s['seconds']:.1f} s  "
            f"lat rms {_fmt(s['lat_rms_mm'], 1)} peak {_fmt(s['lat_peak_mm'], 1)} mm  "
            f"head rms {_fmt(s['head_rms_deg'], 2)} deg  "
            f"steer flips {_fmt(s['flips_per_s'], 2)}/s")


def _values(rows, column):
    return [float(row[column]) for row in rows if row[column] != ""]


def _rms(values, scale):
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / len(values)) * scale


def _sign_changes(values):
    # Zeros carry no sign and are skipped.
    signs = [v > 0 for v in values if v != 0]
    return sum(a != b for a, b in zip(signs, signs[1:]))


def _fmt(x, digits):
    return "--" if x is None else f"{x:.{digits}f}"


def main():
    p = argparse.ArgumentParser(description="Weaving summary per loop log (LANE_FOLLOW rows).")
    p.add_argument("logs", nargs="+", help="CSV files written by loop_log.py")
    for path in p.parse_args().logs:
        gains, rows = read_log(path)
        print(format_line(os.path.basename(path), gains, summarize(rows)))


if __name__ == "__main__":
    main()
