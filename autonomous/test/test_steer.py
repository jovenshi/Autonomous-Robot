"""Tests for the steering response test: control/mixer.py corrections, the
lane controller's speed scaling, tools/steer_report.py, tools/steer_test.py
and tools/lane_sim.py.

The "robot" is tools/lane_sim.Plant set up with known true values (gain per
speed, deadband, drift, dead time, lag), driven through the real Runner and
guided screens with a fake clock, so a whole test takes no real time.

Run from the autonomous/ folder:  python -m pytest test/test_steer.py
"""
import csv
import math
import os
import sys
import types

import numpy as np
import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from control.lane_controller import LaneController, speed_scale  # noqa: E402
from control.mixer import compensate, mix, steer_trim  # noqa: E402
from loop_log import COLUMNS  # noqa: E402
from robot_types import DriveCmd, ImuSample, LaneEstimate, LatestValue  # noqa: E402
from tools import config_writer, lane_sim, steer_report, steer_test  # noqa: E402

TRUE_GAIN = {0.0: 120.0, 0.3: 140.0, 0.5: 150.0, 0.7: 160.0}
TRUE_TRIM = {0.3: 0.02, 0.5: 0.02, 0.7: 0.025}       # drift = -trim x gain
TRUE_DEADBAND, TRUE_DEAD, TRUE_LAG, TRUE_MPS = 0.04, 0.04, 0.10, 0.9


def ns(**overrides):
    values = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_cfg(tmp_path):
    return ns(CONTROL_HZ=100, DISPLAY_SIZE=(800, 480), CALIB_DIR=str(tmp_path / "calibration"),
              LOOP_LOG_DIR=str(tmp_path / "logs"), STEER_TRIM=None, STEER_DEADBAND=0.0,
              STEER_YAW_DPS_PER_UNIT=None, SPEED_MPS_PER_UNIT=None, MOTOR_DEAD_TIME_S=None,
              MOTOR_LAG_S=None, CAMERA_LATENCY_S=None)


# ---------------------------------------------------------------- mixer and controller

def test_trim_interpolates_and_is_zero_at_rest():
    cfg = ns(STEER_TRIM={0.3: 0.02, 0.7: 0.04}, STEER_DEADBAND=0.0)
    assert steer_trim(0.0, cfg) == 0.0
    assert steer_trim(0.15, cfg) == pytest.approx(0.01)      # fades to 0 below the lowest speed
    assert steer_trim(0.5, cfg) == pytest.approx(0.03)
    assert steer_trim(1.0, cfg) == pytest.approx(0.04)
    assert mix(DriveCmd(0.0, 0.0), cfg) == mix(DriveCmd(0.0, 0.0), ns(STEER_TRIM=None,
                                                                       STEER_DEADBAND=0.0))


def test_deadband_lift_is_continuous_and_full_above_the_ramp():
    cfg = ns(STEER_TRIM=None, STEER_DEADBAND=0.05, STEER_DEADBAND_RAMP=0.02)
    assert compensate(0.0, 0.5, cfg) == 0.0
    assert compensate(0.001, 0.5, cfg) == pytest.approx(0.001 + 0.0025)
    assert compensate(-0.1, 0.5, cfg) == pytest.approx(-0.15)
    unset = ns(STEER_TRIM=None, STEER_DEADBAND=0.0)
    assert mix(DriveCmd(0.5, 0.1), unset) == mix(DriveCmd(0.5, 0.1), config)   # default: no change


def test_speed_scaling_only_when_enabled():
    est = LaneEstimate(0.05, 0.0, 1.0, None, 0.0)
    off = ns(STEER_SMOOTH=0.0, LANE_GAIN_SPEED_SCALING=False)
    on = ns(STEER_SMOOTH=0.0, LANE_GAIN_SPEED_SCALING=True, CRUISE_SPEED=0.6,
            LANE_GAIN_MIN_SPEED=0.2)
    assert LaneController(off).update(est, 0.0, 0.01, 0.3) == pytest.approx(off.KP_LAT * 0.05)
    assert LaneController(on).update(est, 0.0, 0.01, 0.3) == pytest.approx(2 * on.KP_LAT * 0.05)
    assert speed_scale(0.05, on) == pytest.approx(3.0)        # floored at LANE_GAIN_MIN_SPEED


# ---------------------------------------------------------------- the simulated robot

class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class SimRobot:
    """Sabertooth + IMU stand-in: motor commands drive a Plant with the true values."""

    def __init__(self, cfg, clock, gains=TRUE_GAIN):
        truth = ns(STEER_GAIN=cfg.STEER_GAIN, STEER_DEADBAND=TRUE_DEADBAND, STEER_TRIM=TRUE_TRIM,
                   STEER_YAW_DPS_PER_UNIT=gains, MOTOR_DEAD_TIME_S=TRUE_DEAD,
                   MOTOR_LAG_S=TRUE_LAG, SPEED_MPS_PER_UNIT=TRUE_MPS, CAMERA_LATENCY_S=0.0)
        self.plant = lane_sim.Plant(truth, 1.0 / cfg.CONTROL_HZ)
        self.clock, self.slot, self.motor = clock, LatestValue(), None
        self.valid, self.stop_after, self.ticks, self.runner = True, None, 0, None
        self.publish()

    def publish(self):
        self.slot.set(ImuSample(0.0, self.plant.rate, self.valid, 3, 0.0, self.clock()))

    def set(self, motor):
        self.motor = motor

    def stop(self):
        self.motor = None

    def tick(self, imu_log):
        motor = self.motor or steer_test.raw_motor(0.0, 0.0, config)
        self.plant.step(motor, (motor.left + motor.right) / 2)
        self.ticks += 1
        if self.stop_after is not None and self.ticks >= self.stop_after:
            self.runner.stop()
        self.clock.sleep(0)          # time advances in Runner's sleep
        self.publish()
        imu_log.poll()


def make_app(tmp_path, gains=TRUE_GAIN, cfg=None):
    cfg = cfg or make_cfg(tmp_path)
    clock = FakeClock()
    robot = SimRobot(cfg, clock, gains)
    imu_log = steer_test.ImuLog(robot.slot, cfg)
    runner = steer_test.Runner(robot, imu_log, cfg, clock=clock, sleep=clock.sleep,
                               on_tick=lambda: robot.tick(imu_log))
    robot.runner = runner
    app = steer_test.SteerApp(cfg, runner, imu_log, background=False, clock=clock)
    return app, robot, cfg


def drive_all_runs(app, distances=True):
    app.press("start")
    while app.page in ("ready", "distance"):
        assert app.draw().shape == (480, 800, 3)
        if app.page == "ready":
            app.press("go")
        else:
            run = app.runs[app.index]
            app.distance_m = TRUE_MPS * run.speed * (run.seconds() - TRUE_DEAD - TRUE_LAG)
            app.press("keep" if distances else "skip")
    assert app.page == "results"


def test_full_test_recovers_the_true_robot(tmp_path):
    app, robot, cfg = make_app(tmp_path)
    assert app.page == "welcome"
    app.draw()
    drive_all_runs(app)
    app.draw()
    r = app.result
    for v, gain in TRUE_GAIN.items():
        assert r["speeds"][v]["gain"] == pytest.approx(gain, rel=0.05)
        assert r["speeds"][v]["deadband"] == pytest.approx(TRUE_DEADBAND, abs=0.01)
    for v, trim in TRUE_TRIM.items():
        assert r["speeds"][v]["trim"] == pytest.approx(trim, abs=0.004)
    assert r["dead_s"] == pytest.approx(TRUE_DEAD, abs=0.02)
    assert r["lag_s"] == pytest.approx(TRUE_LAG, abs=0.03)
    assert r["mps_per_unit"] == pytest.approx(TRUE_MPS, rel=0.02)
    margin = cfg.STEER_TEST_TURN_MARGIN
    assert r["turn_min_steer"] == pytest.approx(TRUE_DEADBAND + margin, abs=0.01)
    assert not any(level == "bad" for level, _ in steer_report.report_lines(r, cfg))
    # the saved folder gives the same answer from the command line
    again = steer_report.analyse_folder(app.folder, cfg)
    assert again["speeds"][0.5]["gain"] == pytest.approx(r["speeds"][0.5]["gain"], rel=1e-3)


def test_apply_writes_config_and_then_mix_cancels_the_drift(tmp_path, monkeypatch):
    target = tmp_path / "config.py"
    target.write_text(open(config_writer.config_path()).read())
    monkeypatch.setattr(config_writer, "config_path", lambda: str(target))
    app, robot, cfg = make_app(tmp_path)
    drive_all_runs(app)
    app.press("apply")
    assert app.page == "applied" and app.draw().shape == (480, 800, 3)
    written = {}
    exec(compile(target.read_text(), "config", "exec"), written)
    assert set(written["STEER_TRIM"]) == {0.3, 0.5, 0.7}
    assert written["SPEED_MPS_PER_UNIT"] == pytest.approx(TRUE_MPS, rel=0.02)
    assert cfg.STEER_DEADBAND == written["STEER_DEADBAND"]
    # the true robot drives straight with the measured trim in mix()
    plant = SimRobot(cfg, FakeClock()).plant
    for _ in range(300):
        plant.step(mix(DriveCmd(0.5, 0.0), cfg), 0.5)
    assert abs(plant.rate) < 0.5                              # was 3 deg/s of drift


def test_a_stopped_run_is_thrown_away(tmp_path):
    app, robot, cfg = make_app(tmp_path)
    app.press("start")
    robot.stop_after = 20
    app.press("go")
    assert app.page == "ready" and "stopped" in app.flash and app.segments == []
    assert robot.motor is None                                # wheels stopped
    robot.stop_after = None
    app.press("go")
    assert app.segments and app.index == 1


def test_imu_failure_stops_the_run(tmp_path):
    app, robot, cfg = make_app(tmp_path)
    app.press("start")
    robot.valid = False
    robot.publish()
    app.press("go")
    assert app.page == "ready" and "IMU" in app.flash and robot.motor is None


def test_welcome_waits_for_a_calibrated_imu(tmp_path):
    app, robot, cfg = make_app(tmp_path)
    robot.slot.set(ImuSample(0.0, 0.0, True, 1, 0.0, app.clock()))
    assert not next(b for b in app.buttons() if b and b.name == "start").enabled
    app.clock.now += cfg.IMU_CALIB_WAIT_S + 1
    assert next(b for b in app.buttons() if b and b.name == "start").enabled


def test_reversed_steering_is_flagged_and_not_applied(tmp_path):
    app, robot, cfg = make_app(tmp_path, gains={v: -g for v, g in TRUE_GAIN.items()})
    drive_all_runs(app, distances=False)
    lines = steer_report.report_lines(app.result, cfg)
    assert any(level == "bad" and "turned RIGHT" in text for level, text in lines)
    assert "STEER_YAW_DPS_PER_UNIT" not in steer_report.config_values(app.result)


def test_plan_covers_every_speed_and_size():
    runs = steer_report.plan_runs(config)
    assert runs[0].kind == "pivot"
    pairs = [(r.speed, r.steer) for r in runs if r.kind == "pair"]
    assert len(pairs) == len(config.STEER_TEST_SPEEDS) * len(config.STEER_TEST_STEERS)
    assert sum(r.kind == "straight" for r in runs) == len(config.STEER_TEST_SPEEDS)


# ---------------------------------------------------------------- camera latency

def write_loop_log(path, delay_s, seconds=12.0, hz=30.0, amplitude=60.0, line=True):
    t = np.arange(0, seconds, 1 / hz) + 50.0
    rate = amplitude * np.sin(2 * math.pi * 1.0 * (t - 50.0))
    yaw = -amplitude / (2 * math.pi) * np.cos(2 * math.pi * (t - 50.0))   # deg, exact integral
    heading = -np.radians(np.interp(t - delay_s, t, yaw))       # camera sees it late
    with open(path, "w", newline="") as f:
        f.write("# gains\n")
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for ti, h, r in zip(t, heading, rate):
            writer.writerow([f"{ti:.4f}", "IDLE", "0.0", f"{h:.6f}" if line else "", "0.9",
                             "0", f"{r:.3f}", "1", "0", "0"])


def test_latency_is_found_from_a_loop_log(tmp_path):
    path = tmp_path / "run.csv"
    write_loop_log(path, 0.08)
    res = steer_report.latency_from_log(str(path), config)
    assert res["latency_s"] == pytest.approx(0.08, abs=0.012) and res["corr"] > 0.9


def test_latency_needs_motion_and_the_line(tmp_path):
    still, blind = tmp_path / "still.csv", tmp_path / "blind.csv"
    write_loop_log(still, 0.08, amplitude=2.0)
    write_loop_log(blind, 0.08, line=False)
    with pytest.raises(ValueError, match="twist"):
        steer_report.latency_from_log(str(still), config)
    with pytest.raises(ValueError, match="line"):
        steer_report.latency_from_log(str(blind), config)


def test_latency_screen_uses_the_newest_log(tmp_path, monkeypatch):
    target = tmp_path / "config.py"
    target.write_text(open(config_writer.config_path()).read())
    monkeypatch.setattr(config_writer, "config_path", lambda: str(target))
    app, robot, cfg = make_app(tmp_path)
    app.press("latency")
    assert app.page == "latency" and "no loop logs" in app.latency_error
    os.makedirs(cfg.LOOP_LOG_DIR)
    write_loop_log(os.path.join(cfg.LOOP_LOG_DIR, "run_20261025_120000.csv"), 0.06)
    app.press("again")
    assert app.latency["latency_s"] == pytest.approx(0.06, abs=0.012)
    app.draw()
    app.press("apply")
    assert cfg.CAMERA_LATENCY_S == pytest.approx(0.06, abs=0.012) and app.page == "welcome"


# ---------------------------------------------------------------- lane_sim

def test_sim_settles_and_reports_guesses():
    t, lat, steer, plant = lane_sim.simulate(ns(STEER_YAW_DPS_PER_UNIT=None), 0.05, 0.5, 8.0)
    assert "STEER_YAW_DPS_PER_UNIT" in plant.guessed
    s = lane_sim.score(t, lat)
    assert s["settle_s"] is not None and s["overshoot_m"] < 0.05


def test_sim_with_measured_drift_and_trim_ends_on_the_line():
    cfg = ns(STEER_YAW_DPS_PER_UNIT=TRUE_GAIN, STEER_TRIM=TRUE_TRIM, STEER_DEADBAND=TRUE_DEADBAND,
             MOTOR_DEAD_TIME_S=TRUE_DEAD, MOTOR_LAG_S=TRUE_LAG, CAMERA_LATENCY_S=0.06,
             SPEED_MPS_PER_UNIT=TRUE_MPS)
    t, lat, _, plant = lane_sim.simulate(cfg, 0.05, 0.5, 10.0)
    assert plant.guessed == [] and abs(lat[-1]) < 0.003


def test_score_and_compare_with_a_log(tmp_path):
    t = np.arange(0, 4, 0.1)
    lat = 0.05 * np.exp(-t) * np.cos(2 * t)
    s = lane_sim.score(t, lat)
    assert s["overshoot_m"] > 0 and 0 < s["settle_s"] < 4
    path = tmp_path / "run.csv"
    with open(path, "w", newline="") as f:
        f.write("# gains\n")
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerow(["0.0", "IDLE", "0.05", "0", "0.9", "", "", "0", "0", "0"])
        for ti, li in zip(t, lat):
            w.writerow([f"{ti + 1:.2f}", "LANE_FOLLOW", f"{li:.5f}", "0", "0.9", "", "", "0",
                        "0", "0.5"])
    lt, llat = lane_sim.log_segment(str(path))
    assert len(lt) == len(t)
    lines = lane_sim.compare_lines(s, lane_sim.score(lt, llat))
    assert "(+0%)" in lines[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
