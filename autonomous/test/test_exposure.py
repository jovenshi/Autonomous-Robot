"""Tests for hal/exposure.py with a fake camera source.

Run from the robot/ folder:  python -m pytest tests
"""
import sys
import os

import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from hal import exposure
from hal.exposure import ExposureControl, cap_exposure, nudge, readback_ok


class FakeSource:
    def __init__(self, exposure_us=8000, gain=2.0):
        self.readback = (exposure_us, gain)
        self.set_calls = []

    def get_exposure(self):
        return self.readback

    def set_exposure(self, exposure_us, gain):
        self.set_calls.append((exposure_us, gain))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(exposure.time, "sleep", lambda s: None)


def make(monkeypatch, mode, source=None, target=None):
    monkeypatch.setattr(config, "EXPOSURE_MODE", mode)
    monkeypatch.setattr(config, "EXPOSURE_TARGET_V", target)
    src = source or FakeSource()
    return ExposureControl(src, "front", config), src


# ---------------------------------------------------------------- pure helpers

def test_readback_ok():
    assert readback_ok(8000, 2.0)
    assert not readback_ok(0, 2.0)
    assert not readback_ok(None, 2.0)
    assert not readback_ok(8000, None)


def test_cap_keeps_brightness():
    assert cap_exposure(20000, 1.0, config) == (10000, 2.0)
    assert cap_exposure(5000, 1.0, config) == (5000, 1.0)


def test_cap_gain_ceiling():
    exp, gain = cap_exposure(200000, 1.0, config)
    assert exp == 10000 and gain == config.EXPOSURE_GAIN_MAX


def test_nudge_inside_tolerance_does_nothing():
    assert nudge(5000, 2.0, 105, 100, config) == (5000, 2.0)


def test_nudge_brightens_exposure_first_then_gain():
    exp, gain = nudge(5000, 2.0, 50, 100, config)
    assert exp == pytest.approx(5500) and gain == 2.0
    exp, gain = nudge(10000, 2.0, 50, 100, config)
    assert exp == 10000 and gain == pytest.approx(2.2)


def test_nudge_darkens_gain_first_then_exposure():
    exp, gain = nudge(5000, 2.2, 200, 100, config)
    assert exp == 5000 and gain == pytest.approx(2.0)
    exp, gain = nudge(5500, 1.0, 200, 100, config)
    assert exp == pytest.approx(5000) and gain == 1.0


def test_nudge_respects_limits():
    assert nudge(config.EXPOSURE_MIN_US, 1.0, 250, 100, config) == (config.EXPOSURE_MIN_US, 1.0)
    assert nudge(10000, config.EXPOSURE_GAIN_MAX, 10, 100, config) == (10000, config.EXPOSURE_GAIN_MAX)


# ---------------------------------------------------------------- lock_after_warmup

def test_auto_mode_does_nothing(monkeypatch):
    ctrl, src = make(monkeypatch, "auto")
    ctrl.lock_after_warmup()
    assert src.set_calls == [] and not ctrl.locked


def test_lock_sets_readback_values(monkeypatch):
    ctrl, src = make(monkeypatch, "locked")
    ctrl.manage_step(120.0, hold=False)            # brightness seen during warm-up
    ctrl.lock_after_warmup()
    assert src.set_calls == [(8000, 2.0)]
    assert ctrl.locked and ctrl.target_v == 120.0


def test_lock_caps_long_exposure(monkeypatch):
    ctrl, src = make(monkeypatch, "locked", FakeSource(30000, 1.0))
    ctrl.lock_after_warmup()
    assert src.set_calls == [(10000, 3.0)]


def test_configured_target_is_kept(monkeypatch):
    ctrl, _ = make(monkeypatch, "locked", target=90)
    ctrl.manage_step(150.0, hold=False)
    ctrl.lock_after_warmup()
    assert ctrl.target_v == 90


def test_bad_readback_uses_fixed_values(monkeypatch, capsys):
    monkeypatch.setattr(config, "USB_LOCK_EXPOSURE_US", 4000)
    monkeypatch.setattr(config, "USB_LOCK_GAIN", 1.5)
    ctrl, src = make(monkeypatch, "locked", FakeSource(0, 0))
    ctrl.lock_after_warmup()
    assert src.set_calls == [(4000, 1.5)] and ctrl.locked
    assert "using fixed" in capsys.readouterr().out


def test_bad_readback_without_fixed_values_stays_auto(monkeypatch, capsys):
    monkeypatch.setattr(config, "USB_LOCK_EXPOSURE_US", None)
    ctrl, src = make(monkeypatch, "locked", FakeSource(None, None))
    ctrl.lock_after_warmup()
    assert src.set_calls == [] and not ctrl.locked
    assert "staying on auto" in capsys.readouterr().out


# ---------------------------------------------------------------- manage_step

def run_frames(ctrl, n, mean_v, hold=False):
    for _ in range(n):
        ctrl.manage_step(mean_v, hold)


def test_managed_nudges_every_check_frames(monkeypatch):
    ctrl, src = make(monkeypatch, "managed", target=100)
    ctrl.lock_after_warmup()
    src.set_calls.clear()
    run_frames(ctrl, config.EXPOSURE_CHECK_FRAMES - 1, 50.0)
    assert src.set_calls == []
    ctrl.manage_step(50.0, False)
    assert len(src.set_calls) == 1
    assert src.set_calls[0][0] == pytest.approx(8800)


def test_hold_freezes_exposure(monkeypatch):
    ctrl, src = make(monkeypatch, "managed", target=100)
    ctrl.lock_after_warmup()
    src.set_calls.clear()
    run_frames(ctrl, config.EXPOSURE_CHECK_FRAMES * 3, 50.0, hold=True)
    assert src.set_calls == []


def test_locked_mode_never_nudges(monkeypatch):
    ctrl, src = make(monkeypatch, "locked", target=100)
    ctrl.lock_after_warmup()
    src.set_calls.clear()
    run_frames(ctrl, config.EXPOSURE_CHECK_FRAMES * 3, 50.0)
    assert src.set_calls == []


def test_no_nudge_before_lock(monkeypatch):
    ctrl, src = make(monkeypatch, "managed", target=100)
    run_frames(ctrl, config.EXPOSURE_CHECK_FRAMES * 3, 50.0)
    assert src.set_calls == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))