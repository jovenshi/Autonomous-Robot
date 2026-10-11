"""Tests for hal/camera.py with fake cameras (no Pi or webcam needed).

Run with:  python test/test_camera.py   (or: python -m pytest)
"""
import os
import sys
import time
import types

import cv2
import numpy as np
import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from hal import camera
from hal.camera import CameraStream, CsiSource, UsbSource
from robot_types import LatestValue


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


# ---------------------------------------------------------------- CameraStream

class FakeSource:
    def __init__(self, fail_every=0):
        self.opened = self.closed = False
        self.reads = 0
        self.fail_every = fail_every

    def open(self):
        self.opened = True

    def read(self):
        time.sleep(0.002)
        self.reads += 1
        if self.fail_every and self.reads % self.fail_every == 0:
            return None
        return np.zeros((240, 320, 3), np.uint8)

    def close(self):
        self.closed = True


def test_stream_stamps_and_publishes():
    slot = LatestValue()
    src = FakeSource()
    stream = CameraStream(src, lambda f: f, slot, "front", config)
    stream.start()
    assert wait_for(lambda: slot.get() is not None and slot.get().seq >= 5)
    stream.stop()
    frame = slot.get()
    assert frame.cam_id == "front"
    assert frame.t <= time.monotonic()
    assert src.opened and src.closed


def test_stream_seq_strictly_increases():
    seen = []
    slot = LatestValue()
    stream = CameraStream(FakeSource(), lambda f: seen.append(f.seq) or f, slot, "side", config)
    stream.start()
    assert wait_for(lambda: len(seen) >= 10)
    stream.stop()
    assert seen == sorted(set(seen)) and seen[0] == 1


def test_failed_reads_are_skipped():
    slot = LatestValue()
    src = FakeSource(fail_every=2)
    stream = CameraStream(src, lambda f: f, slot, "front", config)
    stream.start()
    assert wait_for(lambda: slot.get() is not None and slot.get().seq >= 5)
    stream.stop()
    assert src.reads > slot.get().seq          # some reads produced no frame


def test_processing_error_is_counted_and_thread_continues(capsys):
    calls = []

    def flaky(frame):
        calls.append(frame.seq)
        if frame.seq % 2 == 0:
            raise RuntimeError("bad frame")
        return frame

    slot = LatestValue()
    stream = CameraStream(FakeSource(), flaky, slot, "front", config)
    stream.start()
    assert wait_for(lambda: len(calls) >= 10)
    stream.stop()
    assert stream.error_count >= 4
    assert slot.get().seq % 2 == 1                    # only good frames published
    assert capsys.readouterr().out.count("processing error") == 1


def test_fps_is_measured(monkeypatch):
    monkeypatch.setattr(config, "CAM_FPS_WINDOW_S", 0.2)
    stream = CameraStream(FakeSource(), lambda f: f, LatestValue(), "front", config)
    stream.start()
    assert wait_for(lambda: stream.fps() > 0)
    stream.stop()
    assert 50 < stream.fps() < 600                    # fake source runs at ~400 fps


# ---------------------------------------------------------------- UsbSource

class FakeCapture:
    def __init__(self, path, backend):
        self.path, self.backend = path, backend
        self.props = {}
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def read(self):
        return True, np.zeros((240, 320, 3), np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def usb(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoCapture", FakeCapture)
    src = UsbSource("/dev/v4l/by-id/fake", (320, 240), "YUYV", config)
    src.open()
    return src


def test_usb_open_settings(usb):
    p = usb.cap.props
    assert usb.cap.backend == cv2.CAP_V4L2
    assert p[cv2.CAP_PROP_BUFFERSIZE] == 1
    assert p[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"YUYV")
    assert (p[cv2.CAP_PROP_FRAME_WIDTH], p[cv2.CAP_PROP_FRAME_HEIGHT]) == (320, 240)
    assert p[cv2.CAP_PROP_AUTO_EXPOSURE] == config.USB_AUTO_EXPOSURE_AUTO


def test_usb_exposure_round_trip(usb):
    usb.set_exposure(5000, 10)
    p = usb.cap.props
    assert p[cv2.CAP_PROP_AUTO_EXPOSURE] == config.USB_AUTO_EXPOSURE_MANUAL
    assert p[cv2.CAP_PROP_AUTO_WB] == 0
    assert p[cv2.CAP_PROP_EXPOSURE] == 5000 / config.USB_EXPOSURE_UNIT_US
    assert usb.get_exposure() == (5000, 10)


def test_usb_requires_device_path():
    with pytest.raises(ValueError):
        UsbSource(None, (320, 240), "YUYV", config).open()


def test_usb_close_releases(usb):
    cap = usb.cap
    usb.close()
    assert cap.released and usb.cap is None


# ---------------------------------------------------------------- CsiSource

class FakeRequest:
    def __init__(self, owner):
        self.owner = owner

    def make_array(self, name):
        return np.zeros((480, 640, 3), np.uint8)

    def get_metadata(self):
        return {"ExposureTime": 8000, "AnalogueGain": 2.0, "ColourGains": (1.5, 1.8)}

    def release(self):
        self.owner.released += 1


class FakePicamera2:
    def __init__(self, num):
        self.num = num
        self.camera_controls = {"AfMode": None, "LensPosition": None}
        self.controls_set = []
        self.config_args = None
        self.released = 0
        self.started = self.closed = False

    def create_video_configuration(self, **kwargs):
        self.config_args = kwargs
        return kwargs

    def configure(self, cfg):
        pass

    def start(self):
        self.started = True

    def capture_request(self):
        return FakeRequest(self)

    def set_controls(self, c):
        self.controls_set.append(c)

    def stop(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def csi(monkeypatch):
    af_enum = types.SimpleNamespace(Manual="MANUAL")
    monkeypatch.setitem(sys.modules, "picamera2", types.SimpleNamespace(Picamera2=FakePicamera2))
    monkeypatch.setitem(sys.modules, "libcamera",
                        types.SimpleNamespace(controls=types.SimpleNamespace(AfModeEnum=af_enum)))
    src = CsiSource(0, (640, 480), config)
    src.open()
    return src


def test_csi_configuration(csi):
    args = csi.picam2.config_args
    assert args["main"] == {"size": (640, 480), "format": "RGB888"}
    assert args["queue"] is False
    frame_us = int(1_000_000 / config.CAM_FPS)
    assert args["controls"]["FrameDurationLimits"] == (frame_us, frame_us)


def test_csi_focus_locked(csi):
    assert csi.picam2.controls_set[0] == {"AfMode": "MANUAL",
                                          "LensPosition": config.FRONT_LENS_POSITION}


def test_csi_read_releases_request_and_keeps_metadata(csi):
    img = csi.read()
    assert img.shape == (480, 640, 3)
    assert csi.picam2.released == 1
    assert csi.get_exposure() == (8000, 2.0)


def test_csi_set_exposure_freezes_white_balance(csi):
    csi.read()
    csi.set_exposure(6000, 1.5)
    assert csi.picam2.controls_set[-1] == {"AeEnable": False, "AwbEnable": False,
                                           "ExposureTime": 6000, "AnalogueGain": 1.5,
                                           "ColourGains": (1.5, 1.8)}


def test_csi_close(csi):
    cam = csi.picam2
    csi.close()
    assert cam.closed and csi.picam2 is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
