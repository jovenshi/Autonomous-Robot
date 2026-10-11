"""Camera threads and the two camera sources.

CameraStream runs one camera in its own thread:
    read -> stamp t and seq -> process_fn(frame) -> write the result to its slot.
The newest result overwrites the older one; nothing queues.

CsiSource (Picamera2, front) and UsbSource (OpenCV V4L2, side) expose the same
five methods, so CameraStream never knows which one it has:
    open(), read(), get_exposure(), set_exposure(exposure_us, gain), close()

Both sources start with auto-exposure on, so hal/exposure.py can let it
settle and then lock it with set_exposure().
"""
import threading
import time
import traceback

import cv2

from robot_types import Frame


class CameraStream:
    def __init__(self, source, process_fn, slot, name, cfg):
        # process_fn(frame) -> result; use functools.partial for extra arguments.
        self.source = source
        self.process_fn = process_fn
        self.slot = slot
        self.name = name                 # 'front' or 'side'; becomes Frame.cam_id
        self.cfg = cfg
        self.error_count = 0             # frames whose processing raised
        self._seq = 0
        self._fps = 0.0
        self._window_start = None
        self._window_frames = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        # Opens the source and starts the thread.
        self.source.open()
        self._thread = threading.Thread(target=self._loop, name=f"cam_{self.name}", daemon=True)
        self._thread.start()

    def stop(self):
        # Ends the thread, then releases the camera. Called by shutdown.
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.source.close()

    def fps(self):
        # Measured frame rate, for health checks and the display.
        return self._fps

    def _loop(self):
        while not self._stop.is_set():
            image = self.source.read()
            if image is None:
                time.sleep(self.cfg.CAM_READ_RETRY_S)
                continue
            t = time.monotonic()
            self._seq += 1
            frame = Frame(image=image, t=t, seq=self._seq, cam_id=self.name)
            result = self._process(frame)
            if result is not None:
                self.slot.set(result)
            self._count_frame(t)

    def _process(self, frame):
        # Runs process_fn; on error, skip the frame and keep running.
        # The first error prints a traceback; later ones are only counted.
        try:
            return self.process_fn(frame)
        except Exception:
            self.error_count += 1
            if self.error_count == 1:
                print(f"Camera '{self.name}' processing error (further errors counted only):")
                traceback.print_exc()
            return None

    def _count_frame(self, t):
        # Updates fps once per CAM_FPS_WINDOW_S.
        if self._window_start is None:
            self._window_start = t
            self._window_frames = 0
            return
        self._window_frames += 1
        elapsed = t - self._window_start
        if elapsed >= self.cfg.CAM_FPS_WINDOW_S:
            self._fps = self._window_frames / elapsed
            self._window_start = t
            self._window_frames = 0


class CsiSource:
    """Front camera: Raspberry Pi Camera Module 3 through Picamera2."""

    def __init__(self, cam_num, size, cfg):
        self.cam_num = cam_num
        self.size = size                 # (width, height)
        self.cfg = cfg
        self.picam2 = None
        self._metadata = {}
        self._lock = threading.Lock()

    def open(self):
        # Configures for low latency at CAM_FPS, starts, then locks focus.
        from picamera2 import Picamera2  # imported here so tests and laptops don't need it

        self.picam2 = Picamera2(self.cam_num)
        frame_us = int(1_000_000 / self.cfg.CAM_FPS)
        config = self.picam2.create_video_configuration(
            main={"size": tuple(self.size), "format": "RGB888"},  # RGB888 = BGR byte order, as OpenCV expects
            controls={"FrameDurationLimits": (frame_us, frame_us)},
            queue=False,                 # always wait for a fresh frame, never a queued old one
        )
        self.picam2.configure(config)
        self.picam2.start()
        self._lock_focus()

    def read(self):
        # One BGR frame; keeps its metadata for get_exposure().
        with self._lock:
            request = self.picam2.capture_request()
            try:
                image = request.make_array("main")
                self._metadata = request.get_metadata()
            finally:
                request.release()
        return image

    def get_exposure(self):
        # From the newest frame's metadata.
        md = self._metadata
        return md.get("ExposureTime"), md.get("AnalogueGain")

    def set_exposure(self, exposure_us, gain):
        # Manual exposure; auto-exposure and auto white balance off.
        # White balance is frozen at the newest frame's colour gains.
        controls = {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": int(exposure_us),
            "AnalogueGain": float(gain),
        }
        colour_gains = self._metadata.get("ColourGains")
        if colour_gains is not None:
            controls["ColourGains"] = tuple(colour_gains)
        self.picam2.set_controls(controls)

    def close(self):
        if self.picam2 is not None:
            with self._lock:
                self.picam2.stop()
                self.picam2.close()
            self.picam2 = None

    def _lock_focus(self):
        # Camera Module 3 only: manual focus at FRONT_LENS_POSITION, so the
        # lens never hunts while driving. Skipped on cameras without autofocus.
        if "AfMode" not in self.picam2.camera_controls:
            return
        from libcamera import controls
        self.picam2.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": float(self.cfg.FRONT_LENS_POSITION),
        })


class UsbSource:
    """Side camera: USB webcam through OpenCV's V4L2 backend."""

    def __init__(self, device_path, size, fourcc, cfg):
        self.device_path = device_path   # /dev/v4l/by-id/..., never an index
        self.size = size                 # (width, height)
        self.fourcc = fourcc
        self.cfg = cfg
        self.cap = None
        self._lock = threading.Lock()

    def open(self):
        # Opens by path; sets YUYV, size, fps and a one-frame buffer.
        if self.device_path is None:
            raise ValueError("SIDE_CAM_DEVICE is not set in config.py")
        cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam at {self.device_path}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, self.cfg.CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.cfg.USB_AUTO_EXPOSURE_AUTO)
        self.cap = cap
        self._warn_if_size_differs()

    def read(self):
        # One BGR frame, or None if the read failed.
        with self._lock:
            ok, image = self.cap.read()
        return image if ok else None

    def get_exposure(self):
        # From OpenCV properties. Many webcams report the last manual value
        # here, not what auto-exposure is currently using.
        with self._lock:
            exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            gain = self.cap.get(cv2.CAP_PROP_GAIN)
        return exposure * self.cfg.USB_EXPOSURE_UNIT_US, gain

    def set_exposure(self, exposure_us, gain):
        # Manual exposure; auto-exposure and auto white balance off.
        with self._lock:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.cfg.USB_AUTO_EXPOSURE_MANUAL)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_us / self.cfg.USB_EXPOSURE_UNIT_US)
            self.cap.set(cv2.CAP_PROP_GAIN, gain)

    def close(self):
        if self.cap is not None:
            with self._lock:
                self.cap.release()
            self.cap = None

    def _warn_if_size_differs(self):
        # Webcams silently fall back to another size if one is unsupported.
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width, height) != tuple(self.size):
            print(f"Webcam gave {width}x{height}, asked for {self.size[0]}x{self.size[1]}")