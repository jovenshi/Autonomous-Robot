"""Record the front camera (and with --side the side camera) to PNG frames,
for tuning and replay.

Run from the autonomous/ folder, with the robot pushed slowly by hand (the motors
are never touched):
    python -m tools.record                       # until Ctrl-C
    python -m tools.record --seconds 20 --out recordings/straight_1
    python -m tools.record --side --out recordings/junction_side_1

Exposure is handled exactly as main.py does (EXPOSURE_MODE, locked after
CAM_WARMUP_S), and recording starts only after the lock, so the frames look
like what the robot sees when driving. In 'managed' mode the exposure stays
at its locked value while recording.

--side records the side camera too, into <out>/side/ (a recording of its
own). Both cameras stamp frames with the same clock (time.monotonic), so a
side frame can be matched with where the front camera put the robot at that
moment (tools/side_fixes.py). The side camera stays on auto-exposure: its
manual exposure settings are not confirmed yet (USB_* in config.py).
"""
import argparse
import datetime
import os
import sys
import threading
import time

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from hal.camera import CameraStream, CsiSource, UsbSource
from hal.exposure import ExposureControl
from robot_types import LatestValue
from tools.recording import RecordingWriter


def parse_args():
    p = argparse.ArgumentParser(description="Record the front camera to PNG frames.")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p.add_argument("--out", default=f"recordings/front_{stamp}", help="new folder to create")
    p.add_argument("--seconds", type=float, default=None, help="stop after this long")
    p.add_argument("--png-level", type=int, default=1, help="PNG compression 0-9 (1 = fast)")
    p.add_argument("--queue", type=int, default=60, help="frames buffered before skipping")
    p.add_argument("--side", action="store_true", help="also record the side camera to <out>/side")
    return p.parse_args()


def make_meta(cfg, camera="front"):
    return {
        "camera": camera,
        "frame_size": list(cfg.FRONT_SIZE if camera == "front" else cfg.SIDE_SIZE),
        "cam_fps": cfg.CAM_FPS,
        "exposure_mode": cfg.EXPOSURE_MODE if camera == "front" else "auto",
        "capture_size": list(cfg.FRONT_SIZE if camera == "front" else cfg.SIDE_CAPTURE_SIZE),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def build(cfg, writer, recording):
    # Camera thread whose per-frame work is handing the frame to the writer
    # once recording has been switched on.
    source = CsiSource(cfg.FRONT_CAM_NUM, cfg.FRONT_SIZE, cfg)

    def save(frame):
        if recording.is_set():
            exposure_us, gain = source.get_exposure()
            writer.submit(frame, exposure_us, gain)
        return frame

    stream = CameraStream(source, save, LatestValue(), "front", cfg)
    return stream, ExposureControl(source, "front", cfg)


def build_side(cfg, writer, recording):
    # The side camera's thread, saving like the front one (auto-exposure).
    source = UsbSource(cfg.SIDE_CAM_DEVICE, cfg.SIDE_SIZE, cfg.SIDE_CAM_FOURCC, cfg,
                       cfg.SIDE_CAPTURE_SIZE)

    def save(frame):
        if recording.is_set():
            writer.submit(frame)
        return frame

    return CameraStream(source, save, LatestValue(), "side", cfg)


def wait_and_report(streams, seconds):
    # Prints progress once a second until --seconds or Ctrl-C.
    # streams: [(name, CameraStream, RecordingWriter)].
    start = time.monotonic()
    while seconds is None or time.monotonic() - start < seconds:
        time.sleep(1.0)
        print("  ".join(f"{name}: saved {w.saved} skipped {w.dropped} at {s.fps():.1f} fps"
                        for name, s, w in streams))


def main():
    args = parse_args()
    writer = RecordingWriter(args.out, args.png_level, args.queue)
    recording = threading.Event()
    stream, exposure = build(config, writer, recording)
    streams = [("front", stream, writer)]
    if args.side:
        side_writer = RecordingWriter(os.path.join(args.out, "side"), args.png_level, args.queue)
        streams.append(("side", build_side(config, side_writer, recording), side_writer))
    for _, s, _ in streams:
        s.start()
    try:
        exposure.lock_after_warmup()
        writer.start(make_meta(config))
        if args.side:
            side_writer.start(make_meta(config, "side"))
        recording.set()
        print(f"Recording to {args.out}  (Ctrl-C to stop)")
        wait_and_report(streams, args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        recording.clear()
        for name, s, w in streams:
            s.stop()
            w.close()
            print(f"Done ({name}): {w.saved} frames saved, {w.dropped} skipped")
        print(f"In {args.out}")


if __name__ == "__main__":
    main()
