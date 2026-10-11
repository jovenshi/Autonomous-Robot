"""Record the front camera to a folder of PNG frames, for tuning and replay.

Run from the autonomous/ folder, with the robot pushed slowly by hand (the motors
are never touched):
    python -m tools.record                       # until Ctrl-C
    python -m tools.record --seconds 20 --out recordings/straight_1

Exposure is handled exactly as main.py does (EXPOSURE_MODE, locked after
CAM_WARMUP_S), and recording starts only after the lock, so the frames look
like what the robot sees when driving. In 'managed' mode the exposure stays
at its locked value while recording.
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
from hal.camera import CameraStream, CsiSource
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
    return p.parse_args()


def make_meta(cfg):
    return {
        "camera": "front",
        "frame_size": list(cfg.FRONT_SIZE),
        "cam_fps": cfg.CAM_FPS,
        "exposure_mode": cfg.EXPOSURE_MODE,
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


def wait_and_report(stream, writer, seconds):
    # Prints progress once a second until --seconds or Ctrl-C.
    start = time.monotonic()
    while seconds is None or time.monotonic() - start < seconds:
        time.sleep(1.0)
        print(f"saved {writer.saved}  skipped {writer.dropped}  camera {stream.fps():.1f} fps")


def main():
    args = parse_args()
    writer = RecordingWriter(args.out, args.png_level, args.queue)
    recording = threading.Event()
    stream, exposure = build(config, writer, recording)
    stream.start()
    try:
        exposure.lock_after_warmup()
        writer.start(make_meta(config))
        recording.set()
        print(f"Recording to {args.out}  (Ctrl-C to stop)")
        wait_and_report(stream, writer, args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        recording.clear()
        stream.stop()
        writer.close()
        print(f"Done: {writer.saved} frames saved, {writer.dropped} skipped, in {args.out}")


if __name__ == "__main__":
    main()
