"""Reading and writing recordings, shared by every tool.

A recording is a folder:
    frames.csv   seq, t, file, exposure_us, gain   (one row per saved frame)
    meta.json    camera, frame size, exposure mode, start time
    frame_000123.png ...                           (lossless BGR frames)

Frames are saved by a writer thread through a bounded queue: if the SD card
can't keep up, frames are skipped and counted rather than piling up.
"""
import csv
import json
import os
import queue
import sys
import threading

import cv2

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from robot_types import Frame  # noqa: E402

FRAMES_CSV = "frames.csv"
META_JSON = "meta.json"
CSV_FIELDS = ("seq", "t", "file", "exposure_us", "gain")


class RecordingWriter:
    def __init__(self, out_dir, png_level, queue_size):
        self.out_dir = out_dir
        self.png_level = png_level
        self.saved = 0
        self.dropped = 0
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = None
        self._csv_file = None
        self._csv = None

    def start(self, meta):
        # Creates the folder, writes meta.json and starts the writer thread.
        os.makedirs(self.out_dir, exist_ok=False)
        with open(os.path.join(self.out_dir, META_JSON), "w") as f:
            json.dump(meta, f, indent=2)
        self._csv_file = open(os.path.join(self.out_dir, FRAMES_CSV), "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(CSV_FIELDS)
        self._thread = threading.Thread(target=self._loop, name="recorder", daemon=True)
        self._thread.start()

    def submit(self, frame, exposure_us=None, gain=None):
        # Non-blocking; returns False (and counts a drop) if the queue is full.
        try:
            self._queue.put_nowait((frame, exposure_us, gain))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def close(self):
        # Writes everything still queued, then closes the CSV.
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
        if self._csv_file is not None:
            self._csv_file.close()

    def _loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame, exposure_us, gain = item
            name = f"frame_{frame.seq:06d}.png"
            cv2.imwrite(os.path.join(self.out_dir, name), frame.image,
                        [cv2.IMWRITE_PNG_COMPRESSION, self.png_level])
            self._csv.writerow((frame.seq, f"{frame.t:.6f}", name,
                                "" if exposure_us is None else exposure_us,
                                "" if gain is None else gain))
            self.saved += 1


def read_index(rec_dir):
    # The rows of frames.csv, in recording order.
    with open(os.path.join(rec_dir, FRAMES_CSV), newline="") as f:
        return list(csv.DictReader(f))


def read_meta(rec_dir):
    with open(os.path.join(rec_dir, META_JSON)) as f:
        return json.load(f)


def load_frame(rec_dir, row, cam_id="front"):
    # One recorded frame as a Frame, with its original seq and t.
    image = cv2.imread(os.path.join(rec_dir, row["file"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(os.path.join(rec_dir, row["file"]))
    return Frame(image=image, t=float(row["t"]), seq=int(row["seq"]), cam_id=cam_id)


def iter_frames(rec_dir, step=1):
    # Every step-th frame of a recording.
    for row in read_index(rec_dir)[::step]:
        yield load_frame(rec_dir, row)
