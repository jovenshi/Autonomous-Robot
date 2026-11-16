"""Records a run's camera frames while main.py drives (D-pad Up, UiCommand.RECORD).

One folder per recording under RECORD_DIR, named run_<date>_<time>, in
the tools/record.py format (front frames, side frames in side/), so
tools/replay.py and the tuning tools read it as any other recording.
meta.json also names the mode (QUIZ_INTERSECTION, ...) and the loop log of this run: frame t and the
loop log's t are both time.monotonic(), so every frame matches its loop rows.

The camera threads call front() and side() with every frame; they only hand
the frame to a RecordingWriter queue (never blocking: frames are skipped and
counted if the SD card falls behind). stop() finishes writing in a thread of
its own so the control loop never waits on the card; close() (shutdown)
waits for it.
"""
import datetime
import os
import threading
import time

from robot_types import RecordStatus
from tools.record import make_meta
from tools.recording import RecordingWriter


class RunRecorder:
    def __init__(self, base_dir, cfg, writer_factory=RecordingWriter, clock=time.monotonic):
        self.base_dir = base_dir
        self.cfg = cfg
        self.writer_factory = writer_factory
        self.clock = clock
        self.loop_log_path = None          # set by main once the loop log is open
        self._lock = threading.Lock()      # camera threads vs start/stop
        self._front = None                 # RecordingWriter while recording
        self._side = None
        self._name = None
        self._started_t = None
        self._closing = []                 # threads still writing stopped recordings

    @property
    def active(self):
        return self._front is not None

    def start(self, mode_name, with_side):
        # Opens a new recording -> its folder, or None (logged) if it can't be made.
        if self.active:
            return None
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"run_{stamp}"
        path = os.path.join(self.base_dir, name)
        n = 2
        while os.path.exists(path):        # two recordings in the same second
            name = f"run_{stamp}_{n}"
            path, n = os.path.join(self.base_dir, name), n + 1
        cfg = self.cfg
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            front = self.writer_factory(path, cfg.RECORD_PNG_LEVEL, cfg.RECORD_QUEUE)
            front.start(self._meta("front", mode_name))
            side = None
            if with_side:
                side = self.writer_factory(os.path.join(path, "side"), cfg.RECORD_PNG_LEVEL,
                                           cfg.RECORD_QUEUE)
                side.start(self._meta("side", mode_name))
        except OSError as e:
            print(f"Recording: could not start in {path} ({e})")
            return None
        with self._lock:
            self._front, self._side = front, side
            self._name, self._started_t = name, self.clock()
        print(f"Recording to {path}")
        return path

    def front(self, frame, exposure_us=None, gain=None):
        # Camera thread: one front frame (dropped at once when not recording).
        writer = self._front
        if writer is not None:
            writer.submit(frame, exposure_us, gain)

    def side(self, frame):
        writer = self._side
        if writer is not None:
            writer.submit(frame)

    def status(self):
        # RecordStatus for the display, or None when not recording.
        with self._lock:
            front, side = self._front, self._side
            if front is None:
                return None
            writers = [w for w in (front, side) if w is not None]
            return RecordStatus(name=self._name, started_t=self._started_t,
                                saved=sum(w.saved for w in writers),
                                dropped=sum(w.dropped for w in writers))

    def stop(self, why=""):
        # Stops taking frames now; the frames already queued are written in
        # the background.
        with self._lock:
            writers = [w for w in (self._front, self._side) if w is not None]
            name, started = self._name, self._started_t
            self._front = self._side = self._name = self._started_t = None
        if not writers:
            return
        seconds = self.clock() - started

        def finish():
            for w in writers:
                w.close()
            saved, dropped = sum(w.saved for w in writers), sum(w.dropped for w in writers)
            print(f"Recording {name} done: {seconds:.1f} s, {saved} frames saved, "
                  f"{dropped} skipped")

        print(f"Recording {name} stopped{f' ({why})' if why else ''}; finishing writing")
        thread = threading.Thread(target=finish, name="recorder-close", daemon=True)
        thread.start()
        self._closing = [t for t in self._closing if t.is_alive()] + [thread]

    def close(self):
        # Shutdown: stop, and wait until every recording is on the card.
        self.stop("shutting down")
        for thread in self._closing:
            thread.join()
        self._closing = []

    def _meta(self, camera, mode_name):
        meta = make_meta(self.cfg, camera)
        meta["mode"] = mode_name
        meta["loop_log"] = self.loop_log_path
        meta["source"] = "main.py"
        return meta
