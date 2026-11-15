"""The newest front camera frame in shared memory, for the vision worker.

One block holds a small header and the full-size BGR frame:
    header (float64): seq, t, has_line, x0, y0, x1, y1
where (x0, y0)-(x1, y1) is the lane's fitted line (FrontLaneObs.line_px,
lane-size px) from the camera thread, which the dash finder needs. The
camera thread publishes; the worker process reads the newest frame. A lock
covers each copy (about 0.2 ms for 640 x 480), so a frame is never read half
written, and an event wakes the reader when a new frame is in.

The control process creates the block (SharedFrame.create) and passes
handle() to the worker, which attaches (SharedFrame.attach). Only the
creator unlinks it, in close(), after the camera thread has stopped.
"""
from multiprocessing import shared_memory

import numpy as np

from robot_types import Frame

_HEADER = 7                                # float64 values before the image


class SharedFrame:
    def __init__(self, shm, shape, lock, new_frame, owner):
        self._shm = shm
        self.shape = tuple(shape)
        self._lock = lock
        self._new_frame = new_frame
        self._owner = owner
        self._header = np.ndarray((_HEADER,), np.float64, shm.buf, 0)
        self._image = np.ndarray(self.shape, np.uint8, shm.buf, _HEADER * 8)

    @classmethod
    def create(cls, size, ctx):
        # A new block for (width, height) BGR frames; ctx: the multiprocessing
        # context the worker is started with.
        shape = (size[1], size[0], 3)
        shm = shared_memory.SharedMemory(create=True, size=_HEADER * 8 + int(np.prod(shape)))
        shared = cls(shm, shape, ctx.Lock(), ctx.Event(), owner=True)
        shared._header[:] = 0.0                # seq 0: nothing published yet
        return shared

    @classmethod
    def attach(cls, handle):
        # The worker's side, from handle(). A spawned worker shares its
        # creator's resource tracker, so registering the block again here is
        # harmless (unregistering it here would break the creator's unlink).
        name, shape, lock, new_frame = handle
        shm = shared_memory.SharedMemory(name=name)
        return cls(shm, shape, lock, new_frame, owner=False)

    def handle(self):
        # What the worker process needs to attach (picklable at spawn).
        return self._shm.name, self.shape, self._lock, self._new_frame

    def publish(self, frame, line_px):
        # Camera thread: the newest frame and the lane's fitted line (or None).
        line = (1.0, *line_px[0], *line_px[1]) if line_px is not None else (0.0,) * 5
        with self._lock:
            self._image[:] = frame.image
            self._header[:] = (frame.seq, frame.t, *line)
        self._new_frame.set()

    def wait(self, timeout):
        # Worker: True once a frame has been published since the last wait.
        if not self._new_frame.wait(timeout):
            return False
        self._new_frame.clear()
        return True

    def read(self, last_seq=None):
        # (Frame, line_px or None) for the newest frame, or None if nothing
        # has been published or it is still last_seq.
        with self._lock:
            seq = int(self._header[0])
            if seq == 0 or seq == last_seq:
                return None
            t, has_line, x0, y0, x1, y1 = self._header[1:]
            image = self._image.copy()
        line_px = ((x0, y0), (x1, y1)) if has_line else None
        return Frame(image=image, t=float(t), seq=seq, cam_id="front"), line_px

    def close(self):
        # Releases this process's view; the creator also removes the block.
        self._header = self._image = None
        self._shm.close()
        if self._owner:
            self._shm.unlink()
