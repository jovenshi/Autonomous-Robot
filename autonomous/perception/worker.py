"""Vision worker process: location evidence and signs off the control process.

With VISION_WORKER_ENABLED the front camera thread only measures the lane
and shares each frame (hal/shared_frame.py); this process reads the newest
one and runs process_vision on it (the same code as the in-thread path),
putting a VisionResult in a one-slot queue: a newer result replaces one the
control loop has not taken yet, so it never queues behind.

Nothing here can stop the control loop: publishing and taking results never
wait, an error on one frame is counted and the frame skipped, and a stale
or dead worker only raises VISION_STALE (estimation/health.py).

run_worker is the process entry point; like the display process it imports
config itself and leaves Ctrl-C to the main process.
"""
import multiprocessing
import queue
import signal
import threading
import traceback
from dataclasses import dataclass

from hal.shared_frame import SharedFrame
from perception.pipeline import SignPacer, process_vision


@dataclass
class Worker:
    process: object                       # multiprocessing.Process
    results: object                       # one-slot queue of VisionResult
    stop: object                          # Event: set to end the worker


def start_worker(shared, signs_enabled, cfg):
    # Starts the worker (spawn). shared: SharedFrame made with the same
    # context; signs_enabled: that context's Event (set by Behaviour).
    ctx = multiprocessing.get_context("spawn")
    results, stop = ctx.Queue(maxsize=1), ctx.Event()
    results.cancel_join_thread()                  # never wait on a dead worker at exit
    process = ctx.Process(target=run_worker, args=(shared.handle(), signs_enabled, stop, results),
                          name="vision", daemon=True)
    process.start()
    return Worker(process, results, stop)


def stop_worker(worker, cfg):
    # Asks the worker to end; terminates (then kills) it if it doesn't.
    worker.stop.set()
    worker.process.join(cfg.VISION_STOP_TIMEOUT_S)
    if worker.process.is_alive():
        worker.process.terminate()
        worker.process.join(cfg.VISION_STOP_TIMEOUT_S)
    if worker.process.is_alive():
        worker.process.kill()
        worker.process.join(cfg.VISION_STOP_TIMEOUT_S)


def take_result(worker):
    # The newest VisionResult waiting, or None; never blocks.
    try:
        return worker.results.get_nowait()
    except (queue.Empty, EOFError, OSError, ValueError):
        return None


def run_worker(handle, signs_enabled, stop, results, cfg=None):
    # Process loop: wait for a new frame, process_vision, put the result.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl-C is main's; main stops us
    if cfg is None:
        import config as cfg
    shared = SharedFrame.attach(handle)
    model, pacer, last_seq, errors = _load_model(cfg), SignPacer(), None, 0
    try:
        while not stop.is_set():
            if not shared.wait(cfg.VISION_WAIT_S):
                continue
            newest = shared.read(last_seq)
            if newest is None:
                continue
            frame, line_px = newest
            last_seq = frame.seq
            result, errors = _process(frame, line_px, signs_enabled, model, pacer, cfg, errors)
            if result is not None:
                put_latest(results, result)
    finally:
        shared.close()


def put_latest(results, result):
    # Replaces any result still waiting with this one; never blocks.
    while True:
        try:
            results.put_nowait(result)
            return
        except queue.Full:
            try:
                results.get_nowait()
            except queue.Empty:
                pass


def _process(frame, line_px, signs_enabled, model, pacer, cfg, errors):
    # (VisionResult or None, error count). An error skips the frame; the
    # first prints a traceback, later ones are only counted.
    try:
        return process_vision(frame, line_px, signs_enabled.is_set(), cfg, model, pacer), errors
    except Exception:
        if errors == 0:
            print("Vision worker: processing error (further errors counted only):")
            traceback.print_exc()
        return None, errors + 1


def _load_model(cfg):
    # The sign model, or None (candidates only), as in main.py.
    from perception.signs import load_sign_model
    try:
        return load_sign_model(cfg.SIGN_MODEL_PATH, cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"Vision worker: {e}; signs will be boxed but not read")
        return None
