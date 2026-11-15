"""Stage 7 (docs/quiz3_plan.md): the shared frame and the vision worker
process. Real processes and shared memory, synthetic junction frames; no
camera or robot needed."""
import math
import multiprocessing
import os
import queue
import sys
import threading
import time
import types
from multiprocessing import shared_memory

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
import main  # noqa: E402
from estimation.health import check_health  # noqa: E402
from hal.shared_frame import SharedFrame  # noqa: E402
from perception import worker as worker_mod  # noqa: E402
from perception.color import make_front_masks  # noqa: E402
from perception.front_lane import process_front_lane  # noqa: E402
from perception.pipeline import process_front_frame, process_vision  # noqa: E402
from perception.worker import put_latest, start_worker, stop_worker, take_result  # noqa: E402
from robot_types import Frame, HealthWarning, MotorCmd, VisionResult  # noqa: E402
from test_intersection_quiz import make_quiz_system  # noqa: E402
from test_location import junction_scene  # noqa: E402

CTX = multiprocessing.get_context("spawn")


def junction_frame(s, seq, t=None):
    frame = Frame(junction_scene(s), seq / 30 if t is None else t, seq, "front")
    lane = process_front_lane(make_front_masks(frame, config), config, frame.t, frame.seq)
    return frame, lane.line_px


@pytest.fixture
def shared():
    sf = SharedFrame.create(config.FRONT_SIZE, CTX)
    yield sf
    try:
        sf.close()
    except (FileNotFoundError, TypeError):
        pass                                     # closed by the test already


# ---------------------------------------------------------------- shared frame

def test_shared_frame_round_trip(shared):
    assert shared.read() is None                             # nothing published yet
    frame, line = junction_frame(-1.0, 7, t=12.5)
    shared.publish(frame, line)
    got, got_line = shared.read()
    assert (got.seq, got.t, got.cam_id) == (7, 12.5, "front")
    assert np.array_equal(got.image, frame.image)
    assert np.allclose(got_line, line)
    assert shared.read(last_seq=7) is None                   # nothing newer
    shared.publish(Frame(frame.image, 13.0, 8, "front"), None)
    assert shared.read(last_seq=7)[1] is None                # no lane line


def test_wait_wakes_on_a_new_frame(shared):
    assert not shared.wait(0.01)
    frame, line = junction_frame(-1.0, 1)
    shared.publish(frame, line)
    assert shared.wait(0.01) and not shared.wait(0.01)       # one wake per publish


def _read_in_child(handle, out):
    sf = SharedFrame.attach(handle)
    frame, line = sf.read()
    out.put((frame.seq, int(frame.image.sum()), line))
    sf.close()


def test_another_process_reads_it_and_close_unlinks(shared):
    frame, line = junction_frame(-1.2, 42)
    shared.publish(frame, line)
    out = CTX.Queue()
    child = CTX.Process(target=_read_in_child, args=(shared.handle(), out))
    child.start()
    seq, total, got_line = out.get(timeout=20)
    child.join(10)
    assert (seq, total) == (42, int(frame.image.sum())) and np.allclose(got_line, line)
    name = shared.handle()[0]
    shared.close()                                           # the child's exit didn't unlink it
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


# ---------------------------------------------------------------- worker

def test_put_latest_keeps_only_the_newest():
    q = CTX.Queue(maxsize=1)
    put_latest(q, 1)
    put_latest(q, 2)
    time.sleep(0.05)
    assert q.get(timeout=1) == 2
    with pytest.raises(queue.Empty):
        q.get(timeout=0.05)


def wait_for(worker, seq, timeout=30.0):
    # The worker's result for frame seq.
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = take_result(worker)
        if result is not None and result.seq == seq:
            return result
        time.sleep(0.01)
    raise AssertionError(f"no result for frame {seq}")


def test_worker_process_gives_the_in_thread_result(shared):
    signs = CTX.Event()
    worker = start_worker(shared, signs, config)
    try:
        frame, line = junction_frame(-1.1, 5)
        shared.publish(frame, line)
        got = wait_for(worker, 5)
        expected = process_vision(frame, line, False, config)
        assert got.location.crossing_dist_m == pytest.approx(expected.location.crossing_dist_m)
        assert got.location.dash_dists_m == pytest.approx(expected.location.dash_dists_m)
        assert got.sign is None                              # signs off
        signs.set()                                          # Behaviour switches signs on
        frame, line = junction_frame(-1.05, 6)
        shared.publish(frame, line)
        assert wait_for(worker, 6).sign is not None
    finally:
        stop_worker(worker, config)
    assert not worker.process.is_alive()


def test_worker_skips_a_bad_frame_and_carries_on(shared, monkeypatch, capsys):
    # run_worker in a thread, with process_vision failing once.
    calls = {"n": 0}
    real = worker_mod.process_vision

    def flaky(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("bad frame")
        return real(*args, **kw)

    monkeypatch.setattr(worker_mod, "process_vision", flaky)
    stop, results = threading.Event(), queue.Queue(maxsize=1)
    thread = threading.Thread(target=worker_mod.run_worker,
                              args=(shared.handle(), threading.Event(), stop, results, config))
    thread.start()
    try:
        for seq in (1, 2):
            frame, line = junction_frame(-1.1, seq)
            shared.publish(frame, line)
            time.sleep(0.5)
        assert results.get(timeout=5).seq == 2
    finally:
        stop.set()
        thread.join(5)
    assert "processing error" in capsys.readouterr().out


# ---------------------------------------------------------------- control process

def test_vision_stale_warning():
    args = (None, None, None, None, None, {}, 0, False, 0.0, config)
    assert HealthWarning.VISION_STALE not in check_health(*args)                # no worker
    assert HealthWarning.VISION_STALE not in check_health(*args, vision_age=0.1)
    assert HealthWarning.VISION_STALE in check_health(*args, vision_age=config.VISION_STALE_S + 0.1)
    assert HealthWarning.VISION_STALE in check_health(*args, vision_age=math.inf)


def worker_system(shared):
    sys_ = make_quiz_system()
    sys_.signs_enabled = CTX.Event()
    sys_.shared_frame = shared
    sys_.worker = start_worker(shared, sys_.signs_enabled, config)
    sys_.worker_start_t = time.monotonic()
    return sys_


def step_with_frame(sys_, s, seq):
    # The camera thread's part (lane only, then share the frame), then a loop.
    now = time.monotonic()
    frame = Frame(junction_scene(s), now, seq, "front")
    result = process_front_frame(frame, False, config, with_vision=False)
    assert result.location is None and result.sign is None
    sys_.front_slot.set(result)
    shared = sys_.shared_frame
    shared.publish(frame, result.lane.line_px)
    main.control_step(sys_, now)
    return now


def test_control_loop_uses_the_worker_and_survives_its_death(shared, capsys):
    sys_ = worker_system(shared)
    try:
        # Results arrive: the location reaches the tracker, no VISION_STALE.
        end, seq = time.monotonic() + 30, 0
        while (sys_.vision is None or sys_.vision.location.crossing_dist_m is None) \
                and time.monotonic() < end:
            seq += 1
            step_with_frame(sys_, -1.1, seq)
            time.sleep(0.03)
        assert sys_.vision.location.crossing_dist_m == pytest.approx(1.1, abs=0.04)
        now = step_with_frame(sys_, -1.1, seq + 1)
        assert main._vision_age(sys_, now) < config.VISION_STALE_S
        # The worker dies: the loop goes on sending motor commands, with VISION_STALE.
        sys_.worker.process.kill()
        sys_.worker.process.join(5)
        sent = len(sys_.saber.sent)
        for i in range(3):
            step_with_frame(sys_, -1.1, seq + 2 + i)
        assert len(sys_.saber.sent) == sent + 3
        assert "VISION_STALE" in capsys.readouterr().out
    finally:
        stop_worker(sys_.worker, config)


def test_shutdown_stops_the_worker_and_frees_the_shared_frame(shared):
    sys_ = worker_system(shared)
    name = shared.handle()[0]
    main.shutdown(sys_)
    assert not sys_.worker.process.is_alive()
    assert sys_.saber.calls[:2] == ["stop", "close"]
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_front_result_for_the_display_carries_the_worker_result():
    sys_ = make_quiz_system()
    frame = Frame(junction_scene(-1.1), 1.0, 3, "front")
    front = process_front_frame(frame, False, config, with_vision=False)
    vision = process_vision(frame, front.lane.line_px, False, config)
    assert main._front_for_ui(sys_, front, vision) is front               # no worker
    sys_.worker = types.SimpleNamespace(process=None)
    shown = main._front_for_ui(sys_, front, vision)
    assert shown.location is vision.location and shown.lane is front.lane
