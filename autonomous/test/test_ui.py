"""Tests for ui/ (link, screen, panels, overlay). No robot, no screen needed:
drawing is tested on NumPy images, and the display process runs with SDL's
dummy video driver.

Run with:  python -m pytest test/test_ui.py
"""
import multiprocessing
import os
import queue
import sys
import time

import numpy as np
import pygame
import pytest

# Let `python test/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from robot_types import (DriveCmd, FrontLaneObs, FrontResult, HealthWarning, ImuSample,
                         LaneEstimate, MotorCmd, RoadYaw, SideLaneObs, SideResult, State,
                         UiCommand, UiPacket)
from tools.ui_demo import demo_packet, synthetic_front
from ui.link import make_ui_packet, read_commands, send, start_ui, stop_ui
from ui.overlay import BLUE, PURPLE, draw_front_overlay, draw_no_data, draw_side_overlay, mark_stale
from ui.panels import History, draw_lane_gauge, draw_steer_bar, preflight, road_yaw_rows
from ui.screen import VIEWS, Screen, _route, make_layout

CFG = config
W, H = 320, 240


def blank():
    return np.zeros((H, W, 3), np.uint8)


def bare_packet(t=1.0, state=State.IDLE, **kw):
    # Every optional field empty: nothing measured yet.
    fields = dict(t=t, state=state, stop_reason=None, drive=DriveCmd(0.0, 0.0),
                  motor=MotorCmd(0.0, 0.0), lane=None, road_yaw=None, imu=None, sign_vote=None,
                  fps={}, overruns=0, warnings=(), front=None, side=None)
    fields.update(kw)
    return UiPacket(**fields)


def screen_at(t):
    clock = {"now": t}
    return Screen(CFG, clock=lambda: clock["now"]), clock


# ---------------------------------------------------------------- overlays

def empty_front():
    return FrontResult(blank(), FrontLaneObs(None, None, 0.0, (), None, 0.0, 1), None, None,
                       0.0, 0.0, 1)


def empty_side():
    return SideResult(blank(), SideLaneObs(None, False, False, 0.0, None, 0.0, 1), 0.0, 0.0, 1)


@pytest.mark.parametrize("front", [None, empty_front()], ids=["none", "empty"])
def test_front_overlay_with_nothing_measured(front):
    img = blank()
    draw_front_overlay(img, front)
    assert img.shape == (H, W, 3) and img.any()           # centreline marker drawn


@pytest.mark.parametrize("side", [None, empty_side()], ids=["none", "empty"])
@pytest.mark.parametrize("target", [None, 160.0])
def test_side_overlay_with_nothing_measured(side, target, monkeypatch):
    monkeypatch.setattr(config, "SIDE_TARGET_PX", target)
    draw_side_overlay(blank(), side, config)
    draw_side_overlay(np.zeros((480, 640, 3), np.uint8), side, config, scale=2.0)


def test_front_overlay_draws_points_and_line():
    front = synthetic_front(1.0, 1, 0.02, 0.05, CFG)
    img = front.view.copy()
    draw_front_overlay(img, front)
    assert (img != front.view).any()


def test_stale_and_no_data_marks():
    img = np.full((H, W, 3), 200, np.uint8)
    mark_stale(img, 2.5)
    draw_no_data(blank(), "side")
    assert (img != 200).any()


# ---------------------------------------------------------------- panels

def test_steer_bar_deflects_left_for_positive_steer():
    for steer, side in ((0.5, "left"), (-0.5, "right")):
        img = np.zeros((20, 200, 3), np.uint8)
        draw_steer_bar(img, (0, 0, 200, 20), steer, CFG, label=False)
        cols = np.where((img[10] == BLUE).all(axis=1))[0]
        assert len(cols) and ((cols < 100).all() if side == "left" else (cols > 100).all())


def test_preflight_all_ok_and_each_failure():
    imu = ImuSample(0.0, 0.0, True, 3, 0.0, 1.0)
    ok = bare_packet(imu=imu, fps={"front": 30.0})
    assert all(good for _, good in preflight(ok, CFG))
    bad = {
        "IMU valid": bare_packet(imu=ImuSample(0.0, 0.0, False, 3, 0.0, 1.0), fps={"front": 30.0}),
        "gyro calibrated": bare_packet(imu=ImuSample(0.0, 0.0, True, 2, 0.0, 1.0),
                                           fps={"front": 30.0}),
        "front camera 12 fps": bare_packet(imu=imu, fps={"front": 12.0}),
        "motor serial": bare_packet(imu=imu, fps={"front": 30.0},
                                       warnings=(HealthWarning.MOTOR_SERIAL_ERROR,)),
    }
    for label, packet in bad.items():
        assert dict(preflight(packet, CFG))[label] is False


def test_history_trims_and_breaks_at_gaps_and_missing_values():
    history = History(CFG)
    lane = LaneEstimate(0.01, 0.0, 1.0, None, 0.0)
    times = [0.0, 0.1, 0.2, 1.0, 1.1, 7.0]          # gap at 0.2 -> 1.0; 7.0 trims the start
    for t in times:
        history.add(bare_packet(t=t, lane=lane, drive=DriveCmd(0.5, 0.1)))
    kept = [p[0] for p in history.points]
    assert kept[-1] == 7.0 and kept[0] >= 7.0 - CFG.UI_CHART_S and 0.0 not in kept
    history = History(CFG)
    for t in (0.0, 0.1, 0.2, 1.0, 1.1):
        history.add(bare_packet(t=t, lane=lane if t != 0.1 else None, drive=DriveCmd(0.5, 0.1)))
    assert [len(s) for s in history.segments(1)] == [1, 1, 2]     # None at 0.1, gap before 1.0
    assert [len(s) for s in history.segments(2)] == [3, 2]        # steer: only the gap


def test_lane_gauge_marker_sits_on_the_robots_side_of_the_line():
    # lateral + = rear axle right of the line -> marker right of the centre tick.
    for lateral, side in ((0.05, "right"), (-0.05, "left")):
        img = np.zeros((120, 400, 3), np.uint8)
        draw_lane_gauge(img, (0, 0, 400, 96), LaneEstimate(lateral, 0.0, 0.9, None, 0.0), CFG)
        cols = np.where((img[70] == PURPLE).all(axis=1))[0]
        assert len(cols) and ((cols > 200).all() if side == "right" else (cols < 200).all())
    draw_lane_gauge(np.zeros((120, 400, 3), np.uint8), (0, 0, 400, 96), None, CFG)


def test_road_yaw_row_value_and_age():
    assert road_yaw_rows(bare_packet())[0][:2] == ("road yaw", "no data")
    assert road_yaw_rows(bare_packet(road_yaw=RoadYaw(0.0, 0.0, False)))[0][1] == "not anchored"
    row = road_yaw_rows(bare_packet(t=12.0, road_yaw=RoadYaw(87.25, 10.5, True)))[0]
    assert row[1] == "+87.2°   1.5 s ago"


# ---------------------------------------------------------------- screen

def test_layout_letterboxes_into_640_plus_160_column():
    layout = make_layout((800, 480))
    assert layout["main"] == (0, 0, 640, 480)
    column = [layout[z] for z in ("start", "stop", "light", "next")]
    assert all(x == 640 and w == 160 for x, _, w, _ in column)
    assert column[0][1] == 0 and sum(h for *_, h in column) == 480


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_draws_with_nothing(view):
    screen, _ = screen_at(1.0)
    assert screen.draw(None, view).shape == (480, 800, 3)
    screen.update(bare_packet())
    assert screen.draw(screen.packet, view).shape == (480, 800, 3)


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("elapsed", [1.0, 4.0, 10.0, 20.0], ids=["calib", "idle", "follow", "stop"])
def test_every_view_draws_demo_states(view, elapsed):
    screen, _ = screen_at(elapsed)
    for loop in range(6):
        screen.update(demo_packet(elapsed, elapsed, loop, CFG))
    assert screen.draw(screen.packet, view).shape == (480, 800, 3)


def test_last_view_kept_then_marked_stale(monkeypatch):
    marked = []
    monkeypatch.setattr("ui.screen.mark_stale", lambda img, age: marked.append(age))
    screen, clock = screen_at(1.0)
    screen.update(bare_packet(t=1.0, front=synthetic_front(1.0, 1, 0.0, 0.0, CFG)))
    screen.update(bare_packet(t=1.1))                      # in-between loop: no views
    assert screen.front is not None
    screen.draw(screen.packet, "camera")
    assert marked == []
    clock["now"] = 1.0 + CFG.UI_VIEW_STALE_S + 0.5
    screen.draw(screen.packet, "camera")
    assert marked == [pytest.approx(CFG.UI_VIEW_STALE_S + 0.5)]


def test_next_view_cycles_drive_camera_diagnostic():
    screen, _ = screen_at(0.0)
    screen.view = "drive"
    seen = []
    for _ in VIEWS:
        screen.next_view()
        seen.append(screen.view)
    assert seen == ["camera", "diagnostic", "drive"]


def tap(zone):
    x, y, w, h = make_layout(CFG.DISPLAY_SIZE)[zone]
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x + w // 2, y + h // 2), button=1)


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


def test_taps_and_keys():
    screen, _ = screen_at(0.0)
    assert screen.handle_input([tap("start")]) == UiCommand.START
    assert screen.handle_input([tap("stop")]) == UiCommand.STOP
    assert screen.handle_input([tap("next")]) == UiCommand.NEXT_VIEW
    assert screen.handle_input([tap("light")]) is None
    assert screen.handle_input([key(pygame.K_v)]) == UiCommand.NEXT_VIEW
    screen.update(bare_packet(state=State.IDLE))
    assert screen.handle_input([key(pygame.K_SPACE)]) == UiCommand.START
    screen.update(bare_packet(state=State.LANE_FOLLOW))
    assert screen.handle_input([key(pygame.K_SPACE)]) == UiCommand.STOP
    screen.update(bare_packet(state=State.SAFE_STOP))
    assert screen.handle_input([key(pygame.K_SPACE)]) == UiCommand.START
    assert not screen.quit
    assert screen.handle_input([key(pygame.K_q)]) is None and screen.quit


def test_next_view_stays_in_ui_start_stop_go_to_main():
    screen, _ = screen_at(0.0)
    cmd_q = queue.Queue()
    view = screen.view
    _route(UiCommand.NEXT_VIEW, screen, cmd_q)
    assert screen.view != view and cmd_q.empty()
    _route(UiCommand.START, screen, cmd_q)
    _route(UiCommand.STOP, screen, cmd_q)
    assert read_commands(cmd_q) == [UiCommand.START, UiCommand.STOP]


def test_read_commands_drops_anything_but_start_stop():
    cmd_q = queue.Queue()
    for c in (UiCommand.NEXT_VIEW, UiCommand.START):
        cmd_q.put(c)
    assert read_commands(cmd_q) == [UiCommand.START]
    assert read_commands(cmd_q) == []


# ---------------------------------------------------------------- link and process

def view_packet():
    return demo_packet(10.0, time.monotonic(), 0, CFG)     # LANE_FOLLOW with a 320x240 view


class FakeSys:
    class behaviour:
        state, stop_reason = State.LANE_FOLLOW, None


def test_make_ui_packet_views_only_when_asked():
    front, side = synthetic_front(1.0, 1, 0.0, 0.0, CFG), empty_side()
    args = (FakeSys, 1.0, DriveCmd(0.5, 0.1), MotorCmd(0.4, 0.6), None, None, None, None,
            {"front": 30.0}, 2, [HealthWarning.LANE_CONF_LOW], front, side)
    with_views = make_ui_packet(*args, True)
    numbers_only = make_ui_packet(*args, False)
    assert with_views.front is front and with_views.side is side
    assert numbers_only.front is None and numbers_only.side is None
    assert numbers_only.motor == MotorCmd(0.4, 0.6) and numbers_only.overruns == 2
    assert numbers_only.warnings == (HealthWarning.LANE_CONF_LOW,)


def test_send_never_blocks_with_ui_stalled():
    # A display process that never reads: 1000 sends at ~1 kHz, each must
    # return at once (the packet is dropped while the slot is full).
    ctx = multiprocessing.get_context("spawn")
    packet_q = ctx.Queue(maxsize=1)
    packet_q.cancel_join_thread()
    stalled = ctx.Process(target=time.sleep, args=(10,), daemon=True)
    stalled.start()
    packet = view_packet()
    assert packet.front is not None                       # an image in every packet
    times, start = [], time.monotonic()
    try:
        for i in range(1000):
            t0 = time.perf_counter()
            send(packet_q, packet)
            times.append(time.perf_counter() - t0)
            time.sleep(max(0.0, start + (i + 1) / 1000 - time.monotonic()))
    finally:
        stalled.kill()
    assert_send_under_1ms(times)
    assert time.monotonic() - start < 2.0


def assert_send_under_1ms(times):
    # Target from the control loop budget (4 ms): send() p99 and max < 1 ms.
    s = sorted(times)
    p99, worst = s[int(len(s) * 0.99)], s[-1]
    print(f"send(): p99 {p99 * 1e6:.0f} us, max {worst * 1e6:.0f} us")
    assert p99 < 0.001 and worst < 0.001, f"p99 {p99 * 1e3:.3f} ms, max {worst * 1e3:.3f} ms"


@pytest.fixture
def headless(monkeypatch):
    # The display process inherits this: SDL draws to nothing, no window.
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


def test_ui_process_runs_synthetic_packets_then_dies_unnoticed(headless):
    process, packet_q, cmd_q = start_ui(config)
    try:
        start = time.monotonic()
        for loop in range(90):                             # 3 s of IDLE -> LANE_FOLLOW at 30 Hz
            now = time.monotonic()
            send(packet_q, demo_packet(3.0 + loop / 30, now, loop, config))
            read_commands(cmd_q)
            time.sleep(max(0.0, start + (loop + 1) / 30 - time.monotonic()))
        assert process.is_alive(), "display process crashed on synthetic packets"
        process.kill()                                     # killed mid-run
        process.join(2.0)
        times, packet = [], view_packet()
        for loop in range(300):
            t0 = time.perf_counter()
            send(packet_q, packet)
            times.append(time.perf_counter() - t0)
            assert read_commands(cmd_q) == []
        assert_send_under_1ms(times)
    finally:
        stop_ui(process, config)
    assert not process.is_alive()


def test_stop_ui_ends_the_process(headless):
    process, packet_q, cmd_q = start_ui(config)
    send(packet_q, view_packet())
    time.sleep(1.0)
    stop_ui(process, config)
    assert not process.is_alive()


def test_display_that_cannot_open_exits_ui_process_only(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "no_such_driver")
    process, packet_q, cmd_q = start_ui(config)
    process.join(10.0)
    assert not process.is_alive() and process.exitcode == 0
    send(packet_q, view_packet())                          # still harmless
    assert read_commands(cmd_q) == []


def test_display_env_defaults_never_overwrite(monkeypatch):
    from ui.screen import _set_display_env
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-7")
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    _set_display_env()
    assert os.environ["WAYLAND_DISPLAY"] == "wayland-7"
    assert os.environ["XDG_RUNTIME_DIR"] == f"/run/user/{os.getuid()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
