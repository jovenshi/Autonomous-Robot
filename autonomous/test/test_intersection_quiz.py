"""Stage 6 (docs/quiz3_plan.md): the intersection quiz. Behaviour driven by
LocationObs, the judging pause, and a closed-loop run through main's
control_step on synthetic junction frames. No robot needed."""
import os
import queue
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
import main  # noqa: E402
from behavior.state_machine import Behaviour  # noqa: E402
from control.lane_controller import LaneController  # noqa: E402
from estimation.location import LocationTracker  # noqa: E402
from estimation.road_yaw import RoadYawTracker  # noqa: E402
from perception.pipeline import process_front_frame  # noqa: E402
from robot_types import (Frame, LaneEstimate, LatestValue, LocationObs, RunMode,  # noqa: E402
                         State, UiCommand, Zone)
from test_location import junction_scene  # noqa: E402
from test_main import FakeCam, FakeSaber  # noqa: E402

ORDER = (Zone.CONTINUE, Zone.READ_SIGN, Zone.PAUSE, Zone.ENTERING, Zone.TURN, Zone.EXITING,
         Zone.CONTINUE)
PAUSE = config.ZONE_PAUSE_S


def loc(s):
    # The LocationObs for a pass at s (None: no pass), as the tracker makes it.
    if s is None:
        return LocationObs(Zone.CONTINUE, False, False, False, None, 0.0, "none", 0.0)
    pos = sum(s >= b for b in config.ZONE_BOUNDS_M)
    return LocationObs(ORDER[pos], pos >= 1, pos >= 2, pos >= 4, -s, 1.0, "dashes", 0.0)


def lane(conf=0.9, t=0.0):
    return LaneEstimate(0.0, 0.0, conf, None, t)


def step(beh, now, s=None, commands=(), conf=0.9):
    return beh.step(None, None, lane(conf, now), None, None, None, list(commands), False, now,
                    loc(s) if s != "none" else None)


@pytest.fixture
def quiz():
    beh = Behaviour(config, threading.Event(), threading.Event())
    step(beh, 0.0, commands=[UiCommand.MODE_INTERSECTION, UiCommand.START])
    assert beh.mode == RunMode.QUIZ_INTERSECTION
    assert step(beh, 0.03) == State.LANE_FOLLOW
    return beh


# ---------------------------------------------------------------- behaviour

B = config.ZONE_BOUNDS_M


def mid(i):
    # s in the middle of zone position i (1 = READ_SIGN ... 5 = EXITING)
    return (B[i - 1] + B[i]) / 2


def test_each_zone_calls_for_its_state(quiz):
    # Steps PAUSE + 0.1 s apart, so every judging pause has ended.
    t, seen = 0.03, []
    for s in (None, B[0] - 0.2, mid(1), mid(2), mid(3), mid(4), mid(5), B[5] + 0.05, 0.35, 0.41):
        t += PAUSE + 0.1
        seen.append(step(quiz, t, s))
    assert seen == [State.LANE_FOLLOW, State.LANE_FOLLOW, State.APPROACH, State.APPROACH,
                    State.CROSS, State.CROSS, State.CROSS, State.REACQUIRE, State.REACQUIRE,
                    State.IDLE]
    assert quiz.location is not None and not quiz.paused


def test_judging_pause_at_every_zone_change(quiz):
    step(quiz, 1.0, -1.0)                                  # first zone seen: no pause
    assert not quiz.paused
    step(quiz, 2.0, -0.7)                                  # CONTINUE -> READ_SIGN
    assert quiz.paused
    step(quiz, 2.0 + PAUSE - 0.05, -0.69)
    assert quiz.paused
    step(quiz, 2.0 + PAUSE + 0.01, -0.68)
    assert not quiz.paused
    step(quiz, 5.0, mid(2))                                # READ_SIGN -> PAUSE
    assert quiz.paused and quiz.state == State.APPROACH


def test_no_pause_or_location_outside_the_quiz():
    beh = Behaviour(config)
    step(beh, 0.0, commands=[UiCommand.START])
    step(beh, 0.03)
    for i, s in enumerate((-1.0, -0.5, -0.1, 0.3)):
        assert step(beh, 1.0 + i, s) == State.LANE_FOLLOW  # DRIVE: as before
        assert beh.location is None and not beh.paused


def test_cross_timeout_counts_driving_time_only(quiz):
    step(quiz, 1.0, mid(1))
    step(quiz, 4.0, mid(3))                                # ENTERING: CROSS, paused
    assert quiz.state == State.CROSS and quiz.paused
    t = 4.0 + PAUSE                                        # pause over; now drive
    step(quiz, t, mid(3))
    assert step(quiz, t + config.CROSS_TIMEOUT_S - 0.1, mid(3)) == State.CROSS
    assert step(quiz, t + config.CROSS_TIMEOUT_S + 0.1, mid(3)) == State.REACQUIRE


def test_losing_the_pass(quiz):
    step(quiz, 1.0, mid(1))
    assert quiz.state == State.APPROACH
    assert step(quiz, 4.0, None) == State.LANE_FOLLOW     # approach: back to lane following
    step(quiz, 5.0, mid(3))
    assert quiz.state == State.CROSS
    assert step(quiz, 8.0, None) == State.REACQUIRE       # cross: hand over to the lane


def test_reacquire_hands_back_to_lane_follow_and_the_run_still_ends(quiz):
    step(quiz, 1.0, mid(4))
    step(quiz, 4.0, 0.2)
    assert quiz.state == State.REACQUIRE
    t = 7.0
    for _ in range(config.REACQUIRE_FRAMES):
        t += 0.03
        state = step(quiz, t, 0.25)
    assert state == State.LANE_FOLLOW
    assert step(quiz, t + 0.03, 0.39) == State.LANE_FOLLOW
    assert step(quiz, t + 0.06, config.QUIZ_EXIT_M) == State.IDLE


def test_stop_ends_the_pause_and_mode_needs_idle(quiz):
    step(quiz, 1.0, -1.0)
    step(quiz, 2.0, -0.7)
    assert quiz.paused
    step(quiz, 2.1, -0.7, commands=[UiCommand.MODE_SIGNS])    # ignored while driving
    assert quiz.mode == RunMode.QUIZ_INTERSECTION
    step(quiz, 2.2, -0.7, commands=[UiCommand.STOP])
    assert quiz.state == State.IDLE and not quiz.paused


# ---------------------------------------------------------------- closed loop

MPS_PER_UNIT = 1.0          # the simulated robot's real speed per unit of command


def make_quiz_system():
    flags = threading.Event(), threading.Event()
    return main.System(
        cfg=config, front_slot=LatestValue(), signs_enabled=flags[0], exposure_hold=flags[1],
        front_cam=FakeCam(), front_exposure=None, saber=FakeSaber(),
        lane_ctrl=LaneController(config), road_yaw=RoadYawTracker(config),
        behaviour=Behaviour(config, *flags), commands=queue.Queue(),
        locator=LocationTracker(config),
    )


def run_quiz(start_s, seconds, capsys):
    # The robot on its line at start_s; every loop the camera sees the
    # junction from where the robot really is, and it moves by the speed it
    # was sent. Returns (zones announced in order with the true s at each,
    # states seen, true s at the end, loops paused).
    sys_ = make_quiz_system()
    sys_.commands.put(UiCommand.MODE_INTERSECTION)
    sys_.commands.put(UiCommand.START)
    s_true, dt, zones, states, paused = start_s, 1.0 / config.CONTROL_HZ, [], [], 0
    for i in range(int(seconds * config.CONTROL_HZ)):
        t = 1.0 + i * dt
        frame = Frame(junction_scene(s_true), t, i + 1, "front")
        sys_.front_slot.set(process_front_frame(frame, False, config))
        main.control_step(sys_, t)
        s_true += sys_.last_drive.speed * MPS_PER_UNIT * dt
        loc_ = sys_.behaviour.location
        if loc_ is not None and loc_.dist_to_centre_m is not None and \
                (not zones or zones[-1][0] != loc_.zone):
            zones.append((loc_.zone, s_true))
        if not states or states[-1] != sys_.behaviour.state:
            states.append(sys_.behaviour.state)
        paused += sys_.behaviour.paused
        if sys_.behaviour.state == State.IDLE and len(states) > 2:
            break
    return zones, states, s_true, paused


@pytest.mark.parametrize("start_s", [-1.6, -1.1])
def test_quiz_run_reports_every_zone_in_place_then_stops(start_s, capsys):
    # Two start positions, as in the quiz.
    zones, states, s_end, paused = run_quiz(start_s, 40.0, capsys)
    assert [z for z, _ in zones] == list(ORDER)
    for (zone, s), bound in zip(zones[1:], config.ZONE_BOUNDS_M):
        assert s == pytest.approx(bound, abs=0.03), zone       # each change where it belongs
    # REACQUIRE may hand back to LANE_FOLLOW (lane confident) before the end
    assert states[:4] == [State.LANE_FOLLOW, State.APPROACH, State.CROSS, State.REACQUIRE]
    assert states[-1] == State.IDLE and set(states[4:-1]) <= {State.LANE_FOLLOW}
    assert s_end == pytest.approx(config.QUIZ_EXIT_M, abs=0.03)
    # six judging pauses of ZONE_PAUSE_S
    assert paused == pytest.approx(6 * PAUSE * config.CONTROL_HZ, abs=6 * 2)
    # the terminal reports each zone once, from Read the sign on
    out = capsys.readouterr().out
    reported = [line.split("  (")[0] for line in out.splitlines() if line.startswith("Zone: ")]
    assert reported == [f"Zone: {config.ZONE_PHRASES[z.name]}" for z in ORDER[1:]]


# ---------------------------------------------------------------- display

def test_display_shows_the_zone(tmp_path):
    import cv2
    import numpy as np
    from test_ui import bare_packet, screen_at
    from ui.panels import quiz_rows, zone_label
    from ui.screen import VIEWS
    in_pass = bare_packet(state=State.CROSS, mode=RunMode.QUIZ_INTERSECTION, location=loc(mid(3)))
    no_pass = bare_packet(mode=RunMode.QUIZ_INTERSECTION, location=loc(None))
    assert zone_label(in_pass, config) == "Zone: Entering intersection"
    assert zone_label(no_pass, config) is None
    assert zone_label(bare_packet(), config) is None                 # DRIVE: no location
    rows = dict((label, value) for label, value, _ in quiz_rows(in_pass, config))
    assert rows["zone"] == "Entering intersection"
    assert rows["s (from centre)"] == f"{mid(3):+.2f} m"
    assert dict((l, v) for l, v, _ in quiz_rows(no_pass, config))["zone"] == "no junction yet"
    # every view draws; the banner differs with the zone shown
    frame = Frame(junction_scene(-1.0), 0.0, 1, "front")
    front = process_front_frame(frame, False, config)
    screen, _ = screen_at(0.0)
    screen.update(bare_packet(state=State.CROSS, mode=RunMode.QUIZ_INTERSECTION,
                              location=loc(mid(3)), front=front))
    for view in VIEWS:
        img = screen.draw(screen.packet, view)
        assert img.shape == (config.DISPLAY_SIZE[1], config.DISPLAY_SIZE[0], 3)
        cv2.imwrite(str(tmp_path / f"{view}.png"), img)
    banner_with = screen.draw(screen.packet, "drive")[:30]
    banner_without = screen.draw(no_pass, "drive")[:30]
    assert not np.array_equal(banner_with, banner_without)
