"""Stage 3 (docs/quiz3_plan.md): the sign voter, run modes, sign pacing, and
the sign quiz from camera frame to display. No robot needed."""
import os
import queue
import sys
import threading

import numpy as np
import pygame
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
import main  # noqa: E402
from behavior.sign_voter import SignVoter  # noqa: E402
from behavior.state_machine import Behaviour  # noqa: E402
from perception import signs  # noqa: E402
from perception.pipeline import SignPacer, process_front_frame  # noqa: E402
from robot_types import (RunMode, SignObs, SignType, SignVote, State,  # noqa: E402
                         UiCommand)
from test_main import make_system  # noqa: E402
from test_signs import CARD, model, model_path, near_card, sign_frame  # noqa: E402,F401
from test_ui import bare_packet, key, screen_at  # noqa: E402
from ui.link import read_commands  # noqa: E402
from ui.overlay import GREEN  # noqa: E402
from ui.screen import BANNER_H, VIEWS  # noqa: E402

START, STOP = [UiCommand.START], [UiCommand.STOP]


def obs(sign, seq, box=(100, 200, 140, 104), t=None):
    return SignObs(sign, 2.0 if sign != SignType.NONE else 0.0,
                   box if sign != SignType.NONE else None, seq / 10 if t is None else t, seq)


def feed(voter, signs_seen, first_seq=1, **kw):
    for i, sign in enumerate(signs_seen):
        voter.add(obs(sign, first_seq + i, **kw))
    return voter.decided()


# ---------------------------------------------------------------- SignVoter

def test_decides_after_votes_needed_in_a_row():
    voter = SignVoter(config)
    assert feed(voter, [SignType.LEFT] * (config.SIGN_VOTES_NEEDED - 1)) is None
    vote = feed(voter, [SignType.LEFT], first_seq=10)
    assert vote.sign == SignType.LEFT and vote.since_t == 1.0 and vote.bbox == (100, 200, 140, 104)


def test_same_frame_is_counted_once():
    voter = SignVoter(config)
    for _ in range(10):
        voter.add(obs(SignType.LEFT, seq=5))
    assert voter.decided() is None


def test_a_none_breaks_the_run_before_a_decision():
    voter = SignVoter(config)
    assert feed(voter, [SignType.LEFT, SignType.LEFT, SignType.NONE, SignType.LEFT]) is None


def test_holds_through_missed_frames_then_clears():
    voter = SignVoter(config)
    feed(voter, [SignType.PAUSE] * config.SIGN_VOTES_NEEDED)
    held = feed(voter, [SignType.NONE] * (config.SIGN_CLEAR_FRAMES - 1), first_seq=20)
    assert held.sign == SignType.PAUSE and held.bbox == (100, 200, 140, 104)   # box kept too
    assert feed(voter, [SignType.NONE], first_seq=40) is None


def test_another_sign_replaces_the_first_without_a_reset():
    voter = SignVoter(config)
    feed(voter, [SignType.LEFT] * 5)
    still = feed(voter, [SignType.RIGHT] * (config.SIGN_VOTES_NEEDED - 1), first_seq=20, t=7.0)
    assert still.sign == SignType.LEFT                       # one odd frame changes nothing
    vote = feed(voter, [SignType.RIGHT], first_seq=30, t=8.0)
    assert vote.sign == SignType.RIGHT and vote.since_t == 8.0


def test_box_is_the_median_of_recent_boxes():
    voter = SignVoter(config)
    for i, x in enumerate([100, 102, 400, 101, 99]):         # one wild box
        voter.add(obs(SignType.STRAIGHT, i + 1, box=(x, 200, 140, 104)))
    assert voter.decided().bbox == (101, 200, 140, 104)


def test_reset_forgets_the_decision():
    voter = SignVoter(config)
    feed(voter, [SignType.LEFT] * 5)
    voter.reset()
    assert voter.decided() is None


# ---------------------------------------------------------------- pacing and pipeline

def test_sign_pacer_runs_at_sign_hz():
    pacer = SignPacer()
    due = [pacer.due(i / 30, config) for i in range(30)]     # one second of frames
    assert due[0] and sum(due) == config.SIGN_HZ


def test_front_frame_reads_signs_only_when_enabled(model):
    frame = sign_frame("LEFT")
    assert process_front_frame(frame, False, config, sign_model=model).sign is None
    result = process_front_frame(frame, True, config, sign_model=model)
    assert result.sign.sign == SignType.LEFT and result.lane is not None
    pacer = SignPacer()
    first = process_front_frame(frame, True, config, sign_model=model, pacer=pacer).sign
    later = sign_frame("RIGHT", seq=frame.seq + 1)           # 1/30 s on: not a sign frame
    carried = process_front_frame(later, True, config, sign_model=model, pacer=pacer).sign
    assert carried is first and carried.seq == frame.seq     # the reading is passed on, not redone
    assert process_front_frame(later, False, config, sign_model=model, pacer=pacer).sign is None
    assert pacer.latest is None                              # switched off: nothing lingers


def test_missing_model_is_a_warning_not_a_crash(monkeypatch, capsys):
    monkeypatch.setattr(config, "SIGN_MODEL_PATH", "models/not_there.npz")
    assert main._open_sign_model(config) is None
    assert "Signs will be boxed but not read" in capsys.readouterr().out


# ---------------------------------------------------------------- modes in Behaviour

@pytest.fixture
def beh():
    flags = threading.Event(), threading.Event()
    return Behaviour(config, signs_enabled=flags[0], exposure_hold=flags[1]), flags[0]


def step(b, now, commands=()):
    return b.step(None, None, None, None, None, None, list(commands), False, now)


def test_starts_in_start_mode(beh):
    assert beh[0].mode == RunMode[config.START_MODE]


def test_sign_quiz_stays_idle_with_signs_on(beh):
    b, signs_on = beh
    step(b, 1.0, [UiCommand.MODE_SIGNS])
    assert b.mode == RunMode.QUIZ_SIGNS and signs_on.is_set()
    for t in (1.1, 1.2, 5.0):
        assert step(b, t, START) == State.IDLE               # START never moves the robot
    assert signs_on.is_set()
    step(b, 6.0, [UiCommand.MODE_DRIVE])
    assert b.mode == RunMode.DRIVE and not signs_on.is_set()


def test_mode_changes_only_in_idle(beh):
    b, _ = beh
    step(b, 1.0, START)
    assert step(b, 1.1) == State.LANE_FOLLOW
    step(b, 1.2, [UiCommand.MODE_SIGNS])
    assert b.mode == RunMode.DRIVE and b.state == State.LANE_FOLLOW
    step(b, 1.3, STOP)
    step(b, 1.4, [UiCommand.MODE_INTERSECTION])
    assert b.mode == RunMode.QUIZ_INTERSECTION


def test_start_before_a_mode_change_does_not_carry_over(beh):
    b, _ = beh
    b._start_requested = True
    step(b, 1.0, [UiCommand.MODE_SIGNS])
    step(b, 1.1, [UiCommand.MODE_DRIVE])
    assert step(b, 1.2) == State.IDLE


# ---------------------------------------------------------------- the quiz through control_step

@pytest.fixture
def quiz(model):
    system = make_system()
    system.voter = SignVoter(config)
    system.ui_packet_q = queue.Queue()
    system.commands.put(UiCommand.MODE_SIGNS)
    main.control_step(system, 0.0)
    return system, model


def show(system, model, kind, t, frames=4):
    # frames sign frames at 10 Hz, one control loop each -> the last packet.
    for i in range(frames):
        now = t + i / 10
        frame = sign_frame(kind, seq=int(now * 30) + 1)
        frame = type(frame)(frame.image, now, frame.seq, "front")
        system.front_slot.set(process_front_frame(frame, system.signs_enabled.is_set(), config,
                                                  sign_model=model))
        main.control_step(system, now)
    packet = None
    while not system.ui_packet_q.empty():
        packet = system.ui_packet_q.get()
    return packet


def test_sign_quiz_reads_four_signs_in_any_order_without_moving(quiz):
    system, model = quiz
    t = 1.0
    for kind in ("RIGHT", "PAUSE", "LEFT", "STRAIGHT"):
        packet = show(system, model, kind, t)
        assert packet.mode == RunMode.QUIZ_SIGNS and packet.state == State.IDLE
        assert packet.sign_vote.sign == SignType[kind]       # no reset between signs
        assert near_card(packet.sign_vote.bbox, 120, 200)
        assert packet.sign_vote.since_t - t < 5.0            # well inside the 5 s allowed
        gone = show(system, model, None, t + 1.0, frames=config.SIGN_CLEAR_FRAMES)
        assert gone.sign_vote is None
        t += 3.0
    assert all(cmd.left == 0.0 and cmd.right == 0.0 for cmd in system.saber.sent)


def test_leaving_the_sign_quiz_clears_the_vote(quiz):
    system, model = quiz
    assert show(system, model, "LEFT", 1.0).sign_vote is not None
    system.commands.put(UiCommand.MODE_DRIVE)
    main.control_step(system, 2.0)
    assert system.voter.decided() is None and not system.signs_enabled.is_set()


def test_mode_and_sign_changes_are_printed(model, capsys):
    system = make_system()
    system.voter, system.ui_packet_q = SignVoter(config), queue.Queue()
    system.commands.put(UiCommand.MODE_SIGNS)
    main.control_step(system, 0.0)
    show(system, model, "PAUSE", 1.0)
    out = capsys.readouterr().out
    assert "Mode: Sign Reading" in out and "Sign: Pause" in out


# ---------------------------------------------------------------- display

def test_mode_keys_and_link_forwarding():
    screen, _ = screen_at(0.0)
    sent = [screen.handle_input([key(k)]) for k in (pygame.K_1, pygame.K_2, pygame.K_3)]
    assert sent == [UiCommand.MODE_DRIVE, UiCommand.MODE_INTERSECTION, UiCommand.MODE_SIGNS]
    q = queue.Queue()
    for command in sent + [UiCommand.NEXT_VIEW]:
        q.put(command)
    assert read_commands(q) == sent


@pytest.mark.parametrize("view", VIEWS)
def test_banner_names_the_mode_on_every_view(view):
    screen, _ = screen_at(1.0)
    images = {mode: screen.draw(bare_packet(mode=mode), view)[:BANNER_H, :640]
              for mode in RunMode}
    assert not np.array_equal(images[RunMode.DRIVE], images[RunMode.QUIZ_SIGNS])
    assert not np.array_equal(images[RunMode.QUIZ_SIGNS], images[RunMode.QUIZ_INTERSECTION])
    with_sign = screen.draw(bare_packet(mode=RunMode.QUIZ_SIGNS,
                                        sign_vote=SignVote(SignType.LEFT, None, 0.0)), view)
    assert not np.array_equal(with_sign[:BANNER_H, :640], images[RunMode.QUIZ_SIGNS])


def test_camera_view_draws_the_voted_box_on_the_sign(model):
    frame = sign_frame("LEFT", x=120, y=200)
    front = process_front_frame(frame, True, config, sign_model=model)
    screen, clock = screen_at(frame.t)
    vote = SignVote(SignType.LEFT, front.sign.bbox, frame.t)
    packet = bare_packet(t=frame.t, mode=RunMode.QUIZ_SIGNS, front=front, sign_vote=vote)
    screen.update(packet)
    img = screen.draw(packet, "camera")
    cam_h = 480 - BANNER_H
    scale = cam_h / 480                       # full frame -> the letterboxed camera image
    x0 = (640 - cam_h * 4 // 3) // 2
    bx, by, bw, bh = vote.bbox
    row = BANNER_H + int((by + bh / 2) * scale)
    left_edge = img[row, x0 + int(bx * scale) - 3:x0 + int(bx * scale) + 4]
    assert any(tuple(px) == GREEN for px in left_edge)
    plain = screen.draw(bare_packet(t=frame.t, mode=RunMode.QUIZ_SIGNS, front=front), "camera")
    assert not np.array_equal(img, plain)


def test_stale_sign_reading_is_dropped_from_the_view(model):
    frame = sign_frame("LEFT")
    front = process_front_frame(frame, True, config, sign_model=model)
    screen, clock = screen_at(frame.t)
    screen.update(bare_packet(t=frame.t, front=front))
    assert screen._fresh_sign() is front.sign
    clock["now"] = frame.t + config.UI_VIEW_STALE_S + 0.1
    assert screen._fresh_sign() is None
