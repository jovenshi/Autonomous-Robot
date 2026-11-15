"""Stage 4 (docs/quiz3_plan.md): PS5 controller and announcer in the operator
process. No controller, speaker or robot needed: the controller and sound
player are fakes, phrases are rendered as tones."""
import math
import os
import queue
import shutil
import struct
import sys
import time
import types
import wave

import pygame
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from robot_types import LocationObs, SignType, SignVote, State, UiCommand, Zone  # noqa: E402
from test_ui import bare_packet, screen_at, tap  # noqa: E402
from ui import announcer as announcer_mod  # noqa: E402
from ui.announcer import Announcer, render_phrase, wav_seconds  # noqa: E402
from ui.gamepad import Gamepad, start_or_stop  # noqa: E402
from ui.link import read_commands  # noqa: E402
from ui.screen import _read_input  # noqa: E402

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


# ---------------------------------------------------------------- gamepad

class FakePS5:
    """PS5_Controller's interface as Gamepad uses it. `held` is the set of
    control_request keys check_controls sets, every call (a held button)."""

    def __init__(self, keys=("reqCross", "reqCircle", "reqTriangle", "reqSquare", "reqOptions",
                             "reqL1")):
        self.joystick = None
        self.control_request = {k: False for k in keys}
        self.held = set()
        self.fail = False

    def check_controls(self):
        if self.fail:
            raise pygame.error("device gone")
        for k in self.held:
            self.control_request[k] = True

    def reset_controller_state(self):
        for k in self.control_request:
            self.control_request[k] = False


class FakeJoystick:
    def __init__(self, index):
        self.index = index

    def init(self):
        pass

    def get_name(self):
        return "fake DualSense"


def event(kind, **kw):
    return pygame.event.Event(kind, **kw)


@pytest.fixture
def pad(monkeypatch):
    # A Gamepad on a fake controller, with one joystick attached.
    monkeypatch.setattr(pygame.joystick, "Joystick", FakeJoystick)
    ps5 = FakePS5()
    gamepad = Gamepad(config, controller=ps5)
    gamepad._attach(0)
    return gamepad, ps5


def press(gamepad, ps5, key, state=State.IDLE):
    # One tap: pressed this cycle, released before the next.
    ps5.held = {key}
    out = gamepad.poll([], state)
    ps5.held = set()
    gamepad.poll([event(pygame.JOYBUTTONUP, button=0, joy=0, instance_id=0)], state)
    return out


def test_each_button_sends_its_command(pad):
    gamepad, ps5 = pad
    assert press(gamepad, ps5, "reqSquare") == [UiCommand.MODE_SIGNS]
    assert press(gamepad, ps5, "reqTriangle") == [UiCommand.MODE_INTERSECTION]
    assert press(gamepad, ps5, "reqCircle") == [UiCommand.MODE_DRIVE]
    assert press(gamepad, ps5, "reqOptions") == [UiCommand.NEXT_VIEW]
    assert press(gamepad, ps5, "reqL1") == []                    # not in PS5_BUTTONS


def test_cross_starts_when_stopped_and_stops_when_running(pad):
    gamepad, ps5 = pad
    assert press(gamepad, ps5, "reqCross", State.IDLE) == [UiCommand.START]
    assert press(gamepad, ps5, "reqCross", State.SAFE_STOP) == [UiCommand.START]
    assert press(gamepad, ps5, "reqCross", State.LANE_FOLLOW) == [UiCommand.STOP]
    assert start_or_stop(State.APPROACH) == UiCommand.STOP


def test_held_button_counts_once_until_released(pad):
    # PS5_Controller repeats a held button every 0.5 s: holding Cross must not
    # send START and then STOP.
    gamepad, ps5 = pad
    ps5.held = {"reqCross"}
    assert gamepad.poll([], State.IDLE) == [UiCommand.START]
    for _ in range(20):
        assert gamepad.poll([], State.LANE_FOLLOW) == []
    ps5.held = set()
    gamepad.poll([event(pygame.JOYBUTTONUP, button=0, joy=0, instance_id=0)], State.LANE_FOLLOW)
    ps5.held = {"reqCross"}
    assert gamepad.poll([], State.LANE_FOLLOW) == [UiCommand.STOP]


def test_no_controller_returns_at_once():
    ps5 = FakePS5()
    ps5.held = {"reqCross"}                  # would press, but nothing is attached
    gamepad = Gamepad(config, controller=ps5)
    gamepad._detach("test")
    t0 = time.perf_counter()
    for _ in range(100):
        assert gamepad.poll([], State.IDLE) == []
    assert time.perf_counter() - t0 < 0.05


def test_lost_controller_is_dropped_then_reattached(pad, monkeypatch):
    gamepad, ps5 = pad
    ps5.fail = True
    assert gamepad.poll([], State.IDLE) == [] and ps5.joystick is None
    ps5.fail = False
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: 0)
    gamepad.poll([event(pygame.JOYDEVICEREMOVED, instance_id=0)], State.IDLE)
    assert ps5.joystick is None
    ps5.held = {"reqSquare"}
    assert gamepad.poll([event(pygame.JOYDEVICEADDED, device_index=0)], State.IDLE) == \
        [UiCommand.MODE_SIGNS]
    assert ps5.joystick.index == 0


def test_course_controller_loads_without_a_controller():
    # The real PS5_Controller from manual/ (initialize_controller is not called).
    gamepad = Gamepad(config)
    assert gamepad.ps5 is not None and type(gamepad.ps5).__name__ == "PS5_Controller"
    if gamepad.ps5.joystick is None:
        assert gamepad.poll([], State.IDLE) == []


def test_taps_and_ps5_buttons_both_reach_main(pad):
    gamepad, ps5 = pad
    screen, _ = screen_at(0.0)
    screen.update(bare_packet(state=State.IDLE))
    cmd_q = queue.Queue()
    ps5.held = {"reqSquare"}
    _read_input([tap("start")], screen, gamepad, cmd_q)
    assert read_commands(cmd_q) == [UiCommand.START, UiCommand.MODE_SIGNS]
    view = screen.view
    ps5.held = {"reqOptions"}
    gamepad.latched.clear()
    _read_input([], screen, gamepad, cmd_q)
    assert screen.view != view and read_commands(cmd_q) == []    # NEXT_VIEW stays in the UI


# ---------------------------------------------------------------- announcer

def tone(path, seconds=0.5, rate=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 8)))
                               for i in range(int(seconds * rate))))


class FakeRender:
    def __init__(self, fail=()):
        self.texts, self.fail = [], set(fail)

    def __call__(self, text, path, cfg):
        if text in self.fail:
            raise FileNotFoundError("text2wave")
        self.texts.append(text)
        tone(path)


class FakePlayer:
    def __init__(self):
        self.played, self.closed = [], False

    def play_audio(self, path):
        self.played.append(os.path.basename(path).rsplit("_", 1)[0])

    def stop_sound(self):
        pass

    def close(self):
        self.closed = True


def cfg_in(folder, **kw):
    return types.SimpleNamespace(**{**{k: getattr(config, k) for k in dir(config) if k.isupper()},
                                    "AUDIO_DIR": str(folder), **kw})


def ready(tmp_path, render=None, **kw):
    player = FakePlayer()
    ann = Announcer(cfg_in(tmp_path, **kw), make_player=lambda cfg: player,
                    render=render or FakeRender())
    ann.prepare()
    return ann, player


def loc(zone, dist=0.5):
    return LocationObs(zone, False, False, False, dist, 1.0, "dashes", 0.0)


def vote(sign):
    return SignVote(sign, (10, 10, 40, 30), 0.0)


def test_prepare_renders_each_phrase_once_and_reuses_it(tmp_path):
    render = FakeRender()
    ready(tmp_path, render)
    assert sorted(render.texts) == sorted([*config.SIGN_PHRASES.values(),
                                           *config.ZONE_PHRASES.values()])
    again = FakeRender()
    ready(tmp_path, again)
    assert again.texts == []
    changed = FakeRender()
    ready(tmp_path, changed, SIGN_PHRASES={**config.SIGN_PHRASES, "LEFT": "Turn left"})
    assert changed.texts == ["Turn left"]


def test_new_voted_sign_is_spoken_once(tmp_path):
    ann, player = ready(tmp_path)
    for t in range(5):
        ann.update(bare_packet(sign_vote=vote(SignType.LEFT)), 10.0 + t)
    assert player.played == ["sign_LEFT"]
    ann.update(bare_packet(sign_vote=None), 20.0)                  # vote cleared
    ann.update(bare_packet(sign_vote=vote(SignType.LEFT)), 21.0)   # shown again
    ann.update(bare_packet(sign_vote=vote(SignType.PAUSE)), 22.0)  # another sign wins
    assert player.played == ["sign_LEFT", "sign_LEFT", "sign_PAUSE"]


def test_zone_changes_are_spoken_in_order(tmp_path):
    # The first zone of a pass (Continue travel, before Read the sign) is the
    # baseline, not news; no pass at all is silent.
    ann, player = ready(tmp_path)
    ann.update(bare_packet(location=loc(Zone.CONTINUE, None)), 8.0)       # no pass yet
    zones = (Zone.CONTINUE, Zone.READ_SIGN, Zone.READ_SIGN, Zone.PAUSE, Zone.ENTERING)
    for i, zone in enumerate(zones):
        ann.update(bare_packet(location=loc(zone)), 10.0 + 2 * i)
    ann.update(bare_packet(location=None), 20.0)
    assert player.played == ["zone_READ_SIGN", "zone_PAUSE", "zone_ENTERING"]


def test_a_pass_starting_mid_zone_is_not_announced_until_it_changes(tmp_path):
    ann, player = ready(tmp_path)
    ann.update(bare_packet(location=loc(Zone.READ_SIGN)), 10.0)
    ann.update(bare_packet(location=loc(Zone.PAUSE)), 12.0)
    assert player.played == ["zone_PAUSE"]


def test_phrases_never_overlap(tmp_path):
    # Zone and sign in one packet: the sign waits for the zone's length + gap.
    ann, player = ready(tmp_path)
    ann.update(bare_packet(location=loc(Zone.CONTINUE)), 9.0)            # pass baseline
    packet = bare_packet(location=loc(Zone.READ_SIGN), sign_vote=vote(SignType.RIGHT))
    ann.update(packet, 10.0)
    assert player.played == ["zone_READ_SIGN"]
    ann.update(packet, 10.0 + 0.5 + config.AUDIO_GAP_S - 0.01)
    assert player.played == ["zone_READ_SIGN"]
    ann.update(packet, 10.0 + 0.5 + config.AUDIO_GAP_S)
    assert player.played == ["zone_READ_SIGN", "sign_RIGHT"]


def test_newer_value_replaces_one_still_waiting(tmp_path):
    ann, player = ready(tmp_path)
    ann.update(bare_packet(location=loc(Zone.CONTINUE)), 9.0)            # pass baseline
    ann.update(bare_packet(location=loc(Zone.READ_SIGN)), 10.0)
    ann.update(bare_packet(location=loc(Zone.READ_SIGN), sign_vote=vote(SignType.LEFT)), 10.1)
    ann.update(bare_packet(location=loc(Zone.READ_SIGN), sign_vote=vote(SignType.RIGHT)), 10.2)
    ann.update(bare_packet(location=loc(Zone.READ_SIGN), sign_vote=vote(SignType.RIGHT)), 11.0)
    assert player.played == ["zone_READ_SIGN", "sign_RIGHT"]


def test_failed_render_or_player_leaves_it_silent(tmp_path):
    ann, player = ready(tmp_path, FakeRender(fail={config.SIGN_PHRASES["LEFT"]}))
    ann.update(bare_packet(sign_vote=vote(SignType.LEFT)), 10.0)
    ann.update(bare_packet(sign_vote=vote(SignType.STRAIGHT)), 11.0)
    assert player.played == ["sign_STRAIGHT"]

    def no_sound(cfg):
        raise RuntimeError("no audio device")
    silent = Announcer(cfg_in(tmp_path), make_player=no_sound, render=FakeRender())
    silent.prepare()
    silent.update(bare_packet(sign_vote=vote(SignType.LEFT)), 10.0)    # no error
    assert not silent.enabled
    silent.close()


def test_audio_disabled_renders_and_plays_nothing(tmp_path):
    render = FakeRender()
    ann, player = ready(tmp_path, render, AUDIO_ENABLED=False)
    ann.update(bare_packet(sign_vote=vote(SignType.LEFT)), 10.0)
    assert render.texts == [] and player.played == [] and not os.listdir(tmp_path)


def test_close_closes_the_player(tmp_path):
    ann, player = ready(tmp_path)
    ann.close()
    assert player.closed and ann.player is None
    ann.update(bare_packet(sign_vote=vote(SignType.LEFT)), 10.0)       # harmless after close


@pytest.mark.skipif(shutil.which("text2wave") is None, reason="text2wave not installed")
def test_text2wave_renders_a_wav(tmp_path):
    path = str(tmp_path / "left.wav")
    render_phrase("Left", path, config)
    assert 0.1 < wav_seconds(path) < 5.0 and not os.path.exists(path + ".tmp")


def test_course_sound_controller_opens_on_a_silent_driver(tmp_path):
    # The real USB_SoundController on SDL's dummy audio driver.
    pytest.importorskip("pydub")
    player = announcer_mod._course_player(config)
    try:
        path = str(tmp_path / "tone.wav")
        tone(path)
        player.play_audio(path)
        time.sleep(0.2)
    finally:
        player.stop_sound()
        player.close()
