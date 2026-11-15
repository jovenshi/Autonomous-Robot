"""Speaks a change of zone or of the voted sign, in the operator process.

prepare() renders every phrase in SIGN_PHRASES and ZONE_PHRASES to a WAV in
AUDIO_DIR once at start-up with text2wave (the voice USB_SoundController's own
text-to-speech uses). Each file is named after its text, so a phrase changed in
config.py is rendered again and an unchanged one is reused from the last run.
Playback goes through the course's USB_SoundController (not modified).

A zone is spoken when it changes from one zone to the next within a pass;
the first zone of a pass (Continue travel, before Read the sign) is not
news. A voted sign is spoken whenever it appears or changes.

One phrase at a time: USB_SoundController starts each sound at once on a free
channel, so two phrases asked for together would talk over each other. The
announcer queues them, keeps at most one waiting per kind (a newer zone
replaces a zone still waiting) and starts the next once the last one's length
plus AUDIO_GAP_S has passed.
"""
import os
import subprocess
import wave
import zlib

from ui.course import course_module

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # autonomous/
_RENDER_ERRORS = (OSError, subprocess.SubprocessError, wave.Error, EOFError)


class Announcer:
    def __init__(self, cfg, make_player=None, render=None):
        self.cfg = cfg
        self.enabled = cfg.AUDIO_ENABLED
        self._make_player = make_player or _course_player
        self._render = render or render_phrase
        self.player = None
        self.clips = {}                  # (kind, member name) -> (path, seconds)
        self.pending = []                # (kind, member name), oldest first
        self.busy_until = 0.0
        self.last = {"zone": None, "sign": None}

    def prepare(self):
        # Renders missing WAVs, then opens the sound player. Any failure is
        # logged and leaves the announcer (or that phrase) silent.
        if not self.enabled:
            return
        folder = os.path.join(_HERE, self.cfg.AUDIO_DIR)
        os.makedirs(folder, exist_ok=True)
        for kind, phrases in (("sign", self.cfg.SIGN_PHRASES), ("zone", self.cfg.ZONE_PHRASES)):
            for name, text in phrases.items():
                clip = self._clip(folder, kind, name, text)
                if clip is not None:
                    self.clips[(kind, name)] = clip
        try:
            self.player = self._make_player(self.cfg)
        except Exception as e:
            print(f"Announcer: no sound ({e}); display carries on silent")
            self.enabled = False

    def update(self, packet, now):
        # Called every display cycle with the newest UiPacket (or None).
        if not self.enabled or self.player is None or packet is None:
            return
        location, vote = packet.location, packet.sign_vote
        in_pass = location is not None and location.dist_to_centre_m is not None
        self._note("zone", location.zone if in_pass else None, speak_first=False)
        self._note("sign", vote.sign if vote is not None else None, speak_first=True)
        self._play_next(now)

    def close(self):
        if self.player is not None:
            self.player.stop_sound()
            self.player.close()
            self.player = None

    def _note(self, kind, value, speak_first):
        # A new zone or voted sign queues its phrase; None (no pass, vote
        # cleared) is silent but lets the same value be spoken again later.
        # speak_first False: a value right after None only sets the baseline.
        previous = self.last[kind]
        if value == previous:
            return
        self.last[kind] = value
        if previous is None and not speak_first:
            return
        key = (kind, value.name) if value is not None else None
        if key in self.clips:
            self.pending = [p for p in self.pending if p[0] != kind] + [key]

    def _play_next(self, now):
        if not self.pending or now < self.busy_until:
            return
        path, seconds = self.clips[self.pending.pop(0)]
        self.player.play_audio(path)
        self.busy_until = now + seconds + self.cfg.AUDIO_GAP_S

    def _clip(self, folder, kind, name, text):
        # (path, seconds) for one phrase, rendering it if missing; None on failure.
        path = os.path.join(folder, f"{kind}_{name}_{zlib.crc32(text.encode()):08x}.wav")
        try:
            if not os.path.exists(path):
                self._render(text, path, self.cfg)
            return path, wav_seconds(path)
        except _RENDER_ERRORS as e:
            print(f"Announcer: '{text}' not rendered ({e}); it will not be spoken")
            return None


def render_phrase(text, path, cfg):
    # text2wave -> path. Written to a temporary name first, so an interrupted
    # render never leaves a broken WAV that a later start-up would reuse.
    tmp = path + ".tmp"
    subprocess.run(["text2wave", "-scale", str(cfg.AUDIO_TTS_SCALE), "-o", tmp],
                   input=f"{text}\n", text=True, check=True, timeout=cfg.AUDIO_RENDER_TIMEOUT_S,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp, path)


def wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def _course_player(cfg):
    return course_module("usb_sound_controller").USB_SoundController(volume=cfg.AUDIO_VOLUME)
