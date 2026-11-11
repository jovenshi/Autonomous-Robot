"""Turns per-frame sign readings into one steady decision.

A sign is decided once it is read SIGN_VOTES_NEEDED times in a row. The
decision is then held: it is cleared only after SIGN_CLEAR_FRAMES readings
of NONE in a row, or replaced when a different sign wins the same vote. So
one missed or blurred frame never drops a sign from the display, and a new
sign shown after the first is picked up without any reset.

The box reported is the median of the last SIGN_BOX_FRAMES boxes of the
decided sign, so it stays steady on the screen; it is kept through missed
frames for as long as the decision is held.
"""
from collections import deque

import numpy as np

from robot_types import SignType, SignVote


class SignVoter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        # Forget everything: no decision, no run in progress.
        self._vote = None              # (SignType, since_t) once decided
        self._run_sign = None          # the sign read on the latest frames
        self._run = 0                  # how many frames in a row
        self._none_run = 0             # NONE readings in a row
        self._boxes = deque(maxlen=self.cfg.SIGN_BOX_FRAMES)   # of the run's sign
        self._held_box = None          # last median box of the decided sign
        self._last_seq = None

    def add(self, obs):
        # obs: SignObs. A result already counted (same seq) is ignored.
        if obs is None or obs.seq == self._last_seq:
            return
        self._last_seq = obs.seq
        if obs.sign == SignType.NONE:
            self._add_none()
        else:
            self._add_sign(obs)

    def decided(self):
        # SignVote for the held decision, or None.
        if self._vote is None:
            return None
        sign, since_t = self._vote
        return SignVote(sign=sign, bbox=self._held_box, since_t=since_t)

    def _add_none(self):
        self._run_sign, self._run = None, 0
        self._boxes.clear()
        self._none_run += 1
        if self._none_run >= self.cfg.SIGN_CLEAR_FRAMES:
            self._vote, self._held_box = None, None

    def _add_sign(self, obs):
        self._none_run = 0
        if obs.sign != self._run_sign:
            self._run_sign, self._run = obs.sign, 0
            self._boxes.clear()
        self._run += 1
        if obs.bbox is not None:
            self._boxes.append(obs.bbox)
        won = self._run >= self.cfg.SIGN_VOTES_NEEDED
        if won and (self._vote is None or self._vote[0] != obs.sign):
            self._vote = (obs.sign, obs.t)
        if self._vote is not None and self._vote[0] == obs.sign and self._boxes:
            # Held through missed frames: the box stays where the sign was.
            self._held_box = tuple(int(v) for v in
                                   np.median(np.array(self._boxes), axis=0).round())
