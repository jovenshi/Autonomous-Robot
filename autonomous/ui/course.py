"""Imports the course's reference modules from the repo root (above autonomous/).

Used by the operator process for usb_sound_controller.USB_SoundController and
manual.ps5_controller.PS5_Controller, which are reused as they are.
"""
import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def course_module(name):
    # The repo root is appended, never inserted first: it holds ui.py and
    # display.py, which must not shadow autonomous/ packages.
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)
    return importlib.import_module(name)
