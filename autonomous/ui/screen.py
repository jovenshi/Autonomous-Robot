"""Display process: compose the screen, show it full screen, read taps and keys.

run_ui is the process entry point (started by ui.link.start_ui). Each cycle
it waits on the packet queue with get(timeout) until the next redraw (every
packet goes into the strip chart and keeps the last camera views), composes
the chosen view as one DISPLAY_SIZE BGR image, and shows it with pygame.

Layout (800 x 480): a 4:3 main area (640 x 480; camera views are
letterboxed into it, never stretched) and a 160 px control column on the
right in every view: START, STOP (not an e-stop), the warning light and VIEW.
Views cycle DRIVE (glanceable, default) -> CAMERA -> DIAGNOSTIC (every number).

pygame is imported only inside this process, after the display environment
defaults are set, so the main process never loads it.
"""
import os
import queue
import signal
import time

import cv2
import numpy as np

from robot_types import State, UiCommand
from ui.overlay import (AMBER, BG, GREEN, LINE, MUTED, PANEL, RED, TEXT, draw_front_overlay,
                        draw_no_data, draw_side_overlay, mark_stale)
from ui.panels import (History, draw_chart, draw_header, draw_lane_gauge, draw_preflight,
                       draw_status, draw_steer_bar, preflight)
from ui.text import draw_text

VIEWS = ("drive", "camera", "diagnostic")
COLUMN_ZONES = ("start", "stop", "light", "next")
COLUMN_SPLITS = (0.0, 0.3, 0.66, 0.85, 1.0)        # fractions of screen height
TAP_COMMANDS = {"start": UiCommand.START, "stop": UiCommand.STOP, "next": UiCommand.NEXT_VIEW}
START_STATES = (State.IDLE, State.SAFE_STOP)        # space bar means START here
INSET_FRAC = 0.33                                   # side-camera inset, fraction of main width
GAP = 16                                            # px; margin inside the main area


def make_layout(display_size):
    # {'main': rect, 'start' | 'stop' | 'light' | 'next': rect}
    w, h = display_size
    main_w = min(w, h * 4 // 3)
    bounds = [int(round(f * h)) for f in COLUMN_SPLITS]
    layout = {"main": (0, 0, main_w, h)}
    for name, y0, y1 in zip(COLUMN_ZONES, bounds, bounds[1:]):
        layout[name] = (main_w, y0, w - main_w, y1 - y0)
    return layout


class Screen:
    def __init__(self, cfg, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.layout = make_layout(cfg.DISPLAY_SIZE)
        self.view = cfg.UI_START_VIEW
        self.history = History(cfg)
        self.packet = None               # newest packet
        self.front = None                # last FrontResult received (with view)
        self.side = None
        self.quit = False

    def update(self, packet):
        # Called for every packet drained from the queue.
        self.packet = packet
        self.history.add(packet)
        if packet.front is not None:
            self.front = packet.front
        if packet.side is not None:
            self.side = packet.side

    def next_view(self):
        self.view = VIEWS[(VIEWS.index(self.view) + 1) % len(VIEWS)]

    def draw(self, packet, view):
        # Compose the chosen view at DISPLAY_SIZE; returns the BGR image.
        w, h = self.cfg.DISPLAY_SIZE
        frame = np.empty((h, w, 3), np.uint8)
        frame[:] = BG
        main = _sub(frame, self.layout["main"])
        if packet is None:
            draw_no_data(main, "robot link")
        else:
            {"drive": self._draw_drive_view, "camera": self._draw_camera_view,
             "diagnostic": self._draw_diagnostic_view}[view](main, packet)
        self._draw_column(frame, packet, view)
        return frame

    def handle_input(self, events):
        # Taps and keys -> the first UiCommand, or None. q / window close
        # set self.quit (display only; the robot keeps running).
        import pygame
        command = None
        for event in events:
            found = None
            if event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN:
                found = self._key_command(event.key, pygame)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                found = self._tap_command(event.pos)
            command = command or found
        return command

    # ---------------------------------------------------------------- views

    def _draw_drive_view(self, main, packet):
        # Header; then pre-flight (IDLE) or lane gauge and steering bar;
        # the tuning chart along the bottom.
        h, w = main.shape[:2]
        inner = w - 2 * GAP
        draw_header(main, (0, 0, w, 96), packet)
        if packet.state == State.IDLE:
            draw_preflight(main, (GAP, 104, inner, 200), preflight(packet, self.cfg))
        else:
            draw_lane_gauge(main, (GAP, 104, inner, 96), packet.lane, self.cfg)
            draw_steer_bar(main, (GAP, 240, inner, 28), packet.drive.steer, self.cfg)
        draw_chart(main, (GAP, 300, inner, h - 300 - GAP), self.history, self.cfg)

    def _draw_camera_view(self, main, packet):
        # Front camera filling the main area; side camera as an inset.
        h, w = main.shape[:2]
        main[:] = self._camera_image("front", (w, h))
        iw = int(w * INSET_FRAC)
        ih = iw * 3 // 4
        x0 = w - iw - GAP
        main[GAP:GAP + ih, x0:x0 + iw] = self._camera_image("side", (iw, ih))
        cv2.rectangle(main, (x0 - 1, GAP - 1), (x0 + iw, GAP + ih), LINE, 1)

    def _draw_diagnostic_view(self, main, packet):
        draw_status(main, packet, self.cfg)

    def _camera_image(self, kind, size):
        # The kept view enlarged to size, then annotated at that size; STALE
        # when old, NO DATA if nothing has ever arrived.
        result = self.front if kind == "front" else self.side
        img = np.zeros((size[1], size[0], 3), np.uint8)
        if result is None:
            draw_no_data(img, f"{kind} camera")
            return img
        img[:] = cv2.resize(result.view, size, interpolation=cv2.INTER_LINEAR)
        scale = size[0] / result.view.shape[1]
        if kind == "front":
            draw_front_overlay(img, result, scale)
        else:
            draw_side_overlay(img, result, self.cfg, scale)
        age = self.clock() - result.t
        if age > self.cfg.UI_VIEW_STALE_S:
            mark_stale(img, age)
        return img

    # ---------------------------------------------------------------- control column

    def _draw_column(self, frame, packet, view):
        ready = packet is not None and all(ok for _, ok in preflight(packet, self.cfg))
        _button(frame, self.layout["start"], (40, 95, 36) if ready else PANEL, "START",
                "ready" if ready else "not ready", GREEN if not ready else TEXT)
        _button(frame, self.layout["stop"], (40, 40, 170), "STOP", "not an e-stop", TEXT)
        self._draw_light(frame, packet)
        _button(frame, self.layout["next"], PANEL, "VIEW", view.upper(), TEXT, size=22)

    def _draw_light(self, frame, packet):
        # One glanceable health light: ALL OK, or the warning count and the first one.
        x, y, w, h = self.layout["light"]
        if packet is None:
            return
        n = len(packet.warnings)
        colour = RED if n else GREEN
        cv2.circle(frame, (x + 22, y + 30), 9, colour, -1, cv2.LINE_AA)
        draw_text(frame, f"{n} WARN" if n else "ALL OK", (x + 40, y + 38), 22, colour, bold=True)
        if n:
            draw_text(frame, packet.warnings[0].name.replace("_", " ").lower(),
                      (x + 12, y + 66), 14, MUTED)

    # ---------------------------------------------------------------- input

    def _key_command(self, key, pygame):
        if key == pygame.K_SPACE:
            state = self.packet.state if self.packet is not None else State.IDLE
            return UiCommand.START if state in START_STATES else UiCommand.STOP
        if key == pygame.K_v:
            return UiCommand.NEXT_VIEW
        if key == pygame.K_q:
            self.quit = True
        return None

    def _tap_command(self, pos):
        px, py = pos
        for zone, command in TAP_COMMANDS.items():
            x, y, w, h = self.layout[zone]
            if x <= px < x + w and y <= py < y + h:
                return command
        return None


# ---------------------------------------------------------------- drawing helpers

def _sub(frame, rect):
    x, y, w, h = rect
    return frame[y:y + h, x:x + w]


def _button(frame, rect, fill, title, subtitle, title_colour, size=34):
    x, y, w, h = rect
    cv2.rectangle(frame, (x + 6, y + 6), (x + w - 7, y + h - 7), fill, -1, cv2.LINE_AA)
    cx, cy = x + w // 2, y + h // 2
    draw_text(frame, title, (cx, cy + size // 3), size, title_colour, bold=True, align="centre")
    draw_text(frame, subtitle, (cx, cy + size // 3 + 24), 15, AMBER if subtitle == "not ready"
              else MUTED, align="centre")


# ---------------------------------------------------------------- process

def run_ui(packet_q, cmd_q, cfg=None):
    # Display process loop: newest packet -> Screen.draw -> taps -> wait for
    # UI_HZ. cfg is None when started as a process: it imports config itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)    # Ctrl-C is main's; main stops us
    if cfg is None:
        import config as cfg
    _set_display_env()
    display = open_display(cfg)
    if display is None:
        return
    pygame, surface = display
    screen, parent = Screen(cfg), os.getppid()
    signal.signal(signal.SIGTERM, lambda *_: setattr(screen, "quit", True))   # stop_ui
    period, next_t = 1.0 / cfg.UI_HZ, time.monotonic()
    try:
        while not screen.quit and os.getppid() == parent:
            next_t += period
            _drain(packet_q, screen, next_t)
            show(pygame, surface, screen.draw(screen.packet, screen.view))
            _route(screen.handle_input(pygame.event.get()), screen, cmd_q)
            next_t = max(next_t, time.monotonic())
    finally:
        pygame.quit()


def _set_display_env():
    # Launched over SSH: point SDL at the desktop session. Defaults only;
    # values already set are never overwritten.
    os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def open_display(cfg):
    # Full screen at DISPLAY_SIZE -> (pygame, surface), or None (logged).
    try:
        import pygame
        pygame.display.init()
        info = pygame.display.Info()
        actual, wanted = (info.current_w, info.current_h), tuple(cfg.DISPLAY_SIZE)
        if actual != wanted:
            print(f"UI warning: screen reports {actual[0]}x{actual[1]}, "
                  f"DISPLAY_SIZE is {wanted[0]}x{wanted[1]}")
        surface = pygame.display.set_mode(wanted, pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
        return pygame, surface
    except Exception as e:
        print(f"UI: display did not open ({e}); display process exiting, robot keeps running")
        return None


def show(pygame, surface, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surface.blit(pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB"), (0, 0))
    pygame.display.flip()


def _drain(packet_q, screen, until):
    # Blocks in packet_q.get(timeout) (no busy wait), feeding every packet to
    # the screen until the next redraw time.
    while True:
        remaining = until - time.monotonic()
        try:
            packet = packet_q.get(timeout=remaining) if remaining > 0 else packet_q.get_nowait()
        except (queue.Empty, EOFError, OSError):
            return
        screen.update(packet)
        if remaining <= 0:
            return


def _route(command, screen, cmd_q):
    # NEXT_VIEW stays here; START and STOP go to main (dropped if its queue is full).
    if command == UiCommand.NEXT_VIEW:
        screen.next_view()
    elif command is not None:
        try:
            cmd_q.put_nowait(command)
        except queue.Full:
            pass
