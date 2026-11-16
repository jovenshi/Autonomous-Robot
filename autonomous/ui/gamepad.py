"""PS5 controller in the operator process: buttons -> UiCommands, never blocking.

Reuses the course's PS5_Controller (manual/ps5_controller.py, not modified) for
its button map, debounce and control_request flags. Its initialize_controller
is never called (it waits until a controller connects): the joystick is
attached when pygame reports one (at start-up or JOYDEVICEADDED) and dropped
on JOYDEVICEREMOVED or a read error, so the display runs the same with no
controller, and one can be connected or lost at any time.

PS5_Controller's debounce repeats a held button every 0.5 s. Here a button
counts again only after a release (JOYBUTTONUP; for the D-pad, the hat back
at centre), so holding Cross cannot send START and then STOP.

Each poll also reads the triggers and left stick into `manual_input` (a
ManualInput, None with no controller) for driving by hand while stopped
(control/manual.py), with the same scaling as manual/robot_controller.py.
"""
import time

from robot_types import ManualInput, State, UiCommand
from ui.course import course_module

START_STATES = (State.IDLE, State.SAFE_STOP)        # START_STOP (and the space bar) means START here
TOGGLE = "START_STOP"                               # PS5_BUTTONS value: START or STOP by state


def start_or_stop(state):
    # START in IDLE and SAFE_STOP, STOP anywhere else.
    return UiCommand.START if state in START_STATES else UiCommand.STOP


class Gamepad:
    def __init__(self, cfg, controller=None):
        import pygame
        self.pygame = pygame
        self.buttons = cfg.PS5_BUTTONS
        self.latched = set()            # control_request keys pressed and not yet released
        self.manual_input = None        # newest ManualInput, None with no controller
        self.ps5 = controller if controller is not None else _course_controller()
        if self.ps5 is not None:
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self._attach(0)

    def poll(self, events, state):
        # This cycle's pygame events and the robot's State -> list of UiCommand.
        # Returns at once whether or not a controller is connected.
        self.manual_input = None
        if self.ps5 is None:
            return []
        self._handle_events(events)
        if self.ps5.joystick is None:
            return []
        try:
            self.ps5.check_controls()
        except self.pygame.error as e:
            self._detach(f"read failed ({e})")
            return []
        self.manual_input = read_manual(self.ps5.control_request)
        pressed = [k for k in self.buttons if self.ps5.control_request.get(k)]
        self.ps5.reset_controller_state()
        out = [self._command(k, state) for k in pressed if k not in self.latched]
        self.latched.update(pressed)
        return out

    def _command(self, key, state):
        name = self.buttons[key]
        return start_or_stop(state) if name == TOGGLE else UiCommand[name]

    def _handle_events(self, events):
        pygame = self.pygame
        for event in events:
            if event.type == pygame.JOYBUTTONUP:
                self.latched.clear()
            elif event.type == pygame.JOYHATMOTION and tuple(event.value) == (0, 0):
                self.latched = {k for k in self.latched if not k.startswith("reqArrow")}
            elif event.type == pygame.JOYDEVICEADDED and self.ps5.joystick is None:
                self._attach(event.device_index)
            elif event.type == pygame.JOYDEVICEREMOVED:
                self._detach("disconnected")
                if pygame.joystick.get_count() > 0:
                    self._attach(0)

    def _attach(self, index):
        try:
            joystick = self.pygame.joystick.Joystick(index)
            joystick.init()
        except self.pygame.error as e:
            print(f"Gamepad: controller {index} did not open ({e})")
            return
        self.ps5.joystick = joystick
        self.latched.clear()
        print(f"Gamepad: {joystick.get_name()} connected")

    def _detach(self, why):
        if self.ps5.joystick is not None:
            print(f"Gamepad: {why}; screen and keyboard still work")
        self.ps5.joystick = None
        self.latched.clear()


def read_manual(request):
    # PS5_Controller.control_request -> ManualInput: triggers 0-1 and the left
    # stick X as -1..1 (0 while the stick is inside PS5_Controller's deadzone).
    stick = request.get("reqLeftJoyXValue", 0) / 127 if request.get("reqLeftJoyMade") else 0.0
    return ManualInput(r2=float(request.get("reqR2Value", 0.0)),
                       l2=float(request.get("reqL2Value", 0.0)),
                       stick_x=max(-1.0, min(1.0, stick)), t=time.monotonic())


def _course_controller():
    # The course's PS5_Controller, or None (logged) if it can't be loaded.
    try:
        return course_module("manual.ps5_controller").PS5_Controller()
    except Exception as e:
        print(f"Gamepad: PS5_Controller unavailable ({e}); screen and keyboard only")
        return None
