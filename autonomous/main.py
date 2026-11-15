"""Robot entry point: build the system, run the 30 Hz control loop, shut down.

Front camera, IMU, lane fusion, road yaw, lane controller, sign voter,
junction location (LocationTracker), behaviour, display and the Sabertooth.
Location evidence and signs come from the front camera thread, or with
VISION_WORKER_ENABLED from the vision worker process (perception/worker.py),
which reads the frames the camera thread shares (hal/shared_frame.py). The
side camera (SIDE_CAM_ENABLED, if it opens) classes the ground in SIDE_BOX
(perception/side_surface.py) for the location tracker, display and log.
Modes: DRIVE (straight driving), QUIZ_INTERSECTION (drive through the
junction reporting each zone, docs/quiz3_plan.md stage 6), QUIZ_SIGNS.

Run from the robot/ folder:  python main.py
    Enter            START  (again after SAFE_STOP: once to IDLE, once to drive)
    s then Enter     STOP   (back to IDLE)
    1 / 2 / 3 Enter  mode: drive / intersection quiz / sign quiz (only in IDLE)
    Ctrl-C           quit; motors are stopped first

Test on a stand with the wheels off the ground before the first real run.
"""
import math
import multiprocessing
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace

import config
from behavior.sign_voter import SignVoter
from behavior.state_machine import Behaviour
from control.command import drive_for_state
from control.lane_controller import LaneController
from control.mixer import mix
from estimation.health import check_health
from estimation.lane_fusion import fuse_lane, is_fresh
from estimation.location import LocationTracker
from estimation.road_yaw import RoadYawTracker
from hal.camera import CameraStream, CsiSource, UsbSource
from hal.exposure import ExposureControl
from hal.imu import ImuReader
from hal.sabertooth import SabertoothDriver
from hal.shared_frame import SharedFrame
from loop_log import LoopLog
from perception.front_lane import LaneTracker
from perception.pipeline import SignPacer, process_front_frame
from perception.side_surface import process_side_frame
from perception.signs import load_sign_model
from perception.worker import start_worker, stop_worker, take_result
from robot_types import LatestValue, State, UiCommand, VisionResult
from tools.calib_status import startup_line
from ui.link import make_ui_packet, read_commands, send, start_ui, stop_ui

_LANE_STATES = (State.LANE_FOLLOW, State.APPROACH, State.REACQUIRE)


@dataclass
class System:
    cfg: object
    front_slot: LatestValue
    signs_enabled: threading.Event
    exposure_hold: threading.Event
    front_cam: CameraStream
    front_exposure: ExposureControl
    saber: SabertoothDriver
    lane_ctrl: LaneController
    road_yaw: RoadYawTracker
    behaviour: Behaviour
    commands: queue.Queue
    voter: object = None                  # SignVoter; None in tests without one
    locator: object = None                # LocationTracker; None in tests without one
    shared_frame: object = None           # SharedFrame and Worker with VISION_WORKER_ENABLED;
    worker: object = None                 # None for the in-thread path
    vision: object = None                 # newest VisionResult from the worker
    worker_start_t: float = 0.0           # for VISION_STALE before the first result
    side_cam: object = None               # side CameraStream, or None (off or did not open)
    side_slot: LatestValue = field(default_factory=LatestValue)
    imu: object = None                    # ImuReader; None in tests without one
    imu_slot: LatestValue = field(default_factory=LatestValue)
    loop_log: object = None               # LoopLog, or None when LOOP_LOG_ENABLED is off
    ui_process: object = None             # display process and its queues; None when
    ui_packet_q: object = None            # UI_ENABLED is off (no display process at all)
    ui_cmd_q: object = None
    ui_loops: int = 0                     # loops since start, for UI_IMAGE_EVERY
    prev_state: State = State.IDLE
    overrun_times: deque = field(default_factory=deque)
    last_warnings: tuple = ()
    last_seen: tuple = (None, None, None) # (mode, voted sign, zone) last printed
    last_status_t: float = 0.0
    last_drive: object = None


# ---------------------------------------------------------------- build

def build_system(cfg):
    # IMU first (it calibrates in its own thread), then the cameras (they
    # need warm-up), then the Sabertooth.
    if cfg.FRONT_BANDS is None:
        raise SystemExit("FRONT_BANDS is not set in config.py: run python -m tools.calib_ui first")
    signs_enabled, exposure_hold = threading.Event(), threading.Event()
    exposure_hold.set()                               # robot starts in IDLE
    shared, worker = None, None
    if cfg.VISION_WORKER_ENABLED:
        # The worker reads signs_enabled too, so it must be a process Event.
        ctx = multiprocessing.get_context("spawn")
        signs_enabled = ctx.Event()
        shared = SharedFrame.create(cfg.FRONT_SIZE, ctx)
        worker = start_worker(shared, signs_enabled, cfg)
    drift_request = threading.Event()
    imu_slot = LatestValue()
    imu = ImuReader(imu_slot, drift_request, cfg)
    imu.start()
    ui = start_ui(cfg) if cfg.UI_ENABLED else (None, None, None)
    front_slot = LatestValue()
    front_cam, front_exposure = _build_front_camera(cfg, front_slot, signs_enabled, exposure_hold,
                                                    shared)
    side_slot = LatestValue()
    side_cam = _build_side_camera(cfg, side_slot)

    saber = SabertoothDriver(cfg.SABER_PORT, cfg.SABER_BAUD, cfg.SABER_ADDRESS, cfg)
    saber.open()
    if saber.serial_error:
        print("Warning: Sabertooth port did not open; the robot will SAFE_STOP when started")

    commands = queue.Queue()
    _start_keyboard(commands)
    return System(
        cfg=cfg, front_slot=front_slot, signs_enabled=signs_enabled,
        exposure_hold=exposure_hold, front_cam=front_cam, front_exposure=front_exposure,
        saber=saber, lane_ctrl=LaneController(cfg), road_yaw=RoadYawTracker(cfg),
        behaviour=Behaviour(cfg, signs_enabled, exposure_hold, drift_request),
        commands=commands, voter=SignVoter(cfg), locator=LocationTracker(cfg),
        imu=imu, imu_slot=imu_slot,
        loop_log=_open_loop_log(cfg),
        ui_process=ui[0], ui_packet_q=ui[1], ui_cmd_q=ui[2],
        shared_frame=shared, worker=worker, worker_start_t=time.monotonic(),
        side_cam=side_cam, side_slot=side_slot,
    )


def _open_loop_log(cfg):
    if not cfg.LOOP_LOG_ENABLED:
        return None
    log = LoopLog(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.LOOP_LOG_DIR), cfg)
    print(f"Logging each loop to {log.path}")
    return log


def _build_front_camera(cfg, slot, signs_enabled, exposure_hold, shared=None):
    # Starts the front camera thread and locks exposure after warm-up. With
    # shared (a SharedFrame: the vision worker is on) the thread measures
    # the lane and shares the frame; otherwise it reads location and signs too.
    source = CsiSource(cfg.FRONT_CAM_NUM, cfg.FRONT_SIZE, cfg)
    exposure = ExposureControl(source, "front", cfg)
    tracker = LaneTracker()                          # search window, camera thread only
    sign_model = _open_sign_model(cfg) if shared is None else None
    pacer = SignPacer()

    def process(frame):
        if shared is None:
            result = process_front_frame(frame, signs_enabled.is_set(), cfg, tracker,
                                         sign_model, pacer)
        else:
            result = process_front_frame(frame, False, cfg, tracker, with_vision=False)
            shared.publish(frame, result.lane.line_px)
        exposure.manage_step(result.mean_v, exposure_hold.is_set())
        return result

    stream = CameraStream(source, process, slot, "front", cfg)
    stream.start()
    exposure.lock_after_warmup()
    return stream, exposure


def _build_side_camera(cfg, slot):
    # Starts the side camera thread (auto-exposure: the webcam's manual
    # settings are not confirmed yet), or returns None with a warning: the
    # robot drives without it.
    if not cfg.SIDE_CAM_ENABLED:
        return None
    source = UsbSource(cfg.SIDE_CAM_DEVICE, cfg.SIDE_SIZE, cfg.SIDE_CAM_FOURCC, cfg)
    stream = CameraStream(source, lambda frame: process_side_frame(frame, cfg), slot, "side", cfg)
    try:
        stream.start()
    except Exception as e:
        print(f"Warning: side camera did not start ({e}); carrying on without it")
        return None
    return stream


def _open_sign_model(cfg):
    # The trained sign model, or None with a warning: the robot still
    # drives, and the display shows sign candidates without labels.
    try:
        return load_sign_model(cfg.SIGN_MODEL_PATH, cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: {e}\n         Signs will be boxed but not read.")
        return None


_KEYS = {"s": UiCommand.STOP, "1": UiCommand.MODE_DRIVE, "2": UiCommand.MODE_INTERSECTION,
         "3": UiCommand.MODE_SIGNS}


def _start_keyboard(commands):
    # Terminal stand-in for the display: Enter = START, "s" + Enter = STOP,
    # 1 / 2 / 3 + Enter = drive / intersection quiz / sign quiz mode.
    def read_keys():
        while True:
            try:
                line = input()
            except EOFError:
                return
            commands.put(_KEYS.get(line.strip().lower(), UiCommand.START))

    threading.Thread(target=read_keys, name="keyboard", daemon=True).start()


# ---------------------------------------------------------------- one loop

def control_step(sys, now):
    # Read slots -> fuse -> road yaw -> health -> behaviour -> drive -> mix -> Sabertooth.
    # imu_raw is the latest sample whatever its age (health and the safety
    # check judge staleness); imu is None when it is missing or older than
    # MAX_OBS_AGE_S, so road yaw and steering never use stale yaw.
    cfg = sys.cfg
    front = sys.front_slot.get()
    imu_raw = sys.imu_slot.get()
    imu = imu_raw if is_fresh(imu_raw, now, cfg) else None
    lane = fuse_lane(front, None, now, cfg)
    sys.road_yaw.update(imu, lane, sys.behaviour.state, now)
    road_yaw = sys.road_yaw.get() if sys.imu is not None else None
    commands = _read_commands(sys.commands)
    if sys.ui_cmd_q is not None:
        commands += read_commands(sys.ui_cmd_q)      # taps, merged with the keyboard

    vision = _vision(sys, front)
    side = sys.side_slot.get() if sys.side_cam is not None else None
    fps, overruns = _fps(sys), _recent_overruns(sys, now)
    # The side camera's brightness isn't judged: its auto-exposure differs
    # from the front camera's DARK_WARN_V / BRIGHT_WARN_V.
    warnings = check_health(front, None, imu_raw, lane, road_yaw, fps,
                            overruns, sys.saber.serial_error, now, cfg, _vision_age(sys, now))
    vote = _sign_vote(sys, vision, now)
    location = _location(sys, vision, now, side)
    state = sys.behaviour.step(front, None, lane, road_yaw, vote, imu_raw, commands,
                               sys.saber.serial_error, now, location)
    if not sys.signs_enabled.is_set():
        vote = _sign_vote(sys, vision, now)          # sign reading went off this loop: clear
    if state in _LANE_STATES and state != sys.prev_state:
        sys.lane_ctrl.reset()
    if sys.prev_state == State.IDLE and state != State.IDLE and sys.locator is not None:
        sys.locator.reset()                          # each run starts its own junction pass

    drive = drive_for_state(state, lane, imu, sys.behaviour, sys.lane_ctrl, front, cfg, road_yaw)
    motor = mix(drive, cfg)
    sys.saber.set(motor)                             # every loop, or the timeout stops the motors
    sys.last_drive = drive
    quiz_location = sys.behaviour.location           # None outside QUIZ_INTERSECTION
    if sys.loop_log is not None:
        sys.loop_log.write(now, state, lane, imu, drive, sys.behaviour.mode, location, vote,
                           side.surface if side is not None else None)
    if sys.ui_packet_q is not None:
        include_views = sys.ui_loops % cfg.UI_IMAGE_EVERY == 0
        sys.ui_loops += 1
        send(sys.ui_packet_q, make_ui_packet(sys, now, drive, motor, lane, road_yaw, imu_raw,
                                             vote, fps, overruns, warnings,
                                             _front_for_ui(sys, front, vision), side,
                                             include_views, quiz_location))
    _report(sys, state, lane, imu, road_yaw, warnings, now, vote, quiz_location)
    sys.prev_state = state


def _vision(sys, front):
    # The newest VisionResult: from the worker (kept until a newer one
    # arrives), or from the front camera thread's own result.
    if sys.worker is None:
        if front is None:
            return None
        return VisionResult(front.location, front.sign, front.t, front.seq)
    newest = take_result(sys.worker)
    if newest is not None:
        sys.vision = newest
    return sys.vision


def _vision_age(sys, now):
    # Seconds since the newest worker result's frame (inf once the worker
    # has died), or None without a worker. Only a warning (VISION_STALE):
    # the control loop carries on.
    if sys.worker is None:
        return None
    if not sys.worker.process.is_alive():
        return math.inf
    return now - (sys.vision.t if sys.vision is not None else sys.worker_start_t)


def _front_for_ui(sys, front, vision):
    # The front result for the display, carrying the worker's location and
    # sign so the camera view draws them as with the in-thread path.
    if sys.worker is None or front is None:
        return front
    return replace(front, location=vision.location if vision is not None else None,
                   sign=vision.sign if vision is not None else None)


def _sign_vote(sys, vision, now):
    # Feeds the newest fresh sign reading to the voter -> SignVote or None.
    # With sign reading off the voter is cleared, so an old sign never
    # lingers into the next time it is switched on.
    if sys.voter is None:
        return None
    if not sys.signs_enabled.is_set():
        sys.voter.reset()
        return None
    if vision is not None and vision.sign is not None and is_fresh(vision.sign, now, sys.cfg):
        sys.voter.add(vision.sign)
    return sys.voter.decided()


def _fps(sys):
    # Frame rate of each camera running.
    fps = {"front": sys.front_cam.fps()}
    if sys.side_cam is not None:
        fps["side"] = sys.side_cam.fps()
    return fps


def _location(sys, vision, now, side=None):
    # The newest LocationObs from the camera's evidence, moved on by last
    # loop's speed command where the dashes can't be matched; None without
    # a tracker.
    if sys.locator is None:
        return None
    evidence = vision.location if vision is not None else None
    speed = sys.last_drive.speed if sys.last_drive is not None else 0.0
    sign = vision.sign if vision is not None else None
    return sys.locator.update(evidence, speed, now, sign, side)


def _read_commands(commands):
    # Everything queued since the last loop, without blocking.
    out = []
    while True:
        try:
            out.append(commands.get_nowait())
        except queue.Empty:
            return out


def _recent_overruns(sys, now):
    while sys.overrun_times and now - sys.overrun_times[0] > sys.cfg.OVERRUN_WINDOW_S:
        sys.overrun_times.popleft()
    return len(sys.overrun_times)


def _report(sys, state, lane, imu, road_yaw, warnings, now, vote=None, location=None):
    # Terminal output: state, mode, voted sign, zone and warning changes at
    # once, a status line every STATUS_PRINT_S.
    in_pass = location is not None and location.dist_to_centre_m is not None
    seen = (sys.behaviour.mode, vote.sign if vote is not None else None,
            location.zone if in_pass else None)
    if seen != sys.last_seen:
        if seen[0] != sys.last_seen[0]:
            print(f"Mode: {sys.cfg.MODE_LABELS[seen[0].name]}")
        if seen[1] != sys.last_seen[1]:
            print("Sign: " + (sys.cfg.SIGN_PHRASES[seen[1].name] if seen[1] else "none"))
        if seen[2] != sys.last_seen[2] and None not in (seen[2], sys.last_seen[2]):
            # a change from one zone to the next; the first zone of a pass is not news
            print(f"Zone: {sys.cfg.ZONE_PHRASES[seen[2].name]}  ({_where(location)})")
        sys.last_seen = seen
    if state != sys.prev_state:
        reason = f" ({sys.behaviour.stop_reason})" if sys.behaviour.stop_reason else ""
        print(f"State: {sys.prev_state.name} -> {state.name}{reason}")
    names = tuple(w.name for w in warnings)
    if names != sys.last_warnings:
        print("Warnings: " + (", ".join(names) if names else "none"))
        sys.last_warnings = names
    if now - sys.last_status_t >= sys.cfg.STATUS_PRINT_S:
        print(_status_line(sys, state, lane))
        print("  " + _imu_line(imu, road_yaw, now))
        sys.last_status_t = now


def _where(location):
    # "s -0.27 m, dashes"
    if location.dist_to_centre_m is None:
        return "no junction"
    return f"s {-location.dist_to_centre_m:+.2f} m, {location.source}"


def _status_line(sys, state, lane):
    lat = f"{lane.lateral_err_m * 1000:+6.1f} mm" if lane.lateral_err_m is not None else "   --   "
    head = (f"{lane.heading_err_rad * 57.2958:+5.1f} deg"
            if lane.heading_err_rad is not None else "  --    ")
    drive = sys.last_drive
    return (f"{state.name:<11} lat {lat}  head {head}  conf {lane.conf:.2f}  "
            f"speed {drive.speed:.2f}  steer {drive.steer:+.2f}  "
            f"fps {sys.front_cam.fps():4.1f}  cam errors {sys.front_cam.error_count}")


def _imu_line(imu, road_yaw, now):
    # yaw, yaw rate, road yaw and the time since its last camera anchor, calib, drift.
    if imu is None:
        return "IMU --"
    road = (f"road {road_yaw.yaw_deg:+7.1f} deg ({now - road_yaw.t_anchor:4.1f} s ago)"
            if road_yaw is not None and road_yaw.valid else "road   --")
    return (f"yaw {imu.yaw_deg:+7.1f} deg  rate {imu.yaw_rate_dps:+6.1f} deg/s  {road}  "
            f"valid {int(imu.valid)}  "
            f"calib {imu.calib_gyro}  drift {imu.drift_dps:+.3f} deg/s")


# ---------------------------------------------------------------- loop and shutdown

def run(sys):
    # Fixed-rate loop at CONTROL_HZ. An overrun is counted and the schedule
    # restarts from now rather than trying to catch up.
    period = 1.0 / sys.cfg.CONTROL_HZ
    next_t = time.monotonic()
    while True:
        control_step(sys, time.monotonic())
        next_t += period
        wait = next_t - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        else:
            now = time.monotonic()
            sys.overrun_times.append(now)
            next_t = now


def shutdown(sys):
    # Motors first, then everything else; each step runs even if one fails.
    # The loop log is flushed before the camera, whose stop can hang when the
    # camera has failed, so the log of a bad run is never lost. The shared
    # frame is freed last, once neither the worker nor the camera uses it.
    steps = [sys.saber.stop, sys.saber.close]
    if sys.imu is not None:
        steps.append(sys.imu.close)
    if sys.loop_log is not None:
        steps.append(sys.loop_log.close)
    if sys.ui_process is not None:
        def stop_display():
            stop_ui(sys.ui_process, sys.cfg)
        steps.append(stop_display)
    if sys.worker is not None:
        def stop_vision():
            stop_worker(sys.worker, sys.cfg)
        steps.append(stop_vision)
    steps.append(sys.front_cam.stop)
    if sys.side_cam is not None:
        steps.append(sys.side_cam.stop)
    if sys.shared_frame is not None:
        steps.append(sys.shared_frame.close)        # after the camera: it writes there
    for step in steps:
        try:
            step()
        except Exception as e:
            print(f"Shutdown: {step.__qualname__} failed: {e}")


def main():
    system = None
    try:
        print(startup_line(config))
        system = build_system(config)
        print("Ready. Enter = START, s + Enter = STOP, Ctrl-C = quit")
        print("Modes: 1 + Enter = drive, 2 + Enter = intersection quiz, 3 + Enter = sign quiz")
        run(system)
    except KeyboardInterrupt:
        print("\nStopping")
    finally:
        if system is not None:
            shutdown(system)


if __name__ == "__main__":
    main()