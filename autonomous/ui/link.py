"""Main-process side of the display: start it, build packets, send, read taps.

Nothing here can block or crash the control loop because of the display:
send() drops the packet when the one-slot queue is full, read_commands()
never waits, and both keep working if the display process has died.
pygame is never imported in the main process.
"""
import multiprocessing
import queue

from robot_types import UiCommand, UiPacket
from ui.screen import run_ui

_MAIN_COMMANDS = (UiCommand.START, UiCommand.STOP)    # NEXT_VIEW stays in the UI


def start_ui(cfg):
    # Starts the display process (spawn) -> (process, packet_q, cmd_q).
    ctx = multiprocessing.get_context("spawn")
    packet_q = ctx.Queue(maxsize=1)
    cmd_q = ctx.Queue(maxsize=cfg.UI_CMD_QUEUE_SIZE)
    packet_q.cancel_join_thread()        # never wait on a dead display at exit
    # run_ui imports config itself: the module cannot be sent to another
    # process, and main.py never changes it at runtime.
    process = ctx.Process(target=run_ui, args=(packet_q, cmd_q), name="ui", daemon=True)
    process.start()
    return process, packet_q, cmd_q


def stop_ui(process, cfg):
    # Ends the display process; called by shutdown after the motors stop.
    # SIGTERM lets it close the window cleanly; kill only if it doesn't.
    process.terminate()
    process.join(cfg.UI_STOP_TIMEOUT_S)
    if process.is_alive():
        process.kill()
        process.join(cfg.UI_STOP_TIMEOUT_S)


def make_ui_packet(sys, now, drive, motor, lane, road_yaw, imu, sign_vote, fps, overruns,
                   warnings, front, side, include_views):
    # Arguments in UiPacket field order (state and stop_reason come from
    # sys.behaviour). Numbers every loop; camera results (with their views)
    # only when include_views, so images go every UI_IMAGE_EVERY loops.
    return UiPacket(
        t=now, state=sys.behaviour.state, stop_reason=sys.behaviour.stop_reason,
        drive=drive, motor=motor, lane=lane, road_yaw=road_yaw, imu=imu, sign_vote=sign_vote,
        fps=dict(fps), overruns=overruns, warnings=tuple(warnings),
        front=front if include_views else None, side=side if include_views else None,
    )


def send(packet_q, packet):
    # put_nowait; a full queue (display busy or dead) drops the packet.
    try:
        packet_q.put_nowait(packet)
    except (queue.Full, ValueError, OSError):    # ValueError/OSError: queue closed
        pass


def read_commands(cmd_q):
    # Every UI command waiting, without blocking: START and STOP only.
    out = []
    while True:
        try:
            command = cmd_q.get_nowait()
        except (queue.Empty, EOFError, OSError, ValueError):
            return out
        if command in _MAIN_COMMANDS:
            out.append(command)
