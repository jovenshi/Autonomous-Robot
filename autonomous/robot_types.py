"""Data types passed between blocks.

Blocks exchange only the types defined here (architecture doc, "Data types
passed between blocks"). Named robot_types.py rather than types.py so it never
shadows Python's built-in `types` module.

Sign conventions (fixed once, used everywhere):
    lateral error +  : rear axle is right of the yellow line
    heading error +  : robot points right of the road direction
    steer +          : turn left
    yaw +            : counter-clockwise (left)
    road_yaw = yaw + heading error

Times (t) are seconds from time.monotonic(). seq is the number of the frame a
result came from, so stale data can be dropped and overlays land on the frame
they were measured on.

Where a measurement can be missing, the field is Optional: None means
"not measured this frame", and conf is 0.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Optional, TypeVar

import numpy as np


# ---------------------------------------------------------------- enums

class SignType(Enum):
    NONE = auto()
    PAUSE = auto()             # red octagon ("STOP" in the Oct 9 design)
    LEFT = auto()
    RIGHT = auto()
    STRAIGHT = auto()


class Action(Enum):
    PAUSE = auto()
    LEFT = auto()
    RIGHT = auto()
    STRAIGHT = auto()


class State(Enum):
    IDLE = auto()
    LANE_FOLLOW = auto()
    APPROACH = auto()
    DECIDE = auto()
    STOP_HOLD = auto()
    TURN = auto()
    CROSS = auto()
    REACQUIRE = auto()
    SAFE_STOP = auto()


class HealthWarning(Enum):  # "Warning" in the design; renamed to avoid the builtin
    IMU_INVALID = auto()
    IMU_NOT_CALIBRATED = auto()
    IMU_DRIFT_HIGH = auto()
    ROAD_YAW_STALE = auto()
    CAM_FPS_LOW = auto()
    CAM_TOO_DARK = auto()
    CAM_TOO_BRIGHT = auto()
    CAMS_DISAGREE = auto()
    LANE_CONF_LOW = auto()
    LOOP_OVERRUN = auto()
    MOTOR_SERIAL_ERROR = auto()
    VISION_STALE = auto()


class UiCommand(Enum):
    START = auto()
    STOP = auto()
    NEXT_VIEW = auto()
    MODE_DRIVE = auto()
    MODE_INTERSECTION = auto()
    MODE_SIGNS = auto()


class Zone(Enum):
    # Where the robot is along one junction pass, in driving order:
    # CONTINUE, READ_SIGN, PAUSE, ENTERING, TURN, EXITING, then CONTINUE again.
    CONTINUE = auto()
    READ_SIGN = auto()
    PAUSE = auto()
    ENTERING = auto()
    TURN = auto()
    EXITING = auto()


class RunMode(Enum):
    DRIVE = auto()
    QUIZ_INTERSECTION = auto()
    QUIZ_SIGNS = auto()


class Surface(Enum):
    GRASS = auto()
    ROAD = auto()
    YELLOW = auto()


class AnnotationKind(Enum):
    RECT = auto()              # pts: two opposite corners
    LINE = auto()              # pts: two ends
    POINT = auto()             # pts: one point
    TEXT = auto()              # pts: where the text starts


# ---------------------------------------------------------------- cameras

@dataclass(frozen=True)
class Frame:
    image: np.ndarray          # H x W x 3, BGR, uint8
    t: float                   # s
    seq: int
    cam_id: str                # 'front' or 'side'


@dataclass(frozen=True)
class FrontMasks:
    bgr: np.ndarray            # lane-size colour copy; becomes FrontResult.view
    hsv: np.ndarray            # lane-size HSV image
    yellow: np.ndarray         # masks at lane size, uint8 0 or 255
    road: np.ndarray
    white: Optional[np.ndarray]  # None unless built for sign reading
    red: Optional[np.ndarray]    # None unless built for sign reading
    scale: float               # full frame / lane copy
    mean_v: float              # mean V brightness, 0-255


@dataclass(frozen=True)
class FrontLaneObs:
    lateral_m: Optional[float]         # at the rear axle; + = axle right of line
    heading_err_rad: Optional[float]   # + = robot points right of the road
    conf: float                        # 0-1
    centre_pts_px: tuple[tuple[float, float], ...]  # band centres (x, y), lane-size px
    line_px: Optional[tuple[tuple[float, float], tuple[float, float]]]  # fitted line ends
    t: float
    seq: int


@dataclass(frozen=True)
class Annotation:
    # One thing to draw on a camera view. Perception makes these on the frame
    # it measured; ui/overlay.py draws them without knowing what they mean.
    kind: AnnotationKind
    pts: tuple[tuple[float, float], ...]   # (x, y) in view pixels (lane size)
    text: str = ""
    colour: tuple[int, int, int] = (255, 255, 255)   # BGR
    cam_id: str = "front"                  # 'front' or 'side'


@dataclass(frozen=True)
class LocationEvidence:
    # What one frame says about where the junction is. Distances are metres
    # ahead of the rear axle; None = not seen this frame.
    crossing_dist_m: Optional[float]   # to the crossing road's yellow line
    flare_dist_m: Optional[float]      # to where the road starts to widen
    side_surface: Optional[Surface]    # under the side camera's box, or None
    annotations: tuple[Annotation, ...]
    t: float
    seq: int


@dataclass(frozen=True)
class SignObs:
    sign: SignType
    score: float                                    # SVM margin: best class minus runner-up
    bbox: Optional[tuple[int, int, int, int]]       # x, y, w, h in full-frame px
    t: float
    seq: int
    roi: Optional[tuple[int, int, int, int]] = None   # searched region, x, y, w, h in view px
    scores: tuple[tuple[str, float], ...] = ()        # (class name, score) for every class
    annotations: tuple[Annotation, ...] = ()


@dataclass(frozen=True)
class SignModel:
    hog: object                # cv2.HOGDescriptor built from the model file's settings
    weights: np.ndarray        # one row per class
    biases: np.ndarray         # one per class
    classes: tuple[str, ...]   # SignType names, in row order


@dataclass(frozen=True)
class FrontResult:
    view: np.ndarray                   # 320 x 240 image for the display
    lane: FrontLaneObs
    location: Optional[LocationEvidence]   # None until perception/location.py is built
    sign: Optional[SignObs]            # newest sign reading (see its seq); None while signs are off
    mean_v: float
    t: float
    seq: int


@dataclass(frozen=True)
class SideLaneObs:
    offset_m: Optional[float]          # at the camera
    boundary_found: bool
    opening: bool                      # road leaves to the right here
    conf: float                        # 0-1
    boundary_x_px: Optional[float]
    t: float
    seq: int


@dataclass(frozen=True)
class SideResult:
    view: np.ndarray                   # 320 x 240 image for the display
    side: SideLaneObs
    mean_v: float
    t: float
    seq: int


# ---------------------------------------------------------------- IMU

@dataclass(frozen=True)
class ImuSample:
    yaw_deg: float             # relative, unwrapped, drift-corrected; + = left
    yaw_rate_dps: float
    valid: bool
    calib_gyro: int            # 0-3
    drift_dps: float
    t: float


# ---------------------------------------------------------------- estimation

@dataclass(frozen=True)
class LaneEstimate:
    lateral_err_m: Optional[float]     # at the rear axle
    heading_err_rad: Optional[float]
    conf: float                        # 0-1
    cam_gap_m: Optional[float]         # front minus side, when both are good
    t: float


@dataclass(frozen=True)
class RoadYaw:
    yaw_deg: float
    t_anchor: float            # time of the last camera anchor
    valid: bool


@dataclass(frozen=True)
class LocationObs:
    zone: Zone
    sign_read_ready: bool              # in READ_SIGN: signs can be read
    pause_point_ready: bool            # reached the PAUSE zone
    turn_point_ready: bool             # reached the TURN zone
    dist_to_centre_m: Optional[float]  # rear axle to the junction centre; + = still ahead
    conf: float                        # 0-1
    source: str                        # 'crossing', 'flare', 'side', 'dead_reckoning' or 'none'
    t: float


# ---------------------------------------------------------------- behaviour

@dataclass(frozen=True)
class SignVote:
    sign: SignType
    bbox: Optional[tuple[int, int, int, int]]   # median of recent boxes, full-frame px
    since_t: float                              # when this sign won the vote


# ---------------------------------------------------------------- vision worker

@dataclass(frozen=True)
class VisionResult:
    location: Optional[LocationEvidence]
    sign: Optional[SignObs]
    t: float
    seq: int


# ---------------------------------------------------------------- control

@dataclass(frozen=True)
class DriveCmd:
    speed: float               # 0-1
    steer: float               # -1 to 1; + = turn left


@dataclass(frozen=True)
class MotorCmd:
    left: float                # -1 to 1
    right: float               # -1 to 1


# ---------------------------------------------------------------- display

@dataclass(frozen=True)
class UiPacket:
    t: float
    state: State
    stop_reason: Optional[str]
    drive: DriveCmd
    motor: MotorCmd                    # what was sent to the Sabertooth this loop
    lane: Optional[LaneEstimate]
    road_yaw: Optional[RoadYaw]
    imu: Optional[ImuSample]
    sign_vote: Optional[SignVote]
    fps: dict[str, float]              # per camera: 'front', 'side'
    overruns: int
    warnings: tuple[HealthWarning, ...]
    front: Optional[FrontResult]       # with view, or None
    side: Optional[SideResult]         # with view, or None
    mode: RunMode = RunMode.DRIVE
    location: Optional[LocationObs] = None


# ---------------------------------------------------------------- slots

T = TypeVar("T")


class LatestValue(Generic[T]):
    """Holds one item behind a lock. A newer value overwrites the older one;
    nothing queues."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[T] = None

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value

    def get(self) -> Optional[T]:
        with self._lock:
            return self._value