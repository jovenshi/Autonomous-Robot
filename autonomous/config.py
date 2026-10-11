"""Every tunable lives here, once. Nothing else in the code defines a number.

Callers do `import config as cfg` and pass the module wherever a function
takes `cfg`.

Values are starting points to tune on the course, not tested settings.
    MEASURED  -> set from tools/calibrate.py (or measured on the course)
    TUNED     -> set from tools/tune_colors.py under the course lighting
    CONFIRM   -> depends on a hardware or course fact still open
A value of None means it has not been measured or confirmed yet.

Sizes are (width, height) in pixels.
"""

# ---------------------------------------------------------------- timing

CONTROL_HZ = 30                 # Hz
IMU_HZ = 50                     # Hz
CAM_FPS = 30                    # fps
MAX_OBS_AGE_S = 0.15            # s; raise if the webcam's measured latency needs it
LANE_LOST_TIMEOUT_S = 0.5       # s

# ---------------------------------------------------------------- cameras and geometry

FRONT_CAM_NUM = 0
SIDE_CAM_DEVICE = None          # CONFIRM: your webcam's /dev/v4l/by-id/... path
SIDE_CAM_FOURCC = "YUYV"
FRONT_SIZE = (640, 480)
FRONT_LANE_SIZE = (320, 240)
SIDE_SIZE = (320, 240)
FRONT_LENS_POSITION = 2.0       # dioptres (1 / focus distance in m); MEASURED; Camera Module 3 manual focus
CAM_READ_RETRY_S = 0.01         # s; pause after a failed read before trying again
CAM_FPS_WINDOW_S = 1.0          # s; window over which fps() is measured

# MEASURED: 5 bands of (y_top, y_bottom, forward_m from the rear axle,
# half_road_px, m_per_px); y values in FRONT_LANE_SIZE pixels.
FRONT_BANDS = None

SIDE_ROWS = (40, 80, 120, 160, 200)  # 5 rows spread over the side frame height
SIDE_ROAD_ON_LEFT = None        # MEASURED: True if road appears on the image's left
SIDE_TARGET_PX = None           # MEASURED: edge column with the robot centred on the line
SIDE_M_PER_PX = None            # MEASURED: m/px
SIDE_CAM_X_M = None             # MEASURED: m; + = ahead of the rear axle
MIN_LINE_PX = 30                # pixels per band
ROAD_COL_FRAC = 0.5             # a band column counts as road if this fraction of its rows is road
ROAD_GAP_MAX_PX = 8             # gaps in the road this narrow (px, lane size) are bridged
FRONT_FIT_RMS_MAX_M = 0.03      # m; fit error at which front lane conf falls to 0

# ---------------------------------------------------------------- exposure and lighting

EXPOSURE_MODE = "locked"        # 'locked', 'managed' or 'auto'; CONFIRM once lighting is known
CAM_WARMUP_S = 2.0              # s
EXPOSURE_MAX_US = 10000         # µs
EXPOSURE_TARGET_V = None        # 0-255; None = use the brightness measured at lock time
EXPOSURE_STEP = 0.1             # fraction per step
EXPOSURE_CHECK_FRAMES = 15      # frames
EXPOSURE_MIN_US = 100           # µs; floor when darkening in managed mode
EXPOSURE_TOL_V = 10             # 0-255; managed mode leaves exposure alone within this of the target
EXPOSURE_GAIN_MIN = 1.0         # analogue gain floor (CSI scale; CONFIRM for the webcam)
EXPOSURE_GAIN_MAX = 8.0         # analogue gain ceiling (CSI scale; CONFIRM for the webcam)
USB_LOCK_EXPOSURE_US = None     # MEASURED on the bench: used if a camera's exposure readback is unusable
USB_LOCK_GAIN = None            # MEASURED on the bench: used with USB_LOCK_EXPOSURE_US
# Webcam exposure through OpenCV's V4L2 properties (replaces USB_EXPOSURE_CTRLS)
USB_AUTO_EXPOSURE_MANUAL = 1    # CAP_PROP_AUTO_EXPOSURE value for manual; CONFIRM on the bench
USB_AUTO_EXPOSURE_AUTO = 3      # CAP_PROP_AUTO_EXPOSURE value for auto; CONFIRM on the bench
USB_EXPOSURE_UNIT_US = 100      # µs per CAP_PROP_EXPOSURE step; CONFIRM on the bench
DARK_WARN_V = 40                # 0-255
BRIGHT_WARN_V = 220             # 0-255

# ---------------------------------------------------------------- colour tests (TUNED)
# OpenCV HSV: H 0-179, S and V 0-255

YELLOW_H_MIN = 20
YELLOW_H_MAX = 35
YELLOW_S_MIN = 90
YELLOW_V_MIN = 60
GREEN_CHROM_MIN = 0.40          # G / (R+G+B)
DARK_SUM_MIN = 90               # R+G+B; below = too dark to judge
ROAD_S_MAX = 70
ROAD_V_T_MIN = 40               # clamp on the Otsu threshold
ROAD_V_T_MAX = 110
WHITE_S_MAX = 50
WHITE_V_REL = 0.8               # ratio of the frame's bright percentile
WHITE_V_PCT = 95                # percentile
RED_H_LOW_MAX = 10
RED_H_HIGH_MIN = 170
RED_S_MIN = 100

# ---------------------------------------------------------------- junctions and signs

APPROACH_ON = 0.6               # score; hysteresis with APPROACH_OFF
APPROACH_OFF = 0.3
CROSS_LINE_MIN_FRAC = 0.5       # fraction of width
STOP_LINE_Y_FRAC = 0.85         # fraction of height
SIDE_GREEN_MIN_FRAC = 0.2       # fraction
SIGN_HZ = 10                    # Hz
SIGN_SIDE = "right"             # CONFIRM
SIGN_MIN_AREA_PX = 400          # px²
SIGN_ASPECT_MIN = 0.6           # w / h
SIGN_ASPECT_MAX = 1.6
SIGN_RED_MIN_FRAC = 0.15        # fraction of crop
OCTAGON_CORNERS = (7, 9)        # accepted corner count, inclusive
ARROW_ASYM_MIN = 1.6            # wide end / narrow end
SIGN_VOTES_NEEDED = 3           # frames
DEFAULT_ACTION = "STRAIGHT"     # name of an Action member
STOP_HOLD_S = 3.0               # s

# ---------------------------------------------------------------- speeds and lane control

CRUISE_SPEED = 0.5              # 0-1
APPROACH_SPEED = 0.3
CROSS_SPEED = 0.3
REACQUIRE_SPEED = 0.3
KP_LAT = 4.0                    # steer per m
KH = 0.8                        # steer per rad
KD_YAW = 0.004                  # steer per deg/s
STEER_SMOOTH = 0.5              # 0 = none, near 1 = heavy
STEER_MAX = 0.6                 # steer
REACQUIRE_CONF = 0.6
REACQUIRE_FRAMES = 5
CAM_GAP_WARN_M = 0.03           # m
FUSE_GAP_MIN_CONF = 0.6         # both cameras need this conf before cam_gap_m is reported
SPEED_MIN_SCALE = 0.4           # lowest fraction of base speed from low conf or large heading error
SPEED_HEADING_SLOW_RAD = 0.35   # rad (~20 deg); heading error at which speed reaches SPEED_MIN_SCALE
LANE_CONF_WARN = 0.4            # fused lane conf below this -> LANE_CONF_LOW

# ---------------------------------------------------------------- turns and junction crossing

TURN_SPEED = 0.0                # 0 = pivot; above 0 gives an arc
TURN_ANGLE_DEG = 90             # deg
ADVANCE_AFTER_OPENING_S = 0.4   # s; MEASURED on the course
ADVANCE_AFTER_LINE_S = 0.8      # s; MEASURED on the course
TURN_KP = 0.02                  # steer per deg
TURN_MIN_STEER = 0.25           # steer
TURN_TOL_DEG = 3                # deg
TURN_HANDOFF_DEG = 15           # deg
TURN_HANDOFF_CONF = 0.6         # conf
VISION_TURN_STEER = 0.3         # steer
TURN_TIMEOUT_S = 4.0            # s
HOLD_KP = 0.01                  # steer per deg
CROSS_TIMEOUT_S = 3.0           # s

# ---------------------------------------------------------------- IMU and road yaw

IMU_I2C_ADDR = 0x28
IMU_MAX_RATE_DPS = 300          # deg/s
IMU_MAX_REJECTS = 5             # samples
IMU_STILL_S = 3.0               # s
IMU_DRIFT_MAX_DPS = 0.5         # deg/s; set from tools/imu_drift_test.py
IMU_DRIFT_WARN_DPS = 0.2        # deg/s; set from tools/imu_drift_test.py
IMU_CALIB_WAIT_S = 10           # s
ROAD_YAW_ALPHA = 0.05           # blend per loop
ROAD_YAW_MIN_CONF = 0.7         # conf
ROAD_YAW_MAX_RATE_DPS = 10      # deg/s
ROAD_YAW_STALE_S = 5            # s

# ---------------------------------------------------------------- Sabertooth and motors

SABER_PORT = "/dev/ttyAMA0"     # as in the course Sabertooth example
SABER_BAUD = 9600               # must match the DIP switches
SABER_ADDRESS = 128             # 128-135
SABER_TIMEOUT_MS = 200          # auto-stop; sent as 2 x 100 ms (course example value)
SABER_STARTUP_WAIT_S = 2.0      # s; wait after opening the port (course example value)
SABER_READ_TIMEOUT_S = 0.1      # s; pyserial timeout (course example value)
SABER_HW_DEADBAND = 5           # Sabertooth command 17; stored in the controller (course example value)
LEFT_CHANNEL = 1                # CONFIRM on the bench
RIGHT_CHANNEL = 2               # CONFIRM on the bench
LEFT_INVERT = False             # CONFIRM on the bench
RIGHT_INVERT = False            # CONFIRM on the bench
STEER_GAIN = 1.0
MOTOR_DEADBAND = 0.15           # 0-1; software lift so small commands still move the wheels

# ---------------------------------------------------------------- display and health

UI_ENABLED = True
UI_HZ = 10                      # Hz
UI_IMAGE_EVERY = 3              # loops
DISPLAY_SIZE = None             # CONFIRM: depends on the display model
UI_START_VIEW = "split"
CAM_FPS_MIN = 20                # fps
STATUS_PRINT_S = 1.0            # s; terminal status line interval (until the display exists)
OVERRUN_WINDOW_S = 1.0          # s; overruns counted over this window for LOOP_OVERRUN