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
# The first 2 bands are EXTRAPOLATED from the 5 measured ones (flat-floor
# fit), not calibrated; redo tools/calibrate.py with --top-row to replace them.
FRONT_BANDS = [
    (91, 107, 1.6591, 52, 0.003004),
    (112, 128, 1.3561, 63, 0.002453),
    (133, 149, 1.1471, 75, 0.002073),
    (154, 170, 0.9941, 86, 0.001795),
    (175, 191, 0.8774, 98, 0.001583),
    (196, 212, 0.7854, 110, 0.001415),
    (217, 233, 0.7109, 121, 0.00128),
]

SIDE_ROWS = (40, 80, 120, 160, 200)  # 5 rows spread over the side frame height
SIDE_ROAD_ON_LEFT = None        # MEASURED: True if road appears on the image's left
SIDE_TARGET_PX = None           # MEASURED: edge column with the robot centred on the line
SIDE_M_PER_PX = None            # MEASURED: m/px
SIDE_CAM_X_M = None             # MEASURED: m; + = ahead of the rear axle
MIN_LINE_PX = 30                # pixels per band
LINE_WINDOW_PX = 10             # px (lane size); yellow is only counted this close to its densest column
ROAD_COL_FRAC = 0.5             # a band column counts as road if this fraction of its rows is road
ROAD_GAP_MAX_PX = 8             # gaps in the road this narrow (px, lane size) are bridged
ROAD_LINE_GAP_PX = 25           # px (lane size); wider than the line, so the road span bridges it
FRONT_FIT_RMS_MAX_M = 0.03      # m; fit error at which front lane conf falls to 0
LANE_SEARCH_M = 0.06            # m; yellow is searched for this far either side of the last fit
LANE_SEARCH_MAX_AGE_S = 0.2     # s; an older fit switches the window off (full-width search)
LANE_SEARCH_MIN_CONF = 0.5      # conf a fit needs before it sets the next window
LANE_OUTLIER_M = 0.015          # m; a band point missing the fit by more is dropped
LANE_MIN_KEEP = 3               # band points always kept, however bad
LANE_EDGE_WEIGHT = 0.5          # weight of a road-edge point (dash gap) vs a yellow point
ROAD_WIDTH_M = 0.31             # m, edge to edge; MEASURED (matches the current FRONT_BANDS)

# MEASURED by tools/calib_ui.py (or tools/calibrate.py fit): written there on APPLY.
# FRONT_GROUND_H maps FRONT_LANE_SIZE pixels (lens-corrected when the lens
# values are set) to (forward_m, lateral_m) from the rear axle. Perception
# uses FRONT_BANDS only; the homography is kept for the pre-session check.
FRONT_GROUND_H = None           # MEASURED: 3 x 3 nested list
FRONT_CAM_MATRIX = None         # MEASURED (lens step, optional): 3 x 3, FRONT_SIZE pixels
FRONT_DIST_COEFFS = None        # MEASURED (lens step, optional): OpenCV distortion coefficients

# ---------------------------------------------------------------- calibration (tools/calib_ui.py)

CALIB_DIR = "calibration"       # relative to autonomous/; one folder per session
CALIB_UI_HZ = 15                # Hz; guided screen redraw (live camera view)
CALIB_STILL_PX = 1.5            # px; board corners moving less than this per frame count as still
CALIB_SQUARE_M = 0.025          # m; MEASURED on your print (printers rescale)
CALIB_PATTERN = (9, 6)          # inner corners (across, along); a 10 x 7-square board
CALIB_BAND_COUNT = 5
CALIB_BAND_FILL = 0.75          # fraction of each band slot a band covers
CALIB_MIN_FIT_BOARDS = 4        # board photos needed before a fit is allowed
# Suggested board spots: (nearest inner-corner row from the rear axle, board
# centre right of the centreline tape, role). "fit" photos make the fit; the
# "check" photo is held back to verify it and becomes the taped check spot.
# Slide a board until it is fully in view if needed and enter where it really is.
CALIB_POSITIONS = (
    (0.72, 0.00, "fit"), (0.72, -0.07, "fit"), (0.72, 0.07, "fit"),
    (0.82, 0.00, "fit"), (0.92, 0.00, "fit"), (1.02, 0.00, "fit"),
    (1.12, 0.00, "fit"), (1.12, -0.12, "fit"), (1.12, 0.12, "fit"),
    (0.87, 0.05, "check"),
)
CALIB_BOARD_RMS_WARN_M = 0.003  # m; a board fitting worse than this is flagged
CALIB_BOARD_OFFSET_WARN_M = 0.004   # m; a board's mean miss this large suggests a mis-measured spot
CALIB_HOLDOUT_TOL_M = 0.005     # m; mean error on the held-back photo for a PASS
CALIB_CHECK_TOL_M = 0.005       # m; mean error for the pre-session check to PASS
CALIB_CHECK_MAX_AGE_H = 12      # h; an older PASS shows as "not checked" in the start-up checks
CALIB_LENS_EDGE_RATIO = 2.0     # edge boards fitting this much worse than centre ones -> lens hint
CALIB_CHECK_NEAR_M = None       # m; MEASURED: the taped check spot (set on APPLY)
CALIB_CHECK_LATERAL_M = None    # m; MEASURED: the taped check spot (set on APPLY)
CALIB_LENS_MIN_PHOTOS = 12      # lens step: photos needed
CALIB_LENS_MIN_CELLS = 7        # lens step: of the 3 x 3 image cells, how many corners must reach
CALIB_LENS_NEW_PX = 40          # px; a lens photo must move the board this far from earlier ones
CALIB_LENS_RMS_WARN_PX = 0.8    # px; lens fit error above this: retake blurry or bent photos
CALIB_LENS_NEGLIGIBLE_MM = 1.0  # mm; lens correction moving band points less than this is not worth applying

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
# From the test
YELLOW_H_MIN = 21
YELLOW_H_MAX = 30
YELLOW_S_MIN = 116
YELLOW_V_MIN = 72
ROAD_S_MAX = 100
ROAD_V_T_MIN = 51
ROAD_V_T_MAX = 142

GREEN_CHROM_MIN = 0.40          # G / (R+G+B)
DARK_SUM_MIN = 90               # R+G+B; below = too dark to judge

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

CRUISE_SPEED = 0.2              # 0-1
APPROACH_SPEED = 0.3
CROSS_SPEED = 0.3
REACQUIRE_SPEED = 0.3
KP_LAT = 2.0                    # steer per m
KH = 0.8                        # steer per rad
KD_YAW = 0.004                  # steer per deg/s
STEER_SMOOTH = 0.5              # 0 = none, near 1 = heavy
STEER_MAX = 0.6                 # steer
LANE_STEER_SPEED_RATIO = 0.7    # lane |steer| <= this x speed; below 1 the inner wheel never reverses
LANE_GAIN_SPEED_SCALING = False # scale lateral/heading gains by CRUISE_SPEED / speed; on after tuning in lane_sim
LANE_GAIN_MIN_SPEED = 0.2       # 0-1; speed floor for that scaling
REACQUIRE_CONF = 0.6
REACQUIRE_FRAMES = 5
CAM_GAP_WARN_M = 0.03           # m
FUSE_GAP_MIN_CONF = 0.6         # both cameras need this conf before cam_gap_m is reported
SPEED_MIN_SCALE = 0.4           # lowest fraction of base speed from low conf or large heading error
SPEED_HEADING_SLOW_RAD = 0.35   # rad (~20 deg); heading error at which speed reaches SPEED_MIN_SCALE
LANE_CONF_WARN = 0.4            # fused lane conf below this -> LANE_CONF_LOW
RECOVER_DEG = 20                # deg off the road's direction (IMU) at which the robot pivots back
RECOVER_TIMEOUT_S = 3.0         # s; SAFE_STOP if the pivot back takes longer

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
IMU_YAW_SIGN = -1               # confirmed on the bench: BNO055 heading grows clockwise; -1 makes yaw + = left
IMU_GYRO_SIGN = 1               # confirmed on the bench: gyro z is + counter-clockwise (chip flat, z up)
IMU_MODE_SETTLE_S = 0.05        # s; wait after a mode change (course IMUDevice value)
IMU_JOIN_TIMEOUT_S = 1.0        # s; close() waits this long for the IMU thread to exit
IMU_MAX_RATE_DPS = 300          # deg/s
IMU_MAX_REJECTS = 5             # samples
IMU_DRIFT_SETTLE_S = 0.5        # s; drift_request must be set this long before a window opens
IMU_STILL_S = 2.0               # s; drift window; SETTLE + STILL must be < STOP_HOLD_S
IMU_DRIFT_MAX_DPS = 0.1         # deg/s; bench drift ~0.001 (tools/imu_drift_test.py, motors off and on); higher = bumped
IMU_DRIFT_WARN_DPS = 0.05       # deg/s; above the 0.031 one-step-per-window resolution (1/16 deg / IMU_STILL_S)
IMU_CALIB_WAIT_S = 10           # s
DRIFT_TEST_PRINT_HZ = 10        # Hz; tools/imu_drift_test.py status lines
DRIFT_TEST_SPEED = 0.3          # 0-1; both wheels in --motors mode (robot on a stand)
DRIFT_TEST_SIGN_MIN_DPS = 5.0   # deg/s; both rates above this before their signs are compared
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
LEFT_CHANNEL = 2                # confirmed on the bench
RIGHT_CHANNEL = 1               # confirmed on the bench
LEFT_INVERT = True              # confirmed on the bench
RIGHT_INVERT = True             # confirmed on the bench
STEER_GAIN = 1.0
MOTOR_DEADBAND = 0.15           # 0-1; software lift so small commands still move the wheels

# ---------------------------------------------------------------- steering response (tools/steer_test.py)
# MEASURED by the steering test and written there on APPLY. Per-speed values
# are {speed: value} dicts (speed 0-1), interpolated between speeds.
STEER_YAW_DPS_PER_UNIT = None   # MEASURED: {speed: deg/s per unit steer}; sim and gain choice
STEER_DEADBAND = 0.0            # MEASURED: steer the robot needs before it turns while driving; mix()
STEER_DEADBAND_RAMP = 0.02      # steer over which the deadband lift is blended in (no twitch at 0)
STEER_TRIM = None               # MEASURED: {speed: steer that drives straight}; mix()
MOTOR_DEAD_TIME_S = None        # MEASURED: s from command to first movement; sim
MOTOR_LAG_S = None              # MEASURED: s, first-order time constant of the yaw response; sim
CAMERA_LATENCY_S = None         # MEASURED: s from motion to the lane estimate in the loop; sim
SPEED_MPS_PER_UNIT = None       # MEASURED: forward m/s per unit of speed command; sim, lane filter
STEER_TEST_SPEEDS = (0.3, 0.5, 0.7)        # driving speeds tested (speed 0 = pivot is always tested)
STEER_TEST_STEERS = (0.05, 0.1, 0.2, 0.3)  # steer sizes; each is run left then right
STEER_TEST_LEAD_S = 0.7         # s straight before each steer pair (baseline)
STEER_TEST_HOLD_S = 1.2         # s per steer step
STEER_TEST_SETTLE_S = 0.5       # s stopped between pivot steps
STEER_TEST_STEADY_S = 0.5       # s at the end of a step averaged as its steady yaw rate
STEER_TEST_STRAIGHT_S = 3.0     # s of each straight run (speed and trim)
STEER_TEST_MOVE_DPS = 5.0       # deg/s; a step turning less than this is inside the deadband
STEER_TEST_TURN_MARGIN = 0.05   # steer added to the pivot deadband for TURN_MIN_STEER
STEER_TEST_ASYM_WARN = 0.15     # left/right gain difference (fraction) worth a warning
STEER_TEST_LATENCY_MAX_S = 0.4  # s; camera latency search range
STEER_TEST_IMU_POLL_HZ = 200    # Hz; the test reads every IMU sample at this rate

# ---------------------------------------------------------------- display and health

UI_ENABLED = True
UI_HZ = 10                      # Hz
UI_IMAGE_EVERY = 3              # loops
DISPLAY_SIZE = (800, 480)       # px; official 7" touch display; checked against pygame at startup
UI_VIEW_STALE_S = 1.0           # s; a kept camera view older than this is greyed out with its age
UI_CHART_S = 5.0                # s; strip chart history (lateral error and steer)
UI_CHART_LAT_M = 0.10           # m; lateral range (+/-) of the strip chart and the lane gauge
UI_CHART_GAP_S = 0.3            # s; strip chart breaks the line across a gap this long
UI_CMD_QUEUE_SIZE = 8           # commands; UI -> main queue (START / STOP taps)
UI_STOP_TIMEOUT_S = 1.0         # s; shutdown waits this long for the display process
UI_START_VIEW = "drive"          # "drive", "camera" or "diagnostic"
UI_FONT = "/usr/share/fonts/truetype/piboto/Piboto-Regular.ttf"
UI_FONT_BOLD = "/usr/share/fonts/truetype/piboto/Piboto-Bold.ttf"
CAM_FPS_MIN = 20                # fps
STATUS_PRINT_S = 1.0            # s; terminal status line interval (until the display exists)
LOOP_LOG_ENABLED = True         # one CSV row per control loop (loop_log.py)
LOOP_LOG_DIR = "logs"            # relative to autonomous/; one file per run
OVERRUN_WINDOW_S = 1.0          # s; overruns counted over this window for LOOP_OVERRUN