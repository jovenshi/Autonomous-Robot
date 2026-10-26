# Autonomous Driving System — Architecture & Design

> This file is generated from the live design doc (https://claude.ai/artifact/WqpgJWQFLtZxyHrk99f1nz), which is the source of truth. Edit the live doc and re-export rather than editing this copy.

Oct 9, 2026 · @Jacob Ovenshire

## Overview

The robot straddles the yellow centre line under a 30 Hz control loop: the front camera tracks the line, the right-side camera measures distance to the road edge, and the IMU, re-anchored to the road by the cameras, times the turns. A state machine picks the controller, and a separate display process shows status and debug views.

### Hardware

| Part | Detail | Role in the design |
| --- | --- | --- |
| Front camera | CSI, mounted high on the centreline, facing forward; Picamera2 | Yellow-line offset and heading, junction approach, sign reading |
| Side camera | USB, right side, facing down; sees the black/green road edge only; OpenCV (V4L2) | Distance to the right road edge; spots the road opening on the right at junctions |
| BNO055 IMU | I²C, address 0x28, IMUPLUS mode | Relative yaw for turns and straight crossings; yaw rate for steering damping |
| Drive | Two rear drive wheels on a Sabertooth 2x12 (packetized serial, address 128) powered by an SLA battery; omni wheel at the front | Differential drive; turns pivot about the rear-axle midpoint |
| 7" Pi display | Status and debug now; live annotated camera feeds later | Fed by its own process so it can never delay motor commands |
| Software | Python 3 with OpenCV, NumPy, Picamera2, pyserial, pygame | — |

### Driving geometry

- **Target line:** the robot's centreline sits over the yellow dashed line. The front camera measures that directly; the side camera measures it indirectly, because the right road edge is half a road width from the line.
- **Pivot point:** with only the rear wheels driven and an omni wheel in front, the robot rotates about the midpoint of the rear axle. For a 90° turn the rear axle is placed over the junction centre (where the two yellow lines cross) before pivoting, so the robot finishes centred on the new road's line. Because the robot drives on the centre line, the same advance works for left and right turns.
- **Reference point:** all lateral errors are expressed at the rear axle. Each camera's distance from the axle (`SIDE_CAM_X_M`, and `forward_m` in `FRONT_BANDS`) converts its reading to that point.

### Key design decisions

1. **Two lateral measurements, fused.** The front camera sees the yellow line itself (the true target) but through perspective; the side camera sees the road edge close up with almost no distortion. Fusion learns the side camera's offset from the front camera, weights each by its measured noise, and won't average readings that disagree. Either camera can steer alone; heading falls back to the IMU when the front camera is lost (Side camera in decisions).
2. **The IMU is re-anchored to the road.** On straight road with a confident lane lock, `road_yaw` = IMU yaw + camera heading error is updated continuously. Turn and crossing targets come from `road_yaw`, so IMU drift only matters for the few seconds since the last anchor.
3. **Turns finish on vision.** The IMU brings the robot within `TURN_HANDOFF_DEG` of the target; once the front camera sees the new road's line, REACQUIRE takes over and lane centering removes the remaining error. This is your original idea, made explicit as a state.
4. **Colour tests are built for changing light.** Masks rely on hue, saturation and colour ratios rather than raw brightness, the black-road threshold adapts every frame, and exposure is locked or slowly managed instead of left on auto.
5. **Signs: found by colour and shape, read by HOG + SVM.** White sign-shaped blobs beside the road are the candidates. Each crop is contrast-normalised, described by its edge directions (HOG), and a linear SVM trained on your own recordings picks STOP, LEFT, RIGHT, STRAIGHT or NONE. A STOP also needs red in the crop. Signs are read only near junctions and voted over several frames.
6. **Nothing slow sits in the control path.** Drawing and display run in a separate process fed by a one-slot queue, and the Sabertooth's serial timeout stops the motors if the program stops sending.

### How your coding rules shape the design

- **Under 100 lines:** every function in Modules and functions has one job; the `process_*` and `control_step` orchestrators only call smaller functions.
- **No duplicate helpers or variables:** colour tests live in one module (`color.py`), overlay drawing in one module shared by the display and the replay tool, and every tunable only in `config.py`.
- **Preserve the architecture:** blocks exchange only the data types listed below, so any block can be rewritten without touching its neighbours.
- **Test before merging:** every perception and control function runs on recorded data with the motors stubbed.
- **No guessing:** names here are proposals; check them against any code you already have. Unknowns are collected in Assumptions and open questions.

## Block diagram

*Diagram: System architecture · 3 sensor threads, a 30 Hz control loop, a separate display process (drawn in the live doc).*

Each sensor thread turns raw data into one timestamped result per frame. The control loop fuses them, keeps the IMU anchored to the road, lets the state machine choose lane centering or a turn, and sends left and right commands to the Sabertooth every loop. The display process only ever receives copies, so it cannot slow the loop.

## Sensor errors and how the design handles them

Every sensor is wrong some of the time, so each known error has a specific counter and, where it can be detected, a warning on the display. The two that matter most are IMU drift and lighting.

**IMU drift, layer by layer**

1. The magnetometer is off (IMUPLUS mode), so motor fields cannot move the heading.
2. At standstill (IDLE, STOP\_HOLD) the drift rate is measured and subtracted from then on.
3. On straights, `road_yaw` is re-anchored from the camera, so each turn starts from the road's true direction rather than wherever the gyro has wandered.
4. The turn ends on vision, and REACQUIRE removes whatever error is left. Drift only matters over the few seconds of a turn.

**Lighting, layer by layer**

1. Exposure is not left on auto: it is locked after warm-up, or managed in small steps only on plain road.
2. Colour tests use hue, saturation and colour ratios, which change far less than brightness.
3. The black-road threshold is recomputed every frame, and the white threshold is relative to the frame.
4. A poor frame lowers `conf`, so fusion leans on the other camera and the speed governor slows down.
5. Thresholds are tuned on recordings made under the course's real lighting.

| Sensor | What goes wrong | How the design handles it | Display warning |
| --- | --- | --- | --- |
| IMU | Gyro drift: yaw slowly wanders, so turns over- or under-shoot | Still-drift subtraction, camera-anchored `road_yaw`, turns finished on vision | IMU\_DRIFT\_HIGH, ROAD\_YAW\_STALE |
| IMU | Motor magnetic fields corrupt the magnetometer | IMUPLUS mode: magnetometer not used | — |
| IMU | Spikes or failed reads (I²C errors, vibration) | Plausibility gate holds the last good value; `valid=False` after repeated rejects; vision-only turn fallback; soft mount | IMU\_INVALID |
| IMU | Gyro not yet calibrated at power-up | IDLE waits for `calib_gyro` = 3, or warns after `IMU_CALIB_WAIT_S` | IMU\_NOT\_CALIBRATED |
| Both cameras | Room light changes (windows, lamps, shadows) | Hue, saturation and ratio tests; per-frame darkness threshold; relative white threshold | CAM\_TOO\_DARK, CAM\_TOO\_BRIGHT |
| Both cameras | Auto-exposure lags after a light change, leaving washed-out or dark frames | Exposure locked, or managed only in LANE\_FOLLOW (`exposure_hold` freezes it near junctions); `EXPOSURE_MODE` lets you compare all three on the course | CAM\_TOO\_DARK, CAM\_TOO\_BRIGHT |
| Both cameras | Glare on the mats washes out yellow or looks white | Yellow needs saturation; sign candidates must pass shape, size and side tests; crops contrast-normalised (CLAHE) and read by an SVM trained on crops from varied lighting; multi-frame voting | LANE\_CONF\_LOW |
| Both cameras | Motion blur smears the dashes | Exposure capped at `EXPOSURE_MAX_US`; slower speed near junctions | — |
| Both cameras | Mount knocked, so calibration shifts | The learned side offset absorbs small shifts; an offset stuck at SIDE\_BIAS\_MAX\_M is flagged; re-run `calibrate.py` | CAMS\_DISAGREE |
| Front camera | Gaps between dashes | Road-edge midpoint per band | — |
| Side camera | USB latency and frame buffering | Buffer of one frame, continuous reads, timestamps, `MAX_OBS_AGE_S` | CAM\_FPS\_LOW |
| Side camera | Webcam lowers its frame rate in dim light | Manual exposure through V4L2 controls; fps monitored | CAM\_FPS\_LOW |
| Side camera | The robot's own shadow falls on the grass | Green tested by chromaticity, which changes little in shadow | — |
| Side camera | Road width differs between mats, so the edge-to-line distance is wrong | The learned side offset re-learns the edge-to-line distance on every straight | CAMS\_DISAGREE |
| Drive | Unequal motors, wheel slip, omni-wheel scrub | Closed-loop lane control; turns closed on yaw; deadband compensation | — |
| Drive | Program crash or cable fault leaves the last command running | Sabertooth serial timeout; `shutdown` always stops the motors | MOTOR\_SERIAL\_ERROR |
| Pi | Control loop overruns its period | Overruns counted and shown; profile and move work to a process if needed | LOOP\_OVERRUN |
| Front camera | Glare washes out the line, e.g. driving toward a window | Side lateral plus IMU heading (road\_yaw − yaw) keeps lane following, speed capped at APPROACH\_SPEED; the gate stops a jumping front reading from being averaged in | LANE\_CONF\_LOW, CAMS\_DISAGREE |

## Runtime on the Raspberry Pi 5

Three sensor threads and the control loop share one Python process; the display runs in a second process. The loop always uses the newest result from each sensor and never works through a backlog.

| Thread or process | Work done | Rate | Target time per cycle |
| --- | --- | --- | --- |
| `front_cam` thread | CSI capture, masks, front lane, junction score; sign reader when enabled | 30 fps (signs \~10 Hz) | < 15 ms, < 30 ms with signs |
| `side_cam` thread | USB capture, masks, edge position, opening check | 30 fps | < 5 ms |
| `imu` thread | Read, plausibility gate, unwrap, drift correction | 50 Hz | < 2 ms |
| Main thread (control loop) | Fusion, road yaw, state machine, controllers, Sabertooth output, UI packet | 30 Hz | < 4 ms |
| `ui` process | Draw overlays and status, show them on the 7" display, read taps | \~10 Hz | Its own core; the loop never waits on it |

The times are budgets to check with a profiler on the Pi, not measurements.

### Efficiency rules

- **Capture small.** Side camera at 320×240 in YUYV: uncompressed, so no JPEG smearing at the black/green edge. Front camera at 640×480, with lane and junction work on a 320×240 copy and sign crops taken from the full frame.
- **Work on regions only.** The lower part of the front frame for the lane; the road-side strip for signs.
- **Make masks once.** Colour conversion and masks are computed once per frame and shared by lane, junction and sign code.
- **Vectorised calls only.** OpenCV and NumPy on whole arrays; no per-pixel Python loops.
- **Newest value wins.** Each camera thread writes one result object per frame into its `LatestValue` slot; a newer one overwrites the older one and nothing queues.
- **Keep USB latency down.** Set the capture buffer to one frame (`CAP_PROP_BUFFERSIZE`) and read continuously in the thread, so the newest frame is never stuck behind old ones.
- **Open the webcam by a stable path.** Use `/dev/v4l/by-id/…`. On the Pi 5 the CSI camera stack also creates `/dev/video*` nodes, so an index like 0 may not be the webcam.
- **Exposure off auto.** See Sensor errors. This also stops many webcams from silently lowering their frame rate in dim light.
- **Drop stale data.** The loop ignores any result older than `MAX_OBS_AGE_S`.
- **Isolate the display.** The UI process gets packets through a one-slot queue with `put_nowait`; when it is busy the packet is dropped, so the control loop never blocks on drawing.
- **Threads first, processes if needed.** OpenCV releases Python's GIL during heavy calls, so vision threads run in parallel. If profiling shows contention, move the front camera into its own process behind the same slot interface; nothing else changes.

## Data types passed between blocks

Fourteen small dataclasses, five enums and one `LatestValue` holder are the only things blocks pass to each other. All live in `types.py`; every result carries `t` from `time.monotonic()` and the `seq` number of the frame it came from, so stale data is dropped and overlays always land on the frame they were measured on.

**Sign conventions (fixed once, used everywhere):** lateral error + = rear axle is right of the yellow line; heading error + = robot points right of the road direction; steer + = turn left; yaw + = counter-clockwise (left). Positive errors therefore call for positive steer, and `road_yaw = yaw + heading error` (a robot pointing right of the road has a smaller yaw than the road). Confirm the IMU direction on the bench by turning the robot left by hand.

| Type | Fields (units) | Made by → used by |
| --- | --- | --- |
| `Frame` | `image` (H×W×3 BGR uint8), `t` (s), `seq` (int), `cam_id` ('front' or 'side') | `CameraStream` → perception |
| `FrontMasks` | `hsv`, `yellow`, `road`, `white`, `red` (masks at lane size), `scale` (full frame ÷ lane copy), `mean_v` | `make_front_masks` → front lane, junction, signs |
| `FrontLaneObs` | `lateral_m`, `heading_err_rad`, `conf` (0–1), `centre_pts_px` (band centres), `line_px` (fitted line ends), `t`, `seq` | `process_front_lane` → fusion, overlay |
| `IntersectionObs` | `approach_score` (0–1), `crossing_line_y` (px or None), `t`, `seq` | `process_intersection` → behaviour, overlay |
| `SignObs` | `sign` (SignType), `score` (SVM margin: best class minus runner-up), `bbox` (x, y, w, h in full-frame px), `t`, `seq` | `process_signs` → `SignVoter`, overlay |
| `FrontResult` | `view` (320×240 image), `lane`, `inter`, `sign` (or None), `mean_v`, `t`, `seq` | front thread → front slot (one object per frame) |
| `SideLaneObs` | `offset_m` (at the camera), `boundary_found`, `opening` (bool), `conf` (0–1), `boundary_x_px`, `t`, `seq` | `process_side_lane` → fusion, behaviour, overlay |
| `SideResult` | `view` (320×240 image), `side`, `mean_v`, `t`, `seq` | side thread → side slot |
| `ImuSample` | `yaw_deg` (relative, unwrapped, drift-corrected), `yaw_rate_dps`, `valid` (bool), `calib_gyro` (0–3), `drift_dps`, `t` | `ImuReader` → road yaw, behaviour, controllers |
| `LaneEstimate` | `lateral_err_m` (at the rear axle), `heading_err_rad`, `conf` (0–1), `cam_gap_m` (front minus side before the learned offset, or None), heading\_src ('front', 'imu' or None), `t` | `fuse_lane` → controllers, behaviour, road yaw |
| `RoadYaw` | `yaw_deg`, `t_anchor` (last camera anchor), `valid` (bool) | `RoadYawTracker` → behaviour, display |
| `DriveCmd` | `speed` (0–1), `steer` (−1 to 1) | `drive_for_state` → `mix` |
| `MotorCmd` | `left`, `right` (−1 to 1) | `mix` → `SabertoothDriver` |
| `UiPacket` | `t`, `state`, `stop_reason`, `drive`, `lane`, `road_yaw`, `imu`, `sign_vote`, `fps`, `overruns`, `warnings`, `front` and `side` (results with views, or None) | `make_ui_packet` → display process |

**Enums**

- `SignType`: NONE, STOP, LEFT, RIGHT, STRAIGHT
- `Action`: STOP, LEFT, RIGHT, STRAIGHT
- `State`: IDLE, LANE\_FOLLOW, APPROACH, DECIDE, STOP\_HOLD, TURN, CROSS, REACQUIRE, SAFE\_STOP
- `Warning`: IMU\_INVALID, IMU\_NOT\_CALIBRATED, IMU\_DRIFT\_HIGH, ROAD\_YAW\_STALE, CAM\_FPS\_LOW, CAM\_TOO\_DARK, CAM\_TOO\_BRIGHT, CAMS\_DISAGREE, LANE\_CONF\_LOW, LOOP\_OVERRUN, MOTOR\_SERIAL\_ERROR
- `UiCommand`: START, STOP, NEXT\_VIEW

**`LatestValue[T]`** holds one item behind a lock: `set(value)` replaces it, `get()` returns it or None. Slots: `front`, `side`, `imu`. Two `threading.Event` flags carry decisions back to the camera threads: `signs_enabled` (set in APPROACH) and `exposure_hold` (set outside LANE\_FOLLOW so managed exposure never changes mid-junction).

## Modules and functions

Six packages (hal, perception, estimation, behaviour, control, ui) plus tools and tests. Each function has one job and a fixed signature; the orchestrators (`process_*`, `control_step`, `run_ui`) only call the small functions beneath them.

```
robot/
├── main.py                build_system, control_step, run, shutdown
├── config.py              every tunable (Configuration variables)
├── types.py               dataclasses, enums, LatestValue
├── hal/
│   ├── camera.py          CameraStream, CsiSource, UsbSource
│   ├── exposure.py        lock or manage exposure for either source
│   ├── imu.py             ImuReader
│   └── sabertooth.py      SabertoothDriver
├── perception/
│   ├── color.py           the only place colour tests are applied
│   ├── pipeline.py        process_front_frame, process_side_frame
│   ├── front_lane.py
│   ├── side_lane.py
│   ├── intersection.py
│   └── signs.py
├── estimation/
│   ├── lane_fusion.py
│   ├── road_yaw.py        RoadYawTracker
│   └── health.py          check_health
├── behaviour/
│   ├── sign_voter.py
│   └── state_machine.py
├── control/
│   ├── lane_controller.py
│   ├── turn_controller.py
│   ├── speed_governor.py
│   ├── command.py
│   └── mixer.py
├── ui/
│   ├── link.py            main-process side: start_ui, make_ui_packet, send, read_commands
│   ├── process.py         run_ui (display process entry point)
│   ├── screen.py          views, status panel, touch
│   └── overlay.py         draw_front_overlay, draw_side_overlay (also used by replay)
├── models/                sign_svm.npz, written by tools/train_signs.py
├── tools/                 record, replay, tune_colors, calibrate, imu_drift_test, collect_signs, train_signs
└── tests/
```

### hal/camera.py

`CameraStream(source, process_fn, slot, name)` runs one camera thread. `CsiSource(cam_num, size)` wraps Picamera2 and `UsbSource(device_path, size, fourcc)` wraps `cv2.VideoCapture` with the V4L2 backend; both expose the same five methods, so `CameraStream` never knows which it has.

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `CameraStream.start()` | — | None | Open the source, start the thread |
| `CameraStream._loop()` | — | None | Read → stamp `t` and `seq` → `process_fn(frame)` → write the result to its slot; count fps |
| `CameraStream.fps()` | — | float | Measured frame rate, for health and the display |
| `source.open()` | — | None | CSI: configure and start Picamera2. USB: open by path, set size, YUYV, buffer size 1 |
| `source.read()` | — | image or None | One BGR frame |
| `source.get_exposure()` | — | (exposure\_us, gain) | CSI: from frame metadata. USB: from the V4L2 controls named in config |
| `source.set_exposure(exposure_us, gain)` | numbers | None | Manual exposure with auto-exposure and auto white balance off |
| `source.close()` | — | None | Release the camera |

### hal/exposure.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `lock_after_warmup(source, cfg)` | source | None | Let auto-exposure settle for `CAM_WARMUP_S` on the course, then fix those values, capped at `EXPOSURE_MAX_US` to limit motion blur |
| `manage_step(source, mean_v, hold, cfg)` | source, ROI brightness, hold flag | None | Managed mode only: every `EXPOSURE_CHECK_FRAMES`, nudge exposure by `EXPOSURE_STEP` toward `EXPOSURE_TARGET_V`; does nothing while `hold` is set |

### hal/imu.py — `ImuReader(i2c_bus, address)`

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `start()` | — | None | Put the BNO055 in IMUPLUS mode (magnetometer off), take the first yaw as 0, start the 50 Hz thread |
| `read_raw()` | — | (heading\_deg, gyro\_z\_dps) | One read from the sensor |
| `_to_robot_yaw(heading_deg, prev_yaw)` | raw heading, last yaw | float (deg) | Flip to counter-clockwise-positive and unwrap across 0/360 |
| `_plausible(new_yaw, prev_yaw, dt)` | two yaws, seconds | bool | False if the implied rate exceeds `IMU_MAX_RATE_DPS` |
| `measure_drift(seconds)` | duration | float or None | Only while the motors are stopped (IDLE, STOP\_HOLD): yaw change ÷ time. Stored if below `IMU_DRIFT_MAX_DPS`, otherwise rejected as "not really still" |
| `_loop()` | — | None | Read → convert → subtract stored drift × elapsed → gate → write `ImuSample`; `valid=False` after `IMU_MAX_REJECTS` rejects in a row |

The BNO055 relies on I²C clock stretching, which has caused read errors on Raspberry Pi boards; if reads glitch, lower the I²C clock or use the sensor's UART mode (only `read_raw()` changes). Mount it away from the motors and Sabertooth, on a soft mount to damp omni-wheel vibration.

### hal/sabertooth.py — `SabertoothDriver(port, baud, address)`

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `open()` | — | None | Open the serial port (pyserial), no bauding byte (the 2x12 keeps a stored baud rate), set the serial timeout (command 14, in 100 ms units) |
| `set(cmd)` | `MotorCmd` | None | Apply deadband and invert flags, send one packet per motor |
| `stop()` | — | None | Send speed 0 to both motors; called by `shutdown` and SAFE\_STOP |
| `_side_packet(x, channel)` | −1 to 1, 1 or 2 | bytes | Forward command (0 for motor 1, 4 for motor 2) or reverse (1 or 5) with value round(abs(x) × 127) |
| `_packet(command, value)` | ints | bytes | address, command, value, (address + command + value) & 0x7F |
| `_with_deadband(x)` | −1 to 1 | float | Lift small non-zero commands above `MOTOR_DEADBAND` so the wheels actually move |

Packetized serial sends 4-byte packets (address 128–135, command, value 0–127, 7-bit checksum). With the timeout set, `control_step` must send every loop, even when nothing changed, or the Sabertooth stops the motors. The serial timeout is off at power-up and is not stored, so open() must send it on every run.

**Confirmed setup:** Sabertooth 2x12 in packetized serial mode at address 128, powered from an SLA battery, with the Pi's TX wired to S1 and the Pi's ground to 0V. It runs at 9600 baud from the factory; only command 15 changes that, and the new rate is stored. In this mode S2 is an active-low emergency stop, pulled high internally, so a button from S2 to 0V stops both motors. S1 expects 0–5 V TTL and the Pi drives 3.3 V, so confirm reliable reception on the bench.

| DIP switch | Required | Meaning |
| --- | --- | --- |
| 1 | OFF | Packetized serial needs 1 and 2 both OFF |
| 2 | OFF | Packetized serial needs 1 and 2 both OFF |
| 3 | ON | Lithium cutoff off, for the SLA battery (OFF enables lithium cutoff) |
| 4 | ON | Address 128 needs 4, 5 and 6 all ON |
| 5 | ON | Address 128 needs 4, 5 and 6 all ON |
| 6 | ON | Address 128 needs 4, 5 and 6 all ON |

Reported setting: on, on, on, off, off, off. Read as switches 1 to 6, that is analog mode (1 and 2 ON) and the driver will ignore serial commands. Read as 6 to 1, it is packetized serial at address 128 with lithium cutoff enabled. Check the numbers printed on the switch block.

### perception/color.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `mask_yellow(hsv, cfg)` | HSV image | mask | Hue range plus minimum saturation; the brightness bound is kept loose |
| `mask_green(img, cfg)` | BGR image | mask | Green chromaticity G ÷ (R+G+B) above `GREEN_CHROM_MIN`, ignoring pixels too dark to judge. Ratios barely change with brightness or the robot's own shadow |
| `mask_road(hsv, roi, cfg)` | HSV, region | mask | Dark and unsaturated; the darkness threshold comes from Otsu on the region's V channel each frame, clamped to `ROAD_V_T_MIN`–`ROAD_V_T_MAX` |
| `mask_white(hsv, cfg)` | HSV | mask | Low saturation and brightness above `WHITE_V_REL` × the frame's bright percentile (relative, not absolute) |
| `mask_red(hsv, cfg)` | HSV | mask | Two hue ranges (red wraps around 0/179) plus minimum saturation |
| `roi_brightness(hsv, roi)` | HSV, region | float | Mean V, for exposure management and health |
| `make_front_masks(frame, cfg)` | `Frame` | `FrontMasks` | Downscale, convert once, build all front masks |
| `make_side_masks(frame, cfg)` | `Frame` | (road, green) | Same for the side camera |

### perception/pipeline.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `process_front_frame(frame, signs_enabled, sign_model, cfg, tracker=None)` | `Frame`, flag | `FrontResult` | Front thread entry: masks once, lane, junction, and signs only when the flag is set and `SIGN_HZ` allows |
| `process_side_frame(frame, cfg)` | `Frame` | `SideResult` | Side thread entry: masks, edge, opening |

### perception/front\_lane.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `find_yellow_x(yellow_band)` | mask strip | float or None | Column centroid of yellow pixels; None if fewer than `MIN_LINE_PX` |
| `find_road_edges(road_band)` | mask strip | (left\_x, right\_x), either may be None | Outer extent of the black road in the band |
| `centre_in_band(yellow_x, left_x, right_x, half_road_px)` | x values, band width | float or None | Yellow x if found; else the midpoint of both edges; else one edge ± half the road width. Covers the gaps between dashes |
| `band_to_ground(x_px, band, img_cx)` | x, band calibration | (forward\_m, lateral\_m) | Pixel → metres from the rear axle, using `FRONT_BANDS` |
| `fit_centre_line(points_m)` | list of (forward, lateral) | (slope, intercept) or None | Least-squares line; None with fewer than 2 points |
| `fit_robust(points_m, weights, cfg)` | band points, weights | (fit or None, kept indices) | Weighted line fit (road-edge points count LANE\_EDGE\_WEIGHT). Drops the point farthest from a fit of the OTHER points while it misses by more than LANE\_OUTLIER\_M, keeping at least LANE\_MIN\_KEEP; judging each point without itself stops an outlier in the nearest or farthest band from hiding |
| `LaneTracker` | fit, conf, t | window\_fit(t) → fit or None | The last fit with conf ≥ LANE\_SEARCH\_MIN\_CONF, used as the next frame's search window (±LANE\_SEARCH\_M per band) while it is younger than LANE\_SEARCH\_MAX\_AGE\_S. One per camera thread (main.py) and one per replay |
| `line_to_errors(fit)` | line | (lateral\_m, heading\_err\_rad) | Lateral at forward = 0 (the rear axle) and heading from the slope, signs per the conventions |
| `process_front_lane(masks, cfg, t, seq, tracker=None)` | `FrontMasks` | `FrontLaneObs` | Runs the above over every band (with a tracker, yellow is searched for only near the last fit), then fit\_robust; `conf` from bands found and fit residual; keeps band points in pixels for overlays |

**Stray yellow.** Two guards keep a crossing line, a sign or a yellow patch beside the road from bending the fit. The search window ignores yellow more than `LANE_SEARCH_M` from where the last confident fit puts the line, so a crossing line's pixels inside the window average out on the line itself; if the line jumps outside the window, that band falls back to the road edges, and once the last fit is older than `LANE_SEARCH_MAX_AGE_S` the whole width is searched again. The outlier-resistant fit then drops a band point that disagrees with the others; a dropped band lowers `conf` like a missing one.

### perception/side\_lane.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `find_boundary_x(road, green, rows)` | masks, `SIDE_ROWS` | float or None | Median road → green transition column across the sample rows (which side is road is set by `SIDE_ROAD_ON_LEFT`) |
| `edge_to_offset(x_px, cfg)` | x | float (m) | (x − `SIDE_TARGET_PX`) × `SIDE_M_PER_PX`: the robot's offset from the yellow line, measured at the camera |
| `is_opening(green, cfg)` | mask | bool | Green fraction below `SIDE_GREEN_MIN_FRAC`: a road leaves to the right here |
| `process_side_lane(frame, cfg)` | `Frame` | `SideLaneObs` | Masks → boundary → offset; `conf` from how many rows found the edge |

Passing a right-hand branch, the edge disappears, so `conf` drops and fusion falls back to the front camera automatically.

### perception/intersection.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `crossing_line_row(yellow_mask, cfg)` | mask | int or None | Lowest row where yellow spans more than `CROSS_LINE_MIN_FRAC` of the width: the crossing road's centre line |
| `road_widening(road_mask, cfg)` | mask | float (0–1) | How far black road extends past its normal width in the look-ahead bands, either side |
| `process_intersection(masks, cfg, t, seq)` | `FrontMasks` | `IntersectionObs` | `approach_score` from both cues; `crossing_line_y` is the fallback position fix |

### perception/signs.py

Candidates are still found by colour and shape; a linear SVM on HOG features reads them. HOG records edge directions and normalises each small block, so overall brightness and contrast changes mostly cancel, and CLAHE evens out shadows and glare first. Reading one crop should take about 1 ms on the Pi 5 (an estimate; confirm when profiling).

The SVM has five classes: STOP, LEFT, RIGHT, STRAIGHT and NONE. NONE is trained on false candidates (glare, mat seams, white objects off the course), so rejecting them is learned, not only thresholded.

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `find_sign_candidates(white_mask, cfg)` | mask | list of bbox | Contours → keep four-cornered blobs of sign size and aspect on `SIGN_SIDE` of the road |
| `pick_sign(candidates)` | list of bbox | bbox or None | Largest candidate: the nearest sign on our side, not one meant for another approach |
| `crop_sign(frame, bbox, scale)` | full `Frame`, bbox | image | Crop from the full-resolution frame |
| `normalise_crop(crop, cfg)` | crop | grayscale square | Grayscale, resize to `SIGN_CROP_PX` × `SIGN_CROP_PX`, then CLAHE (`SIGN_CLAHE_CLIP`, `SIGN_CLAHE_TILES`). Training calls this same function, so the Pi and the laptop see identical crops |
| `load_sign_model(path, cfg)` | `SIGN_MODEL_PATH` | `SignModel` | Reads `sign_svm.npz`: weights (one row per class), biases, class names and the HOG settings used in training, and builds the `cv2.HOGDescriptor` from them. Called once by `build_system`; a missing file, or a crop size that differs from `SIGN_CROP_PX`, stops start-up with a clear error |
| `hog_features(gray, model)` | normalised crop | float32 vector | `model.hog.compute(gray)`: 1,764 values at the default settings (64 px window, 16 px blocks, 8 px stride and cells, 9 bins) |
| `classify_sign(features, model)` | vector | (SignType, score) | Class scores = weights · features + biases, one NumPy product, so the Pi needs no scikit-learn. The best class wins; score = best minus runner-up |
| `red_check(crop_hsv, cfg)` | crop | bool | Red fraction ≥ `SIGN_RED_MIN_FRAC` |
| `process_signs(frame, masks, model, cfg)` | `Frame`, `FrontMasks`, `SignModel` | `SignObs` | Candidates → pick → crop → normalise → HOG → classify. NONE when score < `SIGN_MARGIN_MIN`, or when the SVM says STOP but `red_check` fails |

`SignModel` is a small dataclass in `types.py` (`hog`, `weights`, `biases`, `classes`). `build_system` loads it and passes it to `process_front_frame`.

### Training the sign classifier

The model is trained on the laptop and copied to the Pi as one file. Retrain whenever the signs, the lighting or the front camera mount change.

1. **Record.** Drive past every sign type with `tools/record.py` under each lighting condition: room lights, daylight, lamps, your own shadow on the sign. Include several distances and slight angles.
2. **Collect.** `tools/collect_signs.py <recording>` runs `find_sign_candidates`, `pick_sign`, `crop_sign` and `normalise_crop` over every frame and saves each crop as a PNG in `signs_data/unsorted/`, named by recording and frame number.
3. **Sort.** Move each crop into `stop/`, `left/`, `right/`, `straight/` or `none/`. Aim for 50–100 per sign class and at least as many in `none/`; every false candidate belongs there.
4. **Train.** `tools/train_signs.py signs_data/` augments each crop (brightness and contrast changes, rotation within ±8°, scale ±10%, small shifts, slight blur), computes HOG and fits a scikit-learn `LinearSVC`. It never flips crops horizontally: a flipped LEFT is a RIGHT.
5. **Check.** Whole recordings are held out for testing, at least one per lighting condition, because neighbouring frames are near-copies. The tool prints accuracy and a confusion matrix on them, suggests `SIGN_MARGIN_MIN`, and writes `models/sign_svm.npz`.
6. **Deploy.** Copy the file to the Pi and replay a recording with `tools/replay.py` to confirm the sign box and label.

If held-out errors still cluster in glare or shadow with more data, the same sorted crops can train a small CNN; only `hog_features` and `classify_sign` would change.

### estimation/

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `is_fresh(obj, now, cfg)` | any result | bool | `t` within `MAX_OBS_AGE_S` |
| `side_to_axle(offset_m, heading_err_rad, cfg)` | side offset, heading | float (m) | Offset − `SIDE_CAM_X_M` × sin(heading): moves the side reading to the rear axle |
| `fuse_lane(front, side, imu, road_yaw, side_bias, now, cfg)` | `FrontResult`, `SideResult` (either may be None), latest IMU sample, RoadYaw, SideBias | `LaneEstimate` | Lateral error at the axle: side reading plus learned offset, gated on disagreement, weighted by conf ÷ σ². Heading from the front camera, else road\_yaw − yaw (heading\_src). Raw `cam_gap_m` when both are good |
| `LaneFilter.update(front, side, imu, speed_cmd, side_bias, road_yaw, now)` | FrontResult, SideResult (either may be None), latest IMU sample, commanded speed, SideBias, RoadYaw | `LaneEstimate` | Replaces fuse\_lane once built. Predicts lateral and heading from the IMU yaw rate and speed to each new reading's frame time, gates the reading, corrects with it (noise σ² ÷ conf), then outputs the estimate moved forward to now. Coasts up to LANE\_PREDICT\_MAX\_S with no accepted reading |
| `LaneFilter.reset()` | — | None | Called on entering LANE\_FOLLOW or REACQUIRE and after TURN; the filter restarts from the next front reading, or a side reading with heading road\_yaw − yaw |
| `SideBias.update(raw_gap_m, state, heading_err_rad)` | front − side gap before correction, State, heading | None | In LANE\_FOLLOW, with both conf ≥ FUSE\_GAP\_MIN\_CONF and \|heading\| ≤ SIDE\_BIAS\_MAX\_HEADING\_RAD: blend the gap in by SIDE\_BIAS\_ALPHA, clamped to ±SIDE\_BIAS\_MAX\_M |
| `SideBias.get()` | — | float (m) | The learned side offset, added to the side reading in fuse\_lane; starts at 0 each run |
| `RoadYawTracker.update(imu, lane, state, now)` | IMU, estimate, state | None | In LANE\_FOLLOW or APPROACH, with a front-camera heading (heading\_src = front), lane `conf` ≥ `ROAD_YAW_MIN_CONF` and yaw rate under `ROAD_YAW_MAX_RATE_DPS`: blend yaw + heading error into `road_yaw` by `ROAD_YAW_ALPHA` |
| `RoadYawTracker.after_turn(direction)` | +1 left, −1 right | None | New road direction = old ± 90°, used until the camera re-anchors it |
| `RoadYawTracker.get()` | — | `RoadYaw` | Value, last anchor time, valid |
| `check_health(front, side, imu, lane, road_yaw, fps, overruns, cfg)` | latest everything | list of Warning | One small test per warning, each in its own helper. CAMS\_DISAGREE also fires when the side offset sits at SIDE\_BIAS\_MAX\_M, or after FUSE\_GATE\_WARN\_COUNT gate rejections in a row |

### behaviour/sign\_voter.py — `SignVoter(cfg)`

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `add(obs)` | `SignObs` | None | Count consecutive identical signs; ignores a result already counted (same `seq`) |
| `decided()` | — | SignType or None | The sign once seen `SIGN_VOTES_NEEDED` times in a row |
| `reset()` | — | None | Clear after each junction |

### behaviour/state\_machine.py — `Behaviour(cfg)`

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `step(front, side, lane, road_yaw, sign, imu, commands, now)` | latest results | State | Commands first (START, STOP), then safety, then the current state's handler |
| `_check_safety(lane, imu, now)` | estimate, IMU | reason or None | Lane lost longer than `LANE_LOST_TIMEOUT_S`, data stale, or a motor serial error → SAFE\_STOP with the reason |
| `_on_<state>(…)` | per state | next State | One small handler per state (Behaviour state machine section) |
| `choose_action(sign, stop_done, cfg)` | SignType or None, bool | Action | Sign → action; a STOP already served or no sign → `DEFAULT_ACTION` |
| `_enter(state, now)` | new state | None | Record entry time and the position-fix cue; set `yaw_target_deg` from `road_yaw` for TURN (± 90°) and CROSS; set or clear `signs_enabled` and `exposure_hold`; start a drift measurement in IDLE and STOP\_HOLD |

### control/

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `LaneController.update(est, yaw_rate_dps, dt, speed)` | `LaneEstimate`, IMU rate | steer | `KP_LAT`·lateral + `KH`·heading − `KD_YAW`·yaw rate, with the first two terms scaled by CRUISE\_SPEED ÷ max(speed, LANE\_GAIN\_MIN\_SPEED) when LANE\_GAIN\_SPEED\_SCALING is on (off by default); smoothed, clamped to `STEER_MAX` |
| `LaneController.reset()` | — | None | Clear the smoothing filter when a lane state starts |
| `turn_steer(yaw_deg, target_deg, cfg)` | yaw, target | steer | P on yaw error, clamped, never below `TURN_MIN_STEER` so it doesn't stall |
| `turn_done(yaw_deg, target_deg, front_conf, cfg)` | yaw, target, front lane conf | bool | Error under `TURN_TOL_DEG`, or under `TURN_HANDOFF_DEG` once the front camera sees the new line with `conf` ≥ `TURN_HANDOFF_CONF` |
| `vision_turn_steer(front, direction, cfg)` | `FrontResult` | (steer, done) | Fallback if the IMU goes invalid mid-turn: pivot at `VISION_TURN_STEER` until the front camera sees the line with small heading error |
| `hold_steer(yaw_deg, target_deg, cfg)` | yaw, target | steer | Gentler P (`HOLD_KP`) to drive straight on `road_yaw` across a junction |
| `target_speed(state, est, cfg)` | State, estimate | 0–1 | Per-state base speed, scaled down by low `conf`, large heading error, or the cameras disagreeing by more than CAM\_GAP\_WARN\_M (after the learned offset). Capped at APPROACH\_SPEED while heading\_src is imu |
| `drive_for_state(state, est, imu, behaviour, lane_ctrl, front, cfg)` | all of the above | `DriveCmd` | Picks the steering source for the state (lane, hold, turn, vision turn or none) plus the speed |
| `mix(cmd, cfg)` | `DriveCmd` | `MotorCmd` | left = v − `STEER_GAIN`·steer, right = v + `STEER_GAIN`·steer, where steer first gets STEER\_TRIM for the current speed and, if non-zero, a lift of STEER\_DEADBAND; rescaled to ±1; speed 0 with steer pivots about the rear axle |

### ui/

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `start_ui(cfg)` | config | (process, packet\_q, cmd\_q) | Start the display process (multiprocessing, spawn) with a one-slot packet queue and a small command queue |
| `make_ui_packet(sys, include_views)` | system | `UiPacket` | Numbers every loop; camera views only every `UI_IMAGE_EVERY` loops |
| `send(packet_q, packet)` | queue, packet | None | `put_nowait`; when full, drop the packet |
| `read_commands(cmd_q)` | queue | list of UiCommand | Non-blocking |
| `run_ui(packet_q, cmd_q, cfg)` | queues | — | Display process loop: newest packet → `Screen.draw` → taps → wait for `UI_HZ` |
| `Screen.draw(packet, view)` | packet, view | None | Compose the chosen view at `DISPLAY_SIZE` |
| `draw_status(img, packet)` | image, packet | None | State, speeds, errors, IMU, sign vote, fps, warnings |
| `draw_front_overlay(img, front)` | image, `FrontResult` | None | Band points, fitted line, crossing line, sign box and label |
| `draw_side_overlay(img, side)` | image, `SideResult` | None | Detected edge column, target column, opening flag |
| `Screen.handle_input(events)` | pygame events | UiCommand or None | Tap zones and keys |

### main.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| `build_system(cfg)` | config | `System` | Create slots, flags, sources, camera streams, IMU, Sabertooth, trackers, controllers, behaviour, display process |
| `control_step(sys, now)` | system, time | None | Read slots → `fuse_lane` → `road_yaw.update` → `voter.add` → `check_health` → `behaviour.step` → `drive_for_state` → `mix` → `saber.set` (every loop) → UI packet |
| `run(sys)` | system | None | Fixed-rate loop at `CONTROL_HZ`; counts overruns |
| `shutdown(sys)` | system | None | `saber.stop()` first, then the display process and threads; always runs (try/finally, Ctrl-C) |

## Behaviour state machine

Nine states decide who steers and how fast. Every junction exits through REACQUIRE, where the lane controller absorbs leftover turn error, and turn and crossing targets come from the camera-anchored `road_yaw` rather than the raw IMU heading.

*Diagram: Behaviour state machine · 9 states, SAFE_STOP reachable from all (drawn in the live doc).*

DECIDE runs for a single step, so the robot only stops when a STOP sign was voted. For a turn, the advance after the position fix places the rear axle over the junction centre before the pivot; both advance times are measured on the course.

| State | Steering | Speed | Leaves when → next |
| --- | --- | --- | --- |
| IDLE | None | 0 | Measures IMU drift while still. START (tap or key) once `calib_gyro` = 3, or after `IMU_CALIB_WAIT_S` with a warning → LANE\_FOLLOW |
| LANE\_FOLLOW | Lane controller | `CRUISE_SPEED` | `approach_score` ≥ `APPROACH_ON` → APPROACH. `road_yaw` re-anchored continuously; managed exposure allowed |
| APPROACH | Lane controller | `APPROACH_SPEED` | Position fix → DECIDE: the right-side opening holds for SIDE\_OPENING\_FRAMES in a row (exact, preferred; ignored outside APPROACH), or the crossing line passes `STOP_LINE_Y_FRAC` (fallback when there is no right branch). `approach_score` < `APPROACH_OFF` → LANE\_FOLLOW. Entry sets `signs_enabled` and `exposure_hold` |
| DECIDE | — | — | Same step: `choose_action(voter.decided(), stop_done)` → STOP\_HOLD, TURN or CROSS; records which cue fired and whether an opening was seen. RIGHT with no opening seen → SAFE\_STOP (proposed) |
| STOP\_HOLD | None | 0 | `STOP_HOLD_S` elapsed → DECIDE with `stop_done` set. Re-measures IMU drift while still |
| TURN | Advance: `hold_steer` on `road_yaw`. Rotate: `turn_steer` to `road_yaw` ± 90° | Advance: `APPROACH_SPEED` for `ADVANCE_AFTER_OPENING_S` or `ADVANCE_AFTER_LINE_S`, by cue. Rotate: `TURN_SPEED` (0 = pivot) | `turn_done` → REACQUIRE, then `road_yaw.after_turn`. IMU invalid → finish with `vision_turn_steer`. `TURN_TIMEOUT_S` → SAFE\_STOP |
| CROSS | `hold_steer` on `road_yaw` | `CROSS_SPEED` | Right edge back after the opening for SIDE\_EDGE\_BACK\_FRAMES in a row (preferred), or front lane `conf` high past the junction → REACQUIRE. `CROSS_TIMEOUT_S` → REACQUIRE |
| REACQUIRE | Lane controller | `REACQUIRE_SPEED` | `conf` ≥ `REACQUIRE_CONF` for `REACQUIRE_FRAMES` → LANE\_FOLLOW; voter reset, `signs_enabled` and `exposure_hold` cleared |
| SAFE\_STOP | None | 0 | Reason shown on the display; manual START → IDLE |

## Side camera in decisions

The side camera gives the steadiest lateral reading, keeps the robot driving when the front camera loses the line, and supplies the exact moment at junctions. Today the robot drives on the front camera and IMU only; `fuse_lane` already accepts a side reading, so each job below is added without changing the data flow.

### Who decides what

| Decision | Front camera | Side camera | IMU | Rule |
| --- | --- | --- | --- | --- |
| Lateral error (steering) | Absolute reference; teaches the side camera its offset | Main reading on straights once its offset is learned | — | Bias-corrected, noise-weighted average, gated on disagreement |
| Heading error | Primary | Optional edge slope (later) | `road_yaw − yaw` when the front camera is lost | Front first, IMU fallback |
| Junction ahead | Arms: `approach_score ≥ APPROACH_ON` | — | — | Arming alone never moves the robot |
| Position fix | Crossing line (fallback) | Opening for `SIDE_OPENING_FRAMES` in a row (preferred) | — | Opening counts only while armed |
| Right-turn check | Sign vote says RIGHT | Was an opening seen? | — | RIGHT with no opening → SAFE\_STOP (proposed) |
| CROSS exit | Lane `conf` high past the junction (fallback) | Edge back for `SIDE_EDGE_BACK_FRAMES` in a row (preferred) | — | Side cue first; it can't lock onto the crossing road's line |
| Speed | Low `conf`, large heading error | Cameras disagree | — | Any of these scales speed down |
| Health | — | Bias at its limit, or repeated gate rejections | — | `CAMS_DISAGREE` |

### Fusing lateral error

The front camera sees the yellow line itself, so it is right on average but noisy at the axle. The side camera is steady but measures the line through the road width, so it can carry a fixed offset. Fusion uses the front camera to remove that offset and the side camera to remove the noise.

1. **Move to the axle.** `side_to_axle` as today.
2. **Add the learned offset.** `SideBias` tracks the raw gap (front − side, before correction) and adds it to the side reading. It learns only in LANE\_FOLLOW, with both `conf ≥ FUSE_GAP_MIN_CONF` and |heading| ≤ `SIDE_BIAS_MAX_HEADING_RAD`, by `SIDE_BIAS_ALPHA` per loop. It is clamped to ±`SIDE_BIAS_MAX_M`; at the clamp it raises `CAMS_DISAGREE`, since the mount has moved or the calibration is wrong. This also absorbs road-width differences between mats.
3. **Gate.** If the corrected readings differ by more than `FUSE_GATE_M`, don't average them. Keep the one nearer last loop's `lateral_err_m`, since a sudden jump is usually the faulty reading (glare at the front, a mat gap at the side). Repeated rejections raise `CAMS_DISAGREE`.
4. **Weight by measured noise.** Each reading's weight is its `conf` divided by its noise variance, with `FRONT_LAT_SIGMA_M` and `SIDE_LAT_SIGMA_M` measured from straight recordings. Raw `conf` alone isn't comparable between the cameras: they compute it differently.
5. **Report the raw gap.** `cam_gap_m` stays front − side before correction, so the display and `SideBias` see the true disagreement.

```latex
lateral = \frac{w_f \, x_f + w_s \, (x_s + b)}{w_f + w_s}, \quad w = \frac{conf}{\sigma^2}
```

Here x is each camera's lateral at the axle and b is the learned side offset.

This average, the gate and the dropout rules below are the first cut. The [lane filter](#msgdh50esk9.84409) replaces them once built, keeping the same σ values and `SideBias`.

### When a camera drops out

1. **Both cameras good:** fused as above; heading from the front camera.
2. **Front lost** (glare, washed-out line): side lateral plus heading = `road_yaw − yaw`, if `road_yaw` is valid, anchored within `ROAD_YAW_STALE_S`, and the IMU sample is fresh. `heading_src` = `imu` and `target_speed` caps speed at `APPROACH_SPEED`. A downward-facing camera is far less exposed to window glare, so this covers the failure seen near the window.
3. **Side lost** (an opening, a gap in the green): front only, which is today's behaviour.
4. **Both lost:** the lane-lost timer runs, and SAFE\_STOP follows after `LANE_LOST_TIMEOUT_S`.

`RoadYawTracker` anchors only when `heading_src` is `front`. Anchoring from an IMU-derived heading would feed `road_yaw` back into itself.

### Lane filter (replaces the average once built)

With the side camera in, fusion becomes a two-state lane filter, `LaneFilter` in `estimation/lane_filter.py`. It takes each reading from the sensor that is good at it: heading from the front camera, position from the side camera, turning between frames from the gyro. It replaces the weighted average, the gate and the dropout rules above; `SideBias` stays and corrects the side reading before it reaches the filter.

**What it tracks.** Lateral error at the rear axle `y` (m) and heading error `θ` (rad), signs as in Data types, plus a 2×2 uncertainty for the pair. The state is kept at the time of the last reading used; the output is moved forward to `now` each loop.

**Each loop:**

1. **Predict from the IMU.** Move the state forward to each new reading's frame time `t`, then to `now`. No camera is needed for this step. Uncertainty grows by `LANE_Q_LAT` and `LANE_Q_HEAD` per second.
2. **Correct with each fresh reading, oldest first.** The front camera measures `y` and `θ`; the side camera measures `y + SIDE_CAM_X_M · sin θ`. Each reading's noise variance is σ² ÷ conf, using `FRONT_LAT_SIGMA_M`, `FRONT_HEAD_SIGMA_RAD` and `SIDE_LAT_SIGMA_M`. Readings with conf below `LANE_CONF_FLOOR` are skipped.
3. **Gate.** A reading too far from the prediction (normalised error above `LANE_GATE_FRONT` or `LANE_GATE_SIDE`) is rejected and counted. If every reading is rejected for `LANE_REJECT_RESET_N` in a row, restart from the next front reading: a real jump, such as a mis-centred start, must not be locked out. If only one camera keeps being rejected, raise `CAMS_DISAGREE`.
4. **Output** a `LaneEstimate` with the same fields as today, so `LaneController`, `target_speed` and the display don't change. `conf` falls as the lateral uncertainty grows, reaching 0 at `LANE_CONF_SIGMA_M`. `heading_src` is `front` while front readings are being accepted, else `imu`.

```latex
\theta_k = \theta_{k-1} - \omega \, \Delta t, \qquad y_k = y_{k-1} + v \, \sin\theta_k \, \Delta t
```

Here ω is the IMU yaw rate in rad/s (+ = turning left) and v is commanded speed × `SPEED_MPS_PER_UNIT`, measured once since there are no encoders. Errors in v barely matter on a straight, because they are multiplied by sin θ.

**Limits and resets.**

- **Coasting:** with no accepted reading for `LANE_PREDICT_MAX_S`, the output reports the lane lost, and the lane-lost timer and SAFE\_STOP work as today.
- **Stale IMU:** no prediction step; the filter takes readings as they come, which is today's behaviour.
- **Reset** on entering LANE\_FOLLOW or REACQUIRE and after TURN. Start from the next front reading, or from a side reading with heading `road_yaw − yaw` when `road_yaw` is valid.
- `RoadYawTracker` still anchors only on accepted front readings.

| Situation | First-cut fusion (above) | Lane filter |
| --- | --- | --- |
| Both cameras good | Average weighted by conf ÷ σ² | Same weights, applied in the correction step |
| One reading jumps | Keep the reading nearer last loop's lateral | Compare against the prediction, which includes the gyro's turning since last loop |
| Front lost | Side lateral, heading = `road_yaw − yaw` | Gyro carries heading from the last front reading; side camera keeps correcting position |
| Side lost | Front only | Front only, no step in the output |
| Both lost | Hold last steer until `LANE_LOST_TIMEOUT_S` | Coast on gyro and speed for up to `LANE_PREDICT_MAX_S`, then lost |
| Readings of different ages | Averaged as they are | Each applied at its own frame time |

**Tests.** Replay straight recordings with frames dropped and outliers injected, and check the estimate tracks the clean run. Run the closed-loop sim in `test_control.py` with camera latency and dropouts, filter against first-cut fusion.

### Junction decisions

- **Arm and trigger.** The front camera arms (APPROACH); the side camera triggers (DECIDE). The opening must hold for `SIDE_OPENING_FRAMES` frames in a row, and is ignored in LANE\_FOLLOW, so a shadow or a worn patch of green can't start a junction.
- **Right-turn check.** DECIDE records whether an opening was seen. A RIGHT vote with no opening means a misread sign or a junction with no right road; the robot goes to SAFE\_STOP with that reason rather than pivoting into the grass (proposed; see open questions).
- **CROSS exit.** The edge returning for `SIDE_EDGE_BACK_FRAMES` in a row ends CROSS. Front `conf` past the junction is the fallback, for junctions with no right road.
- **Advance reference.** The opening's first debounced frame is the cue `ADVANCE_AFTER_OPENING_S` counts from, and the one to log when measuring it on the course.

### Build order from today's code

1. `perception/side_lane.py` and `process_side_frame`; start the side `CameraStream` in `build_system`; show the side view as the CAMERA inset. Fusion still ignores it (`SIDE_FUSION_ENABLED = False`).
2. Run `tools/calibrate.py` for `SIDE_TARGET_PX`, `SIDE_M_PER_PX` and `SIDE_CAM_X_M`. Record straight runs with both cameras and measure `FRONT_LAT_SIGMA_M`, `SIDE_LAT_SIGMA_M` and the raw gap.
3. `SideBias` and LaneFilter with the disagreement speed scale, tested on replay and in the closed-loop sim. Measure SPEED\_MPS\_PER\_UNIT and FRONT\_HEAD\_SIGMA\_RAD first. Then set `SIDE_FUSION_ENABLED = True` and compare loop logs with and without.
4. Front-lost and both-lost behaviour in the filter (`heading_src`, coasting), tested by covering each camera in turn on the stand.
5. With junction work: opening debounce, the right-turn check and the CROSS exit.

Before the side camera: if loop logs show frequent short front-camera gaps in LANE\_FOLLOW, add the predict step alone to `fuse_lane` first (about 20 lines). It carries over into `LaneFilter` unchanged.

## Display and debug view

The 7" display runs in its own process at about 10 Hz, fed by a one-slot queue, so a slow or crashed display can never delay a motor command. Perception already returns pixel geometry from the exact frame it measured, which is what live annotation of the camera feeds needs later.

### Views (tap or key to cycle)

| View | Shows | Cost |
| --- | --- | --- |
| Status | State and stop reason, speed and steer, lateral and heading error, lane `conf`, IMU yaw, `road_yaw` and its age, gyro calibration, drift, sign vote, fps per camera, loop overruns, warnings | Text only; cheapest |
| Front | Front camera view with band centre points, fitted line, crossing line, sign box and label, plus a status strip | One 320×240 image per packet |
| Side | Side camera view with the detected edge, the target column and the opening flag | One 320×240 image per packet |
| Split (default) | Front and side views with a compact status panel | Both images |

### What live annotation needs, built in now

1. **Draw in the display process only**, never in the camera threads or the control loop.
2. **Overlays come from the same frame.** Each camera writes one result object per frame (view image, observations, `seq`), so the geometry drawn can never belong to a different frame.
3. **Geometry is in processed-image pixels** (320×240) with `scale` to the full frame; the screen scales once to `DISPLAY_SIZE`.
4. **Send small images only.** A 320×240 colour view is 230 KB; both cameras at 10 Hz is under 5 MB/s through the queue. If profiling shows the copy matters, swap the queue for shared memory behind the same `send` function.
5. **One-slot queue, drop when full**, so the display always shows the newest data and never builds a backlog.
6. **One drawing module.** `ui/overlay.py` is also used by `tools/replay.py`, so the debug view on the robot and on recordings is identical.
7. **Headless option.** `UI_ENABLED = False` runs the robot with no display process at all.

### Controls and safety

- Taps or keys send `UiCommand`s (START, STOP, NEXT\_VIEW) back through a second queue, read without blocking by `control_step`.
- A STOP on the screen is a convenience, not an emergency stop. A physical switch on motor power is still needed, and the Sabertooth serial timeout covers a crashed program.
- Rendering uses pygame in full screen (it blits NumPy images and reports touch as mouse events); OpenCV draws the annotations onto each image before it is shown.

## Configuration variables

Every tunable lives once, in `config.py`, and nothing else defines a number. Values are starting points to tune on the course, not tested settings; "measured" means it comes from tools/calib\_ui.py or `tools/calibrate.py`, and colour values come from `tools/tune_colors.py` under the course lighting.

### Timing

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `CONTROL_HZ` | 30 | Hz | `run` |
| `IMU_HZ` | 50 | Hz | `ImuReader` |
| `CAM_FPS` | 30 | fps | camera sources |
| `MAX_OBS_AGE_S` | 0.15; raise if the webcam's measured latency needs it | s | `is_fresh`, `_check_safety` |
| `LANE_LOST_TIMEOUT_S` | 0.5 | s | `_check_safety` |

### Cameras and geometry

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `FRONT_CAM_NUM` | 0 | index | `CsiSource` |
| `SIDE_CAM_DEVICE` | your webcam's `/dev/v4l/by-id/…` entry | path | `UsbSource` |
| `SIDE_CAM_FOURCC` | YUYV | — | `UsbSource` |
| `FRONT_SIZE` | (640, 480) | px | `CsiSource` |
| `FRONT_LANE_SIZE` | (320, 240) | px | `make_front_masks` |
| `SIDE_SIZE` | (320, 240) | px | `UsbSource` |
| `FRONT_BANDS` | 5 bands of (y\_top, y\_bottom, forward\_m from the rear axle, half\_road\_px, m\_per\_px), measured | px, m | `front_lane.py` |
| `SIDE_ROWS` | 5 rows spread over the frame | px | `find_boundary_x` |
| `SIDE_ROAD_ON_LEFT` | measured: True if the road appears on the image's left | bool | `find_boundary_x` |
| `SIDE_TARGET_PX` | measured with the robot centred on the yellow line | px | `edge_to_offset` |
| `SIDE_M_PER_PX` | measured | m/px | `edge_to_offset` |
| `SIDE_CAM_X_M` | measured; + = ahead of the rear axle | m | `side_to_axle` |
| `MIN_LINE_PX` | 30 | pixels per band | `find_yellow_x`, `find_road_edges` |
| `LANE_SEARCH_M, LANE_SEARCH_MAX_AGE_S, LANE_SEARCH_MIN_CONF` | 0.06, 0.2, 0.5 | m, s, conf | LaneTracker search window |
| `LANE_OUTLIER_M, LANE_MIN_KEEP, LANE_EDGE_WEIGHT` | 0.015, 3, 0.5 | m, points, weight | fit\_robust |

### Calibration

Set by `tools/calib_ui.py` on APPLY (or `calibrate.py --apply`); the rest are settings of the tool itself.

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `FRONT_GROUND_H` | measured | 3 × 3, lane px → m | pre-session check, target outline |
| `FRONT_CAM_MATRIX`, `FRONT_DIST_COEFFS` | None (lens step, optional) | FRONT\_SIZE px | fit, check |
| `ROAD_WIDTH_M` | 0.31 (from the current bands) | m | `half_road_px` in `FRONT_BANDS` |
| `CALIB_SQUARE_M` | 0.025; measure your print | m | fit, check, lens |
| `CALIB_PATTERN` | (9, 6) | inner corners | detection |
| `CALIB_POSITIONS` | 9 fit spots 0.72–1.12 m, 1 check spot | (near m, lateral m, role) | PLACE screen |
| `CALIB_MIN_FIT_BOARDS` | 4 | photos | FINISH, fit |
| `CALIB_BAND_COUNT`, `CALIB_BAND_FILL` | 5, 0.75 | — | new band rows |
| `CALIB_BOARD_RMS_WARN_M`, `CALIB_BOARD_OFFSET_WARN_M` | 0.003, 0.004 | m | per-board hints |
| `CALIB_HOLDOUT_TOL_M` | 0.005 | m | check photo PASS |
| `CALIB_CHECK_TOL_M` | 0.005 | m | pre-session check PASS |
| `CALIB_CHECK_NEAR_M`, `CALIB_CHECK_LATERAL_M` | measured (the check photo's spot) | m | pre-session check |
| `CALIB_LENS_EDGE_RATIO` | 2.0 | edge ÷ centre error | lens hint |
| `CALIB_LENS_MIN_PHOTOS`, `CALIB_LENS_MIN_CELLS` | 12, 7 | photos, of 9 cells | lens COMPUTE |
| `CALIB_LENS_NEW_PX` | 40 | px | refuses near-duplicate lens photos |
| `CALIB_LENS_RMS_WARN_PX`, `CALIB_LENS_NEGLIGIBLE_MM` | 0.8, 1.0 | px, mm | lens verdicts |
| `CALIB_UI_HZ`, `CALIB_STILL_PX` | 15, 1.5 | Hz, px | screen redraw, "hold still" |
| `CALIB_DIR` | "calibration" | folder | session photos (git-ignored) |
| `CALIB_CHECK_MAX_AGE_H` | 12 | h | start-up checks: an older PASS shows as not checked |

### Exposure and lighting

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `EXPOSURE_MODE` | 'locked' (or 'managed', 'auto') | — | `exposure.py` |
| `CAM_WARMUP_S` | 2.0 | s | `lock_after_warmup` |
| `EXPOSURE_MAX_US` | 10000 | µs | `lock_after_warmup`, `manage_step` |
| `EXPOSURE_TARGET_V` | brightness at lock time | 0–255 | `manage_step` |
| `EXPOSURE_STEP` | 0.1 | fraction per step | `manage_step` |
| `EXPOSURE_CHECK_FRAMES` | 15 | frames | `manage_step` |
| `USB_EXPOSURE_CTRLS` | control names from `v4l2-ctl --list-ctrls` for your webcam | — | `UsbSource` |
| `DARK_WARN_V`, `BRIGHT_WARN_V` | 40, 220 | 0–255 | `check_health` |

### Colour tests (OpenCV HSV: H 0–179, S and V 0–255)

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `YELLOW_H_MIN`, `YELLOW_H_MAX` | 20, 35 | hue | `mask_yellow` |
| `YELLOW_S_MIN`, `YELLOW_V_MIN` | 90, 60 | S, V | `mask_yellow` |
| `GREEN_CHROM_MIN` | 0.40 | G ÷ (R+G+B) | `mask_green` |
| `DARK_SUM_MIN` | 90 | R+G+B; below = too dark to judge | `mask_green` |
| `ROAD_S_MAX` | 70 | S | `mask_road` |
| `ROAD_V_T_MIN`, `ROAD_V_T_MAX` | 40, 110 | V; clamp on the Otsu threshold | `mask_road` |
| `WHITE_S_MAX`, `WHITE_V_REL`, `WHITE_V_PCT` | 50, 0.8, 95 | S, ratio, percentile | `mask_white` |
| `RED_H_LOW_MAX`, `RED_H_HIGH_MIN` | 10, 170 | hue | `mask_red` |
| `RED_S_MIN` | 100 | S | `mask_red` |

### Junctions and signs

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `APPROACH_ON`, `APPROACH_OFF` | 0.6, 0.3 | score | `_on_lane_follow`, `_on_approach` (hysteresis) |
| `CROSS_LINE_MIN_FRAC` | 0.5 | fraction of width | `crossing_line_row` |
| `STOP_LINE_Y_FRAC` | 0.85 | fraction of height | `_on_approach` (fallback position fix) |
| `SIDE_GREEN_MIN_FRAC` | 0.2 | fraction | `is_opening` |
| `SIDE_OPENING_FRAMES` | 3 | frames in a row | `_on_approach` |
| `SIDE_EDGE_BACK_FRAMES` | 3 | frames in a row | `_on_cross` |
| `SIGN_HZ` | 10 | Hz | `process_front_frame` |
| `SIGN_SIDE` | 'right' (to confirm) | — | `find_sign_candidates` |
| `SIGN_MIN_AREA_PX` | 400 | px² | `find_sign_candidates` |
| `SIGN_ASPECT_MIN`, `SIGN_ASPECT_MAX` | 0.6, 1.6 | w ÷ h | `find_sign_candidates` |
| `SIGN_RED_MIN_FRAC` | 0.15 | fraction of crop | `red_check` |
| `SIGN_CROP_PX` | 64 (must match the model file) | px (square) | `normalise_crop`, `load_sign_model` |
| `SIGN_MARGIN_MIN` | 0.5; use the value `train_signs.py` suggests | SVM margin | `process_signs` |
| `SIGN_CLAHE_CLIP, SIGN_CLAHE_TILES` | 2.0, (4, 4) | clip limit, tile grid | `normalise_crop` |
| `SIGN_MODEL_PATH` | 'models/sign\_svm.npz' | path | `build_system` |
| `SIGN_VOTES_NEEDED` | 3 | frames | `SignVoter` |
| `DEFAULT_ACTION` | STRAIGHT | Action | `choose_action` |
| `STOP_HOLD_S` | 3.0 | s | `_on_stop_hold` |

### Speeds and lane control

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `CRUISE_SPEED` | 0.5 | 0–1 | `target_speed` |
| `APPROACH_SPEED`, `CROSS_SPEED`, `REACQUIRE_SPEED` | 0.3, 0.3, 0.3 | 0–1 | `target_speed` |
| `KP_LAT` | 4.0 | steer per m | `LaneController` |
| `KH` | 0.8 | steer per rad | `LaneController` |
| `KD_YAW` | 0.004 | steer per deg/s | `LaneController` |
| `STEER_SMOOTH` | 0.5 | 0 = none, near 1 = heavy | `LaneController` |
| `STEER_MAX` | 0.6 | steer | `LaneController` |
| `REACQUIRE_CONF`, `REACQUIRE_FRAMES` | 0.6, 5 | conf, frames | `_on_reacquire` |
| `CAM_GAP_WARN_M` | 0.03 | m | `check_health` |
| `SIDE_FUSION_ENABLED` | False until build step 3 | bool | `fuse_lane` |
| `FUSE_GAP_MIN_CONF` | 0.6 | conf | `fuse_lane, SideBias` |
| `FRONT_LAT_SIGMA_M, SIDE_LAT_SIGMA_M` | measured: spread of each camera's lateral on straight recordings | m | `fuse_lane` |
| `FUSE_GATE_M` | 0.04; about 3 × the larger sigma once measured | m | `fuse_lane` |
| `FUSE_GATE_WARN_COUNT` | 15 (0.5 s) | loops | `check_health` |
| `SIDE_BIAS_ALPHA` | 0.01 (settles in about 3 s) | blend per loop | `SideBias` |
| `SIDE_BIAS_MAX_M` | 0.03 | m | `SideBias, check_health` |
| `SIDE_BIAS_MAX_HEADING_RAD` | 0.1 (about 6°) | rad | `SideBias` |

### Lane filter

Used once `LaneFilter` replaces `fuse_lane`. `FRONT_LAT_SIGMA_M` and `SIDE_LAT_SIGMA_M` above are reused.

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `SPEED_MPS_PER_UNIT` | measured by steer\_test.py | m/s per unit of speed command | predict step |
| `FRONT_HEAD_SIGMA_RAD` | measured; start 0.03 | rad | front correction |
| `LANE_Q_LAT` | 0.0004 | m² per s | predict step |
| `LANE_Q_HEAD` | 0.003 | rad² per s | predict step |
| `LANE_CONF_FLOOR` | 0.1 | conf | correction (readings below are skipped) |
| `LANE_GATE_FRONT`, `LANE_GATE_SIDE` | 9.2, 6.6 | normalised error (99% level) | gate |
| `LANE_REJECT_RESET_N` | 5 | loops in a row | gate restart |
| `LANE_PREDICT_MAX_S` | 0.5 | s | coasting limit |
| `LANE_CONF_SIGMA_M` | 0.03 | m | output conf |

### Turns and junction crossing

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `TURN_SPEED` | 0.0 (pivot); above 0 gives an arc | 0–1 | `target_speed`, `mix` |
| `TURN_ANGLE_DEG` | 90 | deg | `_enter` |
| `ADVANCE_AFTER_OPENING_S` | measured; start 0.4 | s | `_on_turn` |
| `ADVANCE_AFTER_LINE_S` | measured; start 0.8 | s | `_on_turn` |
| `TURN_KP` | 0.02 | steer per deg | `turn_steer` |
| `TURN_MIN_STEER` | 0.25 | steer | `turn_steer` |
| `TURN_TOL_DEG` | 3 | deg | `turn_done` |
| `TURN_HANDOFF_DEG`, `TURN_HANDOFF_CONF` | 15, 0.6 | deg, conf | `turn_done` |
| `VISION_TURN_STEER` | 0.3 | steer | `vision_turn_steer` |
| `TURN_TIMEOUT_S` | 4.0 | s | `_on_turn` |
| `HOLD_KP` | 0.01 | steer per deg | `hold_steer` |
| `CROSS_TIMEOUT_S` | 3.0 | s | `_on_cross` |

### IMU and road yaw

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `IMU_I2C_ADDR` | 0x28 | address | `ImuReader` |
| `IMU_MAX_RATE_DPS` | 300 | deg/s | `_plausible` |
| `IMU_MAX_REJECTS` | 5 | samples | `_loop` |
| `IMU_STILL_S` | 3.0 | s | `measure_drift` |
| `IMU_DRIFT_MAX_DPS` | 0.5 | deg/s | `measure_drift` |
| `IMU_DRIFT_WARN_DPS` | 0.2 | deg/s | `check_health` |
| `IMU_CALIB_WAIT_S` | 10 | s | `_on_idle` |
| `ROAD_YAW_ALPHA` | 0.05 | blend per loop | `RoadYawTracker` |
| `ROAD_YAW_MIN_CONF` | 0.7 | conf | `RoadYawTracker` |
| `ROAD_YAW_MAX_RATE_DPS` | 10 | deg/s | `RoadYawTracker` |
| `ROAD_YAW_STALE_S` | 5 | s | `check_health` |

### Sabertooth and motors

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `SABER_PORT` | /dev/ttyAMA0 (GPIO 14/15, enabled with dtparam=uart0=on; on the Pi 5, /dev/serial0 is the debug header) | path | `SabertoothDriver` |
| `SABER_BAUD` | 9600 (2x12 factory default; set by command 15, not by DIP switches) | baud | `SabertoothDriver` |
| `SABER_ADDRESS` | 128 (DIP switches 4, 5, 6 ON) | 128–135 | `SabertoothDriver` |
| `SABER_TIMEOUT_MS` | 300 (sent as 3 × 100 ms) | ms | `open` |
| `LEFT_CHANNEL`, `RIGHT_CHANNEL` | 1, 2; confirm on the bench | motor | `set` |
| `LEFT_INVERT`, `RIGHT_INVERT` | False, False; confirm on the bench | bool | `set` |
| `STEER_GAIN` | 1.0 | — | `mix` |
| `MOTOR_DEADBAND` | 0.15 | 0–1 | `_with_deadband` |

### Steering response

Measured by `steer_test.py` and `steer_report.py` (Steering response test). Per-speed values are dicts keyed by the test speeds, interpolated between them.

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `STEER_YAW_DPS_PER_UNIT` | measured | {speed: deg/s per unit steer} | lane\_sim, choosing gains |
| `STEER_DEADBAND` | 0.0 until measured | steer | `mix` |
| `STEER_DEADBAND_RAMP` | 0.02 | steer | `mix` (deadband blended in, no twitch at 0) |
| `STEER_TRIM` | None until measured | {speed: steer} | `mix` |
| `MOTOR_DEAD_TIME_S`, `MOTOR_LAG_S` | measured | s | lane\_sim |
| `CAMERA_LATENCY_S` | measured (LATENCY) | s | lane\_sim |
| `SPEED_MPS_PER_UNIT` | measured | m/s per unit command | lane\_sim, lane filter, space hints |
| `LANE_GAIN_SPEED_SCALING`, `LANE_GAIN_MIN_SPEED` | False, 0.2 | —, 0–1 | `LaneController` |
| `STEER_TEST_SPEEDS` | (0.3, 0.5, 0.7); speed 0 is always tested | 0–1 | `plan_runs` |
| `STEER_TEST_STEERS` | (0.05, 0.1, 0.2, 0.3), each left then right | steer | `plan_runs` |
| `STEER_TEST_LEAD_S`, `STEER_TEST_HOLD_S`, `STEER_TEST_SETTLE_S` | 0.7, 1.2, 0.5 | s | `plan_runs` |
| `STEER_TEST_STRAIGHT_S` | 3.0 | s | straight runs |
| `STEER_TEST_STEADY_S` | 0.5 | s | steady rate at the end of a step |
| `STEER_TEST_MOVE_DPS` | 5 | deg/s | deadband threshold |
| `STEER_TEST_TURN_MARGIN` | 0.05 | steer | `TURN_MIN_STEER` suggestion |
| `STEER_TEST_ASYM_WARN` | 0.15 | fraction | asymmetry warning |
| `STEER_TEST_LATENCY_MAX_S` | 0.4 | s | latency search range |
| `STEER_TEST_IMU_POLL_HZ` | 200 | Hz | reading every IMU sample |

### Display and health

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| `UI_ENABLED` | True | bool | `build_system` |
| `UI_HZ` | 10 | Hz | `run_ui` |
| `UI_IMAGE_EVERY` | 3 | loops | `make_ui_packet` |
| `DISPLAY_SIZE` | set for your display (open question) | px | `Screen` |
| `UI_START_VIEW` | 'split' | — | `Screen` |
| `CAM_FPS_MIN` | 20 | fps | `check_health` |

## Testing and tools

Every block can be tested without the robot moving, by replaying recorded runs; nothing merges until its unit tests and a replay pass.

| Tool | Job |
| --- | --- |
| `tools/record.py` | Drive or push the robot slowly and log both camera streams and IMU samples with timestamps |
| `tools/replay.py` | Feed a recording through perception → fusion → behaviour with the motors stubbed; draws with `ui/overlay.py` |
| `tools/tune_colors.py` | Sliders over recorded frames to set the colour values; prints them to paste into `config.py` |
| `tools/calibrate.py` | Maths and command line for the front camera calibration (fit, check, lens) from photos of a chessboard at measured spots; computes `FRONT_BANDS``  and FRONT_GROUND_H (side camera values to be added with the side camera build) ` |
| `tools/calib_ui.py` | Guided calibration on the 7" display: board session, --check (pre-session) and --lens, with live board detection, per-board hints and APPLY. Prints a short note at every step (Front camera calibration) |
| `tools/config_writer.py` | Writes measured values into config.py for APPLY / --apply: only the named assignments change, comments kept, config.py.bak backup, checked to parse first |
| `tools/guide_draw.py` | Buttons, wrapped instructions, value adjusters and console notes for guided tool screens, in the driving display's look; meant for the steering response test and IMU checks too |
| `tools/imu_drift_test.py` | Logs yaw with the robot still, first with motors off, then with motors spinning on a stand, to measure drift and interference |
| `tools/steer_test.py` | Guided steering test on the display (wheels move): pivot, straight and left/right runs at each test speed, distance entry, results with APPLY, and camera latency from a loop log (Steering response test) |
| `tools/steer_report.py` | Turns a steer\_test log (and a loop log, for camera latency) into turn gain, deadband, lag, trim and speed values; prints them to paste into config.py |
| `tools/collect_signs.py` | Runs the sign candidate and crop functions over a recording and saves each normalised crop as a PNG for sorting into class folders |
| `tools/train_signs.py` | Laptop only: augments the sorted crops, trains the linear SVM on HOG features, reports held-out accuracy and a confusion matrix, suggests SIGN\_MARGIN\_MIN and writes models/sign\_svm.npz |
| `tools/menu.py` | Tools menu on the display: tiles for driving and every tool, with what needs doing highlighted (Tools menu) |
| `tools/lane_sim.py` | Closed-loop lane-following sim with the measured robot response; --log compares overshoot and settling with a real run (Steering response test) |
| `tools/calib_status.py` | Remembers the newest camera check (CALIB\_DIR/last\_check.json) for main.py's start-up line, the pre-flight list and the menu |

**Order of testing**

1. Record every course layout under the lighting you will actually face (room lights, daylight, shadows), and each sign from several distances.
2. Front camera: run `calib_ui.py` (board session), retake any flagged board, APPLY, and tape the check spot. Then the camera check before every session: `calib_ui.py --check`, or CAMERA CHECK in the tools menu. Until it has passed within `CALIB_CHECK_MAX_AGE_H`, `main.py` prints a warning and the display's pre-flight list shows a cross (Front camera calibration).
3. Collect and sort sign crops, train with `train_signs.py`, and check held-out accuracy before trusting signs on the course.
4. Unit-test each perception function on saved frames with hand-labelled values, including deliberately dark, bright and shadowed frames.
5. Replay whole runs and check the state sequence and sign decisions match what happened.
6. On the bench with the wheels off the ground: steer + turns left, a left turn by hand raises yaw, Sabertooth channels and invert flags are right, and `MOTOR_DEADBAND` just starts the wheels. Then kill the program and confirm the Sabertooth timeout stops the wheels.
7. Run `imu_drift_test.py` and set `IMU_DRIFT_MAX_DPS` and `IMU_DRIFT_WARN_DPS` from the results.
8. Run the steering test on the floor (`steer_test.py`, or the tools menu), APPLY, and measure camera latency from a loop log. Then check `lane_sim.py` against a real 5 cm offset start before tuning gains (Steering response test).
9. Kill the display process while the robot runs on the stand; the control loop must not notice.
10. Side camera: run `calibrate.py` for the side values, record straights to measure `FRONT_LAT_SIGMA_M` and `SIDE_LAT_SIGMA_M`, and replay with `SIDE_FUSION_ENABLED` on and off. Then cover the front camera on the stand and check the side + IMU fallback holds the lane.
11. On the course at `APPROACH_SPEED` everywhere, then raise `CRUISE_SPEED` once lane `conf` stays high. Measure `ADVANCE_AFTER_OPENING_S` and `ADVANCE_AFTER_LINE_S` by watching where the rear axle stops.
12. Profile each thread on the Pi 5 and compare against the budgets in Runtime on the Raspberry Pi 5.

### Tools menu

`python -m tools.menu` puts driving and every tool on the 7" display as tiles, so nobody has to remember the commands. The header shows what needs doing and the matching tile is outlined green: the camera check when it hasn't passed in the last `CALIB_CHECK_MAX_AGE_H` hours (or CALIBRATE before the first calib\_ui calibration), and the steering test until the steering response is measured.

| Tile | Runs | Screen |
| --- | --- | --- |
| DRIVE | `main.py` | the driving display |
| CAMERA CHECK | `calib_ui.py --check` | guided |
| CALIBRATE | `calib_ui.py` | guided |
| STEERING TEST | `steer_test.py` | guided (wheels move) |
| IMU CHECK | `imu_drift_test.py` (sign check) | terminal |
| RECORD | `record.py --seconds 20` | terminal |
| LENS STEP | `calib_ui.py --lens` | guided |
| TUNE COLOURS (or UI DEMO when there is no recording) | `tune_colors.py <newest recording>` / `ui_demo.py` | desktop window / display |

Screen tools take the display over and the menu comes back when they finish. Terminal tools keep the menu up with a STOP button (or Space / Esc) that sends them Ctrl-C; their output stays in the terminal. The menu re-reads config.py when a tool returns, so anything it APPLYed shows at once.

**Start-up checks.** The camera check result is remembered in `calibration/last_check.json`, by the pre-session check (pass or fail) and by APPLY at the end of a board session (its check photo). `main.py` prints one line at start-up ("Camera: camera checked 40 min ago", or a warning with the command to run), and the driving display's pre-flight list adds it as its last row. It is information, not a lock: START still works.

### Front camera calibration

One letter-size chessboard, photographed at about 10 measured spots, fits a single ground homography that covers every band. `tools/calib_ui.py` guides the whole process on the 7" display; `tools/calibrate.py` does the maths and also runs from the command line.

**Why not one photo.** A 9 × 6-corner board with 25 mm squares spans about 12.5 cm forward, but the bands span 0.71–1.15 m, so most bands were extrapolated. A single photo also can't catch a mis-measured position, and lens distortion only shows near the image edges.

**Setup (once, about 10 minutes):**

1. Robot on the course mat, camera in its driving mount. Neither moves until APPLY.
2. Mark the rear-axle midpoint on the mat. Tape a tape measure forward from it along the robot's centreline, lined up with two centreline marks on the chassis.
3. Measure one printed square with a ruler; printers rescale.

**Session** (`python -m tools.calib_ui`, about 15 minutes; stop `main.py` first, since it owns the camera):

| Screen | What happens |
| --- | --- |
| WELCOME | Setup checklist; set the measured square size and the road width |
| PLACE | Live view with the detected corners; the banner gives the spot from `CALIB_POSITIONS` (nearest inner-corner row on the tape, centre left or right). A purple outline shows where the board should appear, from the last calibration. CAPTURE is enabled once the whole board is found |
| CONFIRM | Enter where the board really is: tape reading at the nearest inner-corner row, and the centre's offset from the tape (±1 mm / ±1 cm buttons, or arrow keys). KEEP saves the photo and `positions.csv`; RETAKE goes back |
| RESULTS | Each board's error, scored against a fit made without it, with a hint such as "looks 8 mm farther than entered". Also the check photo's PASS/FAIL, bands old → new and warnings. Tap a row and RETAKE to redo it. BANDS shows the bands on a photo. APPLY writes config.py |
| APPLIED | What was written, the check spot to tape, and the next steps |

- **Spots:** 9 fit spots spanning 0.72–1.12 m (±7 cm sideways near, ±12 cm far) plus 1 check spot. If a board can't be fully seen at its spot, slide it until it is and enter where it really is.
- **The check photo** is never used in the fit. It verifies the result against `CALIB_HOLDOUT_TOL_M`. Before moving it, tape its two near corners: that becomes the pre-session check spot.
- **Nothing is lost:** each photo and `positions.csv` are saved as they are kept, in `calibration/front_<time>/`. `--resume <folder>` carries on a session or refits a finished one.
- **Bands keep their rows:** recalibrating keeps the current `FRONT_BANDS` rows and recomputes their distances, so the bands never move silently. `calibrate.py fit --new-rows` spreads them over the rows the boards covered instead. Bands outside the covered rows are flagged.
- **APPLY** writes `FRONT_BANDS`, `FRONT_GROUND_H`, `CALIB_SQUARE_M`, `ROAD_WIDTH_M` and the check spot into config.py through `tools/config_writer.py`. Only those assignments change, and `config.py.bak` keeps the old file. The file is checked to parse before it is replaced. Perception still uses only `FRONT_BANDS`.

**Pre-session check** (`python -m tools.calib_ui --check`, 30 seconds): put the board on the taped marks. Purple dots show where its corners should appear. CHECK reports PASS or FAIL against `CALIB_CHECK_TOL_M` and which way the board reads off. A FAIL with the board on its marks means the camera has moved: run the session again.

**Lens step** (`python -m tools.calib_ui --lens`, optional): only needed if RESULTS says edge boards fit more than `CALIB_LENS_EDGE_RATIO` times worse than central ones.

- Take at least `CALIB_LENS_MIN_PHOTOS` photos with the board held 40–80 cm from the camera (sharp at the driving focus), tilted 20–45°, covering at least `CALIB_LENS_MIN_CELLS` of the 3 × 3 view cells. The screen shades covered cells and refuses near-duplicate photos.
- COMPUTE reports the fit error in px and how much the correction would change the band measurements, in mm. That is the part of the lens bend a homography can't absorb; under `CALIB_LENS_NEGLIGIBLE_MM` it isn't worth applying.
- APPLY writes `FRONT_CAM_MATRIX` and `FRONT_DIST_COEFFS`; REFIT then refits the last board session with them. Runtime perception is unchanged: the lens correction only shapes how `FRONT_BANDS` are computed.

**Without the display:** take photos with `tools/record.py`, write `positions.csv` by hand, and run `python -m tools.calibrate fit <folder> [--apply]`, `check <photo>` or `lens <folder>`. The output is the same report, as text.

The side camera values (`SIDE_TARGET_PX`, `SIDE_M_PER_PX`, `SIDE_CAM_X_M`) are not in the tool yet; they come with the side camera build.

### Steering response test

`tools/steer_test.py` measures how the robot really turns, guided on the 7" display, so the controller, the sim and the lane filter stop relying on guesses. Before it, the lane gains, `TURN_MIN_STEER` and the sim assumed 150°/s per unit steer, an instant response, no steering deadband and a perfectly straight steer 0. The Sabertooth is open loop, so the real response depends on speed, floor friction, motor mismatch and battery voltage. The analysis is `tools/steer_report.py`.

**The wheels move.** Clear floor: about 3 m ahead and 1 m either side. Stop `main.py` first (it owns the Sabertooth), then `python -m tools.steer_test` (or the tools menu).

| Screen | What happens |
| --- | --- |
| WELCOME | Safety and setup; START waits for a valid, calibrated IMU (or `IMU_CALIB_WAIT_S`) |
| READY | What the next run does and the space it needs; GO, SKIP, or RESULTS to analyse early |
| RUNNING | Live yaw rate and commanded steer; STOP, Space, Esc or Enter stops the wheels at once |
| DISTANCE | After a straight run: the distance from the start mark to the rear axle (±1 / ±10 cm); KEEP, REDO, SKIP |
| RESULTS | Per speed: turn gain, deadband, drift and trim, m/s; dead time and lag; warnings; the values APPLY writes |
| LATENCY | Camera latency from the newest `main.py` loop log; APPLY writes `CAMERA_LATENCY_S` |

**The runs** (from `STEER_TEST_SPEEDS` and `STEER_TEST_STEERS`; 16 runs, about 7 minutes):

1. Pivot: each steer size left then right at speed 0, `STEER_TEST_SETTLE_S` apart. The robot stays in place.
2. For each speed, a straight run of `STEER_TEST_STRAIGHT_S` at steer 0: its drift gives the trim, and the taped distance gives the forward speed.
3. For each speed and steer size, a run of `STEER_TEST_LEAD_S` straight, then `STEER_TEST_HOLD_S` left, then the same right.

The runner feeds the Sabertooth every loop with raw wheel commands (no trim or deadband lift: it measures the robot as it is). It stops at once on STOP or when the IMU stops giving valid, fresh readings; a stopped run is thrown away and can be run again. Every IMU sample and the timing of each commanded piece go to `calibration/steer_<time>/` after every run, so `python -m tools.steer_report <folder> [--apply]` gives the same result later.

**What the analysis works out** (model: yaw-rate change = gain × (|steer| − deadband), signed):

| Measure | How | Sets |
| --- | --- | --- |
| Turn gain, deadband | Steady rate (last `STEER_TEST_STEADY_S`) minus the rate just before; a line fit of change against steer size per speed. Steps under `STEER_TEST_MOVE_DPS` count as not moving | `STEER_YAW_DPS_PER_UNIT`, `STEER_DEADBAND` (driving speeds) |
| Pivot deadband | The same at speed 0, plus `STEER_TEST_TURN_MARGIN` | `TURN_MIN_STEER` |
| Dead time, lag | Steps from steer 0: time to 10% and to 63% of the change (medians) | `MOTOR_DEAD_TIME_S`, `MOTOR_LAG_S` |
| Drift, trim | Mean rate on the straight run after 1 s of spin-up; trim = −drift ÷ gain | `STEER_TRIM` (per speed) |
| Forward speed | Distance ÷ (run time − dead time − lag); slope through 0 across speeds | `SPEED_MPS_PER_UNIT` |
| Left/right asymmetry | Gain to each side | Warning over `STEER_TEST_ASYM_WARN` |
| Reversed steering | Steer + turned right | Error; nothing about steering is applied |
| Camera latency | Loop log: the delay that best lines up −d(heading)/dt with the gyro rate, up to `STEER_TEST_LATENCY_MAX_S` | `CAMERA_LATENCY_S` |

For the latency log: run `main.py` with the robot in front of the line (IDLE, loop log on), twist it left and right by hand about 30° each way once a second for 10 s, quit, then tap LATENCY (or `python -m tools.steer_report latency`). It refuses a log with too little turning or no line in view.

`MOTOR_DEADBAND` (set on the bench) lifts each wheel off zero. `STEER_DEADBAND` is the extra steer needed on the floor before the robot turns, against tyre scrub; it is usually largest when pivoting, which is why the pivot value sets `TURN_MIN_STEER` separately.

**What the numbers change:**

- **`mix()`** adds `STEER_TRIM` for the current speed (interpolated, 0 when stopped so a robot at rest never creeps) and lifts non-zero steer by `STEER_DEADBAND`, blended in over `STEER_DEADBAND_RAMP` so steer near 0 stays smooth. Both do nothing until measured.
- **`tools/lane_sim.py`** runs the real `LaneController`, `drive_for_state` and `mix()` against a robot model built from these numbers (dead time, lag, camera latency, deadband, drift); anything unmeasured uses a stated guess. `python -m tools.lane_sim --log <loop log>` scores the first LANE\_FOLLOW stretch of a real run (start 5 cm off the line) the same way: peak overshoot and settling time should agree within about 20% before tuning `KP_LAT`, `KH` and `KD_YAW` in the sim.
- **Speed scaling:** with `LANE_GAIN_SPEED_SCALING` on, `LaneController` scales its lateral and heading terms by `CRUISE_SPEED ÷ max(speed, LANE_GAIN_MIN_SPEED)`. It is off by default: turn it on once the sim agrees with the robot.

APPLY writes only what was measured and never the steering values if the steering looked reversed. `config.py.bak` keeps the old file.

## Assumptions and open questions

Your notes settled the drive train, both cameras' interfaces and the side camera's view, centre-line driving, the sign set, the language and the display's role. The questions below still change code; each names what it affects. Tick them off as you confirm them.

- [ ] **Sabertooth setup:** confirmed as a Sabertooth 2x12 in packetized serial mode, address 128, SLA battery, driven from the Pi's GPIO UART (`/dev/ttyAMA0`). Still open: the reported DIP switches (on, on, on, off, off, off) read as analog mode if counted from switch 1, so check them against the table in hal/sabertooth.py; and which motor is on M1. Affects `sabertooth.py`.
- [ ] **Display model:** which 7" display is it (model or resolution), is it touch, and will the Pi run the desktop? Affects `DISPLAY_SIZE`, the layout, and how `run_ui` opens its window.
- [ ] **Side camera position:** how far ahead of or behind the rear axle is it, and how high? Can the robot's own shadow fall in its view? Affects `SIDE_CAM_X_M` and the opening-based advance.
- [ ] **Camera models:** which webcam and which CSI camera? Webcam exposure control names and the CSI field of view differ by model.
- [ ] **Junction shapes:** four-way only, or T-junctions too? With no road to the right, the side camera sees no opening, so the front camera's crossing line becomes the only position fix. Affects APPROACH and TURN.
- [ ] **RIGHT sign, no opening seen:** SAFE\_STOP (proposed), or trust the sign and turn on the crossing-line fix? Affects DECIDE.
- [ ] **Stop sign behaviour:** where should the robot stop (at the position fix, or is there a stop line?), for how long, and then what: straight on, or follow another sign? Affects STOP\_HOLD and `choose_action`.
- [ ] **Sign placement:** which side of the road, at what height, and can signs meant for other approaches be seen from yours? Affects `SIGN_SIDE` and `pick_sign`.
- [ ] **Arrow colour:** assumed a dark arrow on the white sign. This no longer changes code: the SVM learns whatever the real signs look like, as long as the training crops come from the actual signs.
- [ ] **Road width:** is it the same on every mat? The learned side offset now absorbs differences up to SIDE\_BIAS\_MAX\_M; if mats differ by more, raise that limit.
- [ ] **Lighting:** room lights only, or daylight from windows too? Decides whether `EXPOSURE_MODE` defaults to locked or managed.
- [ ] **Start and emergency stop:** screen tap, keyboard or a physical button to start? A physical kill switch on motor power is recommended either way.
- [ ] **Existing code:** is any of this already written? Proposed names should match it, not duplicate it.

### Sources

- [Open Panzer Scout ESC: Sabertooth-compatible packet serial reference](https://github.com/OpenPanzerProject/Scout-ESC) (packet format; motor commands 0, 1, 4, 5)
- [Visuino wiki: Sabertooth packet serial timeout](<https://www.visuino.com/wiki/index.php/Sabertooth_Dual_Motor_Shield_(Packet_Serial).SerialTimeout>) (timeout in 100 ms units)
- [Dimension Engineering: Sabertooth 2x12 user's guide](https://www.dimensionengineering.com/datasheets/Sabertooth2x12.pdf) (DIP modes, lithium cutoff, 9600 baud default, commands 14 and 15, S2 emergency stop)
- [Dimension Engineering: Sabertooth packet serial library](https://dimensionengineering.com/software/SabertoothArduinoLibrary/html) (address 128 = switches 4, 5, 6 ON)
- [Raspberry Pi documentation: UART changes for the Pi 5](https://github.com/raspberrypi/documentation/pull/3294/files) (/dev/serial0 points to the debug UART)
