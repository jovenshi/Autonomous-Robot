# Autonomous Driving System — Architecture & Design

Oct 9, 2026 · updated Oct 11, 2026 (display decisions) · @Jacob Ovenshire

## Overview

The robot straddles the yellow centre line under a 30 Hz control loop: the front camera tracks the line, the right-side camera measures distance to the road edge, and the IMU, re-anchored to the road by the cameras, times the turns. A state machine picks the controller, and a separate display process shows status and debug views.

### Hardware

| Part | Detail | Role in the design |
| --- | --- | --- |
| Front camera | CSI, mounted high on the centreline, facing forward; Picamera2 | Yellow-line offset and heading, junction approach, sign reading |
| Side camera | USB, right side, facing down; sees the black/green road edge only; OpenCV (V4L2) | Distance to the right road edge; spots the road opening on the right at junctions |
| BNO055 IMU | I²C, address 0x28, IMUPLUS mode | Relative yaw for turns and straight crossings; yaw rate for steering damping |
| Drive | Two rear drive wheels on a Sabertooth dual motor driver (serial); omni wheel at the front | Differential drive; turns pivot about the rear-axle midpoint |
| 7" Pi display | Official 7" touch display, 800×480; status and debug now, live annotated camera feeds later | Fed by its own process so it can never delay motor commands |
| Software | Python 3 with OpenCV, NumPy, Picamera2, pyserial, pygame | — |

### Driving geometry

- **Target line:** the robot's centreline sits over the yellow dashed line. The front camera measures that directly; the side camera measures it indirectly, because the right road edge is half a road width from the line.
- **Pivot point:** with only the rear wheels driven and an omni wheel in front, the robot rotates about the midpoint of the rear axle. For a 90° turn the rear axle is placed over the junction centre (where the two yellow lines cross) before pivoting, so the robot finishes centred on the new road's line. Because the robot drives on the centre line, the same advance works for left and right turns.
- **Reference point:** all lateral errors are expressed at the rear axle. Each camera's distance from the axle (SIDE_CAM_X_M, and forward_m in FRONT_BANDS) converts its reading to that point.

### Key design decisions

- **Two lateral measurements, fused.** The front camera sees the yellow line itself (the true target) but through perspective; the side camera sees the road edge close up with almost no distortion. Fusion uses both and flags when they disagree.
- **The IMU is re-anchored to the road.** On straight road with a confident lane lock, road_yaw = IMU yaw + camera heading error is updated continuously. Turn and crossing targets come from road_yaw, so IMU drift only matters for the few seconds since the last anchor.
- **Turns finish on vision.** The IMU brings the robot within TURN_HANDOFF_DEG of the target; once the front camera sees the new road's line, REACQUIRE takes over and lane centering removes the remaining error. This is your original idea, made explicit as a state.
- **Colour tests are built for changing light.** Masks rely on hue, saturation and colour ratios rather than raw brightness, the black-road threshold adapts every frame, and exposure is locked or slowly managed instead of left on auto.
- **Signs: colour first, then shape.** A red octagon means STOP; otherwise the arrow's direction comes from which end is wider. Signs are read only near junctions and voted over several frames.
- **Nothing slow sits in the control path.** Drawing and display run in a separate process fed by a one-slot queue, and the Sabertooth's serial timeout stops the motors if the program stops sending.

### How your coding rules shape the design

- **Under 100 lines:** every function in Modules and functions has one job; the process_* and control_step orchestrators only call smaller functions.
- **No duplicate helpers or variables:** colour tests live in one module (color.py), overlay drawing in one module shared by the display and the replay tool, and every tunable only in config.py.
- **Preserve the architecture:** blocks exchange only the data types listed below, so any block can be rewritten without touching its neighbours.
- **Test before merging:** every perception and control function runs on recorded data with the motors stubbed.
- **No guessing:** names here are proposals; check them against any code you already have. Unknowns are collected in Assumptions and open questions.

## Block diagram

System architecture · 3 sensor threads, a 30 Hz control loop, a separate display process

Each sensor thread turns raw data into one timestamped result per frame. The control loop fuses them, keeps the IMU anchored to the road, lets the state machine choose lane centering or a turn, and sends left and right commands to the Sabertooth every loop. The display process only ever receives copies, so it cannot slow the loop.

## Sensor errors and how the design handles them

Every sensor is wrong some of the time, so each known error has a specific counter and, where it can be detected, a warning on the display. The two that matter most are IMU drift and lighting.

**IMU drift, layer by layer**

- The magnetometer is off (IMUPLUS mode), so motor fields cannot move the heading.
- At standstill (IDLE, STOP_HOLD) the drift rate is measured and subtracted from then on.
- On straights, road_yaw is re-anchored from the camera, so each turn starts from the road's true direction rather than wherever the gyro has wandered.
- The turn ends on vision, and REACQUIRE removes whatever error is left. Drift only matters over the few seconds of a turn.

**Lighting, layer by layer**

- Exposure is not left on auto: it is locked after warm-up, or managed in small steps only on plain road.
- Colour tests use hue, saturation and colour ratios, which change far less than brightness.
- The black-road threshold is recomputed every frame, and the white threshold is relative to the frame.
- A poor frame lowers conf, so fusion leans on the other camera and the speed governor slows down.
- Thresholds are tuned on recordings made under the course's real lighting.

| Sensor | What goes wrong | How the design handles it | Display warning |
| --- | --- | --- | --- |
| IMU | Gyro drift: yaw slowly wanders, so turns over- or under-shoot | Still-drift subtraction, camera-anchored road_yaw, turns finished on vision | IMU_DRIFT_HIGH, ROAD_YAW_STALE |
| IMU | Motor magnetic fields corrupt the magnetometer | IMUPLUS mode: magnetometer not used | — |
| IMU | Spikes or failed reads (I²C errors, vibration) | Plausibility gate holds the last good value; valid=False after repeated rejects; vision-only turn fallback; soft mount | IMU_INVALID |
| IMU | Gyro not yet calibrated at power-up | IDLE waits for calib_gyro = 3, or warns after IMU_CALIB_WAIT_S | IMU_NOT_CALIBRATED |
| Both cameras | Room light changes (windows, lamps, shadows) | Hue, saturation and ratio tests; per-frame darkness threshold; relative white threshold | CAM_TOO_DARK, CAM_TOO_BRIGHT |
| Both cameras | Auto-exposure lags after a light change, leaving washed-out or dark frames | Exposure locked, or managed only in LANE_FOLLOW (exposure_hold freezes it near junctions); EXPOSURE_MODE lets you compare all three on the course | CAM_TOO_DARK, CAM_TOO_BRIGHT |
| Both cameras | Glare on the mats washes out yellow or looks white | Yellow needs saturation; sign candidates must pass shape, size and side tests; multi-frame voting | LANE_CONF_LOW |
| Both cameras | Motion blur smears the dashes | Exposure capped at EXPOSURE_MAX_US; slower speed near junctions | — |
| Both cameras | Mount knocked, so calibration shifts | Front and side lateral readings compared on straights; a persistent gap is flagged; re-run calibrate.py | CAMS_DISAGREE |
| Front camera | Gaps between dashes | Road-edge midpoint per band | — |
| Side camera | USB latency and frame buffering | Buffer of one frame, continuous reads, timestamps, MAX_OBS_AGE_S | CAM_FPS_LOW |
| Side camera | Webcam lowers its frame rate in dim light | Manual exposure through V4L2 controls; fps monitored | CAM_FPS_LOW |
| Side camera | The robot's own shadow falls on the grass | Green tested by chromaticity, which changes little in shadow | — |
| Side camera | Road width differs between mats, so the edge-to-line distance is wrong | Same front/side gap check; open question below | CAMS_DISAGREE |
| Drive | Unequal motors, wheel slip, omni-wheel scrub | Closed-loop lane control; turns closed on yaw; deadband compensation | — |
| Drive | Program crash or cable fault leaves the last command running | Sabertooth serial timeout; shutdown always stops the motors | MOTOR_SERIAL_ERROR |
| Pi | Control loop overruns its period | Overruns counted and shown; profile and move work to a process if needed | LOOP_OVERRUN |

## Runtime on the Raspberry Pi 5

Three sensor threads and the control loop share one Python process; the display runs in a second process. The loop always uses the newest result from each sensor and never works through a backlog.

| Thread or process | Work done | Rate | Target time per cycle |
| --- | --- | --- | --- |
| front_cam thread | CSI capture, masks, front lane, junction score; sign reader when enabled | 30 fps (signs ~10 Hz) | < 15 ms, < 30 ms with signs |
| side_cam thread | USB capture, masks, edge position, opening check | 30 fps | < 5 ms |
| imu thread | Read, plausibility gate, unwrap, drift correction | 50 Hz | < 2 ms |
| Main thread (control loop) | Fusion, road yaw, state machine, controllers, Sabertooth output, UI packet | 30 Hz | < 4 ms |
| ui process | Draw overlays and status, show them on the 7" display, read taps | ~10 Hz | Its own core; the loop never waits on it |

The times are budgets to check with a profiler on the Pi, not measurements.

### Efficiency rules

- **Capture small.** Side camera at 320×240 in YUYV: uncompressed, so no JPEG smearing at the black/green edge. Front camera at 640×480, with lane and junction work on a 320×240 copy and sign crops taken from the full frame.
- **Work on regions only.** The lower part of the front frame for the lane; the road-side strip for signs.
- **Make masks once.** Colour conversion and masks are computed once per frame and shared by lane, junction and sign code.
- **Vectorised calls only.** OpenCV and NumPy on whole arrays; no per-pixel Python loops.
- **Newest value wins.** Each camera thread writes one result object per frame into its LatestValue slot; a newer one overwrites the older one and nothing queues.
- **Keep USB latency down.** Set the capture buffer to one frame (CAP_PROP_BUFFERSIZE) and read continuously in the thread, so the newest frame is never stuck behind old ones.
- **Open the webcam by a stable path.** Use /dev/v4l/by-id/…. On the Pi 5 the CSI camera stack also creates /dev/video* nodes, so an index like 0 may not be the webcam.
- **Exposure off auto.** See Sensor errors. This also stops many webcams from silently lowering their frame rate in dim light.
- **Drop stale data.** The loop ignores any result older than MAX_OBS_AGE_S.
- **Isolate the display.** The UI process gets packets through a one-slot queue with put_nowait; when it is busy the packet is dropped, so the control loop never blocks on drawing.
- **Threads first, processes if needed.** OpenCV releases Python's GIL during heavy calls, so vision threads run in parallel. If profiling shows contention, move the front camera into its own process behind the same slot interface; nothing else changes.

## Data types passed between blocks

Fourteen small dataclasses, five enums and one LatestValue holder are the only things blocks pass to each other. All live in types.py; every result carries t from time.monotonic() and the seq number of the frame it came from, so stale data is dropped and overlays always land on the frame they were measured on.

**Sign conventions (fixed once, used everywhere):** lateral error + = rear axle is right of the yellow line; heading error + = robot points right of the road direction; steer + = turn left; yaw + = counter-clockwise (left). Positive errors therefore call for positive steer, and road_yaw = yaw + heading error (a robot pointing right of the road has a smaller yaw than the road). Confirm the IMU direction on the bench by turning the robot left by hand.

| Type | Fields (units) | Made by → used by |
| --- | --- | --- |
| Frame | image (H×W×3 BGR uint8), t (s), seq (int), cam_id ('front' or 'side') | CameraStream → perception |
| FrontMasks | hsv, yellow, road, white, red (masks at lane size), scale (full frame ÷ lane copy), mean_v | make_front_masks → front lane, junction, signs |
| FrontLaneObs | lateral_m, heading_err_rad, conf (0–1), centre_pts_px (band centres), line_px (fitted line ends), t, seq | process_front_lane → fusion, overlay |
| IntersectionObs | approach_score (0–1), crossing_line_y (px or None), t, seq | process_intersection → behaviour, overlay |
| SignObs | sign (SignType), score (0–1), bbox (x, y, w, h in full-frame px), t, seq | process_signs → SignVoter, overlay |
| FrontResult | view (320×240 image), lane, inter, sign (or None), mean_v, t, seq | front thread → front slot (one object per frame) |
| SideLaneObs | offset_m (at the camera), boundary_found, opening (bool), conf (0–1), boundary_x_px, t, seq | process_side_lane → fusion, behaviour, overlay |
| SideResult | view (320×240 image), side, mean_v, t, seq | side thread → side slot |
| ImuSample | yaw_deg (relative, unwrapped, drift-corrected), yaw_rate_dps, valid (bool), calib_gyro (0–3), drift_dps, t | ImuReader → road yaw, behaviour, controllers |
| LaneEstimate | lateral_err_m (at the rear axle), heading_err_rad, conf (0–1), cam_gap_m (front minus side, or None), t | fuse_lane → controllers, behaviour, road yaw |
| RoadYaw | yaw_deg, t_anchor (last camera anchor), valid (bool) | RoadYawTracker → behaviour, display |
| DriveCmd | speed (0–1), steer (−1 to 1) | drive_for_state → mix |
| MotorCmd | left, right (−1 to 1) | mix → SabertoothDriver |
| UiPacket | t, state, stop_reason, drive, motor (the MotorCmd actually sent this loop), lane, road_yaw, imu, sign_vote, fps, overruns, warnings, front and side (results with views, or None on loops without images) | make_ui_packet → display process |

**Enums**

- SignType: NONE, STOP, LEFT, RIGHT, STRAIGHT
- Action: STOP, LEFT, RIGHT, STRAIGHT
- State: IDLE, LANE_FOLLOW, APPROACH, DECIDE, STOP_HOLD, TURN, CROSS, REACQUIRE, SAFE_STOP
- Warning: IMU_INVALID, IMU_NOT_CALIBRATED, IMU_DRIFT_HIGH, ROAD_YAW_STALE, CAM_FPS_LOW, CAM_TOO_DARK, CAM_TOO_BRIGHT, CAMS_DISAGREE, LANE_CONF_LOW, LOOP_OVERRUN, MOTOR_SERIAL_ERROR
- UiCommand: START, STOP, NEXT_VIEW (NEXT_VIEW is handled inside the display process and never sent to main)

**LatestValue[T]** holds one item behind a lock: set(value) replaces it, get() returns it or None. Slots: front, side, imu. Two threading.Event flags carry decisions back to the camera threads: signs_enabled (set in APPROACH) and exposure_hold (set outside LANE_FOLLOW so managed exposure never changes mid-junction).

## Modules and functions

Six packages (hal, perception, estimation, behaviour, control, ui) plus tools and tests. Each function has one job and a fixed signature; the orchestrators (process_*, control_step, run_ui) only call the small functions beneath them.

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
├── tools/                 record, replay, tune_colors, calibrate, imu_drift_test
└── tests/
```

### hal/camera.py

CameraStream(source, process_fn, slot, name) runs one camera thread. CsiSource(cam_num, size) wraps Picamera2 and UsbSource(device_path, size, fourcc) wraps cv2.VideoCapture with the V4L2 backend; both expose the same five methods, so CameraStream never knows which it has.

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| CameraStream.start() | — | None | Open the source, start the thread |
| CameraStream._loop() | — | None | Read → stamp t and seq → process_fn(frame) → write the result to its slot; count fps |
| CameraStream.fps() | — | float | Measured frame rate, for health and the display |
| source.open() | — | None | CSI: configure and start Picamera2. USB: open by path, set size, YUYV, buffer size 1 |
| source.read() | — | image or None | One BGR frame |
| source.get_exposure() | — | (exposure_us, gain) | CSI: from frame metadata. USB: from the V4L2 controls named in config |
| source.set_exposure(exposure_us, gain) | numbers | None | Manual exposure with auto-exposure and auto white balance off |
| source.close() | — | None | Release the camera |

### hal/exposure.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| lock_after_warmup(source, cfg) | source | None | Let auto-exposure settle for CAM_WARMUP_S on the course, then fix those values, capped at EXPOSURE_MAX_US to limit motion blur |
| manage_step(source, mean_v, hold, cfg) | source, ROI brightness, hold flag | None | Managed mode only: every EXPOSURE_CHECK_FRAMES, nudge exposure by EXPOSURE_STEP toward EXPOSURE_TARGET_V; does nothing while hold is set |

### hal/imu.py — ImuReader(i2c_bus, address)

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| start() | — | None | Put the BNO055 in IMUPLUS mode (magnetometer off), take the first yaw as 0, start the 50 Hz thread |
| read_raw() | — | (heading_deg, gyro_z_dps) | One read from the sensor |
| _to_robot_yaw(heading_deg, prev_yaw) | raw heading, last yaw | float (deg) | Flip to counter-clockwise-positive and unwrap across 0/360 |
| _plausible(new_yaw, prev_yaw, dt) | two yaws, seconds | bool | False if the implied rate exceeds IMU_MAX_RATE_DPS |
| measure_drift(seconds) | duration | float or None | Only while the motors are stopped (IDLE, STOP_HOLD): yaw change ÷ time. Stored if below IMU_DRIFT_MAX_DPS, otherwise rejected as "not really still" |
| _loop() | — | None | Read → convert → subtract stored drift × elapsed → gate → write ImuSample; valid=False after IMU_MAX_REJECTS rejects in a row |

The BNO055 relies on I²C clock stretching, which has caused read errors on Raspberry Pi boards; if reads glitch, lower the I²C clock or use the sensor's UART mode (only read_raw() changes). Mount it away from the motors and Sabertooth, on a soft mount to damp omni-wheel vibration.

### hal/sabertooth.py — SabertoothDriver(port, baud, address)

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| open() | — | None | Open the serial port (pyserial), send the bauding byte 0xAA if your model needs it, set the serial timeout (command 14, in 100 ms units) |
| set(cmd) | MotorCmd | None | Apply deadband and invert flags, send one packet per motor |
| stop() | — | None | Send speed 0 to both motors; called by shutdown and SAFE_STOP |
| _side_packet(x, channel) | −1 to 1, 1 or 2 | bytes | Forward command (0 for motor 1, 4 for motor 2) or reverse (1 or 5) with value round(abs(x) × 127) |
| _packet(command, value) | ints | bytes | address, command, value, (address + command + value) & 0x7F |
| _with_deadband(x) | −1 to 1 | float | Lift small non-zero commands above MOTOR_DEADBAND so the wheels actually move |

Packetized serial sends 4-byte packets (address 128–135, command, value 0–127, 7-bit checksum). With the timeout set, control_step must send every loop, even when nothing changed, or the Sabertooth stops the motors. The S1 input is logic-level serial; confirm the voltage levels and DIP switch settings in your model's manual.

### perception/color.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| mask_yellow(hsv, cfg) | HSV image | mask | Hue range plus minimum saturation; the brightness bound is kept loose |
| mask_green(img, cfg) | BGR image | mask | Green chromaticity G ÷ (R+G+B) above GREEN_CHROM_MIN, ignoring pixels too dark to judge. Ratios barely change with brightness or the robot's own shadow |
| mask_road(hsv, roi, cfg) | HSV, region | mask | Dark and unsaturated; the darkness threshold comes from Otsu on the region's V channel each frame, clamped to ROAD_V_T_MIN–ROAD_V_T_MAX |
| mask_white(hsv, cfg) | HSV | mask | Low saturation and brightness above WHITE_V_REL × the frame's bright percentile (relative, not absolute) |
| mask_red(hsv, cfg) | HSV | mask | Two hue ranges (red wraps around 0/179) plus minimum saturation |
| roi_brightness(hsv, roi) | HSV, region | float | Mean V, for exposure management and health |
| make_front_masks(frame, cfg) | Frame | FrontMasks | Downscale, convert once, build all front masks |
| make_side_masks(frame, cfg) | Frame | (road, green) | Same for the side camera |

### perception/pipeline.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| process_front_frame(frame, signs_enabled, cfg) | Frame, flag | FrontResult | Front thread entry: masks once, lane, junction, and signs only when the flag is set and SIGN_HZ allows |
| process_side_frame(frame, cfg) | Frame | SideResult | Side thread entry: masks, edge, opening |

### perception/front_lane.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| find_yellow_x(yellow_band) | mask strip | float or None | Column centroid of yellow pixels; None if fewer than MIN_LINE_PX |
| find_road_edges(road_band) | mask strip | (left_x, right_x), either may be None | Outer extent of the black road in the band |
| centre_in_band(yellow_x, left_x, right_x, half_road_px) | x values, band width | float or None | Yellow x if found; else the midpoint of both edges; else one edge ± half the road width. Covers the gaps between dashes |
| band_to_ground(x_px, band, img_cx) | x, band calibration | (forward_m, lateral_m) | Pixel → metres from the rear axle, using FRONT_BANDS |
| fit_centre_line(points_m) | list of (forward, lateral) | (slope, intercept) or None | Least-squares line; None with fewer than 2 points |
| line_to_errors(fit) | line | (lateral_m, heading_err_rad) | Lateral at forward = 0 (the rear axle) and heading from the slope, signs per the conventions |
| process_front_lane(masks, cfg, t, seq) | FrontMasks | FrontLaneObs | Runs the above over every band; conf from bands found and fit residual; keeps band points in pixels for overlays |

### perception/side_lane.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| find_boundary_x(road, green, rows) | masks, SIDE_ROWS | float or None | Median road → green transition column across the sample rows (which side is road is set by SIDE_ROAD_ON_LEFT) |
| edge_to_offset(x_px, cfg) | x | float (m) | (x − SIDE_TARGET_PX) × SIDE_M_PER_PX: the robot's offset from the yellow line, measured at the camera |
| is_opening(green, cfg) | mask | bool | Green fraction below SIDE_GREEN_MIN_FRAC: a road leaves to the right here |
| process_side_lane(frame, cfg) | Frame | SideLaneObs | Masks → boundary → offset; conf from how many rows found the edge |

Passing a right-hand branch, the edge disappears, so conf drops and fusion falls back to the front camera automatically.

### perception/intersection.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| crossing_line_row(yellow_mask, cfg) | mask | int or None | Lowest row where yellow spans more than CROSS_LINE_MIN_FRAC of the width: the crossing road's centre line |
| road_widening(road_mask, cfg) | mask | float (0–1) | How far black road extends past its normal width in the look-ahead bands, either side |
| process_intersection(masks, cfg, t, seq) | FrontMasks | IntersectionObs | approach_score from both cues; crossing_line_y is the fallback position fix |

### perception/signs.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| find_sign_candidates(white_mask, cfg) | mask | list of bbox | Contours → keep four-cornered blobs of sign size and aspect on SIGN_SIDE of the road |
| pick_sign(candidates) | list of bbox | bbox or None | Largest candidate: the nearest sign on our side, not one meant for another approach |
| crop_sign(frame, bbox, scale) | full Frame, bbox | image | Crop from the full-resolution frame |
| is_stop(crop_hsv, cfg) | crop | bool | Red fraction ≥ SIGN_RED_MIN_FRAC and the red blob simplifies to OCTAGON_CORNERS corners |
| arrow_direction(crop, cfg) | crop | (SignType, score) | Dark arrow mask; compare the widest column in the left vs right half, and the widest row in the top vs bottom half. The arrowhead is the wide end; the axis with the stronger ratio wins (sideways → LEFT or RIGHT, head at top → STRAIGHT); a ratio under ARROW_ASYM_MIN → NONE |
| process_signs(frame, masks, cfg) | Frame, FrontMasks | SignObs | Candidates → pick → STOP if red octagon, else arrow direction |

### estimation/

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| is_fresh(obj, now, cfg) | any result | bool | t within MAX_OBS_AGE_S |
| side_to_axle(offset_m, heading_err_rad, cfg) | side offset, heading | float (m) | Offset − SIDE_CAM_X_M × sin(heading): moves the side reading to the rear axle |
| fuse_lane(front, side, now, cfg) | FrontResult, SideResult (either may be None) | LaneEstimate | Confidence-weighted lateral error at the axle; heading from the front camera; cam_gap_m when both are good |
| RoadYawTracker.update(imu, lane, state, now) | IMU, estimate, state | None | In LANE_FOLLOW or APPROACH, with lane conf ≥ ROAD_YAW_MIN_CONF and yaw rate under ROAD_YAW_MAX_RATE_DPS: blend yaw + heading error into road_yaw by ROAD_YAW_ALPHA |
| RoadYawTracker.after_turn(direction) | +1 left, −1 right | None | New road direction = old ± 90°, used until the camera re-anchors it |
| RoadYawTracker.get() | — | RoadYaw | Value, last anchor time, valid |
| check_health(front, side, imu, lane, road_yaw, fps, overruns, cfg) | latest everything | list of Warning | One small test per warning, each in its own helper |

### behaviour/sign_voter.py — SignVoter(cfg)

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| add(obs) | SignObs | None | Count consecutive identical signs; ignores a result already counted (same seq) |
| decided() | — | SignType or None | The sign once seen SIGN_VOTES_NEEDED times in a row |
| reset() | — | None | Clear after each junction |

### behaviour/state_machine.py — Behaviour(cfg)

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| step(front, side, lane, road_yaw, sign, imu, commands, now) | latest results | State | Commands first (START, STOP), then safety, then the current state's handler |
| _check_safety(lane, imu, now) | estimate, IMU | reason or None | Lane lost longer than LANE_LOST_TIMEOUT_S, data stale, or a motor serial error → SAFE_STOP with the reason |
| _on_<state>(…) | per state | next State | One small handler per state (Behaviour state machine section) |
| choose_action(sign, stop_done, cfg) | SignType or None, bool | Action | Sign → action; a STOP already served or no sign → DEFAULT_ACTION |
| _enter(state, now) | new state | None | Record entry time and the position-fix cue; set yaw_target_deg from road_yaw for TURN (± 90°) and CROSS; set or clear signs_enabled and exposure_hold; start a drift measurement in IDLE and STOP_HOLD |

### control/

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| LaneController.update(est, yaw_rate_dps, dt) | LaneEstimate, IMU rate | steer | KP_LAT·lateral + KH·heading − KD_YAW·yaw rate, smoothed, clamped to STEER_MAX |
| LaneController.reset() | — | None | Clear the smoothing filter when a lane state starts |
| turn_steer(yaw_deg, target_deg, cfg) | yaw, target | steer | P on yaw error, clamped, never below TURN_MIN_STEER so it doesn't stall |
| turn_done(yaw_deg, target_deg, front_conf, cfg) | yaw, target, front lane conf | bool | Error under TURN_TOL_DEG, or under TURN_HANDOFF_DEG once the front camera sees the new line with conf ≥ TURN_HANDOFF_CONF |
| vision_turn_steer(front, direction, cfg) | FrontResult | (steer, done) | Fallback if the IMU goes invalid mid-turn: pivot at VISION_TURN_STEER until the front camera sees the line with small heading error |
| hold_steer(yaw_deg, target_deg, cfg) | yaw, target | steer | Gentler P (HOLD_KP) to drive straight on road_yaw across a junction |
| target_speed(state, est, cfg) | State, estimate | 0–1 | Per-state base speed, scaled down by low conf or large heading error |
| drive_for_state(state, est, imu, behaviour, lane_ctrl, front, cfg) | all of the above | DriveCmd | Picks the steering source for the state (lane, hold, turn, vision turn or none) plus the speed |
| mix(cmd, cfg) | DriveCmd | MotorCmd | left = v − STEER_GAIN·steer, right = v + STEER_GAIN·steer, rescaled to ±1; speed 0 with steer pivots about the rear axle |

### ui/

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| start_ui(cfg) | config | (process, packet_q, cmd_q) | Start the display process (multiprocessing, spawn) with a one-slot packet queue and a small command queue. Launched over SSH, so the UI process sets defaults (never overwrites) for WAYLAND_DISPLAY=wayland-0 and XDG_RUNTIME_DIR=/run/user/<uid> before importing pygame |
| make_ui_packet(sys, include_views) | system | UiPacket | Numbers every loop; camera views only every UI_IMAGE_EVERY loops |
| send(packet_q, packet) | queue, packet | None | put_nowait; when full, drop the packet |
| read_commands(cmd_q) | queue | list of UiCommand | Non-blocking; UI commands only. control_step merges them with the existing terminal keyboard commands |
| run_ui(packet_q, cmd_q, cfg) | queues | — | Display process loop: newest packet → Screen.draw → taps → wait for UI_HZ |
| Screen.draw(packet, view) | packet, view | None | Compose the chosen view at DISPLAY_SIZE |
| draw_status(img, packet) | image, packet | None | State, speeds, errors, IMU, sign vote, fps, warnings |
| draw_front_overlay(img, front) | image, FrontResult | None | Band points, fitted line, crossing line, sign box and label |
| draw_side_overlay(img, side) | image, SideResult | None | Detected edge column, target column, opening flag |
| Screen.handle_input(events) | pygame events | UiCommand or None | Tap zones and keys |

### main.py

| Function | Inputs | Returns | Job |
| --- | --- | --- | --- |
| build_system(cfg) | config | System | Create slots, flags, sources, camera streams, IMU, Sabertooth, trackers, controllers, behaviour, display process |
| control_step(sys, now) | system, time | None | Read slots → fuse_lane → road_yaw.update → voter.add → check_health → behaviour.step → drive_for_state → mix → saber.set (every loop) → UI packet |
| run(sys) | system | None | Fixed-rate loop at CONTROL_HZ; counts overruns |
| shutdown(sys) | system | None | saber.stop() first, then the display process and threads; always runs (try/finally, Ctrl-C) |

## Behaviour state machine

Nine states decide who steers and how fast. Every junction exits through REACQUIRE, where the lane controller absorbs leftover turn error, and turn and crossing targets come from the camera-anchored road_yaw rather than the raw IMU heading.

Behaviour state machine · 9 states, SAFE_STOP reachable from all

DECIDE runs for a single step, so the robot only stops when a STOP sign was voted. For a turn, the advance after the position fix places the rear axle over the junction centre before the pivot; both advance times are measured on the course.

| State | Steering | Speed | Leaves when → next |
| --- | --- | --- | --- |
| IDLE | None | 0 | Measures IMU drift while still. START (tap or key) once calib_gyro = 3, or after IMU_CALIB_WAIT_S with a warning → LANE_FOLLOW |
| LANE_FOLLOW | Lane controller | CRUISE_SPEED | approach_score ≥ APPROACH_ON → APPROACH. road_yaw re-anchored continuously; managed exposure allowed |
| APPROACH | Lane controller | APPROACH_SPEED | Position fix → DECIDE: the right-side opening begins (exact, preferred), or the crossing line passes STOP_LINE_Y_FRAC (fallback when there is no right branch). approach_score < APPROACH_OFF → LANE_FOLLOW. Entry sets signs_enabled and exposure_hold |
| DECIDE | — | — | Same step: choose_action(voter.decided(), stop_done) → STOP_HOLD, TURN or CROSS; records which cue fired |
| STOP_HOLD | None | 0 | STOP_HOLD_S elapsed → DECIDE with stop_done set. Re-measures IMU drift while still |
| TURN | Advance: hold_steer on road_yaw. Rotate: turn_steer to road_yaw ± 90° | Advance: APPROACH_SPEED for ADVANCE_AFTER_OPENING_S or ADVANCE_AFTER_LINE_S, by cue. Rotate: TURN_SPEED (0 = pivot) | turn_done → REACQUIRE, then road_yaw.after_turn. IMU invalid → finish with vision_turn_steer. TURN_TIMEOUT_S → SAFE_STOP |
| CROSS | hold_steer on road_yaw | CROSS_SPEED | Right edge back after the opening, or front lane conf high past the junction → REACQUIRE. CROSS_TIMEOUT_S → REACQUIRE |
| REACQUIRE | Lane controller | REACQUIRE_SPEED | conf ≥ REACQUIRE_CONF for REACQUIRE_FRAMES → LANE_FOLLOW; voter reset, signs_enabled and exposure_hold cleared |
| SAFE_STOP | None | 0 | Reason shown on the display; manual START → IDLE |

## Display and debug view

The 7" display runs in its own process at about 10 Hz, fed by a one-slot queue, so a slow or crashed display can never delay a motor command. Perception already returns pixel geometry from the exact frame it measured, which is what live annotation of the camera feeds needs later.

### Views (tap or key to cycle)

| View | Shows | Cost |
| --- | --- | --- |
| Status | State and stop reason, speed and steer, lateral and heading error, lane conf, IMU yaw, road_yaw and its age, gyro calibration, drift, sign vote, fps per camera, loop overruns, warnings | Text only; cheapest |
| Front | Front camera view with band centre points, fitted line, crossing line, sign box and label, plus a status strip | One 320×240 image per packet |
| Side | Side camera view with the detected edge, the target column and the opening flag | One 320×240 image per packet |
| Split (default) | Front and side views with a compact status panel | Both images |

### What live annotation needs, built in now

- **Draw in the display process only**, never in the camera threads or the control loop.
- **Overlays come from the same frame.** Each camera writes one result object per frame (view image, observations, seq), so the geometry drawn can never belong to a different frame.
- **Geometry is in processed-image pixels** (320×240) with scale to the full frame; the screen scales once to the display.
- **Letterbox, never stretch.** The views are 4:3 and the 800×480 screen is 5:3, so a single view is shown at 640×480 with a 160 px column for START/STOP and key status. Stretching would distort the overlay geometry.
- **Last view kept, marked stale.** On loops without images, front and side are None and the display keeps the last view it received. Once that view is older than UI_VIEW_STALE_S it is greyed out with its age shown; NO DATA appears only if nothing has ever arrived.
- **Send small images only.** A 320×240 colour view is 230 KB; both cameras at 10 Hz is under 5 MB/s through the queue. If profiling shows the copy matters, swap the queue for shared memory behind the same send function.
- **One-slot queue, drop when full**, so the display always shows the newest data and never builds a backlog.
- **One drawing module.** ui/overlay.py is also used by tools/replay.py, so the debug view on the robot and on recordings is identical.
- **Headless option.** UI_ENABLED = False runs the robot with no display process at all.

### Controls and safety

- Taps or keys send START and STOP back through a second queue, read without blocking by control_step and merged with the terminal keyboard commands, so control still works over SSH. NEXT_VIEW stays inside the display process.
- The terminal status line stays in both modes (UI on or off).
- A STOP on the screen is a convenience, not an emergency stop. A physical switch on motor power is still needed, and the Sabertooth serial timeout covers a crashed program.
- Rendering uses pygame in full screen (it blits NumPy images and reports touch as mouse events); OpenCV draws the annotations onto each image before it is shown.

## Configuration variables

Every tunable lives once, in config.py, and nothing else defines a number. Values are starting points to tune on the course, not tested settings; "measured" means it comes from tools/calibrate.py, and colour values come from tools/tune_colors.py under the course lighting.

### Timing

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| CONTROL_HZ | 30 | Hz | run |
| IMU_HZ | 50 | Hz | ImuReader |
| CAM_FPS | 30 | fps | camera sources |
| MAX_OBS_AGE_S | 0.15; raise if the webcam's measured latency needs it | s | is_fresh, _check_safety |
| LANE_LOST_TIMEOUT_S | 0.5 | s | _check_safety |

### Cameras and geometry

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| FRONT_CAM_NUM | 0 | index | CsiSource |
| SIDE_CAM_DEVICE | your webcam's /dev/v4l/by-id/… entry | path | UsbSource |
| SIDE_CAM_FOURCC | YUYV | — | UsbSource |
| FRONT_SIZE | (640, 480) | px | CsiSource |
| FRONT_LANE_SIZE | (320, 240) | px | make_front_masks |
| SIDE_SIZE | (320, 240) | px | UsbSource |
| FRONT_BANDS | 5 bands of (y_top, y_bottom, forward_m from the rear axle, half_road_px, m_per_px), measured | px, m | front_lane.py |
| SIDE_ROWS | 5 rows spread over the frame | px | find_boundary_x |
| SIDE_ROAD_ON_LEFT | measured: True if the road appears on the image's left | bool | find_boundary_x |
| SIDE_TARGET_PX | measured with the robot centred on the yellow line | px | edge_to_offset |
| SIDE_M_PER_PX | measured | m/px | edge_to_offset |
| SIDE_CAM_X_M | measured; + = ahead of the rear axle | m | side_to_axle |
| MIN_LINE_PX | 30 | pixels per band | find_yellow_x, find_road_edges |

### Exposure and lighting

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| EXPOSURE_MODE | 'locked' (or 'managed', 'auto') | — | exposure.py |
| CAM_WARMUP_S | 2.0 | s | lock_after_warmup |
| EXPOSURE_MAX_US | 10000 | µs | lock_after_warmup, manage_step |
| EXPOSURE_TARGET_V | brightness at lock time | 0–255 | manage_step |
| EXPOSURE_STEP | 0.1 | fraction per step | manage_step |
| EXPOSURE_CHECK_FRAMES | 15 | frames | manage_step |
| USB_EXPOSURE_CTRLS | control names from v4l2-ctl --list-ctrls for your webcam | — | UsbSource |
| DARK_WARN_V, BRIGHT_WARN_V | 40, 220 | 0–255 | check_health |

### Colour tests (OpenCV HSV: H 0–179, S and V 0–255)

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| YELLOW_H_MIN, YELLOW_H_MAX | 20, 35 | hue | mask_yellow |
| YELLOW_S_MIN, YELLOW_V_MIN | 90, 60 | S, V | mask_yellow |
| GREEN_CHROM_MIN | 0.40 | G ÷ (R+G+B) | mask_green |
| DARK_SUM_MIN | 90 | R+G+B; below = too dark to judge | mask_green |
| ROAD_S_MAX | 70 | S | mask_road |
| ROAD_V_T_MIN, ROAD_V_T_MAX | 40, 110 | V; clamp on the Otsu threshold | mask_road |
| WHITE_S_MAX, WHITE_V_REL, WHITE_V_PCT | 50, 0.8, 95 | S, ratio, percentile | mask_white |
| RED_H_LOW_MAX, RED_H_HIGH_MIN | 10, 170 | hue | mask_red |
| RED_S_MIN | 100 | S | mask_red |

### Junctions and signs

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| APPROACH_ON, APPROACH_OFF | 0.6, 0.3 | score | _on_lane_follow, _on_approach (hysteresis) |
| CROSS_LINE_MIN_FRAC | 0.5 | fraction of width | crossing_line_row |
| STOP_LINE_Y_FRAC | 0.85 | fraction of height | _on_approach (fallback position fix) |
| SIDE_GREEN_MIN_FRAC | 0.2 | fraction | is_opening |
| SIGN_HZ | 10 | Hz | process_front_frame |
| SIGN_SIDE | 'right' (to confirm) | — | find_sign_candidates |
| SIGN_MIN_AREA_PX | 400 | px² | find_sign_candidates |
| SIGN_ASPECT_MIN, SIGN_ASPECT_MAX | 0.6, 1.6 | w ÷ h | find_sign_candidates |
| SIGN_RED_MIN_FRAC | 0.15 | fraction of crop | is_stop |
| OCTAGON_CORNERS | 7 to 9 | corners | is_stop |
| ARROW_ASYM_MIN | 1.6 | wide end ÷ narrow end | arrow_direction |
| SIGN_VOTES_NEEDED | 3 | frames | SignVoter |
| DEFAULT_ACTION | STRAIGHT | Action | choose_action |
| STOP_HOLD_S | 3.0 | s | _on_stop_hold |

### Speeds and lane control

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| CRUISE_SPEED | 0.5 | 0–1 | target_speed |
| APPROACH_SPEED, CROSS_SPEED, REACQUIRE_SPEED | 0.3, 0.3, 0.3 | 0–1 | target_speed |
| KP_LAT | 4.0 | steer per m | LaneController |
| KH | 0.8 | steer per rad | LaneController |
| KD_YAW | 0.004 | steer per deg/s | LaneController |
| STEER_SMOOTH | 0.5 | 0 = none, near 1 = heavy | LaneController |
| STEER_MAX | 0.6 | steer | LaneController |
| REACQUIRE_CONF, REACQUIRE_FRAMES | 0.6, 5 | conf, frames | _on_reacquire |
| CAM_GAP_WARN_M | 0.03 | m | check_health |

### Turns and junction crossing

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| TURN_SPEED | 0.0 (pivot); above 0 gives an arc | 0–1 | target_speed, mix |
| TURN_ANGLE_DEG | 90 | deg | _enter |
| ADVANCE_AFTER_OPENING_S | measured; start 0.4 | s | _on_turn |
| ADVANCE_AFTER_LINE_S | measured; start 0.8 | s | _on_turn |
| TURN_KP | 0.02 | steer per deg | turn_steer |
| TURN_MIN_STEER | 0.25 | steer | turn_steer |
| TURN_TOL_DEG | 3 | deg | turn_done |
| TURN_HANDOFF_DEG, TURN_HANDOFF_CONF | 15, 0.6 | deg, conf | turn_done |
| VISION_TURN_STEER | 0.3 | steer | vision_turn_steer |
| TURN_TIMEOUT_S | 4.0 | s | _on_turn |
| HOLD_KP | 0.01 | steer per deg | hold_steer |
| CROSS_TIMEOUT_S | 3.0 | s | _on_cross |

### IMU and road yaw

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| IMU_I2C_ADDR | 0x28 | address | ImuReader |
| IMU_MAX_RATE_DPS | 300 | deg/s | _plausible |
| IMU_MAX_REJECTS | 5 | samples | _loop |
| IMU_STILL_S | 3.0 | s | measure_drift |
| IMU_DRIFT_MAX_DPS | 0.5 | deg/s | measure_drift |
| IMU_DRIFT_WARN_DPS | 0.2 | deg/s | check_health |
| IMU_CALIB_WAIT_S | 10 | s | _on_idle |
| ROAD_YAW_ALPHA | 0.05 | blend per loop | RoadYawTracker |
| ROAD_YAW_MIN_CONF | 0.7 | conf | RoadYawTracker |
| ROAD_YAW_MAX_RATE_DPS | 10 | deg/s | RoadYawTracker |
| ROAD_YAW_STALE_S | 5 | s | check_health |

### Sabertooth and motors

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| SABER_PORT | /dev/serial0 | path | SabertoothDriver |
| SABER_BAUD | 9600; must match your DIP switches | baud | SabertoothDriver |
| SABER_ADDRESS | 128 | 128–135 | SabertoothDriver |
| SABER_TIMEOUT_MS | 300 (sent as 3 × 100 ms) | ms | open |
| LEFT_CHANNEL, RIGHT_CHANNEL | 1, 2; confirm on the bench | motor | set |
| LEFT_INVERT, RIGHT_INVERT | False, False; confirm on the bench | bool | set |
| STEER_GAIN | 1.0 | — | mix |
| MOTOR_DEADBAND | 0.15 | 0–1 | _with_deadband |

### Display and health

| Variable | Start value | Units | Used by |
| --- | --- | --- | --- |
| UI_ENABLED | True | bool | build_system |
| UI_HZ | 10 | Hz | run_ui |
| UI_IMAGE_EVERY | 3 | loops | make_ui_packet |
| DISPLAY_SIZE | (800, 480); checked against pygame.display.Info() at startup, warning on mismatch | px | Screen |
| UI_VIEW_STALE_S | 1.0 | s | Screen |
| UI_START_VIEW | 'split' | — | Screen |
| CAM_FPS_MIN | 20 | fps | check_health |

## Testing and tools

Every block can be tested without the robot moving, by replaying recorded runs; nothing merges until its unit tests and a replay pass.

| Tool | Job |
| --- | --- |
| tools/record.py | Drive or push the robot slowly and log both camera streams and IMU samples with timestamps |
| tools/replay.py | Feed a recording through perception → fusion → behaviour with the motors stubbed; draws with ui/overlay.py |
| tools/tune_colors.py | Sliders over recorded frames to set the colour values; prints them to paste into config.py |
| tools/calibrate.py | From photos of a tape measure on the mat, computes FRONT_BANDS, SIDE_TARGET_PX, SIDE_M_PER_PX and SIDE_CAM_X_M |
| tools/ui_demo.py | Feeds synthetic UiPackets to the display process so the screen can be checked with no robot or sensors |
| tools/imu_drift_test.py | Logs yaw with the robot still, first with motors off, then with motors spinning on a stand, to measure drift and interference |

**Order of testing**

- Record every course layout under the lighting you will actually face (room lights, daylight, shadows), and each sign from several distances.
- Unit-test each perception function on saved frames with hand-labelled values, including deliberately dark, bright and shadowed frames.
- Replay whole runs and check the state sequence and sign decisions match what happened.
- On the bench with the wheels off the ground: steer + turns left, a left turn by hand raises yaw, Sabertooth channels and invert flags are right, and MOTOR_DEADBAND just starts the wheels. Then kill the program and confirm the Sabertooth timeout stops the wheels.
- Run imu_drift_test.py and set IMU_DRIFT_MAX_DPS and IMU_DRIFT_WARN_DPS from the results.
- Kill the display process while the robot runs on the stand; the control loop must not notice.
- On the course at APPROACH_SPEED everywhere, then raise CRUISE_SPEED once lane conf stays high. Measure ADVANCE_AFTER_OPENING_S and ADVANCE_AFTER_LINE_S by watching where the rear axle stops.
- Profile each thread on the Pi 5 and compare against the budgets in Runtime on the Raspberry Pi 5.

## Assumptions and open questions

Your notes settled the drive train, both cameras' interfaces and the side camera's view, centre-line driving, the sign set, the language and the display's role. The questions below still change code; each names what it affects. Tick them off as you confirm them.

- [ ] **Sabertooth setup:** packetized serial is assumed (for addressing and the serial timeout), driven from the Pi's UART (/dev/serial0). Which model is it (2x12, 2x25, 2x32, 2x60), what DIP-switch mode and baud rate, is it on the GPIO UART or a USB-serial adapter, and which motor is on M1? Affects sabertooth.py.
- [x] **Display model:** official 7" touch display, 800×480; main.py is launched over SSH, with the display on the Pi's desktop session.
- [ ] **Side camera position:** how far ahead of or behind the rear axle is it, and how high? Can the robot's own shadow fall in its view? Affects SIDE_CAM_X_M and the opening-based advance.
- [ ] **Camera models:** which webcam and which CSI camera? Webcam exposure control names and the CSI field of view differ by model.
- [ ] **Junction shapes:** four-way only, or T-junctions too? With no road to the right, the side camera sees no opening, so the front camera's crossing line becomes the only position fix. Affects APPROACH and TURN.
- [ ] **Stop sign behaviour:** where should the robot stop (at the position fix, or is there a stop line?), for how long, and then what: straight on, or follow another sign? Affects STOP_HOLD and choose_action.
- [ ] **Sign placement:** which side of the road, at what height, and can signs meant for other approaches be seen from yours? Affects SIGN_SIDE and pick_sign.
- [ ] **Arrow colour:** assumed a dark arrow on the white sign. Affects arrow_direction.
- [ ] **Road width:** is it the same on every mat? The side camera's target assumes a fixed edge-to-line distance; if it varies, fusion should lean on the front camera or re-learn the target on straights.
- [ ] **Lighting:** room lights only, or daylight from windows too? Decides whether EXPOSURE_MODE defaults to locked or managed.
- [ ] **Start and emergency stop:** screen tap, keyboard or a physical button to start? A physical kill switch on motor power is recommended either way.
- [ ] **Existing code:** is any of this already written? Proposed names should match it, not duplicate it.

### Sources

- [Open Panzer Scout ESC: Sabertooth-compatible packet serial reference](https://github.com/OpenPanzerProject/Scout-ESC) (packet format; motor commands 0, 1, 4, 5)
- [Visuino wiki: Sabertooth packet serial timeout](https://www.visuino.com/wiki/index.php/Sabertooth_Dual_Motor_Shield_(Packet_Serial).SerialTimeout) (timeout in 100 ms units)
