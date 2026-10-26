# Build status: design vs code

Inventory of `autonomous/` against `docs/DESIGN.md` (Oct 9, 2026 export) and
`docs/junction_plan.md`, taken 2026-10-26 at commit `07500b8`. No code was
changed for this report.

## Summary

- **About 49% done** (69 of 141 design items fully match; counting Partial as half, about 55%).
- **38 items missing**, **17 partial**, **17 conflicts** (Differs). 12 of the conflicts are naming or signature only: the behaviour matches and only the design text needs updating.
- **Config:** 25 design variables are missing from `config.py`, all for unbuilt features. 44 variables in `config.py` are not in the design. 19 shared variables have values that differ from the design's start value; most of those are measured or tuned values, which is expected.
- **Tests:** in this container as cloned, 273 pass, 12 fail, 1 is skipped and 2 files (48 tests) can't be collected. Every failure and collection error comes from the environment: the Raspberry Pi font and `pygame` aren't installed here. With `pygame` installed and a stand-in font at the Piboto path, **all 334 tests pass**.
- **What works today:** straight driving in LANE_FOLLOW on the front camera and IMU, with road-yaw anchoring, health warnings, the display process, loop logs, and the full front-camera calibration and steering-response toolchain.
- **What's missing:** the side camera's perception and fusion, the lane filter, junction detection, sign reading, the six junction states, and the sign tools.

Status key: **Done** = exists and matches the design. **Partial** = exists but pieces are missing. **Missing** = not in the code. **Differs** = exists but doesn't match the design (*sig* = naming or signature only).

---

## 1. Design items, one by one

### Layout and naming

| Design item | Code | Status | Notes |
| --- | --- | --- | --- |
| `robot/` root | `autonomous/` | **Differs** *sig* | Docstrings in `main.py` and `tools/replay.py` still say "Run from the robot/ folder". |
| `types.py` | `robot_types.py` | **Differs** *sig* | Renamed so it never shadows Python's `types` module (explained in its docstring). |
| `behaviour/` package | `behavior/` | **Differs** *sig* | The folder uses US spelling; the class is still `Behaviour`. |
| `tests/` | `test/` | **Differs** *sig* | |
| `ui/process.py` (`run_ui`) | `run_ui` is in `ui/screen.py` | **Differs** *sig* | There's no `ui/process.py`. |
| `models/` (`sign_svm.npz`) | — | see Tools | Written by `tools/train_signs.py` (missing). |
| `estimation/lane_filter.py` | — | see estimation/ | Counted under `LaneFilter` below. |
| `perception/side_lane.py`, `intersection.py`, `signs.py`, `behaviour/sign_voter.py` | — | see below | Not written; counted under their functions below. |

### Data types (`robot_types.py`)

| Design item | Status | Notes |
| --- | --- | --- |
| `Frame` | **Done** | |
| `FrontMasks` | **Done** | Adds `bgr` (the lane-size colour copy that becomes `FrontResult.view`). `white` and `red` are `Optional`: built only for sign frames. |
| `FrontLaneObs` | **Done** | |
| `IntersectionObs` | **Done** | The type exists; nothing fills it yet (`FrontResult.inter` is always None). |
| `SignObs` | **Done** | The `score` comment says "0-1"; the design says SVM margin (best minus runner-up). |
| `FrontResult` | **Done** | |
| `SideLaneObs`, `SideResult` | **Done** | Types exist; nothing produces them yet. |
| `ImuSample` | **Done** | |
| `LaneEstimate` | **Partial** | No `heading_src` field. |
| `RoadYaw`, `DriveCmd`, `MotorCmd` | **Done** | |
| `UiPacket` | **Done** | Adds `motor` (the `MotorCmd` actually sent). |
| `SignModel` (`hog`, `weights`, `biases`, `classes`) | **Missing** | |
| Enums `SignType`, `Action`, `State`, `UiCommand` | **Done** | |
| Enum `Warning` | **Differs** *sig* | Named `HealthWarning` to avoid shadowing the builtin `Warning`. Members match. |
| `LatestValue[T]` | **Done** | |
| Slots `front`, `side`, `imu` | **Partial** | `front` and `imu` exist; no `side` slot is created. |
| Events `signs_enabled`, `exposure_hold` | **Done** | Plus a third Event, `drift_request` (see Extras). |

### hal/camera.py

| Design item | Status | Notes |
| --- | --- | --- |
| `CameraStream(source, process_fn, slot, name)` | **Done** | Takes `cfg` as well. Adds `stop()` and `error_count`; a frame whose processing raises is skipped, not fatal. |
| `CameraStream.start()`, `_loop()`, `fps()` | **Done** | fps is measured over `CAM_FPS_WINDOW_S`. |
| `CsiSource(cam_num, size)`: `open`, `read`, `get_exposure`, `set_exposure`, `close` | **Done** | Takes `cfg` as well. Also locks Camera Module 3 manual focus at `FRONT_LENS_POSITION`. |
| `UsbSource(device_path, size, fourcc)`: `open`, `read`, `set_exposure`, `close` | **Done** | Opens by path with `CAP_V4L2`, YUYV, buffer size 1; warns if the webcam gives a different size. Never started by `main.py` yet. |
| `UsbSource.get_exposure()` "from the V4L2 controls named in config" | **Differs** | Uses OpenCV's `CAP_PROP_EXPOSURE` / `CAP_PROP_GAIN`, scaled by `USB_EXPOSURE_UNIT_US`. `USB_EXPOSURE_CTRLS` was replaced by `USB_AUTO_EXPOSURE_MANUAL/AUTO` and `USB_EXPOSURE_UNIT_US` (the config comment says so). |

### hal/exposure.py

| Design item | Status | Notes |
| --- | --- | --- |
| `lock_after_warmup(source, cfg)` | **Differs** *sig* | A method of `ExposureControl(source, name, cfg)`. Behaviour matches. When capping exposure at `EXPOSURE_MAX_US` it raises gain to keep the brightness, and if readback is unusable it falls back to `USB_LOCK_EXPOSURE_US` / `USB_LOCK_GAIN`. |
| `manage_step(source, mean_v, hold, cfg)` | **Differs** *sig* | `ExposureControl.manage_step(mean_v, hold)`. Behaviour matches, plus a tolerance band (`EXPOSURE_TOL_V`) and gain limits. |
| `EXPOSURE_MODE` 'locked' / 'managed' / 'auto' | **Done** | |
| Exposure for the side camera | **Missing** | Comes with the side camera wiring. |

### hal/imu.py

| Design item | Status | Notes |
| --- | --- | --- |
| `ImuReader(i2c_bus, address)` | **Differs** *sig* | `ImuReader(slot, drift_request, cfg, device_factory=Bno055)`. The bus is fixed as `busio.I2C(board.SCL, board.SDA)` and the address comes from `IMU_I2C_ADDR`. |
| `start()` | **Done** | The thread opens the sensor itself in IMUPLUS mode; the first good heading becomes yaw 0. |
| `read_raw()` | **Differs** *sig* | Returns `(heading_deg, gyro_z_dps, calib_gyro)`, or None on any failure. |
| `_to_robot_yaw(heading_deg, prev_yaw)` | **Done** | Sign from `IMU_YAW_SIGN`, confirmed on the bench. |
| `_plausible(new_yaw, prev_yaw, dt)` | **Done** | `dt` is measured from the last *good* read, so a run of rejects can't lock the reader out. |
| `measure_drift(seconds)` | **Differs** | There's no such function. Drift is measured continuously while the `drift_request` Event is set (by Behaviour in IDLE and STOP_HOLD): after `IMU_DRIFT_SETTLE_S`, then over back-to-back `IMU_STILL_S` windows. A window under `IMU_DRIFT_MAX_DPS` is stored. |
| `_loop()`, `valid=False` after `IMU_MAX_REJECTS` | **Done** | Also has a sanity gate (`_sane`: finite values, heading inside 0–360, gyro rate within limit). |

### hal/sabertooth.py

| Design item | Status | Notes |
| --- | --- | --- |
| `SabertoothDriver(port, baud, address)` | **Done** | Takes `cfg` as well. Latches `serial_error` instead of raising. |
| `open()` | **Differs** | Sends the serial timeout (command 14) as designed, with no bauding byte. It also waits `SABER_STARTUP_WAIT_S` (2 s), and it sends **command 17 (deadband = `SABER_HW_DEADBAND`)**, a setting the controller stores through power cycles. The design doesn't mention either. |
| `set(cmd)` | **Done** | |
| `stop()` | **Done** | Called by `shutdown`. SAFE_STOP doesn't call it; the loop sends 0 through `set()` every loop instead, which has the same effect. |
| `_side_packet`, `_packet`, `_with_deadband` | **Done** | |

### perception/color.py

| Design item | Status | Notes |
| --- | --- | --- |
| `mask_yellow`, `mask_green`, `mask_road`, `mask_white`, `mask_red`, `roi_brightness` | **Done** | |
| `make_front_masks(frame, cfg)` | **Done** | Adds `with_signs=False`: `white` and `red` are built only when it's set. The ROI is the rows covered by `FRONT_BANDS`. |
| `make_side_masks(frame, cfg)` | **Done** | Written but unused until the side camera is wired. |

### perception/pipeline.py

| Design item | Status | Notes |
| --- | --- | --- |
| `process_front_frame(frame, signs_enabled, sign_model, cfg, tracker=None)` | **Partial** | The signature has no `sign_model`. `inter` and `sign` are always None, and nothing paces sign reading at `SIGN_HZ`. Masks and lane work as designed. |
| `process_side_frame(frame, cfg)` | **Missing** | |

### perception/front_lane.py

| Design item | Status | Notes |
| --- | --- | --- |
| `find_yellow_x(yellow_band)` | **Done** | Takes `min_px` as an argument. |
| `find_road_edges(road_band)` "outer extent of the black road" | **Differs** | Returns the edges of the continuous road stretch that contains a seed column (the yellow x, the predicted x or the image centre). Gaps up to `ROAD_GAP_MAX_PX` are bridged, a column counts as road at `ROAD_COL_FRAC`, and an edge touching the image border is None. Signature: `(road_band, seed_x, min_px, cfg)`. More robust than outer extent, but it isn't what the design says. |
| `centre_in_band`, `band_to_ground`, `line_to_errors` | **Done** | |
| `fit_centre_line(points_m)` | **Done** | Only the tests call it; production uses `fit_robust` → `_weighted_fit`. |
| `fit_robust(points_m, weights, cfg)` | **Done** | |
| `LaneTracker` | **Done** | `window_fit(t, cfg)` and `update(fit, conf, t, cfg)`. One per camera thread (`main._build_front_camera`) and one per replay. |
| `process_front_lane(masks, cfg, t, seq, tracker=None)` | **Done** | `conf` = bands found ÷ bands × (1 − rms ÷ `FRONT_FIT_RMS_MAX_M`). |
| Stray-yellow guards (search window, outlier-resistant fit) | **Done** | |

### perception/side_lane.py

| Design item | Status | Notes |
| --- | --- | --- |
| `find_boundary_x(road, green, rows)` | **Missing** | |
| `edge_to_offset(x_px, cfg)` | **Missing** | |
| `is_opening(green, cfg)` | **Missing** | |
| `process_side_lane(frame, cfg)` | **Missing** | |

### perception/intersection.py

| Design item | Status | Notes |
| --- | --- | --- |
| `crossing_line_row(yellow_mask, cfg)` | **Missing** | |
| `road_widening(road_mask, cfg)` | **Missing** | |
| `process_intersection(masks, cfg, t, seq)` | **Missing** | |

### perception/signs.py

| Design item | Status | Notes |
| --- | --- | --- |
| `find_sign_candidates(white_mask, cfg)` | **Missing** | |
| `pick_sign(candidates)` | **Missing** | |
| `crop_sign(frame, bbox, scale)` | **Missing** | |
| `normalise_crop(crop, cfg)` | **Missing** | |
| `load_sign_model(path, cfg)` | **Missing** | |
| `hog_features(gray, model)` | **Missing** | |
| `classify_sign(features, model)` | **Missing** | |
| `red_check(crop_hsv, cfg)` | **Missing** | |
| `process_signs(frame, masks, model, cfg)` | **Missing** | |

### estimation/

| Design item | Status | Notes |
| --- | --- | --- |
| `is_fresh(obj, now, cfg)` | **Done** | |
| `side_to_axle(offset_m, heading_err_rad, cfg)` | **Done** | Raises if `SIDE_CAM_X_M` is None. |
| `fuse_lane(front, side, imu, road_yaw, side_bias, now, cfg)` | **Partial** | It's `fuse_lane(front, side, now, cfg)`: an older first cut. It accepts a side reading. Missing: weights of conf ÷ σ² (it weights by raw conf), the `FUSE_GATE_M` gate with keep-the-nearer rule, `SideBias`, the IMU heading fallback (`road_yaw − yaw`), `heading_src`, and `SIDE_FUSION_ENABLED`. Output `conf` is the max of the readings; `t` is the newest reading's time. With no front heading, the side reading is moved to the axle with heading 0. |
| `LaneFilter.update(...)` | **Missing** | |
| `LaneFilter.reset()` | **Missing** | |
| `SideBias.update(raw_gap_m, state, heading_err_rad)` | **Missing** | |
| `SideBias.get()` | **Missing** | |
| `RoadYawTracker.update(imu, lane, state, now)` | **Partial** | Anchors in LANE_FOLLOW / APPROACH on conf and yaw rate as designed, but doesn't check `heading_src == front` (the field doesn't exist yet). Safe today because heading only ever comes from the front camera; it must be added with the IMU heading fallback. The first anchor jumps straight to the measured value. |
| `RoadYawTracker.after_turn(direction)`, `get()` | **Done** | |
| `check_health(front, side, imu, lane, road_yaw, fps, overruns, cfg)` | **Partial** | Takes `motor_error` and `now` as well. All 11 warnings exist, but CAMS_DISAGREE fires only on `|cam_gap_m| > CAM_GAP_WARN_M`: the triggers for "side offset at `SIDE_BIAS_MAX_M`" and "`FUSE_GATE_WARN_COUNT` gate rejections in a row" are missing. LANE_CONF_LOW uses a new `LANE_CONF_WARN`. |

### behaviour/sign_voter.py

| Design item | Status | Notes |
| --- | --- | --- |
| `SignVoter.add(obs)` | **Missing** | |
| `SignVoter.decided()` | **Missing** | |
| `SignVoter.reset()` | **Missing** | |

### behaviour/state_machine.py

| Design item | Status | Notes |
| --- | --- | --- |
| `Behaviour(cfg)` | **Done** | Also takes the three Events: `signs_enabled`, `exposure_hold`, `drift_request`. |
| `step(front, side, lane, road_yaw, sign, imu, commands, now)` | **Done** | Takes `motor_error` as well. Order: commands, then safety (moving states only), then handler. A STOP command from any state goes to IDLE. |
| `_check_safety(lane, imu, now)` | **Done** | Takes `motor_error` as well. "Data stale" means a stale IMU, checked only in TURN and CROSS; stale camera data shows up as "lane lost" through `fuse_lane`'s freshness test. |
| `_on_idle` | **Done** | Goes at once if there has never been an IMU sample. The `IMU_CALIB_WAIT_S` timer counts from entering IDLE, not from START. The "warning" is a terminal print plus the existing IMU_NOT_CALIBRATED health warning. |
| `_on_lane_follow` | **Done** | APPROACH on `approach_score ≥ APPROACH_ON` (never fires yet: `inter` is always None). |
| `_on_approach` (hysteresis, opening debounce, crossing-line fallback) | **Missing** | Goes to `_not_built`, which raises `NotImplementedError`. |
| `_on_decide` (with the right-turn check) | **Missing** | |
| `_on_stop_hold` | **Missing** | |
| `_on_turn` (advance, rotate, vision fallback, timeout) | **Missing** | |
| `_on_cross` (edge-back debounce, conf fallback, timeout) | **Missing** | |
| `_on_reacquire` | **Missing** | |
| `_on_safe_stop` | **Done** | |
| `choose_action(sign, stop_done, cfg)` | **Done** | `DEFAULT_ACTION` is stored as a name string (`"STRAIGHT"`). |
| `_enter(state, now)` | **Partial** | Sets entry time, `exposure_hold`, `signs_enabled` and drift (through `drift_request`). Missing: `yaw_target_deg` for TURN and CROSS, recording the position-fix cue, and `opening_seen`. |

### control/

| Design item | Status | Notes |
| --- | --- | --- |
| `LaneController.update(est, yaw_rate_dps, dt, speed)` | **Done** | Clamps the raw steer, then smooths it; `dt` is accepted but unused (fixed rate). |
| `LaneController.reset()` | **Done** | `control_step` calls it on entering any lane state. |
| `turn_steer`, `turn_done`, `vision_turn_steer`, `hold_steer` | **Done** | Written and tested; not called yet (TURN and CROSS aren't built). |
| `target_speed(state, est, cfg)` | **Partial** | Per-state base speed × conf factor × heading factor (`SPEED_MIN_SCALE`, `SPEED_HEADING_SLOW_RAD`). Missing: slowing when the cameras disagree by more than `CAM_GAP_WARN_M`, and the `APPROACH_SPEED` cap while `heading_src` is imu. |
| `drive_for_state(state, est, imu, behaviour, lane_ctrl, front, cfg)` | **Partial** | Lane states and stopped states are done; TURN and CROSS raise `NotImplementedError`. In lane states, steer is also capped at `LANE_STEER_SPEED_RATIO × speed` (see Extras). |
| `mix(cmd, cfg)` | **Done** | `STEER_TRIM` interpolation and the `STEER_DEADBAND` lift with ramp, as designed. |

### ui/

| Design item | Status | Notes |
| --- | --- | --- |
| `start_ui(cfg)` | **Done** | Spawn context, one-slot packet queue, command queue of `UI_CMD_QUEUE_SIZE`. |
| `make_ui_packet(sys, include_views)` | **Differs** *sig* | `make_ui_packet(sys, now, drive, motor, lane, road_yaw, imu, sign_vote, fps, overruns, warnings, front, side, include_views)`. |
| `send(packet_q, packet)`, `read_commands(cmd_q)` | **Done** | `read_commands` passes on only START and STOP; NEXT_VIEW stays in the UI process. |
| `run_ui(packet_q, cmd_q, cfg)` | **Differs** *sig* | It's in `ui/screen.py`, and `cfg` is optional (the process imports config itself). Also exits when the parent process dies, and on SIGTERM. |
| `Screen.draw(packet, view)` | **Done** | Returns the composed BGR image. |
| `draw_status(img, packet)` | **Partial** | It's in `ui/panels.py` as `draw_status(img, packet, cfg)`. It shows every number except the sign vote. |
| `draw_front_overlay(img, front)` | **Partial** | Draws band points, the fitted line, the robot centreline and a status strip. Missing: the crossing line, and the sign box and label. |
| `draw_side_overlay(img, side)` | **Partial** | Takes `cfg` (for `SIDE_TARGET_PX`). Draws the edge and target columns; missing the opening flag. |
| `Screen.handle_input(events)` | **Done** | |
| Views: Status, Front, Side, Split (default) | **Differs** | The views are DRIVE (default: gauge, steer bar, pre-flight, strip chart), CAMERA (front with a side inset) and DIAGNOSTIC (every number). `UI_START_VIEW` is `"drive"`, not `'split'`. |
| Headless option (`UI_ENABLED = False`) | **Done** | |
| `tools/replay.py` uses `ui/overlay.py` | **Missing** | Replay and `tune_colors` still use the stop-gap `tools/debug_draw.py` (there's a TODO about it in `ui/overlay.py`). |

### main.py

| Design item | Status | Notes |
| --- | --- | --- |
| `build_system(cfg)` | **Partial** | Builds the IMU, front camera and exposure lock, Sabertooth, trackers, controller, behaviour, display process and loop log. Missing: the side slot, `UsbSource` stream and its exposure, the sign model, `SignVoter` and `SideBias` / `LaneFilter`. Refuses to start without `FRONT_BANDS`. |
| `control_step(sys, now)` | **Partial** | Passes `side=None` and `sign=None`; no `voter.add`. Otherwise follows the designed order, plus a loop-log row and a terminal status line. |
| `run(sys)` | **Done** | Overruns are counted over `OVERRUN_WINDOW_S`. |
| `shutdown(sys)` | **Done** | Order: `saber.stop`, `saber.close`, IMU, loop log, display, camera. Each step runs even if an earlier one fails. |
| Start-up camera-check line | **Done** | `tools/calib_status.startup_line`. |

### Tools

| Design item | Status | Notes |
| --- | --- | --- |
| `tools/record.py` | **Partial** | Front camera only, to PNG frames plus `frames.csv` and `meta.json`. The design asks for **both camera streams and IMU samples**. |
| `tools/replay.py` | **Partial** | Front lane → `fuse_lane` → `drive_for_state(LANE_FOLLOW)` → `mix` only. No behaviour, side camera or IMU; draws with `debug_draw`, not `ui/overlay.py`. |
| `tools/tune_colors.py` | **Partial** | Sliders for yellow and road only. Green, white and red (`GREEN_CHROM_MIN`, `DARK_SUM_MIN`, `WHITE_*`, `RED_*`) aren't tunable, and side-camera recordings aren't supported. |
| `tools/calibrate.py` (front: fit, check, lens) | **Done** | |
| `tools/calibrate.py` side values (`SIDE_TARGET_PX`, `SIDE_M_PER_PX`, `SIDE_CAM_X_M`, `SIDE_ROAD_ON_LEFT`) | **Missing** | The design defers this to the side camera build, but its build step 2 needs it. |
| `tools/calib_ui.py` (session, `--check`, `--lens`, `--resume`, REFIT) | **Done** | |
| `tools/config_writer.py` | **Done** | |
| `tools/guide_draw.py` | **Done** | Used by `calib_ui`, `steer_test` and `menu`. |
| `tools/imu_drift_test.py` (sign check, still, motors) | **Done** | |
| `tools/steer_test.py` | **Done** | |
| `tools/steer_report.py` (incl. `latency`) | **Done** | APPLY skips the steering values when the steering looks reversed, as designed. |
| `tools/collect_signs.py` | **Missing** | |
| `tools/train_signs.py` | **Missing** | |
| `tools/menu.py` (all 8 tiles, highlights, re-reads config) | **Done** | |
| `tools/lane_sim.py` (incl. `--log`) | **Done** | |
| `tools/calib_status.py` | **Done** | |

### Tests named in the design

| Design item | Status | Notes |
| --- | --- | --- |
| Unit tests per block with motors stubbed | **Done** | 334 tests across 16 files; hardware is faked. |
| Closed-loop sim in `test_control.py` "with camera latency and dropouts, filter against first-cut fusion" | **Missing** | `test_control.py` has no closed-loop sim. Sim tests are in `test_steer.py` (through `lane_sim`), with no dropout or filter comparison. |
| Lane-filter replay tests (dropped frames, injected outliers) | **Missing** | |

### Configuration variables

**Missing from `config.py` (25)**, all for unbuilt features:

| Group | Variables |
| --- | --- |
| Side fusion | `SIDE_FUSION_ENABLED`, `FRONT_LAT_SIGMA_M`, `SIDE_LAT_SIGMA_M`, `FUSE_GATE_M`, `FUSE_GATE_WARN_COUNT`, `SIDE_BIAS_ALPHA`, `SIDE_BIAS_MAX_M`, `SIDE_BIAS_MAX_HEADING_RAD` |
| Lane filter | `FRONT_HEAD_SIGMA_RAD`, `LANE_Q_LAT`, `LANE_Q_HEAD`, `LANE_CONF_FLOOR`, `LANE_GATE_FRONT`, `LANE_GATE_SIDE`, `LANE_REJECT_RESET_N`, `LANE_PREDICT_MAX_S`, `LANE_CONF_SIGMA_M` |
| Junctions | `SIDE_OPENING_FRAMES`, `SIDE_EDGE_BACK_FRAMES` |
| Signs | `SIGN_CROP_PX`, `SIGN_MARGIN_MIN`, `SIGN_CLAHE_CLIP`, `SIGN_CLAHE_TILES`, `SIGN_MODEL_PATH` |
| Exposure | `USB_EXPOSURE_CTRLS` (replaced on purpose; see Extras) |

**Values that differ from the design's start value (19 shared variables):**

| Variable | Design | Code | Why (from code comments) |
| --- | --- | --- | --- |
| `YELLOW_H_MIN`, `YELLOW_H_MAX`, `YELLOW_S_MIN`, `YELLOW_V_MIN` | 20, 35, 90, 60 | 21, 33, 116, 72 | Tuned ("From the test") |
| `ROAD_S_MAX`, `ROAD_V_T_MIN`, `ROAD_V_T_MAX` | 70, 40, 110 | 106, 55, 158 | Tuned |
| `CRUISE_SPEED` | 0.5 | 0.6 | No comment |
| `KP_LAT` | 4.0 | 2.0 | No comment |
| `IMU_STILL_S` | 3.0 | 2.0 | Shortened so `IMU_DRIFT_SETTLE_S` + `IMU_STILL_S` fits inside `STOP_HOLD_S` |
| `IMU_DRIFT_MAX_DPS`, `IMU_DRIFT_WARN_DPS` | 0.5, 0.2 | 0.1, 0.05 | Set from `imu_drift_test` (bench drift about 0.001 °/s) |
| `SABER_TIMEOUT_MS` | 300 | 200 | "course example value" |
| `LEFT_CHANNEL`, `RIGHT_CHANNEL` | 1, 2 | 2, 1 | Confirmed on the bench |
| `LEFT_INVERT`, `RIGHT_INVERT` | False, False | True, True | Confirmed on the bench |
| `DISPLAY_SIZE` | open question | (800, 480) | "official 7" touch display" |
| `UI_START_VIEW` | 'split' | 'drive' | Views differ (see ui/) |

Also: the `SABER_BAUD` comment says "must match the DIP switches", but the design says the baud rate is set by command 15, not by DIP switches. The comment is wrong; the value (9600) is fine.

---

## 2. Status counts

| Status | Count |
| --- | --- |
| Done | 69 |
| Partial | 17 |
| Missing | 38 |
| Differs | 17 (12 naming or signature only) |
| **Total** | **141** |

These count rows in the tables above (some rows group closely related functions). The config tables and the 25 missing config names aren't included.

---

## 3. In the code but not in the design

Decide for each whether to keep it (and add it to the design) or remove it.

### Modules

| Module | What it does |
| --- | --- |
| `loop_log.py` | One CSV row per control loop (`logs/run_<time>.csv`), with the gains on line 1. The design refers to "loop logs" (latency, `lane_sim --log`) but doesn't list the module. |
| `ui/panels.py` | DRIVE and DIAGNOSTIC panels: header, lane gauge, steer bar, pre-flight list, strip chart (`History`), `draw_status`. |
| `ui/text.py` | Anti-aliased text through Pillow in `UI_FONT` / `UI_FONT_BOLD`. Adds a Pillow dependency and the Piboto font. |
| `ui/__init__.py` | Package docstring only. |
| `tools/recording.py` | Recording format, writer thread and readers, shared by the tools. |
| `tools/debug_draw.py` | Stop-gap lane drawing for `replay` and `tune_colors`. Flagged in `ui/overlay.py` to be replaced; it breaks the "one drawing module" rule. |
| `tools/ui_demo.py` | Synthetic packets to the display. The Tools menu table names it, but the tools list doesn't. |
| `tools/weave_report.py` | Compares loop logs (lateral RMS, steer sign changes per second). |

### Classes and functions

- **hal:** `Bno055` (device wrapper); `ImuReader.close`, `_sane`, `_accept`, `_reject`, `_track_drift`; `CameraStream.stop`, `_process`, `error_count`; `CsiSource._lock_focus`; `UsbSource._warn_if_size_differs`; `ExposureControl` with `readback_ok`, `cap_exposure`, `nudge`, `_fallback_values`; `SabertoothDriver.send_command`, `set_auto_stop`, `set_deadband` (command 17), `close`, `map_integer`, `_write`, `serial_error`.
- **perception:** `color.front_lane_roi`; `front_lane.fit_rms`, `_confidence`, `_predicted_x`, `_band_centre_x`, `_bridge_gaps`, `_run_containing`, `_line_ends_px`, `_weighted_fit`, `_miss_without`, `LaneTracker.update`.
- **control:** `command._arc_limit` (lane steer ≤ `LANE_STEER_SPEED_RATIO` × speed, so the inner wheel never reverses); `lane_controller.speed_scale`; `mixer.compensate`, `steer_trim`; `speed_governor._conf_factor`, `_heading_factor`.
- **behaviour:** the `drift_request` Event (replaces `measure_drift`), `_apply_commands` (STOP from any state → IDLE), `_not_built`, `_set_flag`.
- **main:** `System` dataclass, `_start_keyboard` (Enter = START, `s` + Enter = STOP in the terminal), `_report` / `_status_line` / `_imu_line` (terminal status every `STATUS_PRINT_S`), `_recent_overruns`, `_open_loop_log`, `_build_front_camera`.
- **ui:** `stop_ui`, `open_display`, `show`, `_drain`, `_route`, `_set_display_env` (SSH → Wayland defaults), `make_layout`, `Screen.update`, `next_view`, `mark_stale`, `draw_no_data`, `cm`, `deg`, colour theme, pre-flight list.
- **types:** `FrontMasks.bgr`, `UiPacket.motor`.

### Config variables (44)

| Group | Variables |
| --- | --- |
| Cameras | `FRONT_LENS_POSITION`, `CAM_READ_RETRY_S`, `CAM_FPS_WINDOW_S` |
| Front lane | `ROAD_COL_FRAC`, `ROAD_GAP_MAX_PX`, `FRONT_FIT_RMS_MAX_M` |
| Exposure | `EXPOSURE_MIN_US`, `EXPOSURE_TOL_V`, `EXPOSURE_GAIN_MIN`, `EXPOSURE_GAIN_MAX`, `USB_LOCK_EXPOSURE_US`, `USB_LOCK_GAIN`, `USB_AUTO_EXPOSURE_MANUAL`, `USB_AUTO_EXPOSURE_AUTO`, `USB_EXPOSURE_UNIT_US` |
| Signs (leftovers?) | `OCTAGON_CORNERS`, `ARROW_ASYM_MIN`: shape-test values from before the HOG + SVM design; nothing uses them |
| Speed and lane | `LANE_STEER_SPEED_RATIO`, `SPEED_MIN_SCALE`, `SPEED_HEADING_SLOW_RAD`, `LANE_CONF_WARN` |
| IMU | `IMU_YAW_SIGN`, `IMU_GYRO_SIGN`, `IMU_MODE_SETTLE_S`, `IMU_JOIN_TIMEOUT_S`, `IMU_DRIFT_SETTLE_S`, `DRIFT_TEST_PRINT_HZ`, `DRIFT_TEST_SPEED`, `DRIFT_TEST_SIGN_MIN_DPS` |
| Sabertooth | `SABER_STARTUP_WAIT_S`, `SABER_READ_TIMEOUT_S`, `SABER_HW_DEADBAND` |
| Display and logs | `UI_VIEW_STALE_S`, `UI_CHART_S`, `UI_CHART_LAT_M`, `UI_CHART_GAP_S`, `UI_CMD_QUEUE_SIZE`, `UI_STOP_TIMEOUT_S`, `UI_FONT`, `UI_FONT_BOLD`, `STATUS_PRINT_S`, `LOOP_LOG_ENABLED`, `LOOP_LOG_DIR`, `OVERRUN_WINDOW_S` |

The five `USB_*` exposure names replace the design's `USB_EXPOSURE_CTRLS`.

### Outside `autonomous/`

`pytest` from `autonomous/` doesn't collect these, but they're in the repo:

- `python3/`: a committed **virtualenv** (1,623 files, aarch64, Python 3.11). It can't run anywhere but the Pi, and it's the only record of dependencies: there's no `requirements.txt`.
- Course reference files at the root: `display.py`, `imu_controller.py`, `sabertooth.py`, `ui.py`, `usb_sound_controller.py`, `test_camera.py`, `test_display.py`, `test_imu.py`, and `manual/` (PS5 manual driving with its own `sabertooth.py`).

---

## 4. Test run

Command: `python -m pytest` from `autonomous/`.

**Environment:** Python 3.13.17 (the Pi venv is 3.11), OpenCV 5.0.0, NumPy 2.5.3, Pillow and pyserial installed. Not installed: `picamera2`, `libcamera`, `adafruit_bno055`, `board`, `busio`, `pygame`, and the Piboto font (`/usr/share/fonts/truetype/piboto/`). No test needs the hardware libraries: cameras, IMU and serial are all faked, so those tests run here.

### Run A: container as cloned

**273 passed, 12 failed, 1 skipped, 2 collection errors (48 tests not run).**

| File | Passed | Failed | Not run | Cause |
| --- | --- | --- | --- | --- |
| test_calibrate.py | 15 | | | |
| test_camera.py | 14 | | | |
| test_control.py | 23 | | | |
| test_estimation.py | 33 | | | |
| test_exposure.py | 17 | | | |
| test_imu.py | 36 | | | |
| test_loop_log.py | 4 | | | |
| test_main.py | 23 | | | |
| test_perception.py | 27 | | | |
| test_sabertooth.py | 12 | | | |
| test_state_machine.py | 34 | | | |
| test_tools.py | 23 | | | |
| test_calib_ui.py | 0 | 8 | 1 skipped | Failures: `OSError: cannot open resource` from Pillow loading `UI_FONT` (Piboto not installed) in `ui/text.py`. Skip: `importorskip("pygame")` at line 218. |
| test_steer.py | 12 | 4 | | Same font error (the guided-screen tests draw text). |
| test_ui.py | | | 40 (collection error) | `import pygame` at module level: `ModuleNotFoundError`. |
| test_tools_menu.py | | | 8 (collection error) | Imports `bare_packet` from `test_ui`, so it fails the same way. |

### Run B: diagnostic, to see whether anything real hid behind those

`pip install pygame` (2.6.1), DejaVu Sans symlinked at the two Piboto paths, `SDL_VIDEODRIVER=dummy`. This changed only the throwaway container; no repo files changed, and the font links were removed afterwards.

**334 passed, 0 failed.** No real failures: everything in Run A was environment.

### Not covered by any test

- Real Picamera2, V4L2 webcam, BNO055 over I²C, the serial port, and the full-screen display. These run only on the Pi.
- On a machine without the Pi font, any test that draws text fails rather than skipping. Worth deciding whether tests should fall back to a default font or skip (see Questions).

---

## 5. Proposed build order

Dependencies first, core driving path before tools. Each batch is one pull request, with its tests and a replay or bench check before merging.

### Core

**PR 1: Types and config groundwork.** Small; unblocks everything below.
- `LaneEstimate.heading_src`; `SignModel` dataclass; side slot in `System`.
- All 25 missing config names, with the design's start values (`None` where measured), `SIDE_FUSION_ENABLED = False`.
- Settle the naming conflicts (*sig* rows) by updating the design, not the code. The live doc is the source of truth, so that's your edit.
- Depends on: nothing.

**PR 2: Side camera perception and wiring** (design build step 1).
- `perception/side_lane.py`: `find_boundary_x`, `edge_to_offset`, `is_opening`, `process_side_lane`.
- `process_side_frame` in `pipeline.py`.
- `build_system`: `UsbSource` stream, its `ExposureControl`, side slot; `control_step` passes `side` to fusion (still ignored, since `SIDE_FUSION_ENABLED` is False), health and behaviour; side fps in health and the display.
- `draw_side_overlay`: opening flag.
- Depends on PR 1. Needs `SIDE_CAM_DEVICE` from the Pi to run (not to test).

**PR 3: First-cut fusion** (design build step 3, first half).
- `SideBias` (`update`, `get`).
- Full `fuse_lane(front, side, imu, road_yaw, side_bias, now, cfg)`: σ weights, `FUSE_GATE_M` gate with the keep-the-nearer rule, learned offset, IMU heading fallback, `heading_src`, `SIDE_FUSION_ENABLED`, gate-rejection count.
- `RoadYawTracker.update`: anchor only on `heading_src == front`.
- `target_speed`: camera-disagreement scale and the `APPROACH_SPEED` cap on imu heading.
- `check_health`: CAMS_DISAGREE on the bias clamp and on `FUSE_GATE_WARN_COUNT`.
- Depends on PR 2. Enabling it on the robot needs PR 10 (side calibration) and PR 9 (two-camera recordings) to measure the σ values.

**PR 4: Lane filter** (design build steps 3 and 4). Optional as a separate step; see question Q14.
- `estimation/lane_filter.py`: `LaneFilter.update`, `reset`; predict, correct, gate, coast, conf from uncertainty.
- Reset hooks: entering LANE_FOLLOW or REACQUIRE, and after TURN.
- Closed-loop sim in `test_control.py` with latency and dropouts, filter against first-cut fusion; replay tests with dropped frames and outliers.
- Depends on PR 3 (`SideBias`, σ values) and a measured `SPEED_MPS_PER_UNIT` (`steer_test`, already built).

**PR 5: Junction perception.**
- `perception/intersection.py`: `crossing_line_row`, `road_widening`, `process_intersection`.
- `process_front_frame` fills `inter`.
- `draw_front_overlay`: crossing line.
- Depends on PR 1. Independent of PRs 2 to 4, so it can run in parallel.

**PR 6: Junction behaviour without signs** (straight crossings and stops).
- `behaviour/sign_voter.py` (`add`, `decided`, `reset`): pure logic, testable with hand-made `SignObs`.
- `_on_approach` (hysteresis, `SIDE_OPENING_FRAMES` debounce, crossing-line fallback), `_on_decide`, `_on_stop_hold`, `_on_cross` (`SIDE_EDGE_BACK_FRAMES`, conf fallback, `CROSS_TIMEOUT_S`), `_on_reacquire`.
- `_enter`: record the cue and `opening_seen`; `yaw_target_deg` for CROSS.
- `drive_for_state` for CROSS (`hold_steer` at `CROSS_SPEED`).
- `control_step`: `voter.add`; sign is None until PR 8, so `DEFAULT_ACTION` (STRAIGHT) drives every junction.
- Depends on PR 5 (approach score, crossing line) and PR 2 (opening, edge back).

**PR 7: Turns.**
- `_on_turn`: advance (`hold_steer`, cue-based `ADVANCE_AFTER_*`), rotate (`turn_steer`, `turn_done`), IMU-invalid fallback (`vision_turn_steer`), `TURN_TIMEOUT_S`.
- `_enter(TURN)`: `yaw_target_deg` = `road_yaw` ± `TURN_ANGLE_DEG`, with the junction plan's fallback to IMU yaw at entry.
- `road_yaw.after_turn`, `LaneFilter.reset` after TURN, the right-turn check in DECIDE.
- `drive_for_state` for TURN (per phase).
- Can be tested on the course before signs by setting `DEFAULT_ACTION` to LEFT or RIGHT.
- Depends on PR 6.

**PR 8: Sign reading.**
- `perception/signs.py`: all 9 functions; `load_sign_model` in `build_system` (start-up error if the file is missing or the crop size differs, unless signs are turned off; see Q31).
- `process_front_frame`: `sign_model` parameter, white and red masks on sign frames, `SIGN_HZ` pacing.
- Overlay sign box and label; sign vote row in `draw_status`.
- Depends on PR 6 (voter). Unit tests can use a tiny synthetic `.npz`; on the robot it needs PR 12's trained model.

### Tools

**PR 9: Recording and replay for two cameras and the IMU.**
- `record.py`: side stream and IMU samples with timestamps; recording format v2 (`frames.csv` per camera, `imu.csv`).
- `replay.py`: full path (perception both cameras → fusion → road yaw → behaviour → drive → mix) with the motors stubbed, drawing with `ui/overlay.py`.
- Switch `tune_colors.py` to `ui/overlay.py`; delete `tools/debug_draw.py`.
- Depends on PR 2. In practice **start this right after PR 2**: PR 3's σ values and every later replay check need these recordings.

**PR 10: Side camera calibration.**
- `calibrate.py` (and, if wanted, a `calib_ui` screen) for `SIDE_TARGET_PX`, `SIDE_M_PER_PX`, `SIDE_CAM_X_M`, `SIDE_ROAD_ON_LEFT`; APPLY through `config_writer`.
- A way to measure `FRONT_LAT_SIGMA_M` and `SIDE_LAT_SIGMA_M` from straight recordings (a `replay` report or a small tool; the design doesn't say which).
- Depends on PR 2 and PR 9.

**PR 11: Colour tuning for every mask.**
- `tune_colors.py`: green (`GREEN_CHROM_MIN`, `DARK_SUM_MIN`), white and red sliders; side-camera recordings.
- Depends on PR 9.

**PR 12: Sign training tools.**
- `tools/collect_signs.py`, `tools/train_signs.py` (laptop only: scikit-learn, augmentation, held-out recordings, confusion matrix, suggested `SIGN_MARGIN_MIN`), `models/` folder.
- Depends on PR 8's `find_sign_candidates`, `pick_sign`, `crop_sign` and `normalise_crop` (the shared crop path), and PR 9 recordings.

**Housekeeping (any time):** a `requirements.txt` (Pi and laptop sets) in place of the committed `python3/` venv; fix the "robot/ folder" docstrings and the `SABER_BAUD` comment; remove `OCTAGON_CORNERS` and `ARROW_ASYM_MIN` if they're not coming back.

```
PR1 ─┬─ PR2 ─┬─ PR3 ── PR4
     │       ├─ PR9 ─┬─ PR10 (σ, side calib → enable PR3 on the robot)
     │       │       └─ PR11
     └─ PR5 ─┴─ PR6 ── PR7
                 └─── PR8 ── PR12 (needs PR9 recordings)
```

---

## 6. Questions

Things I'd have to guess at to build. Each says what it blocks.

### Fusion and the lane filter (PRs 3 and 4)

1. **CAMS_DISAGREE: raw or corrected gap?** `check_health` fires on `|cam_gap_m| > CAM_GAP_WARN_M` today. The design keeps `cam_gap_m` raw (front − side *before* the learned offset), but the offset may legitimately reach `SIDE_BIAS_MAX_M` (0.03 m), which equals `CAM_GAP_WARN_M` (0.03 m). Should the warning (and the `target_speed` slow-down) use the gap *after* the offset?
2. **Which sigma is "the larger" for `FUSE_GATE_M`?** It says about 3 × the larger sigma once measured. Should it stay a hand-set number, or be computed?
3. **First-cut gate "keep the one nearer last loop's lateral":** with no previous estimate (start, after REACQUIRE), which reading wins?
4. **Fused `conf` in the first cut.** The design doesn't define it. Today it's the max of the two readings. Should it be weighted, or the max?
5. **`LaneEstimate.t`:** the newest reading's time (today), the front's time, or `now`?
6. **Side heading when the front is lost:** `side_to_axle` needs a heading. Use `road_yaw − yaw` (as in the dropout rule) or 0 (as today)?
7. **Lane filter "normalised error" and the gates.** 9.2 and 6.6 match χ² at 99% for 2 and 1 degrees of freedom, so I'd read them as *squared* Mahalanobis distance. Confirm.
8. **Lane filter `conf` falls to 0 at `LANE_CONF_SIGMA_M`:** linear in σ, or another shape?
9. **`LaneFilter.update(speed_cmd, …)`:** is that the previous loop's commanded speed (the one actually driving the robot)? This loop's isn't known until after fusion.
10. **"If only one camera keeps being rejected, raise CAMS_DISAGREE":** how many rejections, and is that `FUSE_GATE_WARN_COUNT`?
11. **σ measurement:** which tool measures `FRONT_LAT_SIGMA_M`, `SIDE_LAT_SIGMA_M` and `FRONT_HEAD_SIGMA_RAD`? A replay report, `calibrate.py`, or something new?
12. **Side calibration method:** how are `SIDE_M_PER_PX` and `SIDE_CAM_X_M` measured (a board under the camera, a tape offset, a sweep)? It decides PR 10's screens.
13. **`SIDE_FUSION_ENABLED` and the IMU heading fallback:** is the front-lost fallback (side lateral + `road_yaw − yaw`) gated by `SIDE_FUSION_ENABLED` too, or always on once the side camera runs?
14. **Build the first-cut gate and dropout rules at all, or go straight to `LaneFilter`?** The design builds both in step 3 and says the filter replaces the average "once built". Doing the full first cut and then replacing it is about twice the fusion work. My proposal: PR 3 builds `SideBias` plus a simple σ-weighted average (no gate or dropout logic), and PR 4 is the filter.

### Side camera (PR 2)

15. **`is_opening`: green fraction of what?** The whole frame, the sample rows, or only the side of the boundary where grass should be?
16. **Who makes the side masks?** `process_side_lane(frame, cfg)` takes a `Frame`, so it would make its own masks, but `process_side_frame` is described as "masks, edge, opening". Should `process_side_lane` take the masks instead (like `process_front_lane`)?
17. **Side `conf`:** "from how many rows found the edge". Rows found ÷ `len(SIDE_ROWS)`, and should spread between rows lower it?
18. **`boundary_found` vs `offset_m`:** is `boundary_found` True whenever `offset_m` is not None, or can a boundary be found with too few rows to trust?

### Junctions (PRs 5 to 7)

19. **`approach_score` from "both cues":** how are `road_widening` and the crossing line combined (max, weighted sum, crossing line only as a bonus)? Which rows are the "look-ahead bands": `FRONT_BANDS`, or rows above them? Is "normal width" 2 × `half_road_px` per band?
20. **`STOP_LINE_Y_FRAC`:** a fraction of the 240-px lane image height? Should the line be tracked across frames (it can vanish below the image before passing 0.85)?
21. **Advance after a STOP.** `ADVANCE_AFTER_OPENING_S` counts from the opening's first debounced frame. After a 3 s STOP_HOLD that time has long passed and the robot has stopped short. Does the advance restart after the stop (and at what speed), or does the robot stop only after advancing?
22. **DECIDE speed.** DECIDE lasts one loop and `drive_for_state` sends speed 0 for it, a 33 ms stop at `APPROACH_SPEED` before TURN or CROSS. Should DECIDE hold the previous command instead?
23. **CROSS "past the junction" for the conf fallback** (open in `junction_plan.md`): a time, the crossing line passing `STOP_LINE_Y_FRAC`, or a small heading error?
24. **`TURN_TIMEOUT_S`:** the design table says SAFE_STOP; the junction plan says "proposed, not yet confirmed". Which?
25. **Turn target when `road_yaw` was never anchored:** the junction plan says to use IMU yaw at TURN entry ± 90°; the design doesn't mention it. Adopt it?
26. **Speed during the vision-only turn fallback:** `TURN_SPEED`?
27. **`road_yaw.after_turn` when the turn ended on `vision_turn_steer`** (IMU invalid): still ± 90°, or mark `road_yaw` invalid until the camera re-anchors?
28. **RIGHT sign, no opening seen** (already open in the design): SAFE_STOP or turn on the crossing-line fix?
29. **Junction shapes and stop-sign behaviour** (already open in the design): these decide `_on_approach`, `_on_decide` and `choose_action`'s default at T-junctions, where STRAIGHT may not exist.

### Signs (PRs 8 and 12)

30. **Sign shape.** `find_sign_candidates` keeps "four-cornered blobs", but a STOP sign is usually an octagon, and `config.py` still has `OCTAGON_CORNERS = (7, 9)` from an older design. What do the real signs look like: square white signs with an arrow or STOP printed, or a red octagon? It decides the candidate test and whether `red_check` makes sense.
31. **Missing model at start-up:** the design says a missing `sign_svm.npz` stops start-up. Until PR 12 has produced a model, that would stop all driving. Should signs be optional (a flag, or the voter always NONE) while there's no model?
32. **Voter and NONE:** does a NONE result (or a sign frame with no candidate) break the run of identical signs? Is a frame with no candidates a `SignObs` with NONE, or no `SignObs` at all?
33. **`SIGN_HZ` state:** `process_front_frame` is a function. Should the last-sign-time live in the `LaneTracker`-style per-thread object, a new one, or the camera thread closure?
34. **"Fourteen dataclasses":** with `SignModel` there are fifteen. Just a count to fix in the design.

### Behaviour, safety and display

35. **The STOP command:** the design lists STOP but not where it goes. The code goes to IDLE from any state, including mid-turn. Keep that, or go to SAFE_STOP with reason "stopped from screen"?
36. **`IMU_CALIB_WAIT_S` timer:** from entering IDLE (code) or from pressing START? And with no IMU sample at all, the code starts at once. Should it wait or refuse instead?
37. **"Data stale" in `_check_safety`:** which data? The code checks only the IMU, and only in TURN and CROSS. Should a stale side camera, or a stale front camera in CROSS, count?
38. **ROAD_YAW_STALE when never anchored:** it shows in IDLE on every start-up. Is that intended, or should it only fire while moving?
39. **Display views:** the design's Status / Front / Side / Split vs the code's DRIVE / CAMERA / DIAGNOSTIC. I assume the code is the newer decision and the design should be updated. Confirm.
40. **Sabertooth command 17:** `open()` writes the controller's own deadband (`SABER_HW_DEADBAND = 5`), which it stores through power cycles, on top of the software `MOTOR_DEADBAND`. Keep it (and add it to the design), or drop it?
41. **`SABER_TIMEOUT_MS`:** 200 (code, from the course example) or 300 (design)?
42. **Tests and the Pi font:** should tests that draw text skip, or fall back to a default font, when Piboto isn't installed? And should `test_ui.py` use `importorskip("pygame")` like `test_calib_ui.py`? Today they fail or error off the Pi.

### Open questions in the design that the code already answers

These can probably be ticked off: **Sabertooth setup** (channels and inverts confirmed on the bench, so packetized serial works; the DIP question is settled in practice), **Display model** (`DISPLAY_SIZE = (800, 480)`, official 7" touch display; `ui/screen.py` opens it through the Wayland desktop), **Camera models** (CSI is Camera Module 3, per `CsiSource`), **Existing code** (yes: this report).
