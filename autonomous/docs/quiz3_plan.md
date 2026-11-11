# Quiz 3 plan: sign reading, location zones, modes, audio

**Status:** stage 0 (plan). Saved 2026-11-10 on branch `quiz3`. Nothing is
built yet. Section 1 is the approved brief, kept word for word. Sections 2 to
4 are the inventory taken from the code before any change. Stage 9 rewrites
this file to match what was built.

## 1. Architecture changes approved for this work

**Processes.** Control process (unchanged role; the only place motors are commanded). Vision worker process (new) for location and sign image work. Operator process (today's display process) also reads the PS5 controller and plays audio. The control process must still never import pygame.

**New types in `robot_types.py`**

- Enums: `Zone` (CONTINUE, READ_SIGN, PAUSE, ENTERING, TURN, EXITING), `RunMode` (DRIVE, QUIZ_INTERSECTION, QUIZ_SIGNS), `Surface` (GRASS, ROAD, YELLOW). `SignType.STOP` and `Action.STOP` are renamed PAUSE. `UiCommand` gains MODE_DRIVE, MODE_INTERSECTION, MODE_SIGNS. `HealthWarning` gains VISION_STALE.
- `Annotation`: `kind` (RECT, LINE, POINT, TEXT), `pts` (view pixels), `text`, `colour`, `cam_id`.
- `LocationEvidence`: `crossing_dist_m`, `flare_dist_m` (metres ahead of the rear axle, or None), `side_surface` (or None), `annotations`, `t`, `seq`. It replaces `IntersectionObs` and `FrontResult.inter`.
- `LocationObs`: `zone`, `sign_read_ready`, `pause_point_ready`, `turn_point_ready`, `dist_to_centre_m`, `conf`, `source`, `t`.
- `SignObs` gains `roi`, `scores`, `annotations`. `SignVote`: `sign`, `bbox`, `since_t`. `SignModel`: `hog`, `weights`, `biases`, `classes`.
- `VisionResult`: `location`, `sign`, `t`, `seq`. `UiPacket` gains `mode` and `location`; `sign_vote` becomes a `SignVote`.

**Location.** `s` is the rear axle's distance along the road from the junction centre (negative before). `LocationTracker` sets `s` from the best source available: the front camera's crossing-line distance; the front camera's flare distance plus `FLARE_TO_CENTRE_M`; a side-camera surface change (stage 8); otherwise dead reckoning, `s += speed command × SPEED_MPS_PER_UNIT × dt`. The zone is where `s + ZONE_REF_X_M` falls among `ZONE_BOUNDS_M`. Zones only move forward within one junction pass. Image rows become metres by interpolating between the rows of `FRONT_BANDS`.

**Signs.** The designed pipeline (white-panel candidates, crop from the full frame, grayscale + CLAHE, HOG, linear SVM with a NONE class, red check for Pause), with these changes: search inside `SIGN_ROI` instead of `SIGN_SIDE`; test candidates on aspect (about 1.65 wide to tall) and on how well they fill their bounding rectangle, plus a dark border ring, not on four corners; grow the box by `SIGN_BBOX_PAD_FRAC` to include the border; `SignVoter` holds its decision, clears after `SIGN_CLEAR_FRAMES` of NONE or when another sign wins, and reports the median of recent boxes.

**Modes.** DRIVE: as today. QUIZ_SIGNS: stays in IDLE with motors at 0, signs on continuously. QUIZ_INTERSECTION: LANE_FOLLOW, APPROACH in READ_SIGN and PAUSE, CROSS (`hold_steer` on `road_yaw`) from ENTERING to the end of EXITING, REACQUIRE in CONTINUE, then IDLE `QUIZ_EXIT_M` past the centre. At each zone change, speed is 0 for `ZONE_PAUSE_S`. Mode changes only in IDLE.

## 2. Existing names reused

Names are the code's, not DESIGN.md's (`robot_types.py`, `behavior/`,
`test/`, `HealthWarning`, `run_ui` in `ui/screen.py`).

### Types (`robot_types.py`)

| Name | Reused for |
| --- | --- |
| `Frame`, `FrontMasks` (`white`, `red`, `hsv`, `bgr`, `scale`) | input to `process_signs` and `process_location` |
| `FrontResult` (`view`, `lane`, `sign`, `mean_v`, `t`, `seq`) | still the front thread's one object per frame |
| `SignObs` (`sign`, `score`, `bbox`, `t`, `seq`) | gains fields, keeps these |
| `SignType`, `Action` | `STOP` member renamed `PAUSE`; other members unchanged |
| `State` (all nine members) | no new states; QUIZ_SIGNS stays in `IDLE` |
| `UiCommand.START`, `STOP`, `NEXT_VIEW` | the PS5 buttons send the same commands as the taps |
| `HealthWarning` | gains `VISION_STALE` |
| `UiPacket`, `DriveCmd`, `MotorCmd`, `LaneEstimate`, `RoadYaw`, `ImuSample` | unchanged apart from the `UiPacket` fields listed in section 1 |
| `LatestValue` | slot for the newest `VisionResult` in the control process |

### Functions and classes

| Name | Where | Reused for |
| --- | --- | --- |
| `mask_white`, `mask_red`, `mask_yellow`, `mask_road`, `mask_green` | `perception/color.py` | sign candidates, red check, crossing line, road widening, side surface. No new colour test is written outside this module |
| `make_front_masks(frame, cfg, with_signs)` | `perception/color.py` | masks once per frame; `with_signs=True` on sign frames |
| `make_side_masks` | `perception/color.py` | stage 8 surface box |
| `process_front_frame(frame, signs_enabled, cfg, tracker)` | `perception/pipeline.py` | calls `process_signs` at `SIGN_HZ` (stage 3), `process_location` (stage 5) |
| `band_to_ground`, `find_road_edges` | `perception/front_lane.py` | road widening uses the same edge finder and band calibration |
| `is_fresh` | `estimation/lane_fusion.py` | staleness of a `VisionResult` |
| `check_health` | `estimation/health.py` | gains the VISION_STALE test |
| `RoadYawTracker` | `estimation/road_yaw.py` | CROSS target |
| `hold_steer` | `control/turn_controller.py` | CROSS steering |
| `target_speed`, `_base_speed` | `control/speed_governor.py` | already return `APPROACH_SPEED`, `CROSS_SPEED`, `REACQUIRE_SPEED` |
| `drive_for_state` | `control/command.py` | gains the CROSS branch and the judging pause |
| `Behaviour`: `step`, `_apply_commands`, `_handler`, `_enter`, `_check_safety`, `_set_flag`, `signs_enabled`, `exposure_hold`, `drift_request` | `behavior/state_machine.py` | gains `mode`, `_on_approach`, `_on_cross`, `_on_reacquire` |
| `choose_action` | `behavior/state_machine.py` | unchanged except the PAUSE rename |
| `System`, `build_system`, `control_step`, `shutdown`, `_read_commands` | `main.py` | gain the sign model, voter, tracker, worker |
| `start_ui`, `stop_ui`, `make_ui_packet`, `send`, `read_commands`, `_MAIN_COMMANDS` | `ui/link.py` | new packet fields, new commands forwarded |
| `Screen` (`update`, `draw`, `handle_input`, `_camera_image`, `_draw_column`), `run_ui`, `_route`, `open_display` | `ui/screen.py` | banner on every view; the gamepad and announcer run inside `run_ui`'s loop |
| `draw_front_overlay`, `draw_side_overlay`, `_px`, `_draw_strip`, theme colours | `ui/overlay.py` | `draw_annotations` is added beside them and uses `_px` |
| `draw_text` | `ui/text.py` | banner, sign label, scores |
| `draw_header`, `draw_status` | `ui/panels.py` | mode, zone and sign vote rows |
| `LoopLog`, `COLUMNS`, `_cell` | `loop_log.py` | new columns |
| `CameraStream`, `CsiSource`, `UsbSource`, `ExposureControl` | `hal/` | stage 7 publishes frames from the front `CameraStream`; stage 8 starts the side one |
| `RecordingWriter`, `read_index`, `load_frame`, `iter_frames` | `tools/recording.py` | `collect_signs`, replay, tests |
| `replay_step`, `format_line`, `show`, `run` | `tools/replay.py` | extended with the sign panel |
| `SLIDERS`, `config_with`, `render`, `tint` | `tools/tune_colors.py` | white and red sliders |
| `USB_SoundController.play_audio`, `stop_sound`, `close` | `usb_sound_controller.py` (repo root, not modified) | playback of the pre-rendered WAVs |
| `PS5_Controller.check_controls`, `control_request`, `reset_controller_state` | `manual/ps5_controller.py` (not modified) | button reading. `initialize_controller` is not called: it blocks until a controller connects |

### Config (`config.py`)

`FRONT_BANDS`, `FRONT_SIZE`, `FRONT_LANE_SIZE`, `ROAD_WIDTH_M`, `MIN_LINE_PX`,
`MAX_OBS_AGE_S`, `CONTROL_HZ`, `SIGN_HZ`, `SIGN_MIN_AREA_PX`,
`SIGN_ASPECT_MIN`, `SIGN_ASPECT_MAX` (values move to bracket 1.65),
`SIGN_RED_MIN_FRAC`, `SIGN_VOTES_NEEDED`, `WHITE_S_MAX`, `WHITE_V_REL`,
`WHITE_V_PCT`, `RED_H_LOW_MAX`, `RED_H_HIGH_MIN`, `RED_S_MIN`,
`CROSS_LINE_MIN_FRAC`, `APPROACH_SPEED`, `CROSS_SPEED`, `REACQUIRE_SPEED`,
`REACQUIRE_CONF`, `REACQUIRE_FRAMES`, `CROSS_TIMEOUT_S`, `HOLD_KP`,
`SPEED_MPS_PER_UNIT`, `SIDE_CAM_DEVICE`, `SIDE_CAM_FOURCC`, `SIDE_SIZE`,
`SIDE_CAM_X_M`, `GREEN_CHROM_MIN`, `DARK_SUM_MIN`, `UI_HZ`,
`UI_IMAGE_EVERY`, `UI_CMD_QUEUE_SIZE`, `DISPLAY_SIZE`, `LOOP_LOG_*`.

## 3. New names

### Types (`robot_types.py`), all from section 1

Enums `Zone`, `RunMode`, `Surface`; `AnnotationKind` (RECT, LINE, POINT,
TEXT) for `Annotation.kind`; dataclasses `Annotation`, `LocationEvidence`,
`LocationObs`, `SignVote`, `SignModel`, `VisionResult`; new members and
fields as listed in section 1.

### Modules and functions

| Stage | Module | Names |
| --- | --- | --- |
| 2 | `perception/signs.py` | `find_sign_candidates`, `pick_sign`, `crop_sign`, `normalise_crop`, `load_sign_model`, `hog_features`, `classify_sign`, `red_check`, `process_signs` (DESIGN.md names) |
| 2 | `tools/collect_signs.py`, `tools/train_signs.py`, `models/` | as in DESIGN.md |
| 3 | `behavior/sign_voter.py` | `SignVoter.add`, `decided` (returns a `SignVote` or None), `reset` |
| 3 | `ui/overlay.py` | `draw_annotations(img, annotations, scale)` |
| 4 | `ui/gamepad.py` | `Gamepad` (`poll()` returns a list of `UiCommand`) |
| 4 | `ui/announcer.py` | `Announcer` (`prepare()`, `update(packet)`, `close()`) |
| 5 | `perception/location.py` | `crossing_line_row`, `road_widening`, `row_to_metres`, `process_location` |
| 5 | `estimation/location.py` | `LocationTracker` (`update`, `get`, `reset`) |
| 6 | `behavior/state_machine.py` | `_on_approach`, `_on_cross`, `_on_reacquire`, `Behaviour.mode` |
| 7 | `hal/shared_frame.py` | `SharedFrame` (`publish`, `read`, `close`) |
| 7 | `perception/worker.py` | `start_worker`, `run_worker`, `stop_worker` |

### Config

Named in the brief: `SIGN_ROI`, `SIGN_BBOX_PAD_FRAC`, `SIGN_CLEAR_FRAMES`,
`FLARE_TO_CENTRE_M`, `ZONE_REF_X_M`, `ZONE_BOUNDS_M`, `ZONE_PAUSE_S`,
`QUIZ_EXIT_M`, `PS5_BUTTONS`, `VISION_WORKER_ENABLED`, `SIDE_BOX`.

From DESIGN.md, missing from `config.py` today: `SIGN_CROP_PX`,
`SIGN_MARGIN_MIN`, `SIGN_CLAHE_CLIP`, `SIGN_CLAHE_TILES`, `SIGN_MODEL_PATH`.

Proposed, not named anywhere yet (to confirm):

| Name | Meaning |
| --- | --- |
| `SIGN_FILL_MIN` | contour area ÷ bounding-rectangle area a candidate needs |
| `SIGN_BORDER_PX`, `SIGN_BORDER_DARK_FRAC` | width of the ring tested around a candidate, and the dark fraction it needs |
| `SIGN_BOX_FRAMES` | recent boxes the voter takes the median over |
| `SIGNS_DATA_DIR` | where `collect_signs` writes crops |
| `VISION_STALE_S` | age of the newest `VisionResult` that raises VISION_STALE |
| `AUDIO_ENABLED`, `AUDIO_VOLUME`, `AUDIO_DIR` | announcer switch, volume, folder for the rendered WAVs |
| `SIGN_PHRASES`, `ZONE_PHRASES`, `MODE_LABELS` | spoken and displayed text per sign, zone and mode |
| `START_MODE` | `RunMode` at start-up |

### Removed

`OCTAGON_CORNERS`, `ARROW_ASYM_MIN`, `SIGN_SIDE` (replaced by `SIGN_ROI`),
`IntersectionObs`, `FrontResult.inter`. `APPROACH_ON`, `APPROACH_OFF` and
`STOP_LINE_Y_FRAC` lose their only reader when `_on_lane_follow` moves to
`LocationObs` in stage 6.

## 4. Found while reading

- Baseline: 344 tests pass on the Pi (`python -m pytest test`, 16 s).
- `autonomous/recordings/` is git-ignored, so tests can't read recordings
  from there on another machine. Test frames go in `test/data/` as a small
  number of committed images.
- scikit-learn is not installed on the Pi. `train_signs.py` tests skip
  without it.
- The front camera sees the ground from 0.71 m to 1.66 m ahead of the rear
  axle (`FRONT_BANDS`). Once the junction centre is nearer than 0.71 m the
  crossing line is out of view, so every zone from there on is placed by
  dead reckoning (or the side camera, stage 8).
- `PS5_Controller.initialize_controller` blocks until a controller connects,
  and `USB_SoundController.__init__` calls `pygame.init()`.
- Attached to the Pi on 2026-11-10: a Sonix USB webcam
  (`/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Webcam_Webcam-video-index0`),
  a Plugable USB audio device (ALSA card 2), `text2wave` and `festival`. No
  game controller was connected.

## 5. Open questions

Asked 2026-11-10; answers get written here.
