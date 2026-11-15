# Quiz 3 plan: sign reading, location zones, modes, audio

**Status:** stage 7 done (vision worker), 2026-11-15, branch `quiz3`. The sign quiz (stage 4) and the intersection quiz (stage 6) are complete. Section 1 is the approved brief, kept word for word; section 7 lists where the build differs, including the dash odometry change approved on 2026-11-15.
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

## 5. Answers (2026-11-10)

| # | Question | Answer | Where it lands |
| --- | --- | --- | --- |
| 1 | Zone boundaries | The course zone diagram. Read off it, scaled by `ROAD_WIDTH_M`: Read the sign from 0.73 m before the centre, Pause zone from where the road widens (0.28 m), Entering from 0.17 m, Turn zone ±0.04 m about the crossing line, Exiting to 0.18 m past, then Continue travel. **To confirm with a tape.** | `ZONE_BOUNDS_M` |
| 1 | Point the judges watch | **Still open** | `ZONE_REF_X_M` (None) |
| 2 | PS5 buttons | Cross is both START and STOP; the rest were mine to choose: Square = Sign Reading, Triangle = Intersection Detection, Circle = Drive, Options = next view | `PS5_BUTTONS` |
| 3 | Sign position | In front and to the left, about 1 inch off the ground: the left corner of the junction on the far side of the cross road | `SIGN_ROI`, set from the sign recordings in stage 2 |
| 4 | Side camera | Final position: right side, a couple of inches before the rear axle, angled outward. It can't see the road edge on a straight; it sees road when the robot comes into a junction | `SIDE_CAM_DEVICE` set; stage 8 can go ahead |
| 5 | `SPEED_MPS_PER_UNIT` | Not measured | Needed before stage 6: dead reckoning places every zone from about Pause onward |
| 6 | `test_sound.py` | Passes on the Pi | The announcer can use `USB_SoundController` as it is |
| 7 | Pause sign | The red octagon is in a black box, like the arrows | One candidate test serves all four signs |
| 8 | Crossing line | Yellow dotted line across the junction | `crossing_line_row` must accept a dashed line |
| 9 | Flare to centre | Roughly 0.28 m | `FLARE_TO_CENTRE_M` |
| 10 | `FrontResult.inter` | Renamed `FrontResult.location` (a `LocationEvidence`); `UiPacket.location` is the `LocationObs` | done in stage 1 |
| 11 | Banner and spoken text | As proposed | `MODE_LABELS`, `SIGN_PHRASES`, `ZONE_PHRASES` |
| 12 | `draw_annotations` in stage 2 | Yes | stage 2 |
| 13 | Removals | `SIGN_SIDE` removed in stage 1; `APPROACH_ON`, `APPROACH_OFF`, `STOP_LINE_Y_FRAC` in stage 6 | |
| 14 | Proposed config names | Fine | |

## 6. Changes these answers make to DESIGN.md

`DESIGN.md` is an export and is not edited here. The live doc could not be
edited from the build session on 2026-11-10 (no Claude Docs connector), so
these are the edits it needs.

| Section of the design | Says | Should say |
| --- | --- | --- |
| Overview, Hardware (side camera), Key design decision 1 | The right-side camera faces down and measures the distance to the right road edge on straights | The side camera is angled outward, a couple of inches before the rear axle on the right. It can't see the road edge on a straight. It sees road only coming into a junction, so its job is a surface box (grass, road or yellow) that fixes the robot's position at a junction |
| Side camera in decisions; Fusing lateral error; Lane filter; `side_lane.py`; `SideBias` | Side lateral reading fused with the front camera | Not possible with this mount: on straights lateral error comes from the front camera alone. `SIDE_FUSION_ENABLED`, `SideBias`, `find_boundary_x`, `edge_to_offset` and the side terms of `LaneFilter` have no input. The opening and edge-back cues become surface changes in `SIDE_BOX` |
| Key design decision 5; `perception/signs.py` | Four-cornered white blobs on `SIGN_SIDE` of the road; STOP | White panel inside a black box, about 1.65 wide to tall, searched for inside `SIGN_ROI`; the fourth sign is PAUSE (a red octagon, also in a black box) |
| Open question "Sign placement" | Which side, what height | Front left, about 1 inch off the ground, at the left corner on the far side of the cross road |
| Open question "Junction shapes" | Four-way or T | The quiz junction is four-way with rounded corners; the road starts to widen about 0.28 m before the centre, and a yellow dotted line crosses it |
| Open question "Stop sign behaviour" | Where to stop | The course defines six zones along the road (`Zone`, `ZONE_BOUNDS_M`); the Pause zone runs from the widening to the cross road's near edge |
| Open question "Start and emergency stop" | Tap, key or button | PS5 Cross button starts and stops, as well as the screen and keyboard |
| Runtime table; `ui` process | Display process draws and reads taps | Operator process: display, PS5 controller and audio. A vision worker process does location and sign image work |
| Data types | `IntersectionObs`, `FrontResult.inter`, `SignType.STOP`, `Action.STOP` | `LocationEvidence`, `FrontResult.location`, `PAUSE`; plus the types in section 1 |
| Config: Junctions and signs | `SIGN_SIDE`, `OCTAGON_CORNERS`, `ARROW_ASYM_MIN` | removed; new names in section 3 |

## 7. Built so far: where it differs from DESIGN.md

| Stage | Item | Difference and why |
| --- | --- | --- |
| 2 | `find_sign_candidates(line_mask, cfg)` | **Changed after the first real recordings (Nov 10).** The real signs are a white card with a thin black outline, about 1.33 wide to tall, not a white panel in a thick black box at 1.65. The white-panel test failed on them: with a lamp in view the card is grey next to the glare, and the octagon touches the outline and splits the white. The candidate is now the outline itself: thin dark lines on the full-resolution frame (`color.mask_dark_lines`, a black-hat, so frame brightness doesn't matter), kept when the line's convex hull is a well-filled rectangle of the outline's aspect, mostly empty inside, and not cut off by the region's edge. The hull tolerates breaks from motion blur. On the 15 `signs_*` recordings plus the 4 straights: no box on any of 828 no-sign frames; a box on 37 to 100% of sign frames (misses are signs cut off by the left edge of the view, or blurred while moving). 1.4 ms per frame |
| 2 | `sign_boxes(frame, scale, cfg)` | New: region crop, line mask, candidates, padding, in one place for `process_signs` and `collect_signs`. `crop_sign(frame, box)` takes a full-frame box |
| 2 | White mask | No longer used by sign reading (`mask_white` and its sliders in `tune_colors` remain). `SIGN_BORDER_PX` and `SIGN_BORDER_DARK_FRAC` are replaced by `SIGN_LINE_*`, `SIGN_INK_MAX`, `SIGN_EDGE_PX` |
| 2 | `collect_signs --random N` | Saves random sign-sized background patches for `none/`: the finder gives no false candidates on the no-sign recordings, so there was nothing else to train NONE on |
| 2 | `replay --save FOLDER --step N` | Writes the annotated picture per frame as a JPEG, for use without the Pi desktop |
| 2 | `hog_features(gray, hog)` | Takes the HOG descriptor, not the model, so training can use it before a model exists |
| 2 | `classify_sign` | Returns `(SignType, margin, scores)`: the scores feed `SignObs.scores` |
| 2 | `red_check(crop_hsv, cfg)` | As designed |
| 2 | `process_signs(..., model=None)` | With no model yet, shows candidates only and reports NONE, so replay and the display work before training |
| 2 | `collect_signs` | Saves the padded colour crop, not the normalised one: `train_signs` augments it and then calls the robot's own `normalise_crop`. `--all` saves every candidate |
| 2 | `train_signs` | Holds out recordings by name (`--holdout`), taken from the crop file names |
| 2 | Test data | `test/data/no_sign/`: 4 real course frames (the ones with the most white clutter), since `recordings/` is git-ignored |
| 2 | `ZONE_REF_X_M = 0.0` | The rear axle decides the zone (the robot spans two zones at once, so one point must) |
| 3 | `FrontResult.sign` | Carries the newest sign reading on every frame while signs are on (its `seq` says which frame it is from), not only on sign frames. Signs are read on one frame in three and the slot holds one result, so a reading would otherwise be overwritten before the control loop or display saw it. `SignVoter` already ignores a repeated `seq` |
| 3 | `SignPacer` (`perception/pipeline.py`) | Per-thread state for `SIGN_HZ` pacing and the newest reading (BUILD_STATUS question 33) |
| 3 | `SignVoter.decided()` | Returns a `SignVote`. The box is held through missed frames for as long as the decision is |
| 3 | Mode selection before the PS5 controller | Keys 1 / 2 / 3 on the display keyboard, and `1` / `2` / `3` + Enter in the terminal, send the mode commands. QUIZ_INTERSECTION is selectable but drives like DRIVE until stage 6 |
| 3 | Display | A 30 px mode banner tops every view, with the voted sign on its right. The camera image is letterboxed under it. The crop in the sign panel is cut from the 320 x 240 view (the display process has no full frame), then normalised as on the robot |
| 3 | Missing model | `main.py` warns and carries on: signs are boxed but not read |
| 4 | `Gamepad.poll(events, state)` | Takes this cycle's pygame events (shared with the taps) and the robot's `State`, so `START_STOP` resolves as the space bar does (`start_or_stop`, moved from `ui/screen.py` with `START_STATES`). The joystick is attached on start-up or `JOYDEVICEADDED` and dropped on `JOYDEVICEREMOVED` or a read error |
| 4 | Held buttons | `PS5_Controller` repeats a held button every 0.5 s; `Gamepad` counts a button again only after a `JOYBUTTONUP`, so a long press on Cross cannot START then STOP |
| 4 | `ui/course.py` (`course_module`) | New: imports the course files from the repo root. The root is appended to `sys.path`, not inserted, because the root's `ui.py` would shadow the `ui` package |
| 4 | `Announcer.update(packet, now)` | One phrase at a time: `USB_SoundController` starts each sound at once, so phrases queue (one waiting per kind, newest wins) and the next starts after the last one's length plus `AUDIO_GAP_S`. A vote that clears and comes back is spoken again. WAVs are named `<kind>_<NAME>_<crc of text>.wav` in `AUDIO_DIR` (git-ignored), so a changed phrase re-renders and the rest are reused: about 5 s on a first start-up, none after |
| 4 | Config | New: `AUDIO_TTS_SCALE` (2.0, as `play_text_to_speech`), `AUDIO_RENDER_TIMEOUT_S`, `AUDIO_GAP_S` |
| 4 | `ui/link.py` | No change needed: stage 3 already forwards the mode commands |
| 5 | `crossing_line_row(yellow_mask, road_mask, cfg)` | The junction is a plus (answer, Nov 15): the crossing line is the cross road's dashed centre line through the centre, so `crossing_dist_m` is the distance to the centre. Rows where yellow near road (`CROSS_LINE_ROAD_PX`), with dash gaps closed (`CROSS_LINE_CLOSE_PX`) and a few rows either side searched (`CROSS_LINE_TILT_PX`, for yaw), covers `CROSS_LINE_MIN_FRAC` of the width. The road `CROSS_ROAD_CHECK_PX` rows either side must be `CROSS_ROAD_EXTRA_M` wider than ours, or reach both image edges (close in, the view is only 0.43-0.48 m wide). Each test removed a false line on the straight recordings: grass specks joined along a dark mat seam, our own road being 60% of the width 1 m ahead |
| 5 | `road_widening(road_mask, yellow_mask, cfg)`, `flare_row(widening, cfg)` | Width of the widest road stretch (`front_lane.road_span`, after closing speckle holes of `FLARE_CLOSE_PX`) per `FLARE_STRIP_PX` strip, compared with the road at the bottom of the view, not with `ROAD_WIDTH_M`: on straights the measured width is 2-6 cm over and drifts with yaw. A flare needs `FLARE_WIDE_STRIPS` strips `FLARE_MIN_M` wider (a mat seam is one strip), a normal bottom (`FLARE_BASE_MAX_M`; inside a rounded corner the cross road would read as a flare 0.3 m too far ahead), and to be nearer than any crossing line seen in the same frame (inside the junction the far corners can read as one) |
| 5 | `FLARE_TO_CENTRE_M = 0.18` | Means "from where the flare test fires": it fires about 0.1 m past where the widening starts (the PAUSE bound, 0.28 m). Measured by `tools/replay` on junction_1-3: medians 0.175, 0.195, 0.173 m; the synthetic corners give the same |
| 5 | `FRONT_BANDS` recalibrated (2026-11-15 board session) | The front camera now sees 0.66-1.29 m ahead (was 0.71-1.66, two bands extrapolated). The crossing line is tracked from 1.29 m to 0.66 m, so the camera places READ_SIGN (from 0.73 m); PAUSE and every later zone come from dash odometry. The far rows read distances up to 10% short (the outer bands are extrapolated). The `signs_*` and `straight_*` recordings predate the camera being reseated, so their distances are approximate |
| 5 | **Architecture change (approved 2026-11-15): dash odometry replaces speed dead reckoning** | Section 1's "otherwise dead reckoning, s += speed command x SPEED_MPS_PER_UNIT x dt" is replaced: a fixed m/s per unit changes with battery charge and can't see stops and restarts. Sources now, best first: crossing line, flare, side camera (stage 8), **dash odometry** (how far our own line's dashes slid between frames: measures real motion), **learned speed** (m/s per unit from recent dash travel while driven, for frames whose dashes can't be matched), else s holds. `SPEED_MPS_PER_UNIT` stays for `lane_sim` and the steering tools only |
| 5 | `dash_positions(dash_mask, line_px, cfg)` | Runs of rows with dash colour within `DASH_WINDOW_M` of the lane's fitted line (`FrontLaneObs.line_px`, passed to `process_location`), breaks of `DASH_GAP_ROWS` bridged, `DASH_LEN_MIN_M`-`DASH_LEN_MAX_M` long, not cut by the view's edge. Dash colour is `color.mask_dash` (Lab b* >= `DASH_LAB_B_MIN`): the lane's yellow test missed washed-out dashes in junction_1 and junction_3 (saturation down to 93). `LocationEvidence.dash_dists_m` is new; replay draws a point per dash |
| 5 | `dash_travel(prev, cur, cfg)` (estimation/location.py) | The shift most dashes agree on (`DASH_MATCH_TOL_M`, at least `DASH_MIN_MATCHES`) within -`DASH_MAX_BACK_M` to `DASH_MAX_STEP_M` (under half the 0.115 m dash spacing, so a shift is never ambiguous: 1.7 m/s at 30 fps) |
| 5 | `LocationTracker.update(evidence, speed_cmd, now)` | Motion is counted only when a camera frame arrives (dashes, or the learned speed x the command since the last frame), so it is never counted twice. A pass starts after `LOCATION_START_FRAMES` frames in a row with a fix and ends at `LOCATION_PASS_END_M`. A camera fix more than `LOCATION_BACK_MAX_M` behind s is ignored. `conf` falls only with distance on the learned speed (`LOCATION_DR_CONF_M`). Sources: 'crossing', 'flare', 'dashes', 'speed', 'held'. The frame-age correction on fixes was dropped (a few mm at crossing speed). Not yet called from `main.py`: stage 6 wires it in |
| 5 | Results on the recordings (pushed, no speed command) | junction_1-3: all six zones in order, each change within 1 cm of its bound. Dash odometry against the crossing line over 0.6 m: within 2.1-2.9 cm; on the committed sequence 5 mm over 0.16 m, including a 2 cm rock backwards. Dashes matched on all but 2-3 frames in junction_2-3; junction_1 holds 208 frames from EXITING on, where the lane fit fails near the mat's end by the wall (lane conf 0-0.3), which is a lane-following issue |
| 5 | Pipeline and replay | `process_front_frame` fills `FrontResult.location` (4.7 ms per frame on the Pi with the dash scan; moves to the vision worker in stage 7). `tools/replay` draws the cue lines and dash points, prints cues, s, source and zone per frame, and the measured flare-to-centre at the end |
| 5 | Test data | `test/data/junction/`: 7 real frames (3 approaching, 1 inside the junction, 3 straights that fooled earlier versions). `test/data/junction_seq/`: junction_3 #440-480, every 2nd frame (JPEG), for dash odometry against the crossing line |
| 6 | `Behaviour.step(..., location=None)` | Takes the `LocationObs`; it is used in QUIZ_INTERSECTION only (`Behaviour.location` is None in DRIVE and QUIZ_SIGNS, so DRIVE behaves as before). `_quiz_target` maps the location to a state: READ_SIGN, PAUSE -> APPROACH; ENTERING, TURN, EXITING -> CROSS; CONTINUE after the junction -> REACQUIRE; `QUIZ_EXIT_M` past the centre -> IDLE. `_on_approach`, `_on_cross`, `_on_reacquire` follow it; a lost pass goes back to LANE_FOLLOW (APPROACH) or REACQUIRE (CROSS). REACQUIRE may hand back to LANE_FOLLOW (`REACQUIRE_CONF` for `REACQUIRE_FRAMES`); the run still ends at `QUIZ_EXIT_M` |
| 6 | Judging pause | `Behaviour.paused`: `ZONE_PAUSE_S` from every zone change while moving (6 per run, including into Read the sign and into Continue travel). `drive_for_state` sends speed 0, steer 0 and resets the lane controller while paused. `CROSS_TIMEOUT_S` counts driving time only; on timeout CROSS hands over to REACQUIRE (lane control), and the zones keep being reported |
| 6 | CROSS branch of `drive_for_state` | `hold_steer(imu.yaw_deg, road_yaw.yaw_deg)` at `CROSS_SPEED`; steer 0 without a valid IMU and road yaw. Road yaw is anchored in LANE_FOLLOW and APPROACH and held in CROSS |
| 6 | `main.py` | `System.locator` (`LocationTracker`), updated every loop with the front camera's evidence and last loop's speed command (`_location`), reset when a run leaves IDLE so each run starts its own pass. The terminal prints each zone change with s and its source |
| 6 | Zone reports | A zone is reported (terminal, announcer) when it changes from one zone to the next within a pass: the pass starts in Continue travel before Read the sign, which is not news. The banner shows "Zone: ..." on every view during a pass (in place of the voted sign); the diagnostic view adds zone, s, source and conf; the camera view draws the crossing line, flare and dash points |
| 6 | `loop_log.py` | New columns `mode`, `zone`, `s`, `loc_source`, `sign`; `LoopLog.write` takes mode, location and vote |
| 6 | Removed | `APPROACH_ON`, `APPROACH_OFF`, `STOP_LINE_Y_FRAC` |
| 6 | Closed-loop test | `test/test_intersection_quiz.py` runs `main.control_step` on synthetic junction frames rendered from the robot's true position (moving by the commanded speed) from two start positions: all six zones in order, each change within 3 cm of its bound, six judging pauses, IDLE at `QUIZ_EXIT_M` |
| 6 | Limits | A run must start where the crossing line can be seen before 0.66 m (the camera sees it from 1.29 m; from farther back the robot lane-follows until it does). Starting nearer than 0.66 m, no pass starts and the robot lane-follows until STOP. `APPROACH_SPEED` and `CROSS_SPEED` (0.3) are above `CRUISE_SPEED` (0.2); not changed |
| 7 | `hal/shared_frame.py` (`SharedFrame`: `create`, `attach`, `handle`, `publish`, `wait`, `read`, `close`) | One shared-memory block: header (seq, t, the lane's fitted line) and the full-size frame. The camera thread publishes the line with the frame because the dash finder needs it and only the camera thread tracks the lane. A lock covers each copy; an event wakes the worker. Only the creator unlinks the block. The worker shares its creator's resource tracker (spawn), so it does not unregister the block |
| 7 | `perception/worker.py` (`start_worker`, `run_worker`, `stop_worker`, `take_result`, `put_latest`) | `start_worker` returns a `Worker` (process, one-slot results queue, stop event). `put_latest` replaces a result the control loop has not taken. An error on one frame is counted and the frame skipped (first traceback printed) |
| 7 | `perception/pipeline.process_vision` | New: location evidence and the paced sign reading for one frame -> `VisionResult`; the in-thread path (`process_front_frame`) and the worker both call it. `process_front_frame(..., with_vision=False)` measures the lane only |
| 7 | `main.py` | With `VISION_WORKER_ENABLED`: `signs_enabled` is a process Event (the worker reads it; `Behaviour` is unchanged), the camera thread publishes each frame, and `_vision` takes the worker's newest result (else builds one from the camera thread's result). The tracker and voter use the `VisionResult`; `_front_for_ui` puts the worker's location and sign on the front result for the display. `shutdown`: worker after the display, then the camera, then the shared frame (the camera thread writes into it) |
| 7 | `VISION_STALE` | `check_health(..., vision_age=None)`: raised when the newest result's frame is older than `VISION_STALE_S`, or the worker has died (age inf). A warning only: the control loop carries on, and the location tracker holds without fresh evidence. It shows for the worker's first 2-3 s while it starts |
| 7 | Measured on the Pi (junction_3 at 30 fps, signs on) | 97% of frames processed; publish to result 9-11 ms median (p95 13-15 ms); the camera thread's work falls from 11-12 ms to 5 ms; `take_result` 13 us median in the control loop; crossing distances identical to the in-thread path |
| 7 | Config | New: `VISION_WAIT_S`, `VISION_STOP_TIMEOUT_S`. `VISION_WORKER_ENABLED` stays False (the brief) |
