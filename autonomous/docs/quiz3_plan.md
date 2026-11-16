# Quiz 3 plan: sign reading, location zones, modes, audio

**Status:** built, 2026-11-15, branch `quiz3` (stages 0-8; stage 9 is this
rewrite). The sign quiz and the intersection quiz both work on the robot.
Section 1 is the approved brief, kept word for word. Sections 2 to 6 describe
what was built; section 7 lists, stage by stage, every place the build
departs from section 1 or from DESIGN.md, and why. `docs/BUILD_STATUS.md`
has the item-by-item status against the design.

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

## 2. As built: modules and entry points

Names are the code's (`robot_types.py`, `behavior/`, `test/`,
`HealthWarning`, `run_ui` in `ui/screen.py`).

| Stage | Module | Main names |
| --- | --- | --- |
| 1 | `robot_types.py` | `Zone`, `RunMode`, `Surface`, `AnnotationKind`, `Annotation`, `LocationEvidence` (with `dash_dists_m`), `LocationObs`, `SignVote`, `SignModel`, `VisionResult`; `SignType.PAUSE`, `Action.PAUSE`; `UiCommand.MODE_*`; `HealthWarning.VISION_STALE`; `FrontResult.location`; `UiPacket.mode`, `.location`; `SideResult.surface`, `.annotations` |
| 2 | `perception/signs.py` | `sign_roi`, `sign_boxes`, `find_sign_candidates(line_mask, cfg)`, `pick_sign`, `crop_sign(frame, box)`, `normalise_crop`, `load_sign_model`, `hog_features(gray, hog)`, `classify_sign` -> (sign, margin, scores), `red_check`, `process_signs`, `sign_label` |
| 2 | `perception/color.py` | `mask_dark_lines` (sign outlines) |
| 2 | `tools/collect_signs.py`, `tools/train_signs.py`, `models/sign_svm.npz` | crops (`--all`, `--random`), laptop training (`--holdout`) |
| 3 | `behavior/sign_voter.py` | `SignVoter.add`, `decided` -> `SignVote`, `reset` |
| 3 | `perception/pipeline.py` | `SignPacer` (`SIGN_HZ`) |
| 3 | `ui/overlay.py`, `ui/panels.py`, `ui/screen.py` | `draw_annotations`, `draw_sign_overlay`, `draw_sign_panel`; `draw_banner`, `quiz_rows`, `zone_label`; mode keys 1 / 2 / 3 |
| 4 | `ui/gamepad.py` | `Gamepad.poll(events, state)`, `start_or_stop` |
| 4 | `ui/announcer.py` | `Announcer.prepare`, `update(packet, now)`, `close`; `render_phrase`, `wav_seconds` |
| 4 | `ui/course.py` | `course_module` (the repo root's `usb_sound_controller`, `manual/ps5_controller`) |
| 5 | `perception/location.py` | `row_to_metres`, `crossing_line_row`, `road_widening`, `flare_row`, `dash_positions`, `process_location(masks, cfg, t, seq, line_px)` |
| 5 | `perception/color.py` | `mask_dash` (Lab b*) |
| 5 | `estimation/location.py` | `LocationTracker.update(evidence, speed_cmd, now, sign, side)`, `get`, `reset`, `learned_speed`; `zone_position`, `camera_fix`, `dash_travel` |
| 6 | `behavior/state_machine.py` | `Behaviour.mode`, `.location`, `.paused`; `_quiz_target`, `_on_approach`, `_on_cross`, `_on_reacquire`, `_update_pause` |
| 6 | `control/command.py` | CROSS branch (`hold_steer`), judging pause |
| 6 | `main.py`, `loop_log.py` | `System.locator`, `_location`, zone reports; log columns `mode`, `zone`, `s`, `loc_source`, `sign` |
| 7 | `hal/shared_frame.py` | `SharedFrame.create`, `attach`, `handle`, `publish`, `wait`, `read`, `close` |
| 7 | `perception/worker.py` | `start_worker`, `run_worker`, `stop_worker`, `take_result`, `put_latest`, `Worker` |
| 7 | `perception/pipeline.py`, `main.py` | `process_vision`; `process_front_frame(..., with_vision)`; `_vision`, `_vision_age`, `_front_for_ui` |
| 8 | `perception/side_surface.py` | `classify_surface`, `process_side_frame` |
| 8 | `main.py`, `estimation/health.py` | `_build_side_camera`, `_fps`; `fps_min`; log column `side_surface` |
| 8 | `tools/record.py --side`, `tools/side_fixes.py` | two-camera recordings; `side_changes`, `near_junction`, `summary_lines` |

Tests (`test/`): `test_quiz3_types`, `test_signs`, `test_sign_quiz`,
`test_operator`, `test_location`, `test_intersection_quiz`,
`test_vision_worker`, `test_side`; 533 tests in all pass on the Pi. Real
frames in `test/data/` (`no_sign`, `junction`, `junction_seq`, `side`).

## 3. As built: how it runs

**Processes and threads.** Control process: the 30 Hz loop (the only place
motors are commanded), the front camera thread (lane; with the vision worker
off, also location evidence and signs), the side camera thread (surface box)
and the IMU thread. Operator process (`run_ui`): display, PS5 controller,
announcer; the control process never imports pygame. Vision worker process
(`VISION_WORKER_ENABLED`, off by default): location evidence and signs on
the frames the camera thread shares.

**Where the robot is.** `LocationTracker` sets s (the rear axle's distance
from the junction centre, negative before) on each front frame from the best
source: the crossing line (seen from 1.29 m to 0.66 m ahead with the
2026-11-15 calibration); the flare plus `FLARE_TO_CENTRE_M`; dash odometry
(how far our own line's dashes slid since the last frame); a speed learned
from recent dash travel; else it holds. A pass starts after
`LOCATION_START_FRAMES` fixes in a row and ends at `LOCATION_PASS_END_M`. The
side camera's two fixes are built but off (`SIDE_FIXES_ENABLED`).

**Zones** (rear-axle s, built in `config.py` from `ROBOT_FRONT_M` = 0.33 m,
`JUNCTION_CORNER_M` = 0.28 m and `CROSS_ROAD_EDGE_M` = `ROAD_WIDTH_M` / 2):

| Zone | Starts | On the mat |
| --- | --- | --- |
| Read the sign | when the sign is cut off by the front camera's view, else -0.73 | sign leaving the view |
| Pause zone | -0.61 | the robot's front reaches the corners |
| Entering intersection | -0.48 | the front reaches the cross road |
| Turn zone | -0.10 | 2 cm before the rear wheels are halfway from the cross road's edge to the yellow line |
| Exiting intersection | +0.04 | (course diagram) |
| Continue travel | +0.28 | the rear axle reaches where the road stops widening |

Zones only move forward within a pass. In QUIZ_INTERSECTION the behaviour
follows them (APPROACH in Read the sign and Pause, CROSS from Entering to the
end of Exiting, REACQUIRE after, IDLE at `QUIZ_EXIT_M` = 0.4 m), stopping
`ZONE_PAUSE_S` at each change. A zone is reported (terminal, voice, banner)
when it changes from one zone to the next within a pass.

**Signs.** The candidate is the sign's thin black outline inside `SIGN_ROI`;
the crop is classified by HOG + a linear SVM with a NONE class, with a red
check for Pause; `SignVoter` holds the decision and a steady box. In
QUIZ_SIGNS the robot stays in IDLE with signs read all the time; in
QUIZ_INTERSECTION signs are read through the whole run.

## 4. Found along the way

- Baseline before stage 1: 344 tests on the Pi; now 533.
- `autonomous/recordings/` is git-ignored: tests use the small set of
  committed frames in `test/data/`.
- scikit-learn is not on the Pi; `train_signs.py` runs on a laptop and its
  tests skip without it.
- The real signs are a white card with a thin black outline (about 1.33 wide
  to tall), not a white panel in a thick black box; the outline is the
  candidate (stage 2).
- The front camera was recalibrated on 2026-11-15 (board session): it sees
  0.66-1.29 m ahead, not 0.71-1.66 m. The `signs_*` and `straight_*`
  recordings predate it, so their distances are approximate.
- The far rows read distances up to 10% short (outer `FRONT_BANDS`
  extrapolated); dash odometry is not affected (it compares the same dashes).
- A fixed m/s per unit of speed command was replaced by dash odometry
  (approved 2026-11-15): it changes with battery charge and can't see stops.
- The side webcam gives YUYV at 3 fps (it lists 30) and MJPG at 19; its
  colours are washed out (grass fails the green test). Its exposure is left
  on auto: the `USB_*` manual settings are unconfirmed.
- `PS5_Controller.initialize_controller` blocks until a controller connects,
  so `Gamepad` attaches on pygame's device events instead.
- The `junction_1` recording's lane fit fails near the mat's end by the
  wall (lane conf 0-0.3): a lane-following issue, not a location one.

## 5. Answers

| # | Date | Question | Answer | Where it lands |
| --- | --- | --- | --- | --- |
| 1 | Nov 10 | Zone boundaries | From the course zone diagram (superseded on Nov 15: rows 15-19) | `ZONE_BOUNDS_M` |
| 1 | Nov 10 | Point the judges watch | The rear axle decides the zone; bounds are written for it | `ZONE_REF_X_M = 0.0` |
| 2 | Nov 10 | PS5 buttons | Cross = START and STOP; Square = Sign Reading, Triangle = Intersection Detection, Circle = Drive, Options = next view | `PS5_BUTTONS` |
| 3 | Nov 10 | Sign position | Front left, about 1 inch off the ground, at the left corner on the far side of the cross road | `SIGN_ROI` |
| 4 | Nov 10 | Side camera | Right side, a couple of inches before the rear axle, angled outward; sees road only coming into a junction | `SIDE_CAM_DEVICE`, stage 8 |
| 5 | Nov 10 | `SPEED_MPS_PER_UNIT` | Not measured; on Nov 15 replaced in the tracker by dash odometry | not used by location |
| 6 | Nov 10 | `test_sound.py` | Passes on the Pi | `USB_SoundController` reused as it is |
| 7 | Nov 10 | Pause sign | Red octagon, on the same outlined card as the arrows | one candidate test for all four |
| 8 | Nov 10 | Crossing line | Yellow dotted line | `crossing_line_row` closes dash gaps |
| 9 | Nov 10 | Flare to centre | Roughly 0.28 m (where the widening starts); the flare test fires 0.1 m later, so 0.18 m measured | `FLARE_TO_CENTRE_M`, `JUNCTION_CORNER_M` |
| 10 | Nov 10 | `FrontResult.inter` | Renamed `FrontResult.location` | stage 1 |
| 11 | Nov 10 | Banner and spoken text | As proposed | `MODE_LABELS`, `SIGN_PHRASES`, `ZONE_PHRASES` |
| 12 | Nov 10 | `draw_annotations` in stage 2 | Yes | stage 2 |
| 13 | Nov 10 | Removals | `SIGN_SIDE` in stage 1; `APPROACH_ON`, `APPROACH_OFF`, `STOP_LINE_Y_FRAC` in stage 6 | done |
| 14 | Nov 10 | Proposed config names | Fine | |
| 15 | Nov 15 | Junction shape | A plus: both dashed centre lines run through the centre | `crossing_line_row`, dash odometry |
| 16 | Nov 15 | Read the sign | Starts when the sign leaves the front camera's view | `LocationTracker.sign_cut`, `READ_SIGN_LATEST_M` |
| 17 | Nov 15 | Intersection entrance; robot front | The cross road's near edge; the front is 0.33 m ahead of the rear axle (will change with a front piece) | `CROSS_ROAD_EDGE_M`, `ROBOT_FRONT_M` |
| 18 | Nov 15 | Pause zone start | When the front reaches where the corners start | `JUNCTION_CORNER_M` |
| 19 | Nov 15 | Turn and Exiting (after a run on the mat) | Turn zone 2 cm earlier; Exiting until the road stops widening on the far side | `TURN_ZONE_EARLY_M`, last bound `JUNCTION_CORNER_M` |
| 20 | Nov 15 | Speed constant vs dash odometry | Use dash odometry, with a learned speed as fallback | `estimation/location.py` |

## 6. Edits DESIGN.md needs

`DESIGN.md` is an export and is not edited here; these are the edits the live
doc needs to match the build.

| Section of the design | Says | Should say |
| --- | --- | --- |
| Overview, Hardware (side camera), Key design decision 1 | The right-side camera faces down and measures the distance to the right road edge on straights | The side camera is angled outward, a couple of inches before the rear axle on the right. It can't see the road edge on a straight; it classes a surface box (road or grass) that changes at the junction's corner and past the cross road. Its fixes are less precise than dash odometry and are off |
| Side camera in decisions; Fusing lateral error; Lane filter; `side_lane.py`; `SideBias` | Side lateral reading fused with the front camera | Not possible with this mount: lateral error comes from the front camera alone. The opening and edge-back cues don't exist |
| Hardware (side camera), config | YUYV | MJPG at 19 fps (this webcam gives YUYV at 3) |
| Key design decision 5; `perception/signs.py` | Four-cornered white blobs on `SIGN_SIDE`; STOP | A white card with a thin black outline, found by its outline inside `SIGN_ROI`; the fourth sign is PAUSE (a red octagon on the same card) |
| `perception/intersection.py`; `IntersectionObs`; APPROACH / DECIDE position fix | `approach_score`, crossing line passing `STOP_LINE_Y_FRAC` | `perception/location.py` gives distances (crossing line, flare, dash positions); `estimation/location.py`'s `LocationTracker` gives s and the zone; the zone drives the quiz states |
| Dead reckoning (`SPEED_MPS_PER_UNIT`) | speed command x m/s per unit | Dash odometry from our own line's dashes, with a speed learned from it as fallback |
| Open questions "Sign placement", "Junction shapes", "Stop sign behaviour", "Start and emergency stop" | open | Front left, 1 inch up; a four-way plus junction with rounded corners; the six course zones (section 3 above) with a judging pause at each; PS5 Cross, screen and keyboard |
| Runtime table; `ui` process | Display process draws and reads taps | Operator process: display, PS5 controller and audio. Optional vision worker process for location and sign image work |
| Behaviour table | APPROACH -> DECIDE -> STOP_HOLD / TURN / CROSS | In QUIZ_INTERSECTION: LANE_FOLLOW -> APPROACH -> CROSS -> REACQUIRE -> IDLE by zone; DECIDE, STOP_HOLD and TURN are not built |
| Data types | `IntersectionObs`, `FrontResult.inter`, `SignType.STOP`, `Action.STOP` | `LocationEvidence`, `LocationObs`, `FrontResult.location`, `PAUSE`, and the types in section 1 |
| Display views | Status / Front / Side / Split | DRIVE / CAMERA / DIAGNOSTIC, with a run-mode banner (and the zone during a pass) on every view |

## 7. Where the build differs, stage by stage

| Stage | Item | Difference and why |
| --- | --- | --- |
| 2 | `find_sign_candidates(line_mask, cfg)` | **Changed after the first real recordings (Nov 10).** The real signs are a white card with a thin black outline, about 1.33 wide to tall, not a white panel in a thick black box at 1.65. The white-panel test failed on them: with a lamp in view the card is grey next to the glare, and the octagon touches the outline and splits the white. The candidate is now the outline itself: thin dark lines on the full-resolution frame (`color.mask_dark_lines`, a black-hat, so frame brightness doesn't matter), kept when the line's convex hull is a well-filled rectangle of the outline's aspect, mostly empty inside, and not cut off by the region's edge. The hull tolerates breaks from motion blur. On the 15 `signs_*` recordings plus the 4 straights: no box on any of 828 no-sign frames; a box on 37 to 100% of sign frames (misses are signs cut off by the left edge of the view, or blurred while moving). 1.4 ms per frame |
| 2 | `sign_boxes(frame, scale, cfg)` | New: region crop, line mask, candidates, padding, in one place for `process_signs` and `collect_signs`. `crop_sign(frame, box)` takes a full-frame box |
| 2 | White mask | No longer used by sign reading (`mask_white` remains; `tune_colors` no longer offers it). `SIGN_BORDER_PX` and `SIGN_BORDER_DARK_FRAC` are replaced by `SIGN_LINE_*`, `SIGN_INK_MAX`, `SIGN_EDGE_PX` |
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
| 8 | **Zone positions changed (2026-11-15, after the first runs on the mat)** | `ZONE_BOUNDS_M` is now built in config.py from `ROBOT_FRONT_M` (0.33 m, rear axle to the front; redo after adding a front piece), `JUNCTION_CORNER_M` (0.28 m, where the corners start) and `CROSS_ROAD_EDGE_M` (`ROAD_WIDTH_M` / 2). READ_SIGN: when the sign is cut off by the edge of the front camera's view (a sign box within `SIGN_CUT_EDGE_PX` of the edge, then `SIGN_CUT_READINGS` readings with none), else at `READ_SIGN_LATEST_M` (-0.73). PAUSE: the front reaches the corners (-0.61). ENTERING: the front reaches the cross road (-0.48). TURN: the rear wheels halfway from the cross road's edge to the yellow line (-0.08). EXITING and CONTINUE unchanged (+0.04, +0.18). Sign reading is on through a whole intersection quiz run. On junction_2 and junction_3 READ_SIGN starts at -0.94 and -1.14 m, as the sign is first cut off (hand-checked) |
| 8 | Side camera | Records and runs in MJPG (`SIDE_CAM_FOURCC`): this webcam gives YUYV at 3 fps though it lists 30, MJPG at 19. It stays on auto-exposure (the `USB_*` manual settings are unconfirmed). It looks out to the right; the top of its image is the ground nearest the robot. `SIDE_CAM_FPS_MIN` (15) is its own CAM_FPS_LOW limit (`health.fps_min`, also used by the pre-flight panel). Its brightness is not judged against the front camera's limits. `main._build_side_camera` starts it if `SIDE_CAM_ENABLED` and it opens; otherwise the robot runs without it |
| 8 | `perception/side_surface.py` (`classify_surface`, `process_side_frame`) | Classes `SIDE_BOX` (top right, measured) by the road mask alone: ROAD at `SIDE_ROAD_MIN_FRAC`, GRASS at `SIDE_GRASS_MAX_FRAC` or less, mixed between. The webcam's grass fails the green test (chromaticity 0.345 vs `GREEN_CHROM_MIN` 0.40; road 0.31-0.32), and the yellow test sees the crossing line's dashes on only some runs, so there is no YELLOW fix. `SideResult` gains `surface` and `annotations`; `side` (the side lane) is None. `LocationEvidence.side_surface` stays None: the side result goes to the tracker directly (`update(..., side=)`), as the two cameras are not in step |
| 8 | Side fixes (`LocationTracker`) | ROAD at -0.18 m and GRASS at +0.12 m (`SIDE_FIXES_M`, medians of junction_side_1-3 by `tools/side_fixes.py`; each run within ~4 cm). A change needs `SIDE_SURFACE_FRAMES` in a row (a mixed frame breaks the run); each fix is used once per pass, within `SIDE_FIX_GATE_M` of s, and only when the estimate has held still (no dashes, no learned speed) since its last camera fix. **Off (`SIDE_FIXES_ENABLED = False`)**: with a learned speed the dashes-lost estimate stays within 1 cm and a side fix made it 4-5 cm; when it holds, s is too stale to gate out the false ROAD changes (mat seams, shadows at -0.92 to -0.38 m) |
| 8 | `tools/record.py --side`, `tools/side_fixes.py` | Two-camera recordings (side in `<out>/side/`, same clock). `side_fixes` gives each side frame the front tracker's s and prints every settled change and the medians next to `SIDE_FIXES_M`; `--save` writes annotated side frames for choosing `SIDE_BOX` |
| 8 | Display and log | The side inset draws the box in its class colour with the road fraction; the loop log gains `side_surface` |
| 8 | Test data | `test/data/side/`: 6 real side frames (grass, road, a dark band that reads as mixed) |
| 9 | **Zone positions, second change (2026-11-15, after another run on the mat)** | Turn zone starts `TURN_ZONE_EARLY_M` (0.02 m) earlier: -0.10 m. Exiting lasts until the rear axle reaches `JUNCTION_CORNER_M` past the centre (+0.28 m; was +0.18). `CROSS_TIMEOUT_S` raised from 3 to 6 s: CROSS now spans 0.76 m, 3.2 s at the 0.235 m/s measured at `CROSS_SPEED` in the 16:37 run log |
| after 9 | **Flare fixes can't drag s back (2026-11-15)** | On the robot (run_20261115_175843) false flares 2-8 cm behind s, each inside `LOCATION_BACK_MAX_M`, ratcheted s back 25 cm in a second and made every zone from PAUSE late. Now a flare fix is ignored once the crossing line has fixed s in the pass, and never moves s back (`LocationTracker._accepts`) |
| after 9 | **Zones measured with a tape (recordings/truth_1, truth_2)** | Held at tape spots every 0.1 m: the crossing line is within 1.5 cm out to 1.1 m, so the front calibration is good; the dashes then fall 2-6 cm behind through the junction. `ZONE_BOUNDS_M` = (-0.83, -0.65, -0.55, -0.20, 0.00, 0.30), measured on the mat; `ROBOT_FRONT_M`, `CROSS_ROAD_EDGE_M`, `TURN_ZONE_EARLY_M` removed. READ_SIGN latest -0.83 (was -0.73), Exiting moved earlier by eye on the mat |
| after 9 | **Mat seam read as a junction** | `YELLOW_H_MAX` 34 (from the colour retune) let the grass beyond a mat seam pass as yellow, so the seam read as a flare and crossing line (straight_2: 145 / 60 frames, straight_4: 322 / 170). The colours stay as tuned (34 keeps the lane following straight); instead the junction cues count only yellow with road on both sides in its row (`perception/location.yellow_on_road`): a painted line lies on the road, the grass outside it. Straights: 3 / 0 and 0 / 0 frames, no pass started; the real junctions unchanged. `test/data/seam/` holds four straight_4 frames |
| after 9 | **Side camera: wide view and band edges** | The webcam's 320x240 is a crop of its sensor; it is now captured at `SIDE_CAPTURE_SIZE` (1024x768, the whole sensor, 18.5 fps) and shrunk to 320x240 in `UsbSource.read`. `SIDE_BOX` is a band across the middle of the view; its road fraction (`SideResult.road_frac`) ramps up as the corner's grass leaves (0.5 at -0.25 m) and down as the far side's grass comes in (0.5 at +0.12 m), measured against the tape (`tools/side_fixes.py --tape`). `LocationTracker` finds each crossing (`band_edge`, interpolated between side frames), compares it with the estimate at that moment (`_s_at`, between front frames) and, with `SIDE_FIXES_ENABLED` (now True), corrects s by the difference; each edge once per pass, within `SIDE_FIX_GATE_M` (0.10). The error is reported either way: `LocationObs.side_edge` / `side_error_m`, a terminal line, the loop log (`side_edge`, `side_err_m`) and a "side check" row on the quiz panel. The old settled-surface fixes (`SIDE_FIXES_M`, `SIDE_SURFACE_FRAMES`, `suspect`) are gone. On truth_2 the fixes take the error at TURN from -2.0 to +0.1 cm and after the far edge to under 0.5 cm; between them the dashes still undercount (-4 cm by the centre), which the side view cannot see (the band is all road from -0.16 to +0.09) |
| after 9 | **Side edges need the front camera running** | An edge is matched to the estimate only with front frames within `SIDE_EDGE_MAX_GAP_S` (0.1 s) around it (`LocationTracker._s_at`); straight_3's front frames stop for 3 s at the second junction, and carrying the estimate on over the gap put a wrong 3 cm fix in |
| after 9 | **The quiz drives on after Continue travel** | `QUIZ_EXIT_M` (IDLE 0.4 m past the centre) is gone: after CONTINUE the robot goes REACQUIRE -> LANE_FOLLOW and on to the next junction, which starts a new pass (the tracker ends each pass at `LOCATION_PASS_END_M`, +1.0 m); the run ends on STOP. recordings/straight_3 (two-camera) goes straight through two junctions: replayed, the first gives every zone in order with side checks of +1.4 and +3.0 cm |
