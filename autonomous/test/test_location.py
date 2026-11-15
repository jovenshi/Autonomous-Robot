"""Stage 5 (docs/quiz3_plan.md): location evidence from the front camera and
the LocationTracker. Synthetic junction images drawn in ground coordinates
through FRONT_BANDS, so the true distances are known. No robot needed."""
import math
import os
import sys
import types

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from estimation.location import LocationTracker, camera_fix, zone_position  # noqa: E402
from perception.color import make_front_masks  # noqa: E402
from perception.location import (crossing_line_row, flare_row, process_location,  # noqa: E402
                                 road_widening, row_to_metres)
from perception.pipeline import process_front_frame  # noqa: E402
from robot_types import Frame, LocationEvidence, Zone  # noqa: E402

HALF_ROAD = config.ROAD_WIDTH_M / 2
WIDEN_START_M = -config.ZONE_BOUNDS_M[1]           # PAUSE starts where the road starts to widen
CORNER_R = WIDEN_START_M - HALF_ROAD               # rounded corner radius
LINE_HALF_M, DASH_M = 0.012, 0.04                   # yellow line half width; dash and gap length
ROAD, GRASS, YELLOW, FLOOR = (45, 42, 40), (50, 150, 70), (0, 215, 230), (200, 205, 210)


def cfg_with(**kw):
    return types.SimpleNamespace(**{**{k: getattr(config, k) for k in dir(config) if k.isupper()},
                                    **kw})


# ---------------------------------------------------------------- synthetic junction

def ground_grid(cfg):
    # (forward_m, lateral_m) of every lane-size pixel, through the band
    # table (rows past the outer band centres take the end values).
    w, h = cfg.FRONT_LANE_SIZE
    bands = sorted(cfg.FRONT_BANDS)
    rows = [(b[0] + b[1]) / 2 for b in bands]
    fwd = np.interp(np.arange(h), rows, [b[2] for b in bands])
    mpp = np.interp(np.arange(h), rows, [b[4] for b in bands])
    xs = np.arange(w) - w / 2
    return np.repeat(fwd[:, None], w, axis=1), mpp[:, None] * xs[None, :]


def junction_scene(s, lateral=0.0, yaw_deg=0.0, junction=True, seam_row=None, cfg=config):
    # Full-size BGR frame of the course seen with the rear axle at s (m from
    # the junction centre along the road), lateral offset (m, + right) and
    # yaw (deg, + right). The junction is a plus with rounded corners; both
    # centre lines are dashed and run through it.
    f, l = ground_grid(cfg)
    yaw = math.radians(yaw_deg)
    u = s + f * math.cos(yaw) - l * math.sin(yaw)         # along our road
    v = lateral + f * math.sin(yaw) + l * math.cos(yaw)   # across it
    road = np.abs(v) <= HALF_ROAD
    if junction:
        road |= np.abs(u) <= HALF_ROAD
        corner = HALF_ROAD + CORNER_R
        in_box = (np.abs(u) < corner) & (np.abs(v) < corner)
        road |= in_box & (np.hypot(np.abs(u) - corner, np.abs(v) - corner) > CORNER_R)
    yellow = (np.abs(v) < LINE_HALF_M) & (np.mod(u, 2 * DASH_M) < DASH_M)
    if junction:
        yellow |= (np.abs(u) < LINE_HALF_M) & (np.mod(v, 2 * DASH_M) < DASH_M)
    img = np.empty(f.shape + (3,), np.uint8)
    img[:] = GRASS
    img[road] = ROAD
    img[road & yellow] = YELLOW
    top = min(b[0] for b in cfg.FRONT_BANDS)
    img[:top - 6] = FLOOR
    if seam_row is not None:
        img[seam_row:seam_row + 2] = ROAD                 # a dark mat seam across the view
    return cv2.resize(img, tuple(cfg.FRONT_SIZE), interpolation=cv2.INTER_NEAREST)


def evidence_at(s, seq=1, t=0.0, **kw):
    frame = Frame(junction_scene(s, **kw), t, seq, "front")
    return process_location(make_front_masks(frame, config), config, t, seq)


# ---------------------------------------------------------------- rows to metres

def test_band_centres_map_to_their_distances_and_outside_is_none():
    for top, bottom, forward, _, _ in config.FRONT_BANDS:
        assert row_to_metres((top + bottom) / 2, config) == pytest.approx(forward)
    rows = [(b[0] + b[1]) / 2 for b in config.FRONT_BANDS]
    assert row_to_metres(min(rows) - 1, config) is None
    assert row_to_metres(max(rows) + 1, config) is None
    assert row_to_metres(None, config) is None
    near, far = row_to_metres(200, config), row_to_metres(120, config)
    assert near < far                                       # higher in the image = farther


# ---------------------------------------------------------------- crossing line

@pytest.mark.parametrize("dist", [0.72, 0.85, 1.0, 1.2])
def test_crossing_line_distance(dist):
    ev = evidence_at(-dist)
    assert ev.crossing_dist_m == pytest.approx(dist, abs=0.04)


@pytest.mark.parametrize("lateral,yaw", [(0.05, 0.0), (-0.05, 0.0), (0.0, 6.0), (0.03, -6.0)])
def test_crossing_line_off_centre_and_yawed(lateral, yaw):
    ev = evidence_at(-1.1, lateral=lateral, yaw_deg=yaw)
    assert ev.crossing_dist_m == pytest.approx(1.1, abs=0.06)


def test_no_crossing_line_out_of_view_or_on_a_straight():
    assert evidence_at(-0.5).crossing_dist_m is None        # nearer than the view
    assert evidence_at(-2.2).crossing_dist_m is None        # beyond it
    for lateral in (0.0, 0.05):
        ev = evidence_at(-1.0, lateral=lateral, junction=False)
        assert ev.crossing_dist_m is None and ev.flare_dist_m is None
    assert crossing_line_row(np.zeros((240, 320), np.uint8), np.zeros((240, 320), np.uint8),
                             config) is None


# ---------------------------------------------------------------- flare

@pytest.mark.parametrize("flare_at", [0.75, 0.9, 1.05])
def test_flare_found_just_after_the_widening_starts(flare_at):
    # The widening grows round the corner, so FLARE_MIN_M is reached a little
    # past where it starts (about 0.1 m with these corners): never before it.
    ev = evidence_at(-(flare_at + WIDEN_START_M))
    assert flare_at - 0.02 <= ev.flare_dist_m <= flare_at + 0.13


def test_no_flare_when_the_view_starts_inside_the_junction():
    # Near rows already wide (the cross road), normal road beyond: not a flare.
    for s in (-0.6, -0.3, 0.0, 0.2):
        assert evidence_at(s).flare_dist_m is None


def test_mat_seam_is_not_a_flare():
    for row in (120, 160, 200):
        assert evidence_at(-1.0, junction=False, seam_row=row * 2).flare_dist_m is None


def test_widening_strips_are_nearest_first_and_normal_on_a_straight():
    masks = make_front_masks(Frame(junction_scene(-1.0, junction=False), 0.0, 1, "front"), config)
    strips = road_widening(masks.road, masks.yellow, config)
    assert strips[0][0] > strips[-1][0]
    assert all(extra is not None and abs(extra) < config.FLARE_MIN_M for _, extra in strips)
    assert flare_row(strips, config) is None


def test_evidence_carries_lines_and_labels():
    ev = evidence_at(-1.15, seq=7, t=3.0)
    assert ev.crossing_dist_m is not None and ev.flare_dist_m is not None
    assert ev.seq == 7 and ev.t == 3.0 and ev.side_surface is None
    texts = [a.text for a in ev.annotations if a.text]
    assert any(t.startswith("cross") for t in texts) and any(t.startswith("flare") for t in texts)
    assert evidence_at(-1.0, junction=False).annotations == ()


def test_front_frame_carries_location_evidence():
    frame = Frame(junction_scene(-1.2), 0.0, 4, "front")
    result = process_front_frame(frame, False, config)
    assert result.location.seq == 4
    assert result.location.crossing_dist_m == pytest.approx(1.2, abs=0.04)


# ---------------------------------------------------------------- tracker

def ev(cross=None, flare=None, seq=1, t=0.0):
    return LocationEvidence(cross, flare, None, (), t, seq)


def started(cfg, s=-1.5, t=0.0):
    # A tracker that has begun a pass at s (crossing fixes).
    tracker, seq = LocationTracker(cfg), 0
    for _ in range(cfg.LOCATION_START_FRAMES):
        seq += 1
        obs = tracker.update(ev(cross=-s, seq=seq, t=t), 0.0, t)
    assert obs.dist_to_centre_m == pytest.approx(-s)
    return tracker, seq


def test_zone_positions_follow_the_bounds():
    order = [Zone.CONTINUE, Zone.READ_SIGN, Zone.PAUSE, Zone.ENTERING, Zone.TURN, Zone.EXITING,
             Zone.CONTINUE]
    edges = (-1.0,) + config.ZONE_BOUNDS_M
    for i, s in enumerate(edges):
        assert zone_position(s + 0.001, config) == i, order[i]
    assert zone_position(-0.4, cfg_with(ZONE_REF_X_M=0.3)) == 3     # reference point ahead


def test_flare_fix_adds_flare_to_centre():
    assert camera_fix(ev(flare=1.0), config) == (-(1.0 + config.FLARE_TO_CENTRE_M), "flare")
    assert camera_fix(ev(cross=1.2, flare=0.9), config) == (-1.2, "crossing")
    assert camera_fix(ev(), config) is None


def test_pass_needs_start_frames_in_a_row():
    tracker = LocationTracker(config)
    seq = 0
    for _ in range(config.LOCATION_START_FRAMES - 1):
        seq += 1
        assert tracker.update(ev(cross=1.2, seq=seq), 0.0, 0.0).dist_to_centre_m is None
    seq += 1
    tracker.update(ev(seq=seq), 0.0, 0.0)                                  # run broken
    for _ in range(config.LOCATION_START_FRAMES - 1):
        seq += 1
        assert tracker.update(ev(cross=1.2, seq=seq), 0.0, 0.0).source == "none"
    seq += 1
    obs = tracker.update(ev(cross=1.2, seq=seq), 0.0, 0.0)
    assert obs.source == "crossing" and obs.zone == Zone.CONTINUE and obs.dist_to_centre_m == 1.2


def test_repeated_or_stale_frames_are_not_fixes():
    tracker, seq = started(config, s=-1.0)
    obs = tracker.update(ev(cross=0.9, seq=seq, t=0.0), 0.0, 0.01)        # same frame again
    assert obs.dist_to_centre_m == 1.0
    old = 0.02 - config.MAX_OBS_AGE_S - 0.01
    obs = tracker.update(ev(cross=0.9, seq=seq + 1, t=old), 0.0, 0.02)    # too old
    assert obs.dist_to_centre_m == 1.0 and obs.source == "dead_reckoning"


def test_dead_reckoning_moves_s_with_the_speed_command():
    cfg = cfg_with(SPEED_MPS_PER_UNIT=0.5)
    tracker, _ = started(cfg, s=-0.8)
    obs = tracker.update(None, 0.2, 1.0)                    # 1 s at 0.1 m/s
    assert obs.dist_to_centre_m == pytest.approx(0.7) and obs.source == "dead_reckoning"
    assert obs.conf == pytest.approx(1.0 - 0.1 / cfg.LOCATION_DR_CONF_M)


def test_unmeasured_speed_holds_s():
    tracker, _ = started(config, s=-0.8)
    assert config.SPEED_MPS_PER_UNIT is None
    assert tracker.update(None, 0.3, 5.0).dist_to_centre_m == pytest.approx(0.8)


def test_camera_fix_is_moved_on_by_its_age():
    cfg = cfg_with(SPEED_MPS_PER_UNIT=1.0)
    tracker, seq = started(cfg, s=-1.5)
    obs = tracker.update(ev(cross=1.2, seq=seq + 1, t=0.9), 0.1, 1.0)     # 0.1 s old at 0.1 m/s
    assert obs.dist_to_centre_m == pytest.approx(1.19)


def test_zones_never_step_back_in_a_pass():
    tracker, seq = started(config, s=-0.3)
    assert tracker.get().zone == Zone.READ_SIGN
    seq += 1
    assert tracker.update(ev(cross=0.25, seq=seq), 0.0, 0.0).zone == Zone.PAUSE
    seq += 1
    obs = tracker.update(ev(cross=0.35, seq=seq), 0.0, 0.0)               # fix puts s back
    assert obs.zone == Zone.PAUSE and obs.dist_to_centre_m == 0.35


def test_a_full_pass_reports_every_zone_in_order_then_waits():
    cfg = cfg_with(SPEED_MPS_PER_UNIT=0.5)
    tracker, seq = started(cfg, s=-1.6)
    zones, t = [], 0.0
    while t < 40.0:
        t += 1 / 30
        s_true = -1.6 + 0.1 * t
        seq += 1
        fix = ev(cross=-s_true, seq=seq, t=t) if -s_true >= 0.71 else ev(seq=seq, t=t)
        obs = tracker.update(fix, 0.2, t)
        if not zones or obs.zone != zones[-1]:
            zones.append(obs.zone)
        if obs.dist_to_centre_m is None:
            break
    assert zones == [Zone.CONTINUE, Zone.READ_SIGN, Zone.PAUSE, Zone.ENTERING, Zone.TURN,
                     Zone.EXITING, Zone.CONTINUE]
    assert -1.6 + 0.1 * t == pytest.approx(cfg.LOCATION_PASS_END_M, abs=0.01)
    assert tracker.get().source == "none"


def test_ready_flags_follow_the_zone():
    tracker, seq = started(config, s=-1.0)
    obs = tracker.get()
    assert not (obs.sign_read_ready or obs.pause_point_ready or obs.turn_point_ready)
    for dist, flags in ((0.5, (True, False, False)), (0.2, (True, True, False)),
                        (0.0, (True, True, True))):
        seq += 1
        obs = tracker.update(ev(cross=dist, seq=seq), 0.0, 0.0)
        assert (obs.sign_read_ready, obs.pause_point_ready, obs.turn_point_ready) == flags


def test_synthetic_push_through_the_junction_with_the_camera_alone():
    # Frames every 2 cm from 2 m before the centre: the tracker starts a
    # pass from the camera and is within 4 cm of the truth while a cue is
    # seen. The crossing line leaves the view (0.66 m) just inside READ_SIGN
    # (0.73 m): every later zone is placed by dead reckoning.
    tracker, errors = LocationTracker(config), []
    for i, s in enumerate(np.arange(-2.0, -0.715, 0.02)):
        result = process_front_frame(Frame(junction_scene(s), i * 0.1, i + 1, "front"), False, config)
        obs = tracker.update(result.location, 0.0, i * 0.1)
        if obs.source == "crossing":
            errors.append(obs.dist_to_centre_m + s)
    assert len(errors) > 20
    assert max(abs(e) for e in errors) < 0.04
    assert tracker.get().zone == Zone.READ_SIGN          # the line is seen to 0.66 m


# ---------------------------------------------------------------- replay

def test_replay_shows_cues_zone_and_measures_flare_to_centre(tmp_path, capsys):
    # A synthetic push from 2 m before the centre, recorded and replayed.
    from tools import replay
    from tools.recording import RecordingWriter
    rec = tmp_path / "junction"
    writer = RecordingWriter(str(rec), 1, 100)
    writer.start({"camera": "front"})
    positions = np.arange(-1.92, -0.715, 0.04)
    for i, s in enumerate(positions):
        writer.submit(Frame(junction_scene(s), i * 0.1, i + 1, "front"))
    writer.close()
    replay.run(str(rec), speed=0, window=False, cfg=config, save_dir=str(tmp_path / "out"))
    out = capsys.readouterr().out
    assert "cross" in out and "crossing" in out
    summary = out.strip().splitlines()[-1]
    assert summary.startswith("Flare to centre over")
    measured = float(summary.split("median ")[1].split()[0])
    # The flare test fires a little past where the widening starts, so the
    # measured value is shorter, and matches what junction_1-3 measured.
    assert WIDEN_START_M - 0.13 < measured <= WIDEN_START_M + 0.02
    assert measured == pytest.approx(config.FLARE_TO_CENTRE_M, abs=0.04)
    assert len(os.listdir(tmp_path / "out")) == len(positions)


# ---------------------------------------------------------------- real course frames

JUNCTION_REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "junction")
NO_SIGN_REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "no_sign")


def real_evidence(rec):
    from tools.recording import iter_frames
    return {f.seq: process_location(make_front_masks(f, config), config, f.t, f.seq)
            for f in iter_frames(rec)}


def test_real_junction_frames_see_both_cues():
    # Hand-checked on the pictures: the crossing line on the cross road's
    # dashes, the flare where the near corners have widened.
    found = real_evidence(JUNCTION_REC)
    for seq, cross, flare in ((1, 0.99, 0.82), (2, 1.22, 1.01), (3, 0.96, 0.79)):
        assert found[seq].crossing_dist_m == pytest.approx(cross, abs=0.03)
        assert found[seq].flare_dist_m == pytest.approx(flare, abs=0.03)
        assert found[seq].crossing_dist_m - found[seq].flare_dist_m == \
            pytest.approx(config.FLARE_TO_CENTRE_M, abs=0.04)


def test_real_frame_inside_the_junction_sees_the_line_and_no_flare():
    # junction_3 #503: the line 0.70 m ahead; the far corners beyond it once
    # read as a flare 0.83 m ahead.
    found = real_evidence(JUNCTION_REC)[4]
    assert found.crossing_dist_m == pytest.approx(0.70, abs=0.03)
    assert found.flare_dist_m is None


def test_real_straights_see_no_cue():
    # A speckled road mask, a yawed straight, a thick mat seam, and the
    # no-sign frames with the most clutter.
    found = {**real_evidence(JUNCTION_REC), **{-k: v for k, v in real_evidence(NO_SIGN_REC).items()}}
    for seq in (5, 6, 7, -1, -2, -3, -4):
        assert found[seq].crossing_dist_m is None and found[seq].flare_dist_m is None, seq


def test_tracker_ignores_a_fix_behind_the_robot():
    tracker, seq = started(config, s=-0.85)
    obs = tracker.update(ev(flare=0.83, seq=seq + 1), 0.0, 0.0)     # would put s at -1.01
    assert obs.dist_to_centre_m == pytest.approx(0.85)
    obs = tracker.update(ev(cross=0.90, seq=seq + 2), 0.0, 0.0)     # within LOCATION_BACK_MAX_M
    assert obs.dist_to_centre_m == pytest.approx(0.90)
