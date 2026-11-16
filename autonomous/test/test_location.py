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
from estimation.location import LocationTracker, camera_fix, dash_travel, zone_position  # noqa: E402
from perception.color import make_front_masks  # noqa: E402
from perception.location import (crossing_line_row, dash_positions, flare_row,  # noqa: E402
                                 process_location, road_widening, row_to_metres)
from perception.pipeline import process_front_frame  # noqa: E402
from robot_types import Frame, LocationEvidence, Zone  # noqa: E402

HALF_ROAD = config.ROAD_WIDTH_M / 2
WIDEN_START_M = config.JUNCTION_CORNER_M           # where the road starts to widen
CORNER_R = WIDEN_START_M - HALF_ROAD               # rounded corner radius
LINE_HALF_M = 0.012                                 # yellow line half width
DASH_LEN_M, DASH_PERIOD_M = 0.045, 0.115           # MEASURED on junction_3
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
    yellow = (np.abs(v) < LINE_HALF_M) & (np.mod(u, DASH_PERIOD_M) < DASH_LEN_M)
    if junction:
        yellow |= (np.abs(u) < LINE_HALF_M) & (np.mod(v, DASH_PERIOD_M) < DASH_LEN_M)
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


# ---------------------------------------------------------------- dashes

def true_dashes(s, near=0.70, far=1.25):
    # Centres (m ahead) of our dashes seen from s, inside [near, far].
    first = math.ceil((s + near - DASH_LEN_M / 2) / DASH_PERIOD_M)
    out = [k * DASH_PERIOD_M + DASH_LEN_M / 2 - s for k in range(first, first + 20)]
    return tuple(d for d in out if near <= d <= far)


@pytest.mark.parametrize("s", [-1.6, -1.0, -0.4, 0.0, 0.3])
def test_dash_centres_on_the_synthetic_line(s):
    # Through the junction too: the crossing line's dashes are not ours.
    found = process_front_frame(Frame(junction_scene(s), 0.0, 1, "front"), False,
                                config).location.dash_dists_m
    expected = true_dashes(s, 0.68, 1.27)
    assert len(found) >= len(expected) - 1 >= 3
    for d in found:
        assert min(abs(d - e) for e in true_dashes(s, 0.6, 1.4)) < 0.01


def test_no_dashes_without_a_lane_line():
    masks = make_front_masks(Frame(junction_scene(-1.0), 0.0, 1, "front"), config)
    assert dash_positions(masks.yellow, None, config) == ()
    assert process_location(masks, config, 0.0, 1).dash_dists_m == ()


DASHES = (0.70, 0.815, 0.93, 1.045, 1.16)
B = config.ZONE_BOUNDS_M


def mid(i):
    # s in the middle of zone position i (1 = READ_SIGN ... 5 = EXITING)
    return (B[i - 1] + B[i]) / 2


def moved(dashes, d):
    return tuple(x - d for x in dashes)


def test_dash_travel_finds_the_shift():
    assert dash_travel(DASHES, moved(DASHES, 0.02), config) == pytest.approx(0.02)
    assert dash_travel(DASHES, DASHES, config) == pytest.approx(0.0)
    # the nearest dash left the view and a new one came in at the far end
    assert dash_travel(DASHES, moved(DASHES, 0.03)[1:] + (1.245,), config) == pytest.approx(0.03)
    noisy = [x - 0.02 + n for x, n in zip(DASHES, (0.003, -0.004, 0.002, 0.0, -0.002))]
    assert dash_travel(DASHES, noisy, config) == pytest.approx(0.02, abs=0.004)


def test_dash_travel_refuses_what_it_cannot_tell():
    assert dash_travel(DASHES, (), config) is None
    assert dash_travel(DASHES, (0.80,), config) is None                 # one dash isn't enough
    # over half the spacing: can't be told from a step back, so never trusted
    assert dash_travel(DASHES, moved(DASHES, 0.08), config) is None


# ---------------------------------------------------------------- tracker

def ev(cross=None, flare=None, seq=1, t=0.0, dashes=()):
    return LocationEvidence(cross, flare, None, (), t, seq, tuple(dashes))


def started(cfg, s=-1.5, t=0.0, dashes=()):
    # A tracker that has begun a pass at s (crossing fixes).
    tracker, seq = LocationTracker(cfg), 0
    for _ in range(cfg.LOCATION_START_FRAMES):
        seq += 1
        obs = tracker.update(ev(cross=-s, seq=seq, t=t, dashes=dashes), 0.0, t)
    assert obs.dist_to_centre_m == pytest.approx(-s)
    return tracker, seq


def test_zone_positions_follow_the_bounds():
    order = [Zone.CONTINUE, Zone.READ_SIGN, Zone.PAUSE, Zone.ENTERING, Zone.TURN, Zone.EXITING,
             Zone.CONTINUE]
    edges = (-1.0,) + config.ZONE_BOUNDS_M
    for i, s in enumerate(edges):
        assert zone_position(s + 0.001, config) == i, order[i]
    entering = (config.ZONE_BOUNDS_M[2] + config.ZONE_BOUNDS_M[3]) / 2 - 0.3
    assert zone_position(entering, cfg_with(ZONE_REF_X_M=0.3)) == 3   # reference point ahead


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


def test_repeated_or_stale_frames_change_nothing():
    tracker, seq = started(config, s=-1.0)
    obs = tracker.update(ev(cross=0.9, seq=seq, t=0.0), 0.3, 0.01)        # same frame again
    assert obs.dist_to_centre_m == 1.0
    old = 0.02 - config.MAX_OBS_AGE_S - 0.01
    obs = tracker.update(ev(cross=0.9, seq=seq + 1, t=old), 0.3, 0.02)    # too old
    assert obs.dist_to_centre_m == 1.0 and obs.source == "crossing"


def test_dashes_carry_s_on_after_the_last_fix():
    tracker, seq = started(config, s=-0.70, dashes=DASHES)
    dashes = DASHES
    for i in range(1, 11):
        dashes = moved(dashes, 0.02)
        obs = tracker.update(ev(seq=seq + i, t=i / 30, dashes=dashes), 0.0, i / 30)
    assert obs.dist_to_centre_m == pytest.approx(0.50) and obs.source == "dashes"
    assert obs.conf == 1.0 and obs.zone == [None, Zone.READ_SIGN, Zone.PAUSE, Zone.ENTERING][
        zone_position(-0.50, config)]


def drive(tracker, seq, t, frames, cmd, step, dashes):
    # frames at 30 fps with the speed command cmd; dashes slide by step per
    # frame (dashes None: none seen). Returns (obs, seq, t, dashes).
    for _ in range(frames):
        seq, t = seq + 1, t + 1 / 30
        if dashes is not None:
            dashes = moved(dashes, step)
        obs = tracker.update(ev(seq=seq, t=t, dashes=dashes or ()), cmd, t)
    return obs, seq, t, dashes


def test_learned_speed_covers_frames_without_dashes():
    tracker, seq = started(config, s=-0.70, dashes=DASHES)
    assert tracker.learned_speed() is None
    obs, seq, t, _ = drive(tracker, seq, 0.0, 30, 0.25, 0.004, DASHES)    # 0.12 m/s at 0.25
    assert tracker.learned_speed() == pytest.approx(0.48)
    assert obs.dist_to_centre_m == pytest.approx(0.58)
    obs, seq, t, _ = drive(tracker, seq, t, 15, 0.25, 0.0, None)          # dashes lost for 0.5 s
    assert obs.dist_to_centre_m == pytest.approx(0.52) and obs.source == "speed"
    assert obs.conf == pytest.approx(1.0 - 0.06 / config.LOCATION_DR_CONF_M)


def test_stopped_frames_dont_teach_the_speed_and_nothing_known_holds():
    tracker, seq = started(config, s=-0.70, dashes=DASHES)
    obs, seq, t, _ = drive(tracker, seq, 0.0, 30, 0.0, 0.0, DASHES)        # a judging pause
    assert tracker.learned_speed() is None and obs.source == "dashes"
    obs, seq, t, _ = drive(tracker, seq, t, 10, 0.3, 0.0, None)            # driven, no dashes
    assert obs.dist_to_centre_m == pytest.approx(0.70) and obs.source == "held"


def test_zones_never_step_back_in_a_pass():
    tracker, seq = started(config, s=B[1] - 0.05)
    assert tracker.get().zone == Zone.READ_SIGN
    seq += 1
    assert tracker.update(ev(cross=-(B[1] + 0.03), seq=seq), 0.0, 0.0).zone == Zone.PAUSE
    seq += 1
    obs = tracker.update(ev(cross=-(B[1] - 0.05), seq=seq), 0.0, 0.0)     # fix puts s back
    assert obs.zone == Zone.PAUSE and obs.dist_to_centre_m == pytest.approx(-(B[1] - 0.05))


def test_a_pushed_pass_reports_every_zone_in_order_then_waits():
    # No speed command at all (pushed): the crossing line, then the dashes.
    s_true = -1.25
    tracker, seq = started(config, s=s_true, dashes=true_dashes(s_true))
    zones, t = [], 0.0
    while s_true < 2.0:
        s_true, t, seq = s_true + 0.004, t + 1 / 30, seq + 1
        cross = -s_true if -s_true >= 0.66 else None
        obs = tracker.update(ev(cross=cross, seq=seq, t=t, dashes=true_dashes(s_true)), 0.0, t)
        if not zones or obs.zone != zones[-1]:
            zones.append(obs.zone)
        if obs.dist_to_centre_m is None:
            break
        assert obs.dist_to_centre_m == pytest.approx(-s_true, abs=0.005)
    assert zones == [Zone.CONTINUE, Zone.READ_SIGN, Zone.PAUSE, Zone.ENTERING, Zone.TURN,
                     Zone.EXITING, Zone.CONTINUE]
    assert s_true == pytest.approx(config.LOCATION_PASS_END_M, abs=0.005)
    assert tracker.get().source == "none"


def test_ready_flags_follow_the_zone():
    tracker, seq = started(config, s=-1.0)
    obs = tracker.get()
    assert not (obs.sign_read_ready or obs.pause_point_ready or obs.turn_point_ready)
    for dist, flags in ((-mid(1), (True, False, False)), (-mid(2), (True, True, False)),
                        (-mid(4), (True, True, True))):
        seq += 1
        obs = tracker.update(ev(cross=dist, seq=seq), 0.0, 0.0)
        assert (obs.sign_read_ready, obs.pause_point_ready, obs.turn_point_ready) == flags


def test_synthetic_push_through_the_junction_with_the_camera_alone():
    # Frames every 2 cm from 2 m before the centre to 0.4 m past it, no
    # speed command: the crossing line, then our own dashes, keep s within
    # 3 cm of the truth through every zone.
    tracker, zones, errors = LocationTracker(config), [], []
    for i, s in enumerate(np.arange(-2.0, 0.4, 0.02)):
        result = process_front_frame(Frame(junction_scene(s), i * 0.1, i + 1, "front"), False, config)
        obs = tracker.update(result.location, 0.0, i * 0.1)
        if obs.dist_to_centre_m is not None:
            errors.append(obs.dist_to_centre_m + s)
        if not zones or obs.zone != zones[-1]:
            zones.append(obs.zone)
    assert zones == [Zone.CONTINUE, Zone.READ_SIGN, Zone.PAUSE, Zone.ENTERING, Zone.TURN,
                     Zone.EXITING, Zone.CONTINUE]
    assert len(errors) > 80 and max(abs(e) for e in errors) < 0.03


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


SEAM_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "seam")


@pytest.mark.parametrize("name", sorted(os.listdir(SEAM_DATA)))
@pytest.mark.parametrize("h_max", [30, 34])
def test_mat_seam_on_a_straight_is_no_junction(name, h_max):
    # straight_4: a dark seam across the mat, the next mat's grass a shade
    # yellower. At YELLOW_H_MAX 34 that grass passes the yellow test; only
    # yellow inside the road counts for the junction cues, so the seam is
    # not a flare or a crossing line either way.
    import cv2
    cfg = cfg_with(YELLOW_H_MAX=h_max)
    frame = Frame(cv2.imread(os.path.join(SEAM_DATA, name)), 0.0, 1, "front")
    ev = process_location(make_front_masks(frame, cfg), cfg, 0.0, 1)
    assert ev.flare_dist_m is None and ev.crossing_dist_m is None


def test_yellow_on_road_keeps_only_yellow_between_road():
    from perception.location import yellow_on_road
    road = np.zeros((3, 10), np.uint8)
    road[:, 2:8] = 255
    yellow = np.zeros((3, 10), np.uint8)
    yellow[0, 4:6] = 255                 # our line, on the road
    yellow[1, 0:10] = 255                # grass and road alike
    yellow[2, 8:10] = 255                # grass beside the road
    kept = yellow_on_road(yellow, road)
    assert kept[0].tolist() == yellow[0].tolist()
    assert np.flatnonzero(kept[1]).tolist() == list(range(2, 8))
    assert not kept[2].any()


def test_tracker_ignores_a_fix_behind_the_robot():
    tracker, seq = started(config, s=-0.85)
    obs = tracker.update(ev(flare=0.83, seq=seq + 1), 0.0, 0.0)     # would put s at -1.01
    assert obs.dist_to_centre_m == pytest.approx(0.85)
    obs = tracker.update(ev(cross=0.90, seq=seq + 2), 0.0, 0.0)     # within LOCATION_BACK_MAX_M
    assert obs.dist_to_centre_m == pytest.approx(0.90)


def test_false_flares_cannot_ratchet_s_back():
    # run_20261115_175843: the crossing line put s at -1.03, then flares
    # 2-8 cm behind (each within LOCATION_BACK_MAX_M) pulled it to -1.17.
    tracker, seq = started(config, s=-1.03)
    for i, flare in enumerate((0.92, 0.96, 0.99)):                 # s -1.10, -1.14, -1.17
        obs = tracker.update(ev(flare=flare, seq=seq + 1 + i), 0.0, 0.0)
        assert obs.dist_to_centre_m == pytest.approx(1.03)
    obs = tracker.update(ev(flare=0.70, seq=seq + 9), 0.0, 0.0)     # ahead: still ignored
    assert obs.dist_to_centre_m == pytest.approx(1.03) and obs.source != "flare"


def test_flare_moves_s_forward_only_before_the_crossing_line():
    tracker, seq = LocationTracker(config), 0
    for _ in range(config.LOCATION_START_FRAMES):                  # a pass begun on the flare
        seq += 1
        tracker.update(ev(flare=1.0, seq=seq), 0.0, 0.0)
    s0 = -(1.0 + config.FLARE_TO_CENTRE_M)
    obs = tracker.update(ev(flare=1.05, seq=seq + 1), 0.0, 0.0)     # behind: ignored
    assert -obs.dist_to_centre_m == pytest.approx(s0)
    obs = tracker.update(ev(flare=0.95, seq=seq + 2), 0.0, 0.0)     # ahead: used
    assert obs.source == "flare" and -obs.dist_to_centre_m == pytest.approx(s0 + 0.05)
    obs = tracker.update(ev(cross=1.2, seq=seq + 3), 0.0, 0.0)      # the line, within the allowance
    assert obs.source == "crossing" and obs.dist_to_centre_m == pytest.approx(1.2)


SEQ_REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "junction_seq")


def test_real_dash_odometry_matches_the_crossing_line():
    # junction_3 pushed towards the junction, rocking back ~2 cm on the way:
    # summed dash travel against how far the crossing line came closer.
    from perception.front_lane import LaneTracker
    from tools.recording import iter_frames
    lane, frames = LaneTracker(), []
    for f in iter_frames(SEQ_REC):
        frames.append(process_front_frame(f, False, config, lane).location)
    steps = [dash_travel(a.dash_dists_m, b.dash_dists_m, config) for a, b in zip(frames, frames[1:])]
    assert None not in steps and min(steps) < -0.005                   # the rock back is seen
    closer = frames[0].crossing_dist_m - frames[-1].crossing_dist_m
    assert closer > 0.15
    assert sum(steps) == pytest.approx(closer, abs=0.015)
    # Every frame has several marks to match. Where the crossing line meets
    # ours the runs differ from single dashes, but they are painted on the
    # ground too, so they slide by the same amount.
    assert all(len(f.dash_dists_m) >= 4 for f in frames)


# ---------------------------------------------------------------- sign leaving the view

def sign_obs(seq, box):
    from robot_types import SignObs, SignType
    return SignObs(SignType.LEFT if box else SignType.NONE, 1.0, box, seq / 30, seq)


EDGE_BOX, MID_BOX = (0, 60, 140, 105), (200, 60, 140, 105)


def feed_signs(tracker, seq, boxes, s):
    # One sign reading per frame, with a crossing fix at s.
    for box in boxes:
        seq += 1
        obs = tracker.update(ev(cross=-s, seq=seq), 0.0, 0.0, sign_obs(seq, box))
    return obs, seq


def test_read_sign_starts_when_the_sign_is_cut_off():
    tracker, seq = started(config, s=-1.1)
    obs, seq = feed_signs(tracker, seq, [MID_BOX, EDGE_BOX, None], -1.1)
    assert obs.zone == Zone.CONTINUE                       # one missed reading: not yet
    obs, seq = feed_signs(tracker, seq, [None] * (config.SIGN_CUT_READINGS - 1), -1.1)
    assert obs.zone == Zone.READ_SIGN and obs.sign_read_ready and tracker.sign_cut


def test_a_sign_lost_mid_view_or_flickering_is_not_cut_off():
    tracker, seq = started(config, s=-1.1)
    obs, seq = feed_signs(tracker, seq, [MID_BOX] + [None] * 5, -1.1)      # lost mid-view
    assert obs.zone == Zone.CONTINUE
    obs, seq = feed_signs(tracker, seq, [EDGE_BOX, None, EDGE_BOX, None, EDGE_BOX], -1.1)
    assert obs.zone == Zone.CONTINUE and not tracker.sign_cut
    # the same reading passed on again (paced sign reading) counts once
    tracker.update(ev(cross=1.1, seq=seq + 1), 0.0, 0.0, sign_obs(seq + 1, None))
    obs = tracker.update(ev(cross=1.1, seq=seq + 2), 0.0, 0.0, sign_obs(seq + 1, None))
    assert obs.zone == Zone.CONTINUE


def test_without_a_sign_read_sign_starts_at_its_latest_bound():
    tracker, seq = started(config, s=config.READ_SIGN_LATEST_M - 0.01)
    assert tracker.get().zone == Zone.CONTINUE
    obs = tracker.update(ev(cross=-(config.READ_SIGN_LATEST_M + 0.01), seq=seq + 1), 0.0, 0.0)
    assert obs.zone == Zone.READ_SIGN


def test_a_sign_cut_off_before_the_pass_counts_when_it_starts():
    tracker = LocationTracker(config)
    seq = 0
    for box in (EDGE_BOX, None, None):
        seq += 1
        tracker.update(ev(seq=seq), 0.0, 0.0, sign_obs(seq, box))
    assert tracker.sign_cut and tracker.get().dist_to_centre_m is None
    for _ in range(config.LOCATION_START_FRAMES):
        seq += 1
        obs = tracker.update(ev(cross=1.25, seq=seq), 0.0, 0.0)
    assert obs.zone == Zone.READ_SIGN


@pytest.mark.skipif(not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "recordings",
                                                   "junction_3")),
                    reason="needs recordings/junction_3 (not in git)")
def test_real_sign_cut_off_in_junction_3():
    # Hand-checked: the hexagon sign is first cut by the left edge at #385.
    from perception.front_lane import LaneTracker
    from perception.pipeline import SignPacer
    from perception.signs import load_sign_model
    from tools.recording import iter_frames
    rec = os.path.join(os.path.dirname(__file__), "..", "recordings", "junction_3")
    model, lane, pacer, tracker = (load_sign_model(config.SIGN_MODEL_PATH, config), LaneTracker(),
                                   SignPacer(), LocationTracker(config))
    for f in iter_frames(rec, 1, skip_missing=True):
        res = process_front_frame(f, True, config, lane, model, pacer)
        if tracker.update(res.location, 0.0, f.t, res.sign).zone == Zone.READ_SIGN:
            break
    assert 380 <= f.seq <= 392
