"""Quiz 3 types and config (docs/quiz3_plan.md, stage 1): no robot needed."""
import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
import robot_types  # noqa: E402
from robot_types import (Action, Annotation, AnnotationKind, DriveCmd, FrontResult,  # noqa: E402
                         HealthWarning, LocationEvidence, LocationObs, MotorCmd, RunMode,
                         SignObs, SignType, SignVote, State, Surface, UiCommand, UiPacket,
                         VisionResult, Zone)


def names(enum):
    return [member.name for member in enum]


def test_enums():
    assert names(Zone) == ["CONTINUE", "READ_SIGN", "PAUSE", "ENTERING", "TURN", "EXITING"]
    assert names(RunMode) == ["DRIVE", "QUIZ_INTERSECTION", "QUIZ_SIGNS"]
    assert names(Surface) == ["GRASS", "ROAD", "YELLOW"]
    assert names(AnnotationKind) == ["RECT", "LINE", "POINT", "TEXT"]
    assert {"MODE_DRIVE", "MODE_INTERSECTION", "MODE_SIGNS"} <= set(names(UiCommand))
    assert "VISION_STALE" in names(HealthWarning)


def test_stop_is_renamed_pause():
    assert "PAUSE" in names(SignType) and "STOP" not in names(SignType)
    assert "PAUSE" in names(Action) and "STOP" not in names(Action)
    assert "STOP" in names(UiCommand)          # the stop button keeps its name


def test_location_evidence_replaces_intersection_obs():
    assert not hasattr(robot_types, "IntersectionObs")
    fields = [f.name for f in dataclasses.fields(FrontResult)]
    assert "location" in fields and "inter" not in fields
    evidence = LocationEvidence(0.9, None, Surface.ROAD, (), 1.0, 7)
    assert VisionResult(evidence, None, 1.0, 7).location.crossing_dist_m == 0.9


def test_sign_obs_new_fields_default_empty():
    obs = SignObs(SignType.LEFT, 1.2, (10, 20, 66, 40), 1.0, 7)
    assert obs.roi is None and obs.scores == () and obs.annotations == ()
    note = Annotation(AnnotationKind.RECT, ((1, 2), (3, 4)))
    assert note.cam_id == "front" and note.text == ""


def test_ui_packet_defaults_to_drive_mode():
    packet = UiPacket(t=0.0, state=State.IDLE, stop_reason=None, drive=DriveCmd(0.0, 0.0),
                      motor=MotorCmd(0.0, 0.0), lane=None, road_yaw=None, imu=None,
                      sign_vote=SignVote(SignType.PAUSE, None, 0.0), fps={}, overruns=0,
                      warnings=(), front=None, side=None)
    assert packet.mode == RunMode.DRIVE and packet.location is None
    obs = LocationObs(Zone.READ_SIGN, True, False, False, 0.5, 0.8, "crossing", 0.0)
    assert dataclasses.replace(packet, location=obs).location.zone == Zone.READ_SIGN


def test_zone_config():
    bounds = config.ZONE_BOUNDS_M
    assert len(bounds) == len(Zone) and list(bounds) == sorted(bounds)
    # the flare test fires a little past where the road starts to widen
    assert 0 < config.FLARE_TO_CENTRE_M < config.JUNCTION_CORNER_M
    assert config.LOCATION_PASS_END_M > bounds[-1]     # the pass ends only after "Continue travel"
    assert config.START_MODE in names(RunMode)
    assert config.VISION_WORKER_ENABLED is False


def test_labels_cover_their_enums():
    assert set(config.MODE_LABELS) == set(names(RunMode))
    assert config.MODE_LABELS["QUIZ_SIGNS"] == "Sign Reading"
    assert set(config.ZONE_PHRASES) == set(names(Zone))
    assert set(config.SIGN_PHRASES) == set(names(SignType)) - {"NONE"}


def test_ps5_buttons_name_real_commands():
    commands = set(names(UiCommand)) | {"START_STOP"}
    assert set(config.PS5_BUTTONS.values()) <= commands
    assert config.PS5_BUTTONS["reqCross"] == "START_STOP"
    for mode in ("MODE_DRIVE", "MODE_INTERSECTION", "MODE_SIGNS"):
        assert mode in config.PS5_BUTTONS.values()


def test_sign_config():
    for name in ("OCTAGON_CORNERS", "ARROW_ASYM_MIN", "SIGN_SIDE"):
        assert not hasattr(config, name)
    assert config.SIGN_ASPECT_MIN < 1.65 < config.SIGN_ASPECT_MAX
    assert config.SIGN_MODEL_PATH.endswith(".npz")
