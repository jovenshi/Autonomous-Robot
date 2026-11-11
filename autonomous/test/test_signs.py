"""Tests for perception/signs.py, tools/collect_signs.py, tools/train_signs.py,
the sign part of tools/replay.py and ui/overlay.draw_annotations.

No robot needed: signs are drawn synthetically, like the real ones (a white
card with a thin black outline around a black arrow or a red octagon), and
the no-sign frames in test/data/no_sign are real course frames. The test model
is a nearest-template linear model (weights = each class's HOG template,
bias = -|template|^2 / 2), written and read through the real model file.
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from perception import signs  # noqa: E402
from perception.color import make_front_masks  # noqa: E402
from robot_types import (Annotation, AnnotationKind, Frame, SignModel,  # noqa: E402
                         SignType)
from tools import collect_signs, replay, train_signs  # noqa: E402
from tools.recording import RecordingWriter  # noqa: E402

NO_SIGN_REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "no_sign")
GRASS = (40, 170, 40)                     # BGR
RED = (0, 0, 220)
CARD = (136, 104)                         # full-frame px: the white card
INSET, LINE = 8, 3                        # px: card edge to the outline, outline thickness
SIGN_KINDS = ("PAUSE", "LEFT", "RIGHT", "STRAIGHT")


# ---------------------------------------------------------------- synthetic scenes

def draw_symbol(img, kind, cx, cy, colour_override=None):
    # A black arrow, or a red octagon, centred on (cx, cy).
    if kind == "PAUSE":
        r = 30
        pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.pi / 8 + np.arange(8) * np.pi / 4]
        cv2.fillPoly(img, [np.int32(pts)], colour_override or RED)
        return
    head = {"LEFT": [(-36, 0), (-12, -22), (-12, 22)], "RIGHT": [(36, 0), (12, -22), (12, 22)],
            "STRAIGHT": [(0, -26), (-18, -6), (18, -6)]}[kind]
    shaft = {"LEFT": (-12, -7, 36, 7), "RIGHT": (-36, -7, 12, 7), "STRAIGHT": (-6, -6, 6, 26)}[kind]
    cv2.fillPoly(img, [np.int32([(cx + x, cy + y) for x, y in head])], (0, 0, 0))
    cv2.rectangle(img, (cx + shaft[0], cy + shaft[1]), (cx + shaft[2], cy + shaft[3]), (0, 0, 0), -1)


def sign_frame(kind=None, x=120, y=200, seq=1, outlined=True, octagon_colour=None, gap=0):
    # A 640 x 480 frame of grass with one sign card at (x, y), full-frame px.
    # gap: px missing from the middle of the outline's top line (motion blur).
    img = np.full((480, 640, 3), GRASS, np.uint8)
    if kind is not None:
        w, h = CARD
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), -1)
        if outlined:
            cv2.rectangle(img, (x + INSET, y + INSET), (x + w - 1 - INSET, y + h - 1 - INSET),
                          (0, 0, 0), LINE)
        if gap:
            cv2.rectangle(img, (x + w // 2 - gap // 2, y + INSET - LINE),
                          (x + w // 2 + gap // 2, y + INSET + LINE), (255, 255, 255), -1)
        draw_symbol(img, kind, x + w // 2, y + h // 2, octagon_colour)
    return Frame(img, 1.0 + seq / 30, seq, "front")


def outline_mask(*rects, size=(480, 640)):
    # A line mask with rectangle outlines whose outer edge is (x, y, w, h),
    # 3 px thick.
    mask = np.zeros(size, np.uint8)
    for x, y, w, h in rects:
        mask[y:y + h, x:x + w] = 255
        mask[y + 3:y + h - 3, x + 3:x + w - 3] = 0
    return mask


def masks_of(frame):
    return make_front_masks(frame, config, with_signs=True)


def raw_crop(frame):
    # The padded full-resolution crop, as collect_signs saves it.
    return signs.crop_sign(frame, signs.pick_sign(signs.sign_boxes(frame, 2.0, config)))


def normalised(frame):
    return signs.normalise_crop(raw_crop(frame), config)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    # Nearest-template model over every SignType, saved in the real file format.
    hog = train_signs.make_training_hog(config)
    classes = train_signs.CLASSES
    rows = [np.zeros(hog.getDescriptorSize(), np.float32) if name == "NONE"
            else signs.hog_features(normalised(sign_frame(name)), hog) for name in classes]
    weights = np.array(rows)
    biases = -0.5 * (weights ** 2).sum(axis=1)
    path = str(tmp_path_factory.mktemp("model") / "sign_svm.npz")
    train_signs.save_model(path, weights, biases, classes, config)
    return path


@pytest.fixture
def model(model_path):
    return signs.load_sign_model(model_path, config)


# ---------------------------------------------------------------- candidates

def near_card(box, x, y):
    bx, by, bw, bh = box
    return (abs(bx - x) <= 6 and abs(by - y) <= 6
            and abs(bw - CARD[0]) <= 10 and abs(bh - CARD[1]) <= 10)


def test_box_closely_encloses_the_card():
    boxes = signs.sign_boxes(sign_frame("LEFT", x=120, y=200), 2.0, config)
    assert len(boxes) >= 1 and near_card(boxes[0], 120, 200)


@pytest.mark.parametrize("frame", [
    sign_frame("LEFT", outlined=False),           # a white card with no outline
    sign_frame("LEFT", x=480),                    # outside SIGN_ROI
    sign_frame("LEFT", x=-10),                    # cut off by the frame edge
    sign_frame(None),                             # nothing there
])
def test_no_candidates(frame):
    assert signs.sign_boxes(frame, 2.0, config) == []


def test_outline_with_a_gap_is_still_found():
    boxes = signs.sign_boxes(sign_frame("STRAIGHT", gap=14), 2.0, config)
    assert boxes and near_card(boxes[0], 120, 200)


def test_tilted_sign_is_found():
    frame = sign_frame("RIGHT")
    turn = cv2.getRotationMatrix2D((188, 252), 8, 1.0)
    tilted = cv2.warpAffine(frame.image, turn, (640, 480), borderValue=GRASS)
    assert signs.sign_boxes(Frame(tilted, 1.0, 1, "front"), 2.0, config)


def test_shapes_that_are_not_a_sign_outline():
    good = (300, 300, 120, 90)
    mask = outline_mask((20, 20, 90, 90),          # square
                        (20, 150, 200, 60),        # too long
                        (400, 20, 30, 22),         # too small
                        good)
    cv2.line(mask, (20, 300), (140, 300), 255, 3)  # an L: its hull is a triangle
    cv2.line(mask, (20, 300), (20, 390), 255, 3)
    mask[400:470, 500:600] = 255                   # a solid dark patch: all ink
    # The outline's outer edge, then its inner edge; nothing else.
    assert signs.find_sign_candidates(mask, config) == [good, (302, 302, 116, 86)]


def test_outline_running_off_the_region_edge_is_cut_off():
    whole = outline_mask((30, 100, 120, 90))
    assert signs.find_sign_candidates(whole, config)[0] == (30, 100, 120, 90)
    cut = whole[:, 60:]                            # the left part is out of view
    assert signs.find_sign_candidates(np.ascontiguousarray(cut), config) == []


def test_biggest_candidate_first_and_picked():
    small, big = (20, 20, 80, 60), (200, 200, 160, 120)
    boxes = signs.find_sign_candidates(outline_mask(small, big), config)
    assert boxes[0] == big and small in boxes and boxes.index(small) > boxes.index(big)
    assert signs.pick_sign(boxes) == big and signs.pick_sign([]) is None


def test_ink_frac_and_pad_box_clip():
    mask = np.zeros((20, 20), np.uint8)
    mask[:, :10] = 255
    assert signs.ink_frac(mask, (5, 0, 10, 10)) == pytest.approx(0.5)
    assert signs.pad_box((0, 0, 10, 10), 0.5, 12, 12) == (0, 0, 12, 12)


def test_dark_lines_ignore_frame_brightness():
    # The same sign in a dim frame, and beside a glaring lamp, is still found.
    dim = (sign_frame("LEFT").image * 0.55).astype(np.uint8)
    assert signs.sign_boxes(Frame(dim, 1.0, 1, "front"), 2.0, config)
    lamp = dim.copy()
    cv2.circle(lamp, (560, 100), 60, (255, 255, 255), -1)
    assert signs.sign_boxes(Frame(lamp, 1.0, 1, "front"), 2.0, config)


def test_sign_roi_none_is_whole_view(monkeypatch):
    monkeypatch.setattr(config, "SIGN_ROI", None)
    assert signs.sign_roi(config, 320, 240) == (0, 0, 320, 240)
    assert signs.sign_boxes(sign_frame("LEFT", x=480), 2.0, config)
    monkeypatch.setattr(config, "SIGN_ROI", (300, 0, 100, 240))
    assert signs.sign_roi(config, 320, 240) == (300, 0, 20, 240)


# ---------------------------------------------------------------- crop, model, classify

def test_normalise_crop_is_square_gray():
    gray = normalised(sign_frame("RIGHT"))
    assert gray.shape == (config.SIGN_CROP_PX, config.SIGN_CROP_PX) and gray.dtype == np.uint8


def test_model_round_trip(model):
    assert model.classes == train_signs.CLASSES
    assert model.weights.shape == (len(model.classes), model.hog.getDescriptorSize())


def test_load_model_errors(tmp_path, model_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="train_signs"):
        signs.load_sign_model(str(tmp_path / "missing.npz"), config)
    monkeypatch.setattr(config, "SIGN_CROP_PX", 48)
    with pytest.raises(ValueError, match="64 px"):
        signs.load_sign_model(model_path, config)


def test_classify_sign_margin_and_scores():
    model = SignModel(hog=None, weights=np.eye(3, dtype=np.float32),
                      biases=np.zeros(3, np.float32), classes=("NONE", "LEFT", "PAUSE"))
    sign, margin, scores = signs.classify_sign(np.array([0.1, 0.9, 0.5], np.float32), model)
    assert sign == SignType.LEFT and margin == pytest.approx(0.4)
    assert dict(scores) == pytest.approx({"NONE": 0.1, "LEFT": 0.9, "PAUSE": 0.5})


def test_red_check():
    hsv = np.zeros((10, 10, 3), np.uint8)
    assert not signs.red_check(hsv, config)
    hsv[:5] = (0, 255, 200)
    assert signs.red_check(hsv, config)


# ---------------------------------------------------------------- process_signs

@pytest.mark.parametrize("kind", SIGN_KINDS)
@pytest.mark.parametrize("x, y", [(120, 200), (40, 300), (200, 120)])
def test_reads_every_sign_anywhere_in_the_roi(model, kind, x, y):
    frame = sign_frame(kind, x=x, y=y)
    obs = signs.process_signs(frame, masks_of(frame), model, config)
    assert obs.sign == SignType[kind]
    assert obs.score >= config.SIGN_MARGIN_MIN
    assert near_card(obs.bbox, x, y)
    assert (obs.t, obs.seq) == (frame.t, frame.seq)
    assert {name for name, _ in obs.scores} == set(train_signs.CLASSES)


def test_annotations_roi_box_and_label(model):
    frame = sign_frame("STRAIGHT")
    obs = signs.process_signs(frame, masks_of(frame), model, config)
    kinds = [a.kind for a in obs.annotations]
    assert kinds == [AnnotationKind.RECT, AnnotationKind.RECT, AnnotationKind.TEXT]
    assert obs.annotations[0].pts == ((0, 0), (176, 240))          # SIGN_ROI
    assert obs.annotations[2].text == config.SIGN_PHRASES["STRAIGHT"]
    assert obs.roi == tuple(config.SIGN_ROI)


def test_pause_needs_red(model):
    # Same grey level as the red octagon, so the SVM still says PAUSE.
    grey = int(round(0.299 * RED[2]))
    frame = sign_frame("PAUSE", octagon_colour=(grey, grey, grey))
    obs = signs.process_signs(frame, masks_of(frame), model, config)
    assert obs.sign == SignType.NONE and obs.bbox is not None
    assert max(obs.scores, key=lambda item: item[1])[0] == "PAUSE"


def test_low_margin_is_none(model, monkeypatch):
    monkeypatch.setattr(config, "SIGN_MARGIN_MIN", 1e9)
    frame = sign_frame("LEFT")
    assert signs.process_signs(frame, masks_of(frame), model, config).sign == SignType.NONE


def test_no_model_shows_candidate_only():
    frame = sign_frame("LEFT")
    obs = signs.process_signs(frame, masks_of(frame), None, config)
    assert obs.sign == SignType.NONE and obs.bbox is not None and obs.scores == ()


def test_no_candidate_is_none_with_roi_annotation(model):
    frame = sign_frame(None)
    obs = signs.process_signs(frame, masks_of(frame), model, config)
    assert obs.sign == SignType.NONE and obs.bbox is None and obs.score == 0.0
    assert [a.kind for a in obs.annotations] == [AnnotationKind.RECT]


# ---------------------------------------------------------------- no-sign recording

def test_no_sign_recording_reads_no_sign(model, capsys):
    # Real course frames with the most white clutter: no candidate at all.
    replay.run(NO_SIGN_REC, speed=0, window=False, cfg=config, model=model)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    assert all("sign no sign" in line and "no candidate" in line for line in lines)


def test_replay_sign_step_and_panel(model):
    frame = sign_frame("RIGHT")
    obs, crop = replay.sign_step(frame, model, config)
    assert obs.sign == SignType.RIGHT and crop.shape == (config.SIGN_CROP_PX,) * 2
    panel = replay.sign_panel(obs, crop, 240, config)
    assert panel.shape == (240, 240, 3)
    _, none_crop = replay.sign_step(sign_frame(None), model, config)
    assert none_crop is None


# ---------------------------------------------------------------- collect_signs

def write_recording(path, frames):
    writer = RecordingWriter(str(path), png_level=1, queue_size=100)
    writer.start({"camera": "front"})
    for frame in frames:
        writer.submit(frame)
    writer.close()


def test_collect_signs_saves_one_crop_per_sign_frame(tmp_path):
    rec = tmp_path / "signs_left_1"
    write_recording(rec, [sign_frame("LEFT", seq=1), sign_frame(None, seq=2),
                          sign_frame("RIGHT", seq=3)])
    out = tmp_path / "unsorted"
    assert collect_signs.collect(str(rec), str(out), config) == (3, 2)
    assert sorted(os.listdir(out)) == ["signs_left_1_000001.png", "signs_left_1_000003.png"]
    crop = cv2.imread(str(out / "signs_left_1_000001.png"))
    assert abs(crop.shape[1] - CARD[0]) <= 10 and abs(crop.shape[0] - CARD[1]) <= 10


def test_collect_signs_no_sign_recording(tmp_path):
    assert collect_signs.collect(NO_SIGN_REC, str(tmp_path), config, every_candidate=True) == (4, 0)


def test_collect_random_background_patches(tmp_path):
    out = tmp_path / "none"
    assert collect_signs.collect(NO_SIGN_REC, str(out), config, random_count=3) == (4, 12)
    names = sorted(os.listdir(out))
    assert names[0].endswith("_000001_r0.png") and train_signs.recording_of(names[0]) == "no_sign"
    rx, ry, rw, rh = signs.scale_box(signs.sign_roi(config, 320, 240), 2.0)
    for name in names:
        h, w = cv2.imread(str(out / name)).shape[:2]
        assert w * h >= 0.9 * config.SIGN_MIN_AREA_PX and w <= rw / 2 and w > h


# ---------------------------------------------------------------- train_signs (no scikit-learn)

def test_recording_of():
    assert train_signs.recording_of("signs_left_1_000123.png") == "signs_left_1"
    assert train_signs.recording_of("signs_left_1_000123_c2.png") == "signs_left_1"
    assert train_signs.recording_of("photo.png") is None


def test_load_samples_and_split(tmp_path):
    for name, rec in (("left", "a"), ("left", "b"), ("none", "b")):
        os.makedirs(tmp_path / name, exist_ok=True)
        cv2.imwrite(str(tmp_path / name / f"{rec}_000001.png"), np.zeros((8, 12, 3), np.uint8))
    samples = train_signs.load_samples(str(tmp_path))
    assert [(n, r) for _, n, r in samples] == [("NONE", "b"), ("LEFT", "a"), ("LEFT", "b")]
    train, test = train_signs.split_by_recording(samples, {"b"})
    assert [r for _, _, r in train] == ["a"] and len(test) == 2


def test_augment_never_flips():
    # An image bright on its left stays bright on its left in every copy.
    crop = np.zeros((80, 132, 3), np.uint8)
    crop[:, :50] = 200
    rng = np.random.default_rng(1)
    for _ in range(30):
        out = train_signs.augment(crop, rng, config).astype(float)
        assert out.shape == crop.shape
        assert out[:, :40].mean() > out[:, -40:].mean() + 50


def test_with_augmented_count():
    samples = [(np.zeros((40, 66, 3), np.uint8), "LEFT", "a")] * 2
    out = train_signs.with_augmented(samples, np.random.default_rng(0), config)
    assert len(out) == 2 * (1 + config.SIGN_AUG_COPIES)


def test_suggest_margin_and_confusion():
    predictions = [("LEFT", 2.0), ("LEFT", 0.3), ("RIGHT", 1.5), ("RIGHT", 0.2)]
    truth = ["LEFT", "NONE", "RIGHT", "NONE"]
    margin, acc = train_signs.suggest_margin(predictions, truth)
    assert acc == 1.0 and 0.3 < margin <= 1.5
    matrix = train_signs.confusion(["LEFT", "LEFT"], ["LEFT", "NONE"], ["NONE", "LEFT"])
    assert matrix.tolist() == [[0, 1], [0, 1]]


def test_fit_svm_reads_synthetic_signs(tmp_path):
    pytest.importorskip("sklearn")
    hog = train_signs.make_training_hog(config)
    samples = [(raw_crop(sign_frame(kind, x=40 + 40 * i)), kind, "a")
               for kind in SIGN_KINDS for i in range(3)]
    x, y = train_signs.feature_matrix(samples, hog, config)
    weights, biases, classes = train_signs.fit_svm(x, y, config)
    path = str(tmp_path / "m.npz")
    train_signs.save_model(path, weights, biases, classes, config)
    model = signs.load_sign_model(path, config)
    assert [n for n, _ in train_signs.predict(x, model)] == list(y)


# ---------------------------------------------------------------- draw_annotations

def test_draw_annotations_draws_each_kind():
    from ui.overlay import draw_annotations
    img = np.zeros((100, 100, 3), np.uint8)
    notes = (Annotation(AnnotationKind.RECT, ((10, 10), (30, 30)), "", (0, 255, 0)),
             Annotation(AnnotationKind.LINE, ((50, 10), (90, 10)), "", (255, 0, 0)),
             Annotation(AnnotationKind.POINT, ((50, 50),), "", (0, 0, 255)),
             Annotation(AnnotationKind.TEXT, ((10, 90),), "Left", (255, 255, 255)))
    draw_annotations(img, notes, scale=1.0)
    assert tuple(img[10, 20]) == (0, 255, 0)
    assert tuple(img[10, 70]) == (255, 0, 0)
    assert tuple(img[50, 50]) == (0, 0, 255)
    assert img[75:91, 10:40].any()
    scaled = np.zeros((200, 200, 3), np.uint8)
    draw_annotations(scaled, notes[:1], scale=2.0)
    assert tuple(scaled[20, 40]) == (0, 255, 0)


def test_train_flow_with_stand_in_fitter(tmp_path, monkeypatch, capsys):
    # The whole train() path (folders, holdout, augmentation, report, model
    # file) with a nearest-template fitter standing in for LinearSVC.
    def template_fit(x, y, cfg):
        classes = tuple(c for c in train_signs.CLASSES if c in set(y))
        weights = np.array([x[y == c].mean(axis=0) for c in classes], np.float32)
        return weights, -0.5 * (weights ** 2).sum(axis=1), classes

    monkeypatch.setattr(train_signs, "fit_svm", template_fit)
    monkeypatch.setattr(config, "SIGN_AUG_COPIES", 1)
    data = tmp_path / "signs_data"
    for kind in SIGN_KINDS:
        os.makedirs(data / kind.lower())
        for rec, x in (("rec_a", 40), ("rec_b", 100)):
            cv2.imwrite(str(data / kind.lower() / f"{rec}_000001.png"), raw_crop(sign_frame(kind, x=x)))
    out = str(tmp_path / "models" / "sign_svm.npz")
    train_signs.train(str(data), ["rec_b"], out, config)
    text = capsys.readouterr().out
    assert "Held out: 4 crops, accuracy 1.000" in text and "Suggested SIGN_MARGIN_MIN" in text
    assert signs.load_sign_model(out, config).classes == ("PAUSE", "LEFT", "RIGHT", "STRAIGHT")
