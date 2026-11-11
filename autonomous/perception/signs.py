"""Sign reading: find a sign panel by colour and shape, read it with HOG + SVM.

Each Quiz 3 sign is a white panel about 1.65 wide to tall inside a black
box: an arrow (LEFT, RIGHT, STRAIGHT) or a red octagon (PAUSE). Candidates
are white blobs inside SIGN_ROI that have the panel's aspect, fill their
bounding rectangle and have a dark ring around them. The biggest candidate
is cropped from the full-resolution frame (box grown by SIGN_BBOX_PAD_FRAC
to take in the black border), normalised (grayscale, square, CLAHE),
described by HOG and scored by a linear SVM with a NONE class. A PAUSE also
needs red in the crop.

The SVM is trained on the laptop (tools/train_signs.py), which calls
crop_sign, normalise_crop and hog_features from here, so the Pi and the
laptop see identical crops. The Pi needs no scikit-learn: classifying is one
NumPy product.

Boxes: (x, y, w, h). Candidates and the ROI are in view pixels
(FRONT_LANE_SIZE); SignObs.bbox is in full-frame pixels, as in DESIGN.md.
"""
import os

import cv2
import numpy as np

from perception.color import mask_red
from robot_types import Annotation, AnnotationKind, SignModel, SignObs, SignType

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # autonomous/


# ---------------------------------------------------------------- candidates

def sign_roi(cfg, width, height):
    # SIGN_ROI clipped to the view; the whole view when it is None.
    if cfg.SIGN_ROI is None:
        return (0, 0, width, height)
    x, y, w, h = cfg.SIGN_ROI
    x0, y0 = max(0, x), max(0, y)
    return (x0, y0, max(0, min(width, x + w) - x0), max(0, min(height, y + h) - y0))


def find_sign_candidates(white_mask, dark_mask, cfg):
    # Boxes of white blobs centred inside SIGN_ROI that pass the size,
    # aspect, fill and dark-border tests, biggest first. dark_mask: the
    # frame's road mask (dark and unsaturated), which the black box passes.
    height, width = white_mask.shape[:2]
    roi = sign_roi(cfg, width, height)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (_centre_inside(box, roi) and _panel_shaped(box, area, cfg)
                and border_dark_frac(dark_mask, box, cfg.SIGN_BORDER_PX) >= cfg.SIGN_BORDER_DARK_FRAC):
            found.append((area, box))
    found.sort(key=lambda item: -item[0])
    return [box for _, box in found]


def _centre_inside(box, roi):
    x, y, w, h = box
    rx, ry, rw, rh = roi
    cx, cy = x + w / 2, y + h / 2
    return rx <= cx < rx + rw and ry <= cy < ry + rh


def _panel_shaped(box, area, cfg):
    # Big enough, about the panel's aspect, and filling its rectangle (the
    # contour's area includes the arrow or octagon inside it).
    _, _, w, h = box
    if area < cfg.SIGN_MIN_AREA_PX or h == 0:
        return False
    return (cfg.SIGN_ASPECT_MIN <= w / h <= cfg.SIGN_ASPECT_MAX
            and area / (w * h) >= cfg.SIGN_FILL_MIN)


def border_dark_frac(dark_mask, box, ring_px):
    # Fraction of the ring ring_px wide around box that is dark. Only ring
    # pixels inside the image count; 0 when none are.
    height, width = dark_mask.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, x - ring_px), max(0, y - ring_px)
    x1, y1 = min(width, x + w + ring_px), min(height, y + h + ring_px)
    outer = dark_mask[y0:y1, x0:x1]
    inner = dark_mask[y:y + h, x:x + w]
    ring_total = outer.size - inner.size
    if ring_total <= 0:
        return 0.0
    ring_dark = cv2.countNonZero(outer) - cv2.countNonZero(inner)
    return ring_dark / ring_total


def pick_sign(candidates):
    # The biggest candidate: the nearest sign. candidates come biggest first.
    return candidates[0] if candidates else None


def pad_box(box, frac, width, height):
    # box grown by frac of its size on every side, clipped to the image.
    x, y, w, h = box
    dx, dy = int(round(w * frac)), int(round(h * frac))
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(width, x + w + dx), min(height, y + h + dy)
    return (x0, y0, x1 - x0, y1 - y0)


def scale_box(box, scale):
    # View-pixel box -> full-frame pixels.
    return tuple(int(round(v * scale)) for v in box)


# ---------------------------------------------------------------- crop and features

def crop_sign(frame, box, scale):
    # The full-resolution crop under a view-pixel box (already padded).
    x, y, w, h = scale_box(box, scale)
    return frame.image[y:y + h, x:x + w]


def normalise_crop(crop, cfg):
    # Grayscale, SIGN_CROP_PX square, then CLAHE. Training uses this too.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (cfg.SIGN_CROP_PX, cfg.SIGN_CROP_PX), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=cfg.SIGN_CLAHE_CLIP, tileGridSize=tuple(cfg.SIGN_CLAHE_TILES))
    return clahe.apply(gray)


def make_hog(crop_px, block_px, stride_px, cell_px, bins):
    # One HOG window covering the whole normalised crop.
    return cv2.HOGDescriptor((crop_px, crop_px), (block_px, block_px), (stride_px, stride_px),
                             (cell_px, cell_px), bins)


def hog_features(gray, hog):
    # HOG of one normalised crop as a float32 vector.
    return hog.compute(gray).ravel().astype(np.float32)


# ---------------------------------------------------------------- model

def load_sign_model(path, cfg):
    # Reads the .npz written by tools/train_signs.py. A relative path is
    # relative to autonomous/. Raises with a clear message when the file is
    # missing or was trained for another crop size.
    full = path if os.path.isabs(path) else os.path.join(_HERE, path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"Sign model {full} not found: train it with tools/train_signs.py")
    data = np.load(full)
    crop_px = int(data["crop_px"])
    if crop_px != cfg.SIGN_CROP_PX:
        raise ValueError(f"Sign model {full} was trained on {crop_px} px crops; "
                         f"SIGN_CROP_PX is {cfg.SIGN_CROP_PX}")
    classes = tuple(str(c) for c in data["classes"])
    unknown = [c for c in classes if c not in SignType.__members__]
    if unknown:
        raise ValueError(f"Sign model {full} has unknown classes {unknown}")
    hog = make_hog(crop_px, *(int(v) for v in data["hog"]))
    weights = data["weights"].astype(np.float32)
    if weights.shape != (len(classes), hog.getDescriptorSize()):
        raise ValueError(f"Sign model {full}: weights {weights.shape} do not match its HOG settings")
    return SignModel(hog=hog, weights=weights, biases=data["biases"].astype(np.float32),
                     classes=classes)


def classify_sign(features, model):
    # -> (SignType, margin, scores). Class scores = weights . features +
    # biases; the best class wins; margin = best minus runner-up; scores is
    # ((class name, score), ...) in the model's order.
    scores = model.weights @ features + model.biases
    order = np.argsort(scores)[::-1]
    margin = float(scores[order[0]] - scores[order[1]]) if len(scores) > 1 else float("inf")
    named = tuple((name, float(s)) for name, s in zip(model.classes, scores))
    return SignType[model.classes[order[0]]], margin, named


def red_check(crop_hsv, cfg):
    # True when at least SIGN_RED_MIN_FRAC of the crop is red.
    red = mask_red(crop_hsv, cfg)
    return red.size > 0 and cv2.countNonZero(red) / red.size >= cfg.SIGN_RED_MIN_FRAC


# ---------------------------------------------------------------- one frame

def process_signs(frame, masks, model, cfg):
    # Candidates -> pick -> crop -> normalise -> HOG -> classify. NONE when
    # there is no candidate, the margin is under SIGN_MARGIN_MIN, or the SVM
    # says PAUSE without red in the crop. masks must be built with_signs.
    # model None (no model trained yet): candidates only, always NONE.
    height, width = masks.white.shape[:2]
    roi = sign_roi(cfg, width, height)
    box = pick_sign(find_sign_candidates(masks.white, masks.road, cfg))
    if box is None:
        return _obs(frame, SignType.NONE, 0.0, None, roi, (), cfg)
    padded = pad_box(box, cfg.SIGN_BBOX_PAD_FRAC, width, height)
    if model is None:
        return _obs(frame, SignType.NONE, 0.0, padded, roi, (), cfg, masks.scale)
    crop = crop_sign(frame, padded, masks.scale)
    sign, margin, scores = classify_sign(hog_features(normalise_crop(crop, cfg), model.hog), model)
    if margin < cfg.SIGN_MARGIN_MIN:
        sign = SignType.NONE
    elif sign == SignType.PAUSE and not red_check(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), cfg):
        sign = SignType.NONE
    return _obs(frame, sign, margin, padded, roi, scores, cfg, masks.scale)


def sign_label(sign, cfg):
    # The text shown (and later spoken) for a sign; "no sign" for NONE.
    return cfg.SIGN_PHRASES.get(sign.name, "no sign")


def _obs(frame, sign, margin, view_box, roi, scores, cfg, scale=1.0):
    # SignObs with its annotations: the ROI, and the candidate box with its label.
    notes = [_rect(roi, cfg.SIGN_ROI_COLOUR)]
    bbox = None
    if view_box is not None:
        colour = cfg.SIGN_NONE_COLOUR if sign == SignType.NONE else cfg.SIGN_BOX_COLOUR
        notes.append(_rect(view_box, colour))
        x, y, _, _ = view_box
        notes.append(Annotation(AnnotationKind.TEXT, ((x, max(0, y - 2)),),
                                sign_label(sign, cfg), colour))
        bbox = scale_box(view_box, scale)
    return SignObs(sign=sign, score=margin, bbox=bbox, t=frame.t, seq=frame.seq,
                   roi=roi, scores=scores, annotations=tuple(notes))


def _rect(box, colour):
    x, y, w, h = box
    return Annotation(AnnotationKind.RECT, ((x, y), (x + w, y + h)), "", colour)
