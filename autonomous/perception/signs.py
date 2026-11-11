"""Sign reading: find a sign by its black outline, read it with HOG + SVM.

Each Quiz 3 sign is a white card with a thin black outline around its
symbol: an arrow (LEFT, RIGHT, STRAIGHT) or a red octagon (PAUSE). The
outline is the candidate. Inside SIGN_ROI, thin dark lines are found on the
full-resolution frame (perception/color.mask_dark_lines), and a line whose
convex hull is a rectangle of the outline's aspect, well filled, and mostly
empty inside, is a sign outline. The hull makes a small break in the line
(motion blur) harmless. The biggest candidate is cropped from the full frame
(box grown by SIGN_BBOX_PAD_FRAC to take in the card's white margin),
normalised (grayscale, square, CLAHE), described by HOG and scored by a
linear SVM with a NONE class. A PAUSE also needs red in the crop.

A white-panel test was tried first (docs/quiz3_plan.md, section 7): it fails
when a lamp is in view (the card is then grey next to the glare) and when
the octagon touches the outline.

The SVM is trained on the laptop (tools/train_signs.py), which calls
crop_sign, normalise_crop and hog_features from here, so the Pi and the
laptop see identical crops. The Pi needs no scikit-learn: classifying is one
NumPy product.

Boxes are (x, y, w, h). SIGN_ROI, SignObs.roi and annotations are in view
pixels (FRONT_LANE_SIZE); candidates and SignObs.bbox are in full-frame
pixels, as in DESIGN.md.
"""
import os

import cv2
import numpy as np

from perception.color import mask_dark_lines, mask_red
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


def find_sign_candidates(line_mask, cfg):
    # Boxes of sign outlines in a thin-dark-line mask, biggest first, in the
    # mask's pixels. Inner and outer edges of each line are both tried: when
    # the outline touches something outside it (a hand, the card's shadow),
    # its inner edge is still a clean rectangle.
    height, width = line_mask.shape[:2]
    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        if (_whole(box, width, height, cfg) and _outline_shaped(contour, box, cfg)
                and ink_frac(line_mask, box) <= cfg.SIGN_INK_MAX):
            found.append((box[2] * box[3], box))
    found.sort(key=lambda item: -item[0])
    return [box for _, box in found]


def _whole(box, width, height, cfg):
    # False when the box reaches the edge of the searched region: the sign
    # is cut off there.
    x, y, w, h = box
    edge = cfg.SIGN_EDGE_PX
    return x >= edge and y >= edge and x + w <= width - edge and y + h <= height - edge


def _outline_shaped(contour, box, cfg):
    # Big enough, wider than tall, and its convex hull is a rectangle of the
    # outline's aspect (measured on the best-fit rotated rectangle, so a
    # tilted sign still passes).
    _, _, w, h = box
    if w * h < cfg.SIGN_MIN_AREA_PX or w <= h:
        return False
    hull = cv2.convexHull(contour)
    _, (side_a, side_b), _ = cv2.minAreaRect(hull)
    if min(side_a, side_b) <= 0:
        return False
    aspect = max(side_a, side_b) / min(side_a, side_b)
    fill = cv2.contourArea(hull) / (side_a * side_b)
    return cfg.SIGN_ASPECT_MIN <= aspect <= cfg.SIGN_ASPECT_MAX and fill >= cfg.SIGN_FILL_MIN


def ink_frac(line_mask, box):
    # Fraction of the box that is line. An outline with its symbol is mostly
    # empty; a dark textured patch is not.
    x, y, w, h = box
    return cv2.countNonZero(line_mask[y:y + h, x:x + w]) / (w * h)


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
    # A box in one image's pixels -> another's (view to full frame: x scale;
    # full frame to view: x 1 / scale).
    return tuple(int(round(v * scale)) for v in box)


def sign_boxes(frame, scale, cfg):
    # Every sign found in SIGN_ROI of this frame, biggest first, as padded
    # full-frame boxes. scale: full frame / view (FrontMasks.scale).
    height, width = frame.image.shape[:2]
    x0, y0, w, h = scale_box(sign_roi(cfg, int(round(width / scale)), int(round(height / scale))),
                             scale)
    lines = mask_dark_lines(frame.image[y0:y0 + h, x0:x0 + w], cfg)
    return [pad_box((x + x0, y + y0, bw, bh), cfg.SIGN_BBOX_PAD_FRAC, width, height)
            for x, y, bw, bh in find_sign_candidates(lines, cfg)]


# ---------------------------------------------------------------- crop and features

def crop_sign(frame, box):
    # The full-resolution crop under a full-frame box (already padded).
    x, y, w, h = box
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
    # says PAUSE without red in the crop. masks: the frame's FrontMasks (for
    # its view size and scale). model None (no model trained yet):
    # candidates only, always NONE.
    height, width = masks.bgr.shape[:2]
    roi = sign_roi(cfg, width, height)
    box = pick_sign(sign_boxes(frame, masks.scale, cfg))
    if box is None:
        return _obs(frame, SignType.NONE, 0.0, None, roi, (), cfg, masks.scale)
    if model is None:
        return _obs(frame, SignType.NONE, 0.0, box, roi, (), cfg, masks.scale)
    crop = crop_sign(frame, box)
    sign, margin, scores = classify_sign(hog_features(normalise_crop(crop, cfg), model.hog), model)
    if margin < cfg.SIGN_MARGIN_MIN:
        sign = SignType.NONE
    elif sign == SignType.PAUSE and not red_check(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), cfg):
        sign = SignType.NONE
    return _obs(frame, sign, margin, box, roi, scores, cfg, masks.scale)


def sign_label(sign, cfg):
    # The text shown (and later spoken) for a sign; "no sign" for NONE.
    return cfg.SIGN_PHRASES.get(sign.name, "no sign")


def _obs(frame, sign, margin, bbox, roi, scores, cfg, scale):
    # SignObs with its annotations (view px): the ROI, and the sign's box
    # with its label. bbox: padded full-frame box, or None.
    notes = [_rect(roi, cfg.SIGN_ROI_COLOUR)]
    if bbox is not None:
        colour = cfg.SIGN_NONE_COLOUR if sign == SignType.NONE else cfg.SIGN_BOX_COLOUR
        view_box = scale_box(bbox, 1.0 / scale)
        notes.append(_rect(view_box, colour))
        notes.append(Annotation(AnnotationKind.TEXT, ((view_box[0], max(0, view_box[1] - 2)),),
                                sign_label(sign, cfg), colour))
    return SignObs(sign=sign, score=margin, bbox=bbox, t=frame.t, seq=frame.seq,
                   roi=roi, scores=scores, annotations=tuple(notes))


def _rect(box, colour):
    x, y, w, h = box
    return Annotation(AnnotationKind.RECT, ((x, y), (x + w, y + h)), "", colour)
